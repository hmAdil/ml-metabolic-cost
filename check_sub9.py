"""
Checks two open hypotheses about subject 9 (see write-up, section 6.6).

Part A - offset hypothesis: is the leave-one-subject-out under-prediction for a held-out
         subject a roughly constant shift (a person-specific baseline) rather than random error?
Part B - EMG quality hypothesis: is subject 9's poor EMG-only result caused by one or a few
         EMG channels? (The dataset's own SNR fields are checked separately in snr_compare.py.)

Run from the repository root:  python check_sub9.py
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
df = pd.read_csv("features/all_subjects.csv")
feats = [c for c in df.columns if c.startswith(("emg_", "grf_", "rom_"))]
emg = [c for c in feats if c.startswith("emg_")]
subjects = sorted(df["subject"].unique())


def lasso():
    return make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=50000, random_state=0))


# ---------------- Part A ----------------
print("=" * 70)
print("PART A: is the held-out error a constant shift (offset)?")
print("=" * 70)
rows = []
preds = {}
for s in subjects:
    tr, te = df[df["subject"] != s], df[df["subject"] == s].reset_index(drop=True)
    m = lasso().fit(tr[feats].values, tr["target"].values)
    p = m.predict(te[feats].values)
    a = te["target"].values
    res = a - p                               # actual - predicted
    preds[s] = (a, p)
    # honest calibration: estimate the offset from the first k tests, test on the rest
    cal, raw = {}, {}
    for k in (1, 3, 5):
        off = res[:k].mean()
        cal[k] = mean_squared_error(a[k:], p[k:] + off)
        raw[k] = mean_squared_error(a[k:], p[k:])      # same tests, no calibration
    rows.append({
        "held_out": s,
        "MSE": mean_squared_error(a, p),
        "mean_resid": res.mean(),
        "std_resid": res.std(),
        "frac_under": (res > 0).mean(),
        "corr": np.corrcoef(a, p)[0, 1],
        "MSE_oracle_shift": mean_squared_error(a, p + res.mean()),
        "raw_k1": raw[1], "calib_k1": cal[1],
        "raw_k3": raw[3], "calib_k3": cal[3],
        "raw_k5": raw[5], "calib_k5": cal[5],
    })
A = pd.DataFrame(rows).set_index("held_out")
pd.set_option("display.width", 200)
print("\nmean_resid > 0 means the model predicts too LOW on average.")
print("frac_under = share of tests the model under-predicts; corr = actual vs predicted.\n")
print(A[["MSE", "mean_resid", "std_resid", "frac_under", "corr", "MSE_oracle_shift"]].round(3).to_string())
print("\nCalibration: estimate the shift from the first k tests of the held-out person, then score the")
print("remaining tests with and without it (raw_k vs calib_k are on the same tests, so they compare fairly):")
print(A[["raw_k1", "calib_k1", "raw_k3", "calib_k3", "raw_k5", "calib_k5"]].round(3).to_string())

# ---------------- Part B ----------------
print("\n" + "=" * 70)
print("PART B: is subject 9's weak EMG-only result driven by particular EMG channels?")
print("=" * 70)


def cv_mse(sub, cols):
    d = df[df["subject"] == sub]
    kf = KFold(10, shuffle=True, random_state=0)
    p = cross_val_predict(lasso(), d[cols].values, d["target"].values, cv=kf)
    return mean_squared_error(d["target"].values, p)


print("\nEMG-only 10-fold CV MSE per subject (this script's seed; may differ slightly from ablation.py):")
base = {s: cv_mse(s, emg) for s in subjects}
print({int(s): round(v, 3) for s, v in base.items()})

print("\nSubject 9, EMG-only, dropping one muscle at a time (MSE; baseline %.3f):" % base[9])
drop = {c: cv_mse(9, [x for x in emg if x != c]) for c in emg}
print(pd.Series(drop).sort_values().round(3).to_string())

print("\nSubject 9's EMG features vs the other four subjects (z = (Sub9 mean - others' mean) / others' std):")
oth, s9 = df[df["subject"] != 9], df[df["subject"] == 9]
z = ((s9[emg].mean() - oth[emg].mean()) / oth[emg].std()).sort_values(key=abs, ascending=False)
print(z.round(2).head(6).to_string())
print("\nWithin-person variability (coefficient of variation = std/mean) of each EMG feature, Sub9 vs others:")
cv = pd.DataFrame({"Sub9": s9[emg].std() / s9[emg].mean(),
                   "others_avg": pd.concat([df[df.subject == s][emg].std() / df[df.subject == s][emg].mean()
                                            for s in subjects if s != 9], axis=1).mean(axis=1)})
cv["ratio"] = cv["Sub9"] / cv["others_avg"]
print(cv.sort_values("ratio", ascending=False).round(2).head(6).to_string())

print("\n" + "-" * 70)
print("The dataset's EMG signal-quality (SNR) fields are not in features/all_subjects.csv.")
print("They are in Sub_N/SubN_segEnergetics.mat under .../EMG/SNR/<muscle>; see snr_compare.py.")