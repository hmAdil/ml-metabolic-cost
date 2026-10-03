import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("features/all_subjects.csv")

feature_sets = {
    "EMG only": [c for c in df.columns if c.startswith("emg_")],
    "GRF + joint angles": [c for c in df.columns if c.startswith(("grf_", "rom_"))],
    "All": [c for c in df.columns if c.startswith(("emg_", "grf_", "rom_"))],
}
subjects = sorted(df["subject"].unique())


def lasso():
    # scaler and alpha tuning are refit inside every fold (no leakage)
    return make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=50000, random_state=0))


header = f"{'':20s}" + "".join(f"{'Sub' + str(s):>8s}" for s in subjects) + f"{'avg':>8s}{'LOSO':>9s}"
print("MSE (lower is better). Per-subject 10-fold CV, then pooled leave-one-subject-out (LOSO)")
print(header)

for name, cols in feature_sets.items():
    per_subject = []
    for s in subjects:
        d = df[df["subject"] == s]
        pred = cross_val_predict(lasso(), d[cols].values, d["target"].values,
                                 cv=KFold(10, shuffle=True, random_state=0))
        per_subject.append(mean_squared_error(d["target"].values, pred))
    pred = cross_val_predict(lasso(), df[cols].values, df["target"].values,
                             groups=df["subject"].values, cv=LeaveOneGroupOut())
    loso = mean_squared_error(df["target"].values, pred)
    line = f"{name:20s}" + "".join(f"{m:8.4f}" for m in per_subject)
    print(line + f"{sum(per_subject) / len(per_subject):8.4f}{loso:9.4f}")