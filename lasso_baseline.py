import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("features/all_subjects.csv")
feats = [c for c in df.columns if c.startswith(("emg_", "grf_", "rom_"))]


def lasso():
    # scaler + lasso (alpha tuned by inner 5-fold CV) are refit inside every outer fold
    return make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=50000, random_state=0))


def report(name, y, X, groups=None, cv=None):
    p_lasso = cross_val_predict(lasso(), X, y, groups=groups, cv=cv)
    p_mean = cross_val_predict(DummyRegressor(strategy="mean"), X, y, groups=groups, cv=cv)
    print(f"{name}: lasso MSE={mean_squared_error(y, p_lasso):.4f} | "
          f"predict-the-mean MSE={mean_squared_error(y, p_mean):.4f}")
    return p_lasso


print("=== Per subject, 10-fold CV ===")
for s, d in df.groupby("subject"):
    report(f"Sub{s}", d["target"].values, d[feats].values,
           cv=KFold(10, shuffle=True, random_state=0))

print("\n=== Pooled, leave-one-subject-out ===")
X, y, g = df[feats].values, df["target"].values, df["subject"].values
pred = report("All subjects", y, X, groups=g, cv=LeaveOneGroupOut())
for s in sorted(df["subject"].unique()):
    m = g == s
    print(f"  held-out Sub{s}: MSE={mean_squared_error(y[m], pred[m]):.4f}")