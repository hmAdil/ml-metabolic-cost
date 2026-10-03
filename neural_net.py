import warnings
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")  # lbfgs convergence chatter

df = pd.read_csv("features/all_subjects.csv")
feats = [c for c in df.columns if c.startswith(("emg_", "grf_", "rom_"))]


def lasso():
    return make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=50000, random_state=0))


def neural_net():
    # one hidden layer, tanh, linear output; scaler and target scaling are refit per fold
    base = make_pipeline(
        StandardScaler(),
        MLPRegressor(activation="tanh", solver="lbfgs", max_iter=5000, random_state=0),
    )
    model = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
    grid = {
        "regressor__mlpregressor__hidden_layer_sizes": [(2,), (3,), (4,), (6,), (8,)],
        "regressor__mlpregressor__alpha": [0.01, 0.1, 1.0],
    }
    inner = KFold(5, shuffle=True, random_state=0)
    return GridSearchCV(model, grid, cv=inner, scoring="neg_mean_squared_error")


def evaluate(X, y, groups=None, cv=None):
    preds = {
        "lasso": cross_val_predict(lasso(), X, y, groups=groups, cv=cv, n_jobs=-1),
        "nn": cross_val_predict(neural_net(), X, y, groups=groups, cv=cv, n_jobs=-1),
        "mean": cross_val_predict(DummyRegressor(strategy="mean"), X, y, groups=groups, cv=cv, n_jobs=-1),
    }
    return preds


def mse(y, p):
    return mean_squared_error(y, p)


if __name__ == "__main__":
    print("=== Per subject, 10-fold CV (MSE) ===")
    print(f"{'':6s} {'lasso':>8s} {'NN':>8s} {'mean':>8s}")
    for s, d in df.groupby("subject"):
        y = d["target"].values
        p = evaluate(d[feats].values, y, cv=KFold(10, shuffle=True, random_state=0))
        print(f"Sub{s:<3d} {mse(y, p['lasso']):8.4f} {mse(y, p['nn']):8.4f} {mse(y, p['mean']):8.4f}")

    print("\n=== Pooled, leave-one-subject-out (MSE) ===")
    X, y, g = df[feats].values, df["target"].values, df["subject"].values
    p = evaluate(X, y, groups=g, cv=LeaveOneGroupOut())
    print(f"{'All':6s} {mse(y, p['lasso']):8.4f} {mse(y, p['nn']):8.4f} {mse(y, p['mean']):8.4f}")
    for s in sorted(df["subject"].unique()):
        m = g == s
        print(f"Sub{s:<3d} {mse(y[m], p['lasso'][m]):8.4f} {mse(y[m], p['nn'][m]):8.4f} {mse(y[m], p['mean'][m]):8.4f}")