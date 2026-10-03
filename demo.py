import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description="Predict metabolic cost for a subject the model has never seen.")
parser.add_argument("--subject", type=int, default=1, help="subject to hold out (default: 1)")
args = parser.parse_args()

df = pd.read_csv("features/all_subjects.csv")
feats = [c for c in df.columns if c.startswith(("emg_", "grf_", "rom_"))]

if args.subject not in df["subject"].unique():
    raise SystemExit(f"Subject {args.subject} not in dataset. Available: {sorted(df['subject'].unique())}")

train = df[df["subject"] != args.subject]
test = df[df["subject"] == args.subject].reset_index(drop=True)

# Train on the other subjects only
model = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=50000, random_state=0))
model.fit(train[feats].values, train["target"].values)
pred = model.predict(test[feats].values)
actual = test["target"].values

# Results table
print(f"\nHeld-out Sub{args.subject} (model trained on subjects {sorted(int(s) for s in train['subject'].unique())})")
print(f"{'test':6s}{'actual':>9s}{'predicted':>11s}{'error':>9s}")
for t, a, p in zip(test["test"], actual, pred):
    print(f"{t:6s}{a:9.3f}{p:11.3f}{p - a:9.3f}")

mse = mean_squared_error(actual, pred)
mse_mean = mean_squared_error(actual, np.full_like(actual, train["target"].mean()))
print(f"\nMSE: {mse:.4f}   (predicting the training average instead: {mse_mean:.4f})")

# Which features did the lasso rely on most?
coefs = pd.Series(model[-1].coef_, index=feats)
top = coefs[coefs != 0].sort_values(key=abs, ascending=False).head(5)
print(f"Features kept by lasso: {int((coefs != 0).sum())} of {len(feats)}. Top 5 by weight:")
print(top.round(3).to_string())

# Plot
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
lim = [min(actual.min(), pred.min()) - 0.2, max(actual.max(), pred.max()) + 0.2]
ax[0].scatter(actual, pred)
ax[0].plot(lim, lim, "k--", linewidth=1)
ax[0].set(xlim=lim, ylim=lim, xlabel="Actual metabolic cost (W/kg)",
          ylabel="Predicted (W/kg)", title=f"Sub{args.subject}: predicted vs actual")
x = np.arange(1, len(actual) + 1)
ax[1].plot(x, actual, "o-", label="Actual")
ax[1].plot(x, pred, "s--", label="Predicted")
ax[1].set(xlabel="Test number (T1-T30)", ylabel="Metabolic cost (W/kg)",
          title=f"Per test (MSE = {mse:.3f})")
ax[1].legend()
plt.tight_layout()
os.makedirs("results", exist_ok=True)
out = os.path.join("results", f"demo_sub{args.subject}.png")
plt.savefig(out, dpi=150)
print(f"\nPlot saved to {out}")
plt.show()