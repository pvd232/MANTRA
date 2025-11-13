#!/usr/bin/env python3
# 03_synthetic_prelim_min.py  (MCH-only scaffolding)
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

FIGDIR = Path("figures")
FIGDIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)

# 1) PCA scatter (synthetic placeholder)
means = np.array([[0, 0], [3, 1.3], [-2.2, 2.5]])
covs = [
    np.array([[1.0, 0.2], [0.2, 0.6]]),
    np.array([[0.8, -0.1], [-0.1, 0.8]]),
    np.array([[0.7, 0.25], [0.25, 0.9]]),
]
X = np.vstack([rng.multivariate_normal(m, c, size=450) for m, c in zip(means, covs)])
labs = np.repeat([1, 2, 3], 450)
plt.figure(figsize=(6, 5))
for i in [1, 2, 3]:
    idx = labs == i
    plt.scatter(X[idx, 0], X[idx, 1], s=6, alpha=0.6, label=f"cluster {i}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(frameon=False)
plt.title("PCA")
plt.tight_layout()
plt.savefig(FIGDIR / "pca_scatter.png", dpi=180)
plt.close()

# 2) Baseline metrics bar (synthetic, MCH-only)
rows = [
    ("TWAS-only", 0.05, 0.24, 0.22),
    ("Program-mean", 0.08, 0.30, 0.28),
    ("Linear no-manifold", 0.12, 0.36, 0.34),
]
df = pd.DataFrame(rows, columns=["method", "r2", "pearson", "spearman"])
df.to_csv(FIGDIR / "baseline_metrics_MCH.csv", index=False)
x = np.arange(len(df := df))
plt.figure(figsize=(6.8, 4.2))
plt.bar(x - 0.2, df["r2"], width=0.18, label="R^2")
plt.bar(x, df["pearson"], width=0.18, label="Pearson")
plt.bar(x + 0.2, df["spearman"], width=0.18, label="Spearman")
plt.xticks(x, df["method"], rotation=10)
plt.ylabel("score")
plt.title("Baseline performance MCH")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(FIGDIR / "baseline_metrics_bar.png", dpi=180)
plt.close()

# 3) Calibration + ROC
n = 800
ytrue = rng.normal(0, 1, n)
ypred = 0.71 * ytrue + rng.normal(0, 0.58, n)  # slope ~0.73
plt.figure(figsize=(5.6, 4.6))
plt.scatter(ytrue, ypred, s=10, alpha=0.6)
m, b = np.polyfit(ytrue, ypred, 1)
xs = np.linspace(ytrue.min(), ytrue.max(), 100)
plt.plot(xs, m * xs + b, lw=2)
plt.xlabel("true ΔMCH")
plt.ylabel("predicted ΔMCH")
plt.title(f"Calibration (slope ≈ {m:.2f})")
plt.tight_layout()
plt.savefig(FIGDIR / "calibration_scatter.png", dpi=180)
plt.close()

y = rng.integers(0, 2, n)
s = y * 0.8 + rng.normal(0, 0.7, n)
fpr, tpr, _ = roc_curve(y, s)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y, s)
aupr = auc(rec, prec)
plt.figure(figsize=(5.2, 4.6))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "--", lw=1)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC sign correctness")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(FIGDIR / "roc_sign.png", dpi=180)
plt.close()
print(
    "Wrote figures to ./figures/: pca_scatter.png, baseline_metrics_bar.png, calibration_scatter.png, roc_sign.png"
)
