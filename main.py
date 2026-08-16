"""
Validating a body-composition phenotype-classification framework.

A self-contained, reproducible sample analysis that demonstrates the difference
between *training a classifier* and *validating a classification framework*:

  1. Regression of a field proxy metric against a reference method, with a full
     assumption-diagnostics battery (linearity, homoscedasticity, normality of
     residuals, independence, influence).
  2. Measurement-agreement analysis (Bland-Altman) between proxy and reference.
  3. Classification-framework validation of the proxy for a high-adiposity
     phenotype: discrimination (AUC + bootstrap 95% CI), calibration
     (calibration curve, slope, calibration-in-the-large, Hosmer-Lemeshow).

The data are synthetic and illustrative. Every statistic printed here is written
to results.json and injected into the sample report, so no number is hand-typed.

Author: Dr. Sandeep Grover
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RNG = np.random.default_rng(20260816)
N = 420
OUT_FIG_PNG = "figure1_validation.png"
OUT_FIG_SVG = "figure1_validation.svg"
OUT_JSON = "results.json"

# ---------------------------------------------------------------------------
# 1. Synthetic body-composition cohort
#    reference = DXA percent body fat (the reference method)
#    proxy     = a field estimate (e.g. anthropometric / bioimpedance-style),
#                built with a small constant bias, mild proportional bias, and
#                heteroscedastic measurement error, as real proxies behave.
# ---------------------------------------------------------------------------
age = RNG.normal(44, 12, N).clip(19, 78)
female = RNG.integers(0, 2, N)
reference = (18 + 0.18 * age + 8.5 * female + RNG.normal(0, 4.2, N)).clip(6, 55)

# proxy: constant bias +1.4, proportional bias (slope 0.93), error grows with fat
meas_error = RNG.normal(0, 1.6 + 0.06 * reference)
proxy = 1.4 + 0.93 * reference + meas_error

df = pd.DataFrame({"age": age, "female": female,
                   "reference": reference, "proxy": proxy})

# ---------------------------------------------------------------------------
# 2. Regression of proxy on reference + assumption diagnostics
# ---------------------------------------------------------------------------
X = sm.add_constant(df["reference"])
ols = sm.OLS(df["proxy"], X).fit()
slope = float(ols.params["reference"])
intercept = float(ols.params["const"])
slope_ci = [float(c) for c in ols.conf_int().loc["reference"]]
r2 = float(ols.rsquared)
pearson_r = float(np.sqrt(r2)) * (1 if slope > 0 else -1)

resid = ols.resid.values
fitted = ols.fittedvalues.values

# homoscedasticity: Breusch-Pagan
bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)
# normality of residuals: Shapiro-Wilk
shapiro_stat, shapiro_p = stats.shapiro(resid)
# independence: Durbin-Watson
dw = float(durbin_watson(resid))
# influence: Cook's distance
infl = ols.get_influence()
cooks = infl.cooks_distance[0]
cooks_thresh = 4.0 / N
n_influential = int(np.sum(cooks > cooks_thresh))
# multicollinearity (illustrative, on the adjusted model)
Xmulti = sm.add_constant(df[["reference", "age", "female"]])
vif = {Xmulti.columns[i]: float(variance_inflation_factor(Xmulti.values, i))
       for i in range(1, Xmulti.shape[1])}
max_vif = float(max(vif.values()))

# ---------------------------------------------------------------------------
# 3. Bland-Altman agreement (proxy vs reference)
# ---------------------------------------------------------------------------
diff = df["proxy"].values - df["reference"].values
mean_meas = (df["proxy"].values + df["reference"].values) / 2.0
bias = float(np.mean(diff))
sd_diff = float(np.std(diff, ddof=1))
loa_low = bias - 1.96 * sd_diff
loa_high = bias + 1.96 * sd_diff
# proportional bias: regress difference on mean
ba_fit = sm.OLS(diff, sm.add_constant(mean_meas)).fit()
ba_prop_slope = float(ba_fit.params[1])
ba_prop_p = float(ba_fit.pvalues[1])

# ---------------------------------------------------------------------------
# 4. Classification-framework validation
#    phenotype: high adiposity = reference >= 30% body fat
#    the proxy-based logistic model is the classification framework under test.
#    Discrimination and calibration are estimated on a HELD-OUT validation set
#    (stratified 60/40 split) so the numbers are out-of-sample, not apparent.
# ---------------------------------------------------------------------------
y_all = (df["reference"].values >= 30.0).astype(int)
prevalence = float(y_all.mean())

# stratified 60/40 development / validation split
dev_idx, val_idx = [], []
for cls in (0, 1):
    members = np.where(y_all == cls)[0]
    members = RNG.permutation(members)
    cut = int(round(0.6 * len(members)))
    dev_idx.extend(members[:cut]); val_idx.extend(members[cut:])
dev_idx = np.array(sorted(dev_idx)); val_idx = np.array(sorted(val_idx))
n_dev, n_val = len(dev_idx), len(val_idx)

# develop the framework on the development set only
Xc_dev = sm.add_constant(df["proxy"].values[dev_idx])
logit = sm.Logit(y_all[dev_idx], Xc_dev).fit(disp=0)
# apply, unchanged, to the held-out validation set
Xc_val = sm.add_constant(df["proxy"].values[val_idx])
p_hat = pd.Series(logit.predict(Xc_val))
y = y_all[val_idx]

def auc_score(y_true, scores):
    order = np.argsort(scores)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(scores)
    auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

auc = auc_score(y, p_hat.values)
# bootstrap 95% CI for AUC on the held-out validation set
boot = []
idx = np.arange(n_val)
for _ in range(2000):
    b = RNG.choice(idx, n_val, replace=True)
    if y[b].sum() in (0, y[b].shape[0]):
        continue
    boot.append(auc_score(y[b], p_hat.values[b]))
auc_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

# calibration slope + calibration-in-the-large
eps = 1e-6
lp = np.log((p_hat.values + eps) / (1 - p_hat.values + eps))
cal = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
cal_intercept = float(cal.params[0])
cal_slope = float(cal.params[1])

# Hosmer-Lemeshow (10 groups)
def hosmer_lemeshow(y_true, p, g=10):
    dfhl = pd.DataFrame({"y": y_true, "p": p})
    dfhl["grp"] = pd.qcut(dfhl["p"], g, duplicates="drop")
    obs = dfhl.groupby("grp", observed=True)["y"].sum()
    exp = dfhl.groupby("grp", observed=True)["p"].sum()
    n = dfhl.groupby("grp", observed=True)["y"].count()
    hl = (((obs - exp) ** 2) / (exp * (1 - exp / n))).sum()
    ddof = len(obs) - 2
    p_val = 1 - stats.chi2.cdf(hl, ddof)
    return float(hl), float(p_val), int(ddof)

hl_stat, hl_p, hl_df = hosmer_lemeshow(y, p_hat.values)

# operating point at Youden's J (grid spans 0..1 so the ROC anchors at both corners)
thr_grid = np.linspace(0.0, 1.0, 201)
sens_grid, spec_grid = [], []
for t in thr_grid:
    pred = (p_hat.values >= t).astype(int)
    tp = np.sum((pred == 1) & (y == 1)); fn = np.sum((pred == 0) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0)); fp = np.sum((pred == 1) & (y == 0))
    sens_grid.append(tp / (tp + fn)); spec_grid.append(tn / (tn + fp))
sens_grid = np.array(sens_grid); spec_grid = np.array(spec_grid)
youden = sens_grid + spec_grid - 1
j_idx = int(np.argmax(youden))
best_thr = float(thr_grid[j_idx]); best_sens = float(sens_grid[j_idx]); best_spec = float(spec_grid[j_idx])

results = {
    "n": N, "n_dev": n_dev, "n_val": n_val,
    "prevalence_pct": round(prevalence * 100, 1),
    "regression": {
        "slope": round(slope, 3), "slope_ci": [round(slope_ci[0], 3), round(slope_ci[1], 3)],
        "intercept": round(intercept, 2), "r2": round(r2, 3), "pearson_r": round(pearson_r, 3),
    },
    "diagnostics": {
        "breusch_pagan_p": round(bp_p, 3), "shapiro_p": round(shapiro_p, 3),
        "durbin_watson": round(dw, 2), "cooks_influential_n": n_influential,
        "cooks_threshold": round(cooks_thresh, 4), "max_vif": round(max_vif, 2),
    },
    "bland_altman": {
        "bias": round(bias, 2), "sd_diff": round(sd_diff, 2),
        "loa_low": round(loa_low, 2), "loa_high": round(loa_high, 2),
        "prop_bias_slope": round(ba_prop_slope, 3), "prop_bias_p": round(ba_prop_p, 3),
    },
    "classification": {
        "auc": round(auc, 3), "auc_ci": [round(auc_ci[0], 3), round(auc_ci[1], 3)],
        "cal_slope": round(cal_slope, 2), "cal_intercept": round(cal_intercept, 2),
        "hl_stat": round(hl_stat, 2), "hl_p": round(hl_p, 3), "hl_df": hl_df,
        "youden_threshold": round(best_thr, 2), "sens": round(best_sens, 3), "spec": round(best_spec, 3),
    },
}
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

# ---------------------------------------------------------------------------
# 5. Publication-grade multi-panel figure
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300,
})
NAVY = "#1f3b57"; ACC = "#c0392b"; GREY = "#7f8c8d"; FILL = "#4a6f8a"

fig = plt.figure(figsize=(11, 7.2))
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34,
              left=0.06, right=0.985, top=0.90, bottom=0.09)

# Panel A: regression proxy vs reference
axA = fig.add_subplot(gs[0, 0])
axA.scatter(df["reference"], df["proxy"], s=14, alpha=0.5, color=FILL, edgecolor="none")
xs = np.linspace(df["reference"].min(), df["reference"].max(), 100)
pred = ols.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
axA.plot(xs, pred["mean"], color=NAVY, lw=1.8)
axA.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"], color=NAVY, alpha=0.15)
axA.plot(xs, xs, ls="--", color=GREY, lw=1.0)
axA.set_xlabel("Reference (DXA % body fat)"); axA.set_ylabel("Proxy metric (% body fat)")
axA.set_title("A  Proxy vs reference regression", loc="left")
axA.text(0.04, 0.94, f"$R^2$ = {r2:.3f}\nslope = {slope:.2f} ({slope_ci[0]:.2f}, {slope_ci[1]:.2f})",
         transform=axA.transAxes, va="top", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, alpha=0.8))

# Panel B: residuals vs fitted
axB = fig.add_subplot(gs[0, 1])
axB.scatter(fitted, resid, s=14, alpha=0.5, color=FILL, edgecolor="none")
axB.axhline(0, color=NAVY, lw=1.2)
sm_l = sm.nonparametric.lowess(resid, fitted, frac=0.6)
axB.plot(sm_l[:, 0], sm_l[:, 1], color=ACC, lw=1.5)
axB.set_xlabel("Fitted values"); axB.set_ylabel("Residuals")
axB.set_title("B  Residuals vs fitted", loc="left")
axB.text(0.04, 0.06, f"Breusch-Pagan p = {bp_p:.3f}", transform=axB.transAxes,
         va="bottom", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, alpha=0.8))

# Panel C: Normal Q-Q
axC = fig.add_subplot(gs[0, 2])
(osm, osr), (sl, ic, _) = stats.probplot(resid, dist="norm")
axC.scatter(osm, osr, s=14, alpha=0.5, color=FILL, edgecolor="none")
axC.plot(osm, sl * osm + ic, color=NAVY, lw=1.6)
axC.set_xlabel("Theoretical quantiles"); axC.set_ylabel("Sample quantiles")
axC.set_title("C  Normal Q-Q of residuals", loc="left")
axC.text(0.04, 0.94, f"Shapiro-Wilk p = {shapiro_p:.3f}", transform=axC.transAxes,
         va="top", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, alpha=0.8))

# Panel D: ROC
axD = fig.add_subplot(gs[1, 0])
fpr = 1 - spec_grid
order = np.argsort(fpr)
axD.plot(fpr[order], sens_grid[order], color=NAVY, lw=1.9)
axD.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1.0)
axD.scatter([1 - best_spec], [best_sens], color=ACC, zorder=5, s=30)
axD.set_xlabel("1 - specificity"); axD.set_ylabel("Sensitivity")
axD.set_title(f"D  Discrimination, held-out (n={n_val})", loc="left")
axD.text(0.96, 0.08, f"AUC = {auc:.3f}\n({auc_ci[0]:.3f}, {auc_ci[1]:.3f})",
         transform=axD.transAxes, ha="right", va="bottom", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, alpha=0.8))

# Panel E: calibration curve
axE = fig.add_subplot(gs[1, 1])
cal_df = pd.DataFrame({"y": y, "p": p_hat.values})
cal_df["bin"] = pd.qcut(cal_df["p"], 10, duplicates="drop")
grp = cal_df.groupby("bin", observed=True).agg(obs=("y", "mean"), pred=("p", "mean"))
axE.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1.0)
axE.plot(grp["pred"], grp["obs"], marker="o", color=NAVY, lw=1.6, ms=5)
axE.set_xlabel("Predicted probability"); axE.set_ylabel("Observed frequency")
axE.set_title(f"E  Calibration, held-out (n={n_val})", loc="left")
axE.text(0.04, 0.94, f"slope = {cal_slope:.2f}\nintercept = {cal_intercept:.2f}\nH-L p = {hl_p:.3f}",
         transform=axE.transAxes, va="top", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, alpha=0.8))

# Panel F: Bland-Altman
axF = fig.add_subplot(gs[1, 2])
axF.scatter(mean_meas, diff, s=14, alpha=0.5, color=FILL, edgecolor="none")
axF.axhline(bias, color=NAVY, lw=1.5)
axF.axhline(loa_high, color=ACC, ls="--", lw=1.2)
axF.axhline(loa_low, color=ACC, ls="--", lw=1.2)
axF.set_xlabel("Mean of proxy and reference (% body fat)"); axF.set_ylabel("Proxy - reference")
axF.set_title("F  Measurement agreement (Bland-Altman)", loc="left")
axF.text(0.04, 0.94, f"bias = {bias:.2f}\nLoA {loa_low:.1f} to {loa_high:.1f}",
         transform=axF.transAxes, va="top", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, alpha=0.8))

fig.suptitle("Validation of a body-composition phenotype-classification framework (illustrative sample data)",
             fontsize=12, fontweight="bold", y=0.975)
fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_FIG_SVG, bbox_inches="tight", facecolor="white")
print(f"\nSaved {OUT_FIG_PNG}, {OUT_FIG_SVG}, {OUT_JSON}")
