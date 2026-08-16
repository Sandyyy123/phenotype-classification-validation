"""Build the self-contained HTML sample report, injecting every number from
results.json and embedding the figure as base64 (no hand-typed statistics)."""
import json, base64, pathlib

here = pathlib.Path(__file__).parent
R = json.load(open(here / "results.json"))
png_b64 = base64.b64encode((here / "figure1_validation.png").read_bytes()).decode()

reg, dg, ba, cl = R["regression"], R["diagnostics"], R["bland_altman"], R["classification"]

def ci(pair):
    return f"{pair[0]:.3f} to {pair[1]:.3f}"

bp = dg["breusch_pagan_p"]
bp_txt = f"mild heteroscedasticity (Breusch-Pagan p = {bp:.3f})" if bp < 0.05 \
         else f"no evidence of heteroscedasticity (Breusch-Pagan p = {bp:.3f})"

html = f"""<meta charset="utf-8">
<title>Body-Composition Framework Validation</title>
<style>
  :root {{ --navy:#1f3b57; --acc:#c0392b; --ink:#1a1a1a; --muted:#5a6b7a; --line:#d9e0e6; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:Georgia,'Times New Roman',serif; color:var(--ink);
         background:#fff; line-height:1.62; margin:0; }}
  .wrap {{ max-width:840px; margin:0 auto; padding:44px 34px 72px; }}
  .sample-strip {{ background:#fff4e5; border:1px solid #f0c98a; color:#8a5a10;
         font-family:'Segoe UI',Arial,sans-serif; font-size:13px; padding:9px 14px;
         border-radius:6px; margin-bottom:30px; }}
  h1 {{ font-size:25px; line-height:1.3; color:var(--navy); margin:0 0 6px; }}
  .byline {{ font-family:'Segoe UI',Arial,sans-serif; font-size:14px; color:var(--muted);
         margin:0 0 4px; }}
  .rule {{ height:3px; background:var(--navy); width:70px; margin:16px 0 30px; }}
  h2 {{ font-family:'Segoe UI',Arial,sans-serif; font-size:13px; letter-spacing:.09em;
         text-transform:uppercase; color:var(--navy); margin:34px 0 10px;
         border-bottom:1px solid var(--line); padding-bottom:5px; }}
  p {{ margin:0 0 14px; }}
  .lead {{ font-size:16px; color:#33414d; }}
  figure {{ margin:26px 0 8px; }}
  figure img {{ width:100%; border:1px solid var(--line); border-radius:6px; }}
  figcaption {{ font-family:'Segoe UI',Arial,sans-serif; font-size:12.5px; color:var(--muted);
         line-height:1.55; margin-top:10px; }}
  .kv {{ font-family:'Segoe UI',Arial,sans-serif; font-size:13.5px; width:100%;
         border-collapse:collapse; margin:6px 0 16px; }}
  .kv td {{ border-bottom:1px solid var(--line); padding:7px 10px; vertical-align:top; }}
  .kv td:first-child {{ color:var(--muted); width:44%; }}
  .kv td:last-child {{ color:var(--ink); font-variant-numeric:tabular-nums; }}
  b.n {{ color:var(--navy); }}
  .foot {{ margin-top:46px; border-top:1px solid var(--line); padding-top:16px;
         font-family:'Segoe UI',Arial,sans-serif; font-size:13px; color:var(--muted); }}
  .foot .name {{ color:var(--navy); font-weight:600; }}
</style>
<div class="wrap">

  <div class="sample-strip"><b>Sample output: illustrative data only.</b>
     Prepared to show the format and level of rigor you would receive. Not the client's data.</div>

  <h1>Validating a Body-Composition Phenotype-Classification Framework</h1>
  <p class="byline">A worked example: regression and diagnostics, measurement agreement, and out-of-sample classification validity</p>
  <div class="rule"></div>

  <p class="lead">This short report demonstrates the distinction between <em>training a classifier</em>
  and <em>validating a classification framework</em>. Using an illustrative cohort of
  <b class="n">{R['n']}</b> adults with paired measurements of body fat by a reference method and a
  field proxy metric, it asks a measurement-science question: does the proxy discriminate a
  high-adiposity phenotype, is it calibrated out-of-sample, and can it substitute for the reference
  at the individual level?</p>

  <h2>Methods</h2>
  <p><b>Data and phenotype.</b> Paired reference (DXA percent body fat) and proxy measurements were
  available for {R['n']} adults. The target phenotype, high adiposity, was defined a priori as a
  reference value of at least 30% body fat (cohort prevalence {R['prevalence_pct']}%).</p>

  <p><b>Agreement and regression.</b> The proxy was regressed on the reference by ordinary least
  squares. Regression assumptions were checked with residual-versus-fitted plots and a LOWESS smoother
  (linearity), the Breusch-Pagan test (homoscedasticity), the Shapiro-Wilk test and a normal Q-Q plot
  (residual normality), the Durbin-Watson statistic (independence), Cook's distance against a 4/n
  threshold (influence), and variance inflation factors in the covariate-adjusted model
  (multicollinearity). Method agreement was quantified with a Bland-Altman analysis (mean bias and
  95% limits of agreement), and proportional bias was tested by regressing the paired differences on
  the paired means.</p>

  <p><b>Classification-framework validation.</b> A logistic model of the phenotype on the proxy was
  developed on a stratified 60% development sample (n = {R['n_dev']}) and applied, unchanged, to the
  held-out 40% validation sample (n = {R['n_val']}). On the validation sample we estimated
  discrimination (area under the ROC curve with a 2000-sample bootstrap 95% confidence interval),
  calibration (calibration curve by decile, calibration slope, calibration-in-the-large, and the
  Hosmer-Lemeshow test), and an operating point selected by Youden's J. Analyses used Python
  (numpy, pandas, statsmodels, scipy, matplotlib) under a fixed random seed; the full script is
  included for reproducibility.</p>

  <h2>Results</h2>
  <p><b>Agreement and diagnostics.</b> The proxy tracked the reference closely (slope
  <b class="n">{reg['slope']:.2f}</b>, 95% CI {reg['slope_ci'][0]:.2f} to {reg['slope_ci'][1]:.2f};
  R&sup2; = <b class="n">{reg['r2']:.3f}</b>; Pearson r = {reg['pearson_r']:.2f}). Residuals were
  approximately normal (Shapiro-Wilk p = {dg['shapiro_p']:.3f}) and showed no serial dependence
  (Durbin-Watson {dg['durbin_watson']:.2f}), with {bp_txt}. Cook's distance flagged
  {dg['cooks_influential_n']} of {R['n']} observations above the 4/n threshold, and multicollinearity
  was negligible (maximum VIF {dg['max_vif']:.2f}).</p>

  <table class="kv">
    <tr><td>Regression slope (95% CI)</td><td>{reg['slope']:.2f} ({reg['slope_ci'][0]:.2f}, {reg['slope_ci'][1]:.2f})</td></tr>
    <tr><td>R&sup2; / Pearson r</td><td>{reg['r2']:.3f} / {reg['pearson_r']:.2f}</td></tr>
    <tr><td>Breusch-Pagan p (homoscedasticity)</td><td>{dg['breusch_pagan_p']:.3f}</td></tr>
    <tr><td>Shapiro-Wilk p (residual normality)</td><td>{dg['shapiro_p']:.3f}</td></tr>
    <tr><td>Durbin-Watson (independence)</td><td>{dg['durbin_watson']:.2f}</td></tr>
    <tr><td>Influential points (Cook's &gt; 4/n)</td><td>{dg['cooks_influential_n']} of {R['n']}</td></tr>
  </table>

  <p><b>Measurement agreement.</b> The mean bias was {ba['bias']:.2f} percentage points, but the 95%
  limits of agreement were wide ({ba['loa_low']:.1f} to {ba['loa_high']:.1f}), and a significant
  proportional bias was present (difference-on-mean slope {ba['prop_bias_slope']:.3f},
  p {'&lt; 0.001' if ba['prop_bias_p'] < 0.001 else '= %.3f' % ba['prop_bias_p']}): the proxy
  increasingly over-reads at higher adiposity.</p>

  <p><b>Out-of-sample classification validity.</b> On the held-out validation sample the framework
  discriminated the phenotype well (AUC <b class="n">{cl['auc']:.3f}</b>, 95% CI {ci(cl['auc_ci'])})
  and was adequately calibrated (calibration slope {cl['cal_slope']:.2f}, calibration-in-the-large
  {cl['cal_intercept']:.2f}, Hosmer-Lemeshow p = {cl['hl_p']:.3f}). At the Youden-optimal threshold
  ({cl['youden_threshold']:.2f}) sensitivity was {cl['sens']:.3f} and specificity {cl['spec']:.3f}.</p>

  <table class="kv">
    <tr><td>AUC (95% CI), held-out n = {R['n_val']}</td><td>{cl['auc']:.3f} ({ci(cl['auc_ci'])})</td></tr>
    <tr><td>Calibration slope / intercept</td><td>{cl['cal_slope']:.2f} / {cl['cal_intercept']:.2f}</td></tr>
    <tr><td>Hosmer-Lemeshow p</td><td>{cl['hl_p']:.3f}</td></tr>
    <tr><td>Sensitivity / specificity at Youden's J</td><td>{cl['sens']:.3f} / {cl['spec']:.3f}</td></tr>
    <tr><td>Bland-Altman bias (95% LoA)</td><td>{ba['bias']:.2f} ({ba['loa_low']:.1f} to {ba['loa_high']:.1f})</td></tr>
  </table>

  <figure>
    <img src="data:image/png;base64,{png_b64}" alt="Six-panel validation figure">
    <figcaption><b>Figure 1.</b> Validation of a body-composition phenotype-classification framework
    on illustrative sample data (n = {R['n']}; held-out validation n = {R['n_val']}).
    (A) OLS regression of the proxy on the reference with 95% confidence band and identity line.
    (B) Residuals versus fitted values with a LOWESS smoother. (C) Normal Q-Q plot of residuals.
    (D) ROC curve on the held-out sample; the marked point is the Youden-optimal operating point.
    (E) Calibration curve by decile of predicted risk. (F) Bland-Altman plot of proxy minus
    reference against their mean, with mean bias (solid) and 95% limits of agreement (dashed).</figcaption>
  </figure>

  <h2>Interpretation</h2>
  <p>The proxy discriminates the high-adiposity phenotype strongly and is well calibrated out of
  sample, so as a <em>classification framework</em> for ranking or triage it is valid. Yet the wide
  limits of agreement and the proportional bias show it is <em>not</em> interchangeable with the
  reference for individual-level measurement. That is the core measurement-science point: a framework
  can be valid for classification while remaining an imperfect measurement instrument, and
  discrimination alone would hide the gap. A full engagement would add external validation on an
  independent cohort, decision-curve (net-benefit) analysis, and subgroup checks for construct
  validity across sex and age.</p>

  <div class="foot">
    <span class="name">Dr. Sandeep Grover</span> &nbsp;&middot;&nbsp; PhD, data science &nbsp;&middot;&nbsp;
    20+ peer-reviewed publications in statistical inference, genetic epidemiology and classification validation
    <br>Feel free to share with your team.
  </div>

</div>
"""
out_repo = here / "sample_report.html"
out_repo.write_text(html, encoding="utf-8")
dl = pathlib.Path("/mnt/c/Users/grove/Downloads/phenotype_classification_validation_sample.html")
dl.write_text(html, encoding="utf-8")
print("Wrote", out_repo)
print("Wrote", dl)
print("size KB:", round(len(html) / 1024, 1))
