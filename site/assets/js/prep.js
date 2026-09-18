/* =============================================================================
   prep.js — turns window.SURGE_DATA into the matrices every page fits on,
   and exposes ONE consistent penalty scale so Ridge, Lasso and Elastic Net
   are directly comparable on the same λ slider.

   Common objective (identical to scikit-learn's ElasticNet):

       (1/2n)·‖y − Xw‖²  +  α·ρ·‖w‖₁  +  (α(1−ρ)/2)·‖w‖²

   ρ = 1 → Lasso, ρ = 0 → Ridge, 0 < ρ < 1 → Elastic Net.
   Note ML.ridge() solves (XᵀX + cI)⁻¹Xᵀy, so the closed form needs c = n·α.
============================================================================= */
(function (global) {
  'use strict';
  const D = global.SURGE_DATA;
  const col = name => D.columns.indexOf(name);

  const FEATURES = [
    {key: 'is_peak',                label: 'is_peak',        pretty: 'Peak hour',        group: 'demand'},
    {key: 'is_rain',                label: 'is_rain',        pretty: 'Rain',             group: 'weather'},
    {key: 'traffic_speed_kmph',     label: 'traffic_speed',  pretty: 'Traffic speed',    group: 'weather'},
    {key: 'drivers_available_500m', label: 'drivers_avail',  pretty: 'Drivers nearby',   group: 'supply'},
    {key: 'is_event_nearby',        label: 'is_event',       pretty: 'Event nearby',     group: 'demand'},
    {key: 'is_airport_pickup',      label: 'is_airport',     pretty: 'Airport pickup',   group: 'demand'},
    {key: 'is_weekend',             label: 'is_weekend',     pretty: 'Weekend',          group: 'noise'},
    {key: 'open_requests_500m',     label: 'open_requests',  pretty: 'Open requests',    group: 'demand'},
    {key: 'is_bad_weather',         label: 'is_bad_weather', pretty: 'Bad weather (derived)', group: 'weather'}
  ];
  const BASE8 = FEATURES.slice(0, 8).map(f => f.key);
  const TARGET = 'surge_additive_inr';
  const N_TRAIN = 150;

  const raw = name => D.rows.map(r => r[col(name)]);

  /** Build design matrix + target for a list of feature keys, split by time. */
  function build(keys) {
    const idx = keys.map(col);
    const yi = col(TARGET);
    const X = D.rows.map(r => idx.map(j => r[j]));
    const y = D.rows.map(r => r[yi]);
    const Xtr = X.slice(0, N_TRAIN), Xte = X.slice(N_TRAIN);
    const ytr = y.slice(0, N_TRAIN), yte = y.slice(N_TRAIN);
    const sc = ML.standardize(Xtr);
    const Zte = ML.applyScale(Xte, sc.mean, sc.std);
    const c = ML.center(ytr);
    return {
      keys, X, y, Xtr, Xte, ytr, yte,
      Ztr: sc.Z, Zte, mean: sc.mean, std: sc.std,
      yc: c.yc, yBar: c.mean, n: Xtr.length, p: keys.length,
      labels: keys.map(k => (FEATURES.find(f => f.key === k) || {label: k}).label)
    };
  }

  /** Fit on the standardized training matrix. Returns coefficients in scaled space. */
  function fit(d, method, alpha, l1Ratio) {
    const n = d.n;
    switch (method) {
      case 'ols':   return ML.ols(d.Ztr, d.yc).w;
      case 'ridge': return ML.ridge(d.Ztr, d.yc, Math.max(alpha, 0) * n).w;
      case 'lasso': return ML.elasticNet(d.Ztr, d.yc, Math.max(alpha, 1e-9), 1.0).w;
      case 'enet':  return ML.elasticNet(d.Ztr, d.yc, Math.max(alpha, 1e-9),
                                         l1Ratio == null ? 0.5 : l1Ratio).w;
      default: throw new Error('unknown method ' + method);
    }
  }

  /** Scores for a coefficient vector in scaled space. */
  function score(d, w) {
    const trHat = ML.predict(d.Ztr, w, d.yBar);
    const teHat = ML.predict(d.Zte, w, d.yBar);
    return {
      trainRMSE: ML.rmse(d.ytr, trHat), testRMSE: ML.rmse(d.yte, teHat),
      trainR2: ML.r2(d.ytr, trHat), testR2: ML.r2(d.yte, teHat),
      testMAE: ML.mae(d.yte, teHat),
      nonzero: w.filter(v => Math.abs(v) > 1e-8).length,
      trHat, teHat
    };
  }

  /** Coefficients converted back to raw units (₹ per unit of the original feature). */
  const unscale = (d, w) => w.map((v, j) => v / d.std[j]);

  /** Predict a single hand-built row given in RAW feature units. */
  function predictRow(d, w, rowObj) {
    let s = d.yBar;
    d.keys.forEach((k, j) => { s += w[j] * ((rowObj[k] - d.mean[j]) / d.std[j]); });
    return s;
  }

  /**
   * Fit at a given SHRINKAGE BUDGET s = ‖w‖₁ / ‖w_OLS‖₁  (the x-axis used in
   * Elements of Statistical Learning, Fig 3.8/3.10).  Comparing Ridge and Lasso
   * at equal α is misleading — a quadratic penalty and an absolute-value penalty
   * bite at completely different rates — so we instead binary-search the α that
   * puts each estimator at the same total coefficient mass.  What remains visible
   * is the only thing that actually differs: how each one ALLOCATES that mass.
   */
  function fitAtBudget(d, method, s, l1Ratio) {
    const l1 = w => w.reduce((a, v) => a + Math.abs(v), 0);
    const wOls = fit(d, 'ols');
    const target = s * l1(wOls);
    if (s >= 0.999) return wOls;
    if (s <= 0.0005) return wOls.map(() => 0);
    let lo = 1e-6, hi = 1e4;
    for (let k = 0; k < 8 && l1(fit(d, method, hi, l1Ratio)) > target; k++) hi *= 10;
    let w = wOls;
    for (let it = 0; it < 44; it++) {
      const mid = Math.sqrt(lo * hi);
      w = fit(d, method, mid, l1Ratio);
      if (l1(w) > target) lo = mid; else hi = mid;
    }
    return {w, alpha: Math.sqrt(lo * hi)};
  }

  /** Log-spaced penalty grid. */
  const lamGrid = (lo, hi, m = 36) =>
    Array.from({length: m}, (_, i) => lo * Math.pow(hi / lo, i / (m - 1)));

  global.Prep = {D, FEATURES, BASE8, TARGET, N_TRAIN, col, raw,
                 build, fit, score, unscale, predictRow, lamGrid, fitAtBudget};
})(window);
