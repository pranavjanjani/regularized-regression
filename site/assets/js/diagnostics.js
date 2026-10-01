/* =============================================================================
   diagnostics.js — model selection and solver diagnostics.

   Conventions, so the numbers line up with the rest of the site:
     objective      (1/2n)‖y−Zw‖² + αρ‖w‖₁ + (α(1−ρ)/2)‖w‖²
     ridge closed form needs n·α  (ML.ridge takes the sklearn-style constant)
     df(λ)          ridge: Σ dⱼ²/(dⱼ²+λ_sk) with λ_sk = n·α
                    lasso/enet: the number of non-zero coefficients
                    (Zou, Hastie & Tibshirani 2007)
     AIC/BIC        n·log(RSS/n) + {2, log n}·df   — dropped constants, so only
                    differences along the λ axis are meaningful
============================================================================= */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const F = (v, d = 3) => (v == null || !isFinite(v)) ? '—' : v.toFixed(d);
  const KEYS = Prep.FEATURES.map(f => f.key);
  const COL = {ridge: 'var(--c-ridge)', lasso: 'var(--c-lasso)', enet: 'var(--c-enet)'};

  const S = {method: 'ridge', alpha: 0.1, n: 150};
  let D, SV;                                  // data + singular values of Ztr

  function rebuild() {
    const full = Prep.build(KEYS);
    const Xtr = full.Xtr.slice(0, S.n);
    const ytr = full.ytr.slice(0, S.n);
    const sc = ML.standardize(Xtr);
    const c = ML.center(ytr);
    D = {
      Ztr: sc.Z, ytr, yc: c.yc, yBar: c.mean,
      Zte: ML.applyScale(full.Xte, sc.mean, sc.std), yte: full.yte,
      labels: full.labels, keys: KEYS, n: Xtr.length, p: KEYS.length,
      mean: sc.mean, std: sc.std,
      ysd: Math.sqrt(ML.mean(ytr.map(v => (v - c.mean) ** 2)))
    };
    const G = ML.matmul(ML.T(D.Ztr), D.Ztr);
    SV = ML.eigSym(G).values.map(v => Math.sqrt(Math.max(v, 0)));   // singular values of Z
    D.gram = G.map(r => r.map(v => v / D.n));
  }

  const rho = () => S.method === 'ridge' ? 0 : S.method === 'lasso' ? 1 : 0.5;

  function fitAt(alpha) {
    return S.method === 'ridge'
      ? ML.ridge(D.Ztr, D.yc, alpha * D.n).w
      : ML.elasticNet(D.Ztr, D.yc, Math.max(alpha, 1e-9), rho()).w;
  }
  const rss = w => D.yc.reduce((s, v, i) =>
    s + (v - D.Ztr[i].reduce((a, z, j) => a + z * w[j], 0)) ** 2, 0);
  const l1 = w => w.reduce((a, v) => a + Math.abs(v), 0);
  const l2sq = w => w.reduce((a, v) => a + v * v, 0);
  const nnz = w => w.filter(v => Math.abs(v) > 1e-8).length;

  /** Effective degrees of freedom. */
  function dfAt(alpha, w) {
    if (S.method === 'ridge') {
      const lam = alpha * D.n;
      return SV.reduce((s, d) => s + d * d / (d * d + lam), 0);
    }
    return nnz(w);
  }

  const GRID = () => Prep.lamGrid(0.001, 100, 36);

  /** One pass over the λ grid, collecting everything the page needs. */
  function sweep() {
    const out = [];
    GRID().forEach(a => {
      const w = fitAt(a);
      const r = rss(w);
      const df = dfAt(a, w);
      const trHat = ML.predict(D.Ztr, w, D.yBar);
      const teHat = ML.predict(D.Zte, w, D.yBar);
      const data = r / (2 * D.n);
      const pen = rho() * a * l1(w) + (1 - rho()) * a / 2 * l2sq(w);
      out.push({a, w, rss: r, df,
        aic: D.n * Math.log(r / D.n) + 2 * df,
        bic: D.n * Math.log(r / D.n) + Math.log(D.n) * df,
        trainRMSE: ML.rmse(D.ytr, trHat), testRMSE: ML.rmse(D.yte, teHat),
        l1: l1(w), l2: Math.sqrt(l2sq(w)), nnz: nnz(w), data, pen, obj: data + pen});
    });
    return out;
  }

  function cvCurve() {
    return GRID().map(a => ({a, cv: ML.kFoldCV(D.Ztr, D.ytr, 5, (Z, t) =>
      S.method === 'ridge' ? ML.ridge(Z, t, a * Z.length).w
                           : ML.elasticNet(Z, t, Math.max(a, 1e-9), rho()).w)}));
  }

  const argmin = (arr, key) => arr.reduce((b, x) => (b == null || key(x) < key(b)) ? x : b, null);
  const norm01 = ys => {
    const lo = Math.min(...ys), hi = Math.max(...ys);
    return ys.map(v => hi > lo ? (v - lo) / (hi - lo) : 0);
  };

  /* ======================================================================
     1 · four criteria
  ====================================================================== */
  let SWEEP = null, CV = null;
  function drawSelection() {
    SWEEP = sweep();
    CV = cvCurve();
    const xs = SWEEP.map(r => r.a);
    const series = [
      {name: '5-fold CV', color: 'var(--accent)', ys: CV.map(r => r.cv)},
      {name: 'AIC', color: 'var(--c-ridge)', ys: SWEEP.map(r => r.aic)},
      {name: 'BIC', color: 'var(--c-enet)', ys: SWEEP.map(r => r.bic)},
      {name: 'held-out (not for selecting)', color: 'var(--fg-3)',
       ys: SWEEP.map(r => r.testRMSE), dashed: true}
    ].map(s => ({name: s.name, color: s.color, dashed: s.dashed,
                 points: norm01(s.ys).map((v, i) => [xs[i], v])}));

    const bestCV = argmin(CV, r => r.cv);
    const bestAIC = argmin(SWEEP, r => r.aic);
    const bestBIC = argmin(SWEEP, r => r.bic);
    const bestTE = argmin(SWEEP, r => r.testRMSE);

    Plot.line($('selChart'), series, {height: 340, logX: true, dots: false,
      xLabel: 'penalty α (log scale)', yLabel: 'criterion, rescaled to [0,1]',
      yDomain: [-0.04, 1.06], vLine: bestCV.a, vLabel: 'CV'});
    /* extra verticals for the other three */
    const svg = $('selChart').querySelector('svg');
    const g = svg.querySelector('g');
    const m = {l: 54, r: 16, t: 16, b: 42};
    const iw = svg.viewBox.baseVal.width - m.l - m.r, ih = 340 - m.t - m.b;
    const lo = Math.log10(xs[0]), hi = Math.log10(xs[xs.length - 1]);
    const sx = a => (Math.log10(a) - lo) / (hi - lo) * iw;
    const NS = 'http://www.w3.org/2000/svg';
    [[bestAIC.a, 'AIC', 'var(--c-ridge)'], [bestBIC.a, 'BIC', 'var(--c-enet)'],
     [bestTE.a, 'held-out', 'var(--fg-3)']].forEach(([a, lab, col], k) => {
      const ln = document.createElementNS(NS, 'line');
      ln.setAttribute('x1', sx(a)); ln.setAttribute('x2', sx(a));
      ln.setAttribute('y1', 0); ln.setAttribute('y2', ih);
      ln.setAttribute('stroke', col); ln.setAttribute('stroke-width', 1.5);
      ln.setAttribute('stroke-dasharray', '5 3'); ln.setAttribute('opacity', .85);
      g.appendChild(ln);
      const t = document.createElementNS(NS, 'text');
      t.setAttribute('x', sx(a) + 4); t.setAttribute('y', 26 + k * 13);
      t.setAttribute('fill', col); t.setAttribute('font-size', 10);
      t.setAttribute('font-family', 'var(--mono)');
      t.textContent = lab;
      g.appendChild(t);
    });
    Plot.legend($('selLeg'), series.map(s => ({name: s.name, color: s.color})));

    const rows = [
      ['5-fold cross-validation', bestCV.a, 'CV RMSE ₹' + F(bestCV.cv, 2),
       'Resamples the training set. Makes no assumption about df, but costs 5 refits per α.'],
      ['AIC', bestAIC.a, F(bestAIC.aic, 1),
       'Penalizes 2 per degree of freedom. Aims at prediction, and tends to keep more.'],
      ['BIC', bestBIC.a, F(bestBIC.bic, 1),
       'Penalizes log n ≈ ' + F(Math.log(D.n), 1) + ' per df. Aims at the true model, and keeps less.'],
      ['Held-out error', bestTE.a, '₹' + F(bestTE.testRMSE, 2),
       'Shown for reference only. Selecting on it spends the test set and makes the reported number optimistic.']
    ];
    $('selTable').querySelector('tbody').innerHTML = rows.map(([name, a, val, why], i) => {
      const r = SWEEP.reduce((b, x) => Math.abs(x.a - a) < Math.abs(b.a - a) ? x : b, SWEEP[0]);
      const grey = i === 3 ? ' style="color:var(--fg-3)"' : '';
      return `<tr${grey}><td>${name}</td><td class="num">${F(a, 4)}</td>
        <td class="num">${F(r.df, 2)}</td><td class="num">${val}</td>
        <td class="num">₹${F(r.testRMSE, 2)}</td><td class="small">${why}</td></tr>`;
    }).join('');

    const span = [bestAIC.a, bestBIC.a, bestCV.a];
    const teOver = span.map(a =>
      SWEEP.reduce((b, x) => Math.abs(x.a - a) < Math.abs(b.a - a) ? x : b, SWEEP[0]).testRMSE);
    const spread = Math.max(...teOver) - Math.min(...teOver);
    $('selNote').innerHTML =
      `The three usable criteria choose α = ${F(bestAIC.a, 4)} (AIC), ${F(bestCV.a, 4)} (CV) and `
      + `${F(bestBIC.a, 4)} (BIC) — and in that order, which is not luck: BIC charges `
      + `log n ≈ ${F(Math.log(D.n), 1)} per degree of freedom against AIC's 2, so it can never `
      + `choose less regularization.</p><p>Across the whole range they disagree over, held-out `
      + `error moves by <strong>₹${F(spread, 2)}</strong>. `
      + (spread < 1
          ? 'That is nothing on fares of a few hundred rupees. The criteria are arguing about a '
            + 'choice that does not matter, which is itself the finding worth reporting — and the '
            + 'reason to take the simplest model in the region rather than the exact minimum of '
            + 'any one curve.'
          : 'That is a real gap, so the choice of criterion matters here. Say which you used and '
            + 'why.');
  }

  /* ======================================================================
     2 · effective df
  ====================================================================== */
  function drawDf() {
    const pts = SWEEP.map(r => [r.a, r.df]);
    Plot.line($('dfChart'), [{name: 'df(λ)', color: COL[S.method], points: pts}],
      {height: 280, logX: true, dots: false, yDomain: [0, D.p * 1.08],
       xLabel: 'penalty α (log scale)', yLabel: 'effective degrees of freedom',
       vLine: S.alpha, vLabel: 'α = ' + F(S.alpha, 3)});

    const lam = S.alpha * D.n;
    const items = SV.map((d, j) => ({
      label: 'd' + (j + 1) + ' = ' + F(d, 1),
      value: S.method === 'ridge' ? d * d / (d * d + lam) : 1,
      color: COL[S.method]
    }));
    Plot.barsH($('svChart'), items, {labelWidth: 96, height: 280, xDomain: [0, 1.05]});

    const w = fitAt(S.alpha);
    const dfNow = dfAt(S.alpha, w);
    $('dfNote').innerHTML = S.method === 'ridge'
      ? `At α = ${F(S.alpha, 3)} the model is using <strong>${F(dfNow, 2)} of ${D.p}</strong> `
        + `degrees of freedom. Note the bars: the largest direction keeps `
        + `${F(100 * SV[0] * SV[0] / (SV[0] * SV[0] + lam), 0)}% of its length while the smallest `
        + `keeps ${F(100 * SV[D.p - 1] * SV[D.p - 1] / (SV[D.p - 1] * SV[D.p - 1] + lam), 0)}%. `
        + `Ridge spends its budget almost entirely on the directions the data pins down.`
      : `For an ℓ₁ penalty, df is simply the count of surviving coefficients — currently `
        + `<strong>${dfNow} of ${D.p}</strong> — so the curve falls in integer steps instead of `
        + `smoothly. The shrinkage-factor bars are not meaningful here, so they are drawn flat; `
        + `switch to Ridge to see them.`;
  }

  /* ======================================================================
     3 · everything vs λ
  ====================================================================== */
  function drawSweep() {
    const xs = SWEEP.map(r => r.a);
    const mk = (host, legHost, defs, opts) => {
      const series = defs.map(d => ({name: d.name, color: d.color, dashed: d.dashed,
        points: xs.map((a, i) => [a, d.ys[i]])}));
      Plot.line(host, series, Object.assign({height: 250, logX: true, dots: false,
        xLabel: 'α (log scale)', vLine: S.alpha}, opts));
      if (legHost) Plot.legend(legHost, series.map(s => ({name: s.name, color: s.color})));
    };

    mk($('swErr'), $('swErrLeg'), [
      {name: 'train RMSE', color: 'var(--c-ols)', dashed: true, ys: SWEEP.map(r => r.trainRMSE)},
      {name: '5-fold CV', color: 'var(--accent)', ys: CV.map(r => r.cv)},
      {name: 'held-out', color: 'var(--rose)', ys: SWEEP.map(r => r.testRMSE)}
    ], {yLabel: 'RMSE (₹)'});

    mk($('swNorm'), $('swNormLeg'), [
      {name: '‖w‖₁', color: 'var(--c-lasso)', ys: SWEEP.map(r => r.l1)},
      {name: '‖w‖₂', color: 'var(--c-ridge)', ys: SWEEP.map(r => r.l2)}
    ], {yLabel: 'norm'});

    mk($('swSparse'), null, [
      {name: 'non-zero', color: COL[S.method], ys: SWEEP.map(r => r.nnz)}
    ], {yLabel: 'coefficients ≠ 0', yDomain: [-0.3, D.p + 0.3]});

    mk($('swLoss'), $('swLossLeg'), [
      {name: 'data fit', color: 'var(--c-ols)', ys: SWEEP.map(r => r.data)},
      {name: 'penalty', color: COL[S.method], ys: SWEEP.map(r => r.pen)},
      {name: 'total objective', color: 'var(--accent)', ys: SWEEP.map(r => r.obj)}
    ], {yLabel: 'objective'});

    const last = SWEEP[SWEEP.length - 1];
    $('swNote').innerHTML = S.method === 'ridge'
      ? 'Read the sparsity panel first: it is a flat line at ' + D.p + '. Ridge never removes a '
        + 'coefficient, however hard you push — so the only way it can simplify the model is by '
        + 'shrinking, which is what the norms panel shows. Switch to Lasso and the same panel '
        + 'becomes a staircase.'
      : 'The sparsity staircase is the whole difference from Ridge. Each step down is a feature '
        + 'leaving the model for good. Note also the objective panel: the penalty term rises to a '
        + 'peak and then <em>falls</em> — once a coefficient is zeroed it stops being charged for, '
        + 'so at large α the model pays almost no penalty because it has almost nothing left.';
  }

  /* ======================================================================
     4 · prior → posterior, one coefficient
  ====================================================================== */
  function drawBayes() {
    const j = KEYS.indexOf($('bFeat').value);
    const tau = Math.pow(10, +$('bTau').value);
    const sigma = +$('bSigma').value;
    $('bTauV').textContent = F(tau, tau < 1 ? 3 : 2);
    $('bSigmaV').textContent = sigma;

    /* work in standardized-y units, then report in rupees */
    const sigZ = sigma / D.ysd;
    const post = ML.bayesLinear(D.Ztr, D.yc.map(v => v / D.ysd), tau * tau, sigZ * sigZ);
    if (!post) return;
    const mu = post.mu[j], sd = post.sd[j];
    const lamEq = sigZ * sigZ / (tau * tau);

    Plot.density($('gErr'), [{name: 'noise', color: 'var(--c-ols)',
      fn: x => ML.gaussPdf(x, 0, sigma)}],
      {height: 190, xDomain: [-3.2 * sigma, 3.2 * sigma], xLabel: 'residual (₹)', yLabel: ''});

    Plot.density($('gPri'), [{name: 'prior', color: 'var(--c-ridge)',
      fn: x => ML.gaussPdf(x, 0, tau)}],
      {height: 190, xDomain: [-3.2 * tau, 3.2 * tau], xLabel: 'wⱼ (standardized)', yLabel: ''});

    const lo = mu - 4 * sd, hi = mu + 4 * sd;
    Plot.density($('gPost'), [{name: 'posterior', color: 'var(--accent)',
      fn: x => ML.gaussPdf(x, mu, sd)}],
      {height: 190, xDomain: [lo, hi], xLabel: 'wⱼ (standardized)', yLabel: '',
       marks: [{x: 0, color: 'var(--rose)', label: 'zero'},
               {x: mu, color: 'var(--accent)', label: 'mode'}]});

    const coversZero = (mu - 1.96 * sd) * (mu + 1.96 * sd) < 0;
    $('gStats').innerHTML = [
      ['posterior mean (standardized)', F(mu, 4)],
      ['posterior sd', '± ' + F(sd, 4)],
      ['95% credible interval', `[${F(mu - 1.96 * sd, 3)}, ${F(mu + 1.96 * sd, 3)}]`],
      ['in rupees per s.d. of the feature', '₹' + F(mu * D.ysd, 2) + ' ± ' + F(1.96 * sd * D.ysd, 2)],
      ['equivalent ridge λ = σ²/τ²', F(lamEq, 4)],
      ['interval covers zero?', coversZero ? 'yes' : 'no']
    ].map(([k, v]) => `<div class="kv"><span class="k">${k}</span><span class="v${
      k === 'interval covers zero?' ? (coversZero ? ' warn' : ' good') : ''}">${v}</span></div>`).join('');

    $('gNote').innerHTML = coversZero
      ? `The posterior for <code>${D.labels[j]}</code> straddles zero, so the data does not settle `
        + `the sign of this coefficient. Reporting it as an effect would be overclaiming — and note `
        + `that Lasso would simply have set it to zero, which is the same judgement expressed as a `
        + `decision rather than as a distribution.`
      : `The posterior for <code>${D.labels[j]}</code> is clear of zero at 95%, so the data has `
        + `settled its sign. The prior panel is much wider than the posterior, which is what it `
        + `looks like when the likelihood has done the work — tighten τ until the two are `
        + `comparable and the prior starts to pull the posterior toward zero.`;
  }

  /* ======================================================================
     5 · matrices
  ====================================================================== */
  const matHTML = (M, hi) => {
    const cells = M.map((row, i) => row.map((v, j) => {
      const on = hi && i === j;
      return `<div class="cell${on ? ' on' : ''}">${F(v, 2)}</div>`;
    }).join('')).join('');
    return `<div class="mat" style="grid-template-columns:repeat(${M[0].length},auto);
      font-size:.72rem">${cells}</div>`;
  };

  function drawMatrices() {
    const G = D.gram;
    const a = S.alpha;
    const Greg = G.map((r, i) => r.map((v, j) => i === j ? v + a : v));
    $('mGram').innerHTML = `<div style="overflow-x:auto">${matHTML(G, false)}</div>`;
    $('mGramReg').innerHTML = `<div style="overflow-x:auto">${matHTML(Greg, true)}</div>`;

    const w = fitAt(a);
    /* Var(ŵ) = σ²(XᵀX+λI)⁻¹ XᵀX (XᵀX+λI)⁻¹, with σ² from the residuals */
    const r = rss(w);
    const dfNow = dfAt(a, w);
    const s2 = r / Math.max(D.n - dfNow, 1);
    const XtX = ML.matmul(ML.T(D.Ztr), D.Ztr);
    const lam = a * D.n;
    const A = XtX.map((row, i) => row.map((v, j) => i === j ? v + lam : v));
    const Ai = ML.inv(A);
    let se = null;
    if (Ai) {
      const V = ML.matmul(ML.matmul(Ai, XtX), Ai).map(row => row.map(v => v * s2));
      se = V.map((row, i) => Math.sqrt(Math.max(row[i], 0)));
    }
    Plot.barsH($('mCoefs'), D.labels.map((l, i) => ({label: l, value: w[i], color: COL[S.method]})),
      {labelWidth: 126, height: Math.max(190, D.p * 27 + 40)});

    $('mStats').innerHTML = [
      ['σ̂² from residuals', F(s2, 2)],
      ['residual df (n − df)', F(D.n - dfNow, 2)],
      ['condition number of XᵀX', F(ML.condition(XtX), 1)],
      ['condition number of XᵀX + λI', F(ML.condition(A), 1)],
      ['largest standard error', se ? '± ' + F(Math.max(...se), 3) : '—'],
      ['coefficients within 2 s.e. of zero',
       se ? String(w.filter((v, i) => Math.abs(v) < 2 * se[i]).length) + ' of ' + D.p : '—']
    ].map(([k, v]) => `<div class="kv"><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');

    const c0 = ML.condition(XtX), c1 = ML.condition(A);
    $('mNote').innerHTML =
      `Adding α to the diagonal takes the condition number from <strong>${F(c0, 1)}</strong> to `
      + `<strong>${F(c1, 1)}</strong>. That single number is the mechanical reason ridge is stable: `
      + `the inverse is no longer amplifying the directions the data barely constrains.`
      + (se ? ` Of the ${D.p} coefficients, ${w.filter((v, i) => Math.abs(v) < 2 * se[i]).length} `
        + `are within two standard errors of zero — those are the ones an ℓ₁ penalty would be `
        + `most likely to remove.` : '');
  }

  /* ======================================================================
     6 · convergence
  ====================================================================== */
  function drawConverge() {
    if (S.method === 'ridge') {
      ['cObj', 'cDelta'].forEach(id => {
        $(id).innerHTML = '<p class="muted small" style="margin:0;padding:26px 0">'
          + 'Ridge is a closed form — one linear solve, no iteration, nothing to converge. '
          + 'Switch to Lasso or Elastic Net.</p>';
      });
      $('cStats').innerHTML = '';
      $('cNote').innerHTML = 'That ridge needs no convergence check is a real operational '
        + 'advantage, and part of the argument for it in '
        + '<a href="assignments.html#deliverables">Deliverable 3</a>: there is no tolerance to '
        + 'get wrong and no iteration limit to silently hit.';
      return;
    }

    /* instrumented coordinate descent: one record per sweep */
    const a = Math.max(S.alpha, 1e-9), rr = rho();
    const l1p = a * rr, l2p = a * (1 - rr);
    const n = D.n, p = D.p;
    const w = new Array(p).fill(0);
    const nrm = new Array(p).fill(0);
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += D.Ztr[i][j] ** 2;
      nrm[j] = s / n;
    }
    const res = D.yc.slice();
    const objOf = () => {
      let q = 0;
      for (let i = 0; i < n; i++) q += res[i] * res[i];
      return q / (2 * n) + l1p * l1(w) + l2p / 2 * l2sq(w);
    };
    const trace = [];
    const MAX_SWEEPS = 300;
    for (let sweepI = 0; sweepI < MAX_SWEEPS; sweepI++) {
      let maxD = 0;
      for (let j = 0; j < p; j++) {
        let r0 = 0;
        for (let i = 0; i < n; i++) r0 += D.Ztr[i][j] * (res[i] + D.Ztr[i][j] * w[j]);
        r0 /= n;
        const wn = ML.softThreshold(r0, l1p) / (nrm[j] + l2p);
        const d = wn - w[j];
        if (d !== 0) {
          for (let i = 0; i < n; i++) res[i] -= D.Ztr[i][j] * d;
          w[j] = wn;
          if (Math.abs(d) > maxD) maxD = Math.abs(d);
        }
      }
      trace.push({sweep: sweepI + 1, obj: objOf(), delta: maxD, nnz: nnz(w)});
      if (maxD < 1e-10) break;
    }

    Plot.line($('cObj'), [{name: 'objective', color: 'var(--accent)',
      points: trace.map(t => [t.sweep, t.obj])}],
      {height: 250, dots: trace.length < 30, xLabel: 'sweep', yLabel: 'objective'});

    /* Plotted as log10|Δw| because convergence is geometric: a straight line here is
       the expected shape, and a flattening one means the tolerance was reached. */
    const pos = trace.filter(t => t.delta > 0);
    Plot.line($('cDelta'), [{name: 'log₁₀ max |Δw|', color: 'var(--c-lasso)',
      points: pos.map(t => [t.sweep, Math.log10(t.delta)])}],
      {height: 250, dots: pos.length < 30, xLabel: 'sweep',
       yLabel: 'log₁₀ of max |Δw|'});

    const hit = trace.find(t => t.delta < 1e-7);
    $('cStats').innerHTML = [
      ['sweeps recorded', String(trace.length) + (trace.length >= MAX_SWEEPS ? ' (cap)' : '')],
      ['first sweep below 1e−7', hit ? String(hit.sweep) : 'not within ' + MAX_SWEEPS],
      ['final max |Δw|', trace[trace.length - 1].delta.toExponential(2)],
      ['objective fell by', F(trace[0].obj - trace[trace.length - 1].obj, 5)],
      ['non-zero at sweep 1 → final',
       trace[0].nnz + ' → ' + trace[trace.length - 1].nnz]
    ].map(([k, v]) => `<div class="kv"><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');

    $('cNote').innerHTML = trace[0].nnz !== trace[trace.length - 1].nnz
      ? `The support changed while converging: ${trace[0].nnz} coefficients were non-zero after the `
        + `first sweep and ${trace[trace.length - 1].nnz} at the end. <strong>This is why a loose `
        + `tolerance is dangerous for ℓ₁.</strong> Stop early and you report a feature as selected `
        + `that the converged solution drops, or the reverse.`
      : `The support settled in the first sweep and only the magnitudes moved after that, which is `
        + `the common case at a moderate α. It is not guaranteed — raise α until a coefficient is `
        + `near the threshold and the support will keep changing for several sweeps.`;
  }

  /* ======================================================================
     wiring
  ====================================================================== */
  $('bFeat').innerHTML = Prep.FEATURES.map(f =>
    `<option value="${f.key}">${f.label} — ${f.pretty}</option>`).join('');
  $('bFeat').value = 'is_weekend';

  function redrawAll() {
    drawSelection(); drawDf(); drawSweep(); drawBayes(); drawMatrices(); drawConverge();
  }
  function redrawAlphaOnly() {
    drawDf(); drawSweep(); drawMatrices(); drawConverge();
  }

  $('dMethod').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    [...e.currentTarget.children].forEach(c => c.classList.toggle('on', c === b));
    S.method = b.dataset.m; redrawAll();
  });
  $('dAlpha').addEventListener('input', e => {
    S.alpha = Math.pow(10, +e.target.value);
    $('dAlphaV').textContent = F(S.alpha, S.alpha < 1 ? 3 : 2);
    redrawAlphaOnly();
  });
  $('dN').addEventListener('input', e => {
    S.n = +e.target.value; $('dNV').textContent = S.n; rebuild(); redrawAll();
  });
  ['bFeat', 'bTau', 'bSigma'].forEach(id =>
    $(id).addEventListener(id === 'bFeat' ? 'change' : 'input', drawBayes));

  UI.onDraw(() => {
    if (!UI.visible($('selChart'))) return;     // diagnostics tab not on screen
    rebuild(); redrawAll();
  });
})();
