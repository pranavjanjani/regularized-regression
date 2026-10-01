/* =============================================================================
   logistic.js — the logistic regression lab.

   The solver is ML.logistic (proximal gradient + FISTA), verified against
   scikit-learn to ~6e-7 on six fits. Note for anyone reproducing in Python:
   liblinear penalizes the intercept by default, so compare against lbfgs or
   saga, or the coefficients will disagree by far more than solver tolerance.
============================================================================= */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const F = (v, d = 3) => (v == null || !isFinite(v)) ? '—' : v.toFixed(d);
  const PCT = v => isFinite(v) ? (100 * v).toFixed(1) + '%' : '—';
  const KEYS = Prep.FEATURES.map(f => f.key);
  const OPTS = Prep.FEATURES.map(f =>
    `<option value="${f.key}">${f.label} — ${f.pretty}</option>`).join('');

  /* One fit of the full 9-feature model, cached per (alpha, rho). */
  const cache = new Map();
  function fit(d, alpha, rho) {
    const k = alpha + '|' + rho;
    if (!cache.has(k)) cache.set(k, ML.logistic(d.Ztr, d.clsTr, {alpha, l1Ratio: rho}));
    return cache.get(k);
  }

  let CUT = 100;
  let D = Prep.buildBinary(KEYS, CUT);
  const rebuild = () => { cache.clear(); D = Prep.buildBinary(KEYS, CUT); };

  /* ======================================================================
     1 · the label
  ====================================================================== */
  function drawLabel() {
    const posTr = D.clsTr.reduce((a, b) => a + b, 0);
    const posTe = D.clsTe.reduce((a, b) => a + b, 0);
    const base = Math.max(1 - D.baseRate, D.baseRate);
    const distinct = new Set(D.surge.slice(0, Prep.N_TRAIN).map(v => v.toFixed(1))).size;

    $('cutV').textContent = CUT;
    $('dPos').textContent = posTr;
    $('dPosSub').textContent = `of 150 · ${PCT(posTr / 150)}`;
    $('dBase').textContent = PCT(base);
    $('dBase2').textContent = PCT(base);
    $('dPosTe').textContent = posTe;
    $('dLost').textContent = distinct;

    Plot.histogram($('cutHist'), D.surge.slice(0, Prep.N_TRAIN), {
      height: 260, bins: 22, xLabel: 'surge (₹)', yLabel: 'trips',
      color: 'var(--accent)'});
    // mark the cut with an overlay line
    const h = $('cutHist').querySelector('svg');
    if (h) {
      const lo = Math.min(...D.surge.slice(0, Prep.N_TRAIN));
      const hi = Math.max(...D.surge.slice(0, Prep.N_TRAIN));
      const m = {l: 54, r: 16, t: 16, b: 42};
      const W = h.viewBox.baseVal.width, H = h.viewBox.baseVal.height;
      const x = m.l + (CUT - lo) / (hi - lo) * (W - m.l - m.r);
      const NS = 'http://www.w3.org/2000/svg';
      const ln = document.createElementNS(NS, 'line');
      ln.setAttribute('x1', x); ln.setAttribute('x2', x);
      ln.setAttribute('y1', m.t); ln.setAttribute('y2', H - m.b);
      ln.setAttribute('stroke', 'var(--rose)'); ln.setAttribute('stroke-width', 2.2);
      ln.setAttribute('stroke-dasharray', '5 4');
      h.appendChild(ln);
      const tx = document.createElementNS(NS, 'text');
      tx.setAttribute('x', x + 5); tx.setAttribute('y', m.t + 12);
      tx.setAttribute('fill', 'var(--rose)'); tx.setAttribute('font-size', 10.5);
      tx.setAttribute('font-family', 'var(--mono)');
      tx.textContent = 'cut ₹' + CUT;
      h.appendChild(tx);
    }

    Plot.barsH($('cutBal'), [
      {label: 'no alert (0)', value: 150 - posTr, color: 'var(--c-ridge)'},
      {label: 'alert (1)', value: posTr, color: 'var(--rose)'}
    ], {labelWidth: 110, height: 150});

    $('cutBalNote').innerHTML = posTr < 15 || posTr > 135
      ? '<strong>Badly imbalanced.</strong> With this few examples of one class, accuracy is '
        + 'meaningless and the fit is unstable. Move the cut back toward the middle.'
      : `At a ₹${CUT} cut the classes are ${PCT(posTr / 150)} / ${PCT(1 - posTr / 150)} — `
        + 'workable, and close enough to balanced that accuracy is at least interpretable.';
  }

  /* ======================================================================
     2 · the sigmoid
  ====================================================================== */
  function drawSigmoid() {
    const key = $('sigFeat').value;
    const thr = +$('sigThresh').value;
    $('sigThreshV').textContent = thr.toFixed(2);
    const d1 = Prep.buildBinary([key], CUT);
    const m = ML.logistic(d1.Ztr, d1.clsTr, {alpha: 0.01, l1Ratio: 0});

    /* the comparison: ordinary least squares on the same 0/1 label */
    const ols = ML.ols(d1.Ztr, d1.clsTr);
    const olsBar = ML.mean(d1.clsTr);
    const olsPred = d1.Ztr.map(r => olsBar + r[0] * ols.w[0]);
    const outside = olsPred.filter(v => v < 0 || v > 1).length;

    const xs = d1.Ztr.map(r => r[0]);
    const lo = Math.min(...xs) - 0.4, hi = Math.max(...xs) + 0.4;
    const curve = [], line = [];
    for (let i = 0; i <= 160; i++) {
      const z = lo + (hi - lo) * i / 160;
      curve.push([z, ML.sigmoid(m.b + m.w[0] * z)]);
      line.push([z, olsBar + ols.w[0] * z]);
    }
    /* jitter the 0/1 points a little so overlapping observations are visible */
    let seed = 7;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    const pts = d1.Ztr.map((r, i) => ({
      x: r[0], y: d1.clsTr[i] * 0.92 + 0.04 + (rnd() - 0.5) * 0.06,
      color: d1.clsTr[i] ? 'var(--rose)' : 'var(--c-ridge)', r: 3.2,
      tip: `${key} = ${d1.Xtr[i][0]}\nsurge ₹${D.surge[i].toFixed(0)}\nlabel ${d1.clsTr[i]}`
    }));

    const series = [
      {name: 'logistic σ(xw+b)', color: 'var(--accent)', width: 2.8, points: curve},
      {name: 'least squares on the 0/1 label', color: 'var(--c-ols)', dashed: true, points: line},
      {name: 'threshold', color: 'var(--rose)', dashed: true,
       points: [[lo, thr], [hi, thr]]}
    ];
    Plot.line($('sigChart'), series, {height: 330, dots: false,
      xLabel: key + ' (standardized)', yLabel: 'P(alert)', yDomain: [-0.28, 1.28]});
    /* overlay the observations */
    const svg = $('sigChart').querySelector('svg');
    const g = svg.querySelector('g');
    const iw = svg.viewBox.baseVal.width - 54 - 16, ih = 330 - 16 - 42;
    const sx = v => (v - lo) / (hi - lo) * iw;
    const sy = v => ih - (v + 0.28) / 1.56 * ih;
    const NS = 'http://www.w3.org/2000/svg';
    pts.forEach(p => {
      const c = document.createElementNS(NS, 'circle');
      c.setAttribute('cx', sx(p.x)); c.setAttribute('cy', sy(p.y));
      c.setAttribute('r', p.r); c.setAttribute('fill', p.color);
      c.setAttribute('opacity', 0.6);
      g.appendChild(c);
    });
    Plot.legend($('sigLeg'), [
      {name: 'logistic fit', color: 'var(--accent)'},
      {name: 'OLS on 0/1 (inappropriate)', color: 'var(--c-ols)'},
      {name: 'alert = 1', color: 'var(--rose)'},
      {name: 'no alert = 0', color: 'var(--c-ridge)'}]);

    const prob = ML.predictProba(d1.Ztr, m.w, m.b);
    const met = ML.classMetrics(d1.clsTr, prob, thr);
    $('sAcc').textContent = PCT(met.accuracy);
    $('sLoss').textContent = F(ML.logLoss(d1.clsTr, prob), 4);
    $('sOut').textContent = outside;
    $('sCoef').textContent = F(m.w[0], 3);

    $('sigNote').innerHTML = outside > 0
      ? `The dashed least-squares line leaves [0, 1] for <strong>${outside} of 150</strong> training
         trips — it is predicting probabilities below zero or above one, which is not a probability.
         The sigmoid cannot do that by construction. Coefficient
         <span class="hl">${F(m.w[0], 3)}</span> is in log-odds per standard deviation: a one-s.d.
         rise in <code>${key}</code> multiplies the odds of an alert by
         <span class="hl">${F(Math.exp(m.w[0]), 2)}×</span>.`
      : `On this feature the least-squares line happens to stay inside [0, 1], but it has no reason
         to. The logistic coefficient <span class="hl">${F(m.w[0], 3)}</span> is in log-odds: a
         one-s.d. rise multiplies the odds by <span class="hl">${F(Math.exp(m.w[0]), 2)}×</span>.`;
  }

  /* ======================================================================
     3 · decision boundary
  ====================================================================== */
  function drawBoundary() {
    if ($('bFa').value === $('bFb').value) {
      $('bFb').value = KEYS.find(k => k !== $('bFa').value);
    }
    const keys = [$('bFa').value, $('bFb').value];
    const alpha = Math.pow(10, +$('bAlpha').value);
    const rho = +document.querySelector('#bRho button.on').dataset.r;
    $('bAlphaV').textContent = F(alpha, alpha < 1 ? 3 : 2);

    const d2 = Prep.buildBinary(keys, CUT);
    const m = ML.logistic(d2.Ztr, d2.clsTr, {alpha, l1Ratio: rho});
    const labels = keys.map(k => Prep.FEATURES.find(f => f.key === k).label);

    const xs = d2.Ztr.map(r => r[0]), ys = d2.Ztr.map(r => r[1]);
    const pad = 0.5;
    const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad;
    const y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;

    const pts = d2.Ztr.map((r, i) => ({
      x: r[0], y: r[1],
      color: d2.clsTr[i] ? 'var(--rose)' : 'var(--c-ridge)', r: 3.6,
      tip: `${labels[0]} = ${d2.Xtr[i][0]}\n${labels[1]} = ${d2.Xtr[i][1]}\n`
         + `surge ₹${D.surge[i].toFixed(0)}  label ${d2.clsTr[i]}`
    }));

    /* boundary: b + w1 x + w2 y = 0  -> solve for y where possible */
    const bLines = [];
    const levels = [[0.5, 'var(--fg)', 2.4], [0.25, 'var(--c-ridge)', 1.2], [0.75, 'var(--rose)', 1.2]];
    levels.forEach(([lev, col, wid]) => {
      const c = Math.log(lev / (1 - lev));          // b + w·x = c at this probability
      const seg = [];
      if (Math.abs(m.w[1]) > 1e-8) {
        for (const xv of [x0, x1]) seg.push([xv, (c - m.b - m.w[0] * xv) / m.w[1]]);
      } else if (Math.abs(m.w[0]) > 1e-8) {
        const xv = (c - m.b) / m.w[0];
        seg.push([xv, y0], [xv, y1]);
      }
      if (seg.length === 2) bLines.push({name: 'P = ' + lev, color: col, width: wid,
        dashed: lev !== 0.5, points: seg});
    });

    Plot.scatter($('bChart'), pts, {height: 340, xLabel: labels[0] + ' (standardized)',
      yLabel: labels[1] + ' (standardized)', xDomain: [x0, x1], yDomain: [y0, y1], opacity: .72});
    /* draw the boundary lines into the same clipped group */
    const svg = $('bChart').querySelector('svg');
    const g = svg.querySelector('g[clip-path]') || svg.querySelector('g');
    const iw = svg.viewBox.baseVal.width - 54 - 16, ih = 340 - 16 - 42;
    const sx = v => (v - x0) / (x1 - x0) * iw;
    const sy = v => ih - (v - y0) / (y1 - y0) * ih;
    const NS = 'http://www.w3.org/2000/svg';
    bLines.forEach(l => {
      const ln = document.createElementNS(NS, 'line');
      ln.setAttribute('x1', sx(l.points[0][0])); ln.setAttribute('y1', sy(l.points[0][1]));
      ln.setAttribute('x2', sx(l.points[1][0])); ln.setAttribute('y2', sy(l.points[1][1]));
      ln.setAttribute('stroke', l.color); ln.setAttribute('stroke-width', l.width);
      if (l.dashed) ln.setAttribute('stroke-dasharray', '5 4');
      g.appendChild(ln);
    });
    Plot.legend($('bLeg'), [
      {name: 'alert = 1', color: 'var(--rose)'}, {name: 'no alert = 0', color: 'var(--c-ridge)'},
      {name: 'P = 0.5 boundary', color: 'var(--fg)'},
      {name: 'P = 0.25 / 0.75', color: 'var(--c-lasso)'}]);

    Plot.barsH($('bCoefs'), labels.map((l, i) => ({label: l, value: m.w[i],
      color: rho >= 1 ? 'var(--c-lasso)' : rho > 0 ? 'var(--c-enet)' : 'var(--c-ridge)'}))
      .concat([{label: 'intercept', value: m.b, color: 'var(--c-ols)'}]),
      {labelWidth: 120, height: 150});

    const prob = ML.predictProba(d2.Ztr, m.w, m.b);
    const met = ML.classMetrics(d2.clsTr, prob, 0.5);
    const roc = ML.rocCurve(d2.clsTr, prob);
    const zeros = m.w.filter(v => Math.abs(v) < 1e-8).length;
    $('bStats').innerHTML = [
      ['log-loss (train)', F(ML.logLoss(d2.clsTr, prob), 4)],
      ['accuracy', PCT(met.accuracy)],
      ['AUC', F(roc.auc, 4)],
      ['‖w‖₂', F(Math.hypot(m.w[0], m.w[1]), 3)],
      ['coefficients at zero', String(zeros)],
      ['solver iterations', String(m.iters)]
    ].map(([k, v]) => `<div class="kv"><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');

    const steep = Math.hypot(m.w[0], m.w[1]);
    $('bNote').innerHTML = zeros > 0
      ? `<strong>One predictor has been eliminated</strong>, so the boundary is now perpendicular to
         the surviving axis — the model has stopped using the other feature entirely. Only ℓ₁ and
         Elastic Net can do this; ℓ₂ would have kept both forever.`
      : steep < 0.35
        ? `At α = ${F(alpha, 3)} the coefficients are so shrunk that the boundary is barely
           committed: predicted probabilities all sit near the base rate of
           ${PCT(D.baseRate)}, and the P = 0.25 and P = 0.75 lines have moved far apart. The model
           has been penalized into saying "I don't know".`
        : `The solid line is where the model is exactly 50/50. The dashed lines at P = 0.25 and
           P = 0.75 show how fast confidence changes — they spread apart as α grows, because
           shrinking w flattens the sigmoid. Raise α until a coefficient hits zero
           (ℓ₁ or Elastic Net only).`;
  }

  /* ======================================================================
     4 · coefficient paths
  ====================================================================== */
  function drawPaths() {
    const rho = +document.querySelector('#pRho button.on').dataset.r;
    const grid = Prep.lamGrid(0.0008, 3, 30);
    const series = D.labels.map((l, j) => ({name: l, points: [],
      color: Plot.colors[j % Plot.colors.length]}));
    const order = [];
    grid.forEach(a => {
      const m = ML.logistic(D.Ztr, D.clsTr, {alpha: a, l1Ratio: rho});
      m.w.forEach((v, j) => {
        series[j].points.push([a, v]);
        if (Math.abs(v) < 1e-8 && !order.includes(j)) order.push(j);
      });
    });
    Plot.line($('pChart'), series, {height: 360, logX: true, dots: false, zeroLine: true,
      xLabel: 'penalty α (log scale)', yLabel: 'coefficient (log-odds per s.d.)'});
    Plot.legend($('pLeg'), series.map(s => ({name: s.name, color: s.color})));

    $('pNote').innerHTML = rho === 0
      ? 'With a pure ℓ₂ penalty every coefficient shrinks toward zero and none arrives — the same '
        + 'behaviour as ridge in <a href="lab.html#paths">Lab §2</a>, for the same reason: a '
        + 'quadratic penalty has no gradient at the origin.'
      : order.length
        ? 'Elimination order: ' + order.map(j => '<code>' + D.labels[j] + '</code>').join(' → ')
          + '. <strong>Note what this is not.</strong> In the regression lab the first casualty was '
          + '<code>is_weekend</code>, the feature with no real effect. Here the redundant weather '
          + 'features tend to go first instead, because <code>traffic_speed</code> already carries '
          + 'their information and ℓ₁ will not pay twice for it. An ℓ₁ ordering is a ranking of '
          + '<em>marginal usefulness given the others</em>, not of true importance — which is the '
          + 'selection-instability caveat from <a href="theory.html#lasso-caveats">Theory §5.3</a> '
          + 'showing up in a new place.'
        : 'No coefficient reached zero across this α range — extend the grid or raise ρ.';
  }

  /* ======================================================================
     5 · ROC and the threshold
  ====================================================================== */
  function drawThreshold() {
    const thr = +$('tThresh').value;
    const onTest = document.querySelector('#tSplit button.on').dataset.s === 'test';
    $('tThreshV').textContent = thr.toFixed(2);
    const m = fit(D, 0.02, 0);
    const Z = onTest ? D.Zte : D.Ztr;
    const y = onTest ? D.clsTe : D.clsTr;
    const prob = ML.predictProba(Z, m.w, m.b);
    const roc = ML.rocCurve(y, prob);
    const met = ML.classMetrics(y, prob, thr);

    Plot.line($('tRoc'), [
      {name: 'ROC', color: 'var(--accent)', width: 2.6, points: roc.points},
      {name: 'chance', color: 'var(--c-ols)', dashed: true, points: [[0, 0], [1, 1]]},
      {name: 'here', color: 'var(--rose)', points: [[1 - met.specificity, met.recall]],
       width: 0, dotR: 6}
    ], {height: 300, dots: false, xDomain: [0, 1], yDomain: [0, 1.02],
        xLabel: 'false positive rate', yLabel: 'true positive rate'});
    /* the current operating point needs its own marker */
    const svg = $('tRoc').querySelector('svg');
    const g = svg.querySelector('g[clip-path]') || svg.querySelector('g');
    const iw = svg.viewBox.baseVal.width - 54 - 16, ih = 300 - 16 - 42;
    const NS = 'http://www.w3.org/2000/svg';
    const c = document.createElementNS(NS, 'circle');
    c.setAttribute('cx', (1 - met.specificity) * iw);
    c.setAttribute('cy', ih - (met.recall / 1.02) * ih);
    c.setAttribute('r', 6); c.setAttribute('fill', 'var(--rose)');
    c.setAttribute('stroke', 'var(--bg-2)'); c.setAttribute('stroke-width', 2);
    g.appendChild(c);

    const n = y.length;
    $('tConf').innerHTML = `
      <table style="font-size:.85rem;border-collapse:collapse;width:100%">
        <thead><tr><th></th><th class="num">predicted alert</th><th class="num">predicted none</th></tr></thead>
        <tbody>
          <tr><td><strong>actually alert</strong></td>
            <td class="num" style="color:var(--accent);font-weight:700">${met.tp}</td>
            <td class="num" style="color:var(--rose);font-weight:700">${met.fn}</td></tr>
          <tr><td><strong>actually none</strong></td>
            <td class="num" style="color:var(--rose);font-weight:700">${met.fp}</td>
            <td class="num" style="color:var(--accent);font-weight:700">${met.tn}</td></tr>
        </tbody></table>`;
    $('tStats').innerHTML = [
      ['AUC (threshold-free)', F(roc.auc, 4)],
      ['log-loss', F(ML.logLoss(y, prob), 4)],
      ['accuracy', PCT(met.accuracy)],
      ['base rate (always "none")', PCT(1 - y.reduce((a, b) => a + b, 0) / n)],
      ['precision', PCT(met.precision)],
      ['recall (alerts caught)', PCT(met.recall)],
      ['F1', F(met.f1, 3)]
    ].map(([k, v]) => `<div class="kv"><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');

    $('tNote').innerHTML = met.fn > met.fp
      ? `At a threshold of ${thr.toFixed(2)} the model misses <strong>${met.fn}</strong> real alerts
         and raises <strong>${met.fp}</strong> false ones. If a missed alert costs more than a
         needless one — which for dispatch it usually does — <em>lower</em> the threshold and accept
         more false alarms. Nothing about the fitted model changes when you do; only the policy does.`
      : `At ${thr.toFixed(2)} you are catching <strong>${met.recall ? PCT(met.recall) : '—'}</strong>
         of real alerts at the cost of <strong>${met.fp}</strong> false ones. Accuracy here is
         ${PCT(met.accuracy)} against a base rate of
         ${PCT(1 - y.reduce((a, b) => a + b, 0) / n)} — quote both or the number is meaningless.`;
  }

  /* ======================================================================
     6 · comparison
  ====================================================================== */
  function drawCompare() {
    const grid = Prep.lamGrid(0.001, 2, 22);
    const rows = [];
    const defs = [['No penalty', null, 0], ['ℓ₂ (ridge)', 0, 0], ['ℓ₁ (lasso)', 1, 0],
                  ['Elastic Net', 0.5, 0]];
    defs.forEach(([name, rho]) => {
      let alpha = 1e-6, best = Infinity;
      if (rho !== null) {
        grid.forEach(a => {
          const cv = Prep.kFoldLogLoss(D.Ztr, D.clsTr, 5,
            (Z, y) => ML.logistic(Z, y, {alpha: a, l1Ratio: rho}));
          if (cv < best) { best = cv; alpha = a; }
        });
      }
      const m = ML.logistic(D.Ztr, D.clsTr, {alpha, l1Ratio: rho === null ? 0 : rho});
      const p = ML.predictProba(D.Zte, m.w, m.b);
      const met = ML.classMetrics(D.clsTe, p, 0.5);
      const dropped = D.labels.filter((l, i) => Math.abs(m.w[i]) < 1e-8);
      rows.push({name, alpha: rho === null ? null : alpha,
        loss: ML.logLoss(D.clsTe, p), auc: ML.rocCurve(D.clsTe, p).auc,
        acc: met.accuracy, nz: m.w.filter(v => Math.abs(v) > 1e-8).length, dropped});
    });

    /* and the row that matters: predict rupees, then threshold */
    const lin = Prep.build(KEYS);
    const linA = (() => {
      let a = 0.01, best = Infinity;
      Prep.lamGrid(0.002, 50, 24).forEach(x => {
        const cv = ML.kFoldCV(lin.Ztr, lin.ytr, 5, (Z, t) => ML.ridge(Z, t, x * Z.length).w);
        if (cv < best) { best = cv; a = x; }
      });
      return a;
    })();
    const wLin = Prep.fit(lin, 'ridge', linA);
    const yhat = ML.predict(lin.Zte, wLin, lin.yBar);
    const hard = yhat.map(v => (v > CUT ? 1 : 0));
    const linMet = ML.classMetrics(D.clsTe, hard, 0.5);
    /* a calibrated probability is not available from a point prediction, so
       log-loss and AUC are reported from the ranking of the predicted rupees */
    const linAuc = ML.rocCurve(D.clsTe, yhat).auc;
    rows.push({name: 'Linear model, then cut at ₹' + CUT, alpha: linA,
      loss: NaN, auc: linAuc, acc: linMet.accuracy, nz: wLin.filter(v => Math.abs(v) > 1e-8).length,
      dropped: [], isLinear: true});

    const bestAuc = Math.max(...rows.map(r => r.auc));
    $('logCmpTable').querySelector('tbody').innerHTML = rows.map(r => `
      <tr>
        <td>${r.isLinear ? '<em>' + r.name + '</em>' : r.name}</td>
        <td class="num">${r.alpha == null ? '—' : F(r.alpha, 4)}</td>
        <td class="num">${isFinite(r.loss) ? F(r.loss, 4) : '<span class="muted">n/a</span>'}</td>
        <td class="num"${r.auc === bestAuc ? ' style="color:var(--accent);font-weight:700"' : ''}>${F(r.auc, 4)}</td>
        <td class="num">${PCT(r.acc)}</td>
        <td class="num">${r.nz} / 9</td>
        <td class="small">${r.dropped.length ? r.dropped.map(x => '<code>' + x + '</code>').join(' ') : '<span class="muted">none</span>'}</td>
      </tr>`).join('');

    const clsBest = Math.max(...rows.filter(r => !r.isLinear).map(r => r.auc));
    const linRow = rows[rows.length - 1];
    const spread = clsBest - Math.min(...rows.filter(r => !r.isLinear).map(r => r.auc));
    $('logCmpNote').innerHTML =
      `The four classifiers span <strong>${F(spread, 4)}</strong> of AUC — `
      + (spread < 0.02
          ? 'indistinguishable, so pick on interpretability rather than on the fourth decimal. '
          : 'a visible gap worth investigating. ')
      + (linRow.auc >= clsBest - 0.005
          ? `<strong>And note the last row.</strong> Predicting the surge in rupees and cutting at
             ₹${CUT} reaches AUC ${F(linRow.auc, 4)} against the best classifier's
             ${F(clsBest, 4)} — it matches or beats a model built specifically for the binary task,
             while also telling you <em>how far above</em> the cut each cell is. That is the cost of
             dichotomising, measured: you gave up resolution and bought nothing.`
          : `The purpose-built classifier beats thresholding the regression
             (${F(clsBest, 4)} vs ${F(linRow.auc, 4)} AUC), so here the binary model does earn its
             place. Note the log-loss column is blank for the linear row: a point prediction in
             rupees is not a calibrated probability, so only its ranking can be scored.`);
  }

  /* ======================================================================
     wiring
  ====================================================================== */
  $('sigFeat').innerHTML = OPTS; $('sigFeat').value = 'traffic_speed_kmph';
  $('bFa').innerHTML = OPTS; $('bFa').value = 'traffic_speed_kmph';
  $('bFb').innerHTML = OPTS; $('bFb').value = 'open_requests_500m';

  const segs = [['bRho', drawBoundary], ['pRho', drawPaths], ['tSplit', drawThreshold]];
  segs.forEach(([id, fn]) => $(id).addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    [...e.currentTarget.children].forEach(c => c.classList.toggle('on', c === b));
    fn();
  }));
  $('cutSlider').addEventListener('input', e => {
    CUT = +e.target.value; rebuild();
    drawLabel(); drawSigmoid(); drawBoundary(); drawPaths(); drawThreshold(); drawCompare();
  });
  $('sigFeat').addEventListener('change', drawSigmoid);
  $('sigThresh').addEventListener('input', drawSigmoid);
  ['bFa', 'bFb'].forEach(id => $(id).addEventListener('change', drawBoundary));
  $('bAlpha').addEventListener('input', drawBoundary);
  $('tThresh').addEventListener('input', drawThreshold);

  UI.onDraw(() => {
    if (!UI.visible($('cutHist'))) return;      // logistic tab not on screen
    drawLabel(); drawSigmoid(); drawBoundary(); drawPaths(); drawThreshold(); drawCompare();
  });
})();
