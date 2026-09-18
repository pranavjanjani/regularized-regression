/* =============================================================================
   lab.js — the six interactive panels. Everything refits on every input event;
   the dataset is small enough (150 x 9) that this is comfortably real-time.
============================================================================= */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const money = v => '₹' + v.toFixed(1);
  const COL = {ols: 'var(--c-ols)', ridge: 'var(--c-ridge)', lasso: 'var(--c-lasso)', enet: 'var(--c-enet)'};
  const NAME = {ols: 'OLS / MLE', ridge: 'Ridge', lasso: 'Lasso', enet: 'Elastic Net'};

  /* Shared: tune α by 5-fold CV over a log grid. */
  function tune(keys, method, rho) {
    if (method === 'ols') return {alpha: 0, cv: null};
    const d = Prep.build(keys);
    const grid = Prep.lamGrid(0.001, 160, 40);
    let best = null;
    grid.forEach(a => {
      const cv = ML.kFoldCV(d.Ztr, d.ytr, 5, (Xtr, ytr) => {
        const n = Xtr.length;
        if (method === 'ridge') return ML.ridge(Xtr, ytr, a * n).w;
        return ML.elasticNet(Xtr, ytr, a, method === 'lasso' ? 1 : (rho == null ? 0.5 : rho)).w;
      });
      if (!best || cv < best.cv) best = {alpha: a, cv};
    });
    return best;
  }

  /* ========================= 1 · WORKBENCH ========================= */
  const wb = {method: 'ridge', alpha: 1, rho: 0.5,
              keys: Prep.BASE8.slice(), last: null};

  (function initFeatures() {
    const host = $('wbFeatures');
    host.innerHTML = Prep.FEATURES.map(f =>
      `<span class="chip${wb.keys.includes(f.key) ? ' on' : ''}" data-k="${f.key}"
        title="${f.pretty}">${f.label}</span>`).join('');
    host.addEventListener('click', e => {
      const c = e.target.closest('.chip'); if (!c) return;
      const k = c.dataset.k;
      if (wb.keys.includes(k)) {
        if (wb.keys.length <= 2) return;                  // keep the problem well-posed
        wb.keys = wb.keys.filter(x => x !== k);
      } else {
        wb.keys = Prep.FEATURES.map(f => f.key).filter(x => wb.keys.includes(x) || x === k);
      }
      c.classList.toggle('on');
      drawWB();
    });
  })();

  function drawWB() {
    const d = Prep.build(wb.keys);
    const w = Prep.fit(d, wb.method, wb.alpha, wb.rho);
    const s = Prep.score(d, w);
    wb.last = {d, w, s};

    $('wbTrainR').textContent = money(s.trainRMSE);
    $('wbTestR').textContent = money(s.testRMSE);
    $('wbR2').textContent = s.testR2.toFixed(3);
    $('wbNZ').textContent = s.nonzero;
    $('wbNZsub').textContent = 'of ' + d.p + ' features';

    const col = COL[wb.method];
    Plot.barsH($('wbCoefs'), d.labels.map((l, i) => ({label: l, value: w[i], color: col})),
      {labelWidth: 126, height: Math.max(170, d.p * 28 + 42)});

    const pts = d.yte.map((y, i) => ({
      x: y, y: s.teHat[i], color: col,
      tip: `actual ₹${y.toFixed(0)}\npredicted ₹${s.teHat[i].toFixed(0)}\nerror ₹${(s.teHat[i] - y).toFixed(0)}`
    }));
    Plot.scatter($('wbScatter'), pts, {height: Math.max(170, d.p * 28 + 42), diagonal: true,
      xLabel: 'actual surge (₹)', yLabel: 'predicted (₹)',
      xDomain: [-10, 165], yDomain: [-10, 165]});

    const gap = s.trainRMSE - s.testRMSE;
    const dropped = d.labels.filter((l, i) => Math.abs(w[i]) < 1e-8);
    let note;
    if (wb.method === 'ols') {
      note = 'OLS has no penalty to tune; it fits the training split as tightly as the geometry '
        + 'allows. Compare its train RMSE with the others, then its test RMSE.';
    } else if (dropped.length) {
      note = `Dropped at α = ${wb.alpha.toFixed(2)}: <strong>${dropped.map(x => '<code>' + x + '</code>').join(', ')}</strong>. `
        + 'Lower α until each returns, and note the order: that is the importance ranking the '
        + 'penalty induces.';
    } else {
      note = 'Every feature still has a non-zero coefficient. Raise α to see which the penalty '
        + 'drops first.';
    }
    if (s.testRMSE > s.trainRMSE * 1.35) {
      note += ' <strong>Test error is well above training error</strong>, so the model is fitting '
        + 'the training split harder than it generalises.';
    }
    $('wbNote').innerHTML = note;
  }

  $('wbMethod').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    [...e.currentTarget.children].forEach(c => c.classList.toggle('on', c === b));
    wb.method = b.dataset.m;
    $('wbAlphaWrap').style.display = wb.method === 'ols' ? 'none' : '';
    $('wbRhoWrap').style.display = wb.method === 'enet' ? '' : 'none';
    drawWB();
  });
  $('wbAlpha').addEventListener('input', e => {
    wb.alpha = Math.pow(10, +e.target.value);
    $('wbAlphaV').textContent = wb.alpha < 1 ? wb.alpha.toFixed(3) : wb.alpha.toFixed(2);
    drawWB();
  });
  $('wbRho').addEventListener('input', e => {
    wb.rho = +e.target.value; $('wbRhoV').textContent = wb.rho.toFixed(2); drawWB();
  });
  $('wbCV').addEventListener('click', () => {
    if (wb.method === 'ols') return;
    const b = tune(wb.keys, wb.method, wb.rho);
    wb.alpha = b.alpha;
    $('wbAlpha').value = Math.log10(b.alpha);
    $('wbAlphaV').textContent = b.alpha < 1 ? b.alpha.toFixed(3) : b.alpha.toFixed(2);
    drawWB();
    $('wbNote').innerHTML = `5-fold CV on the training split picked <strong>α = ${b.alpha.toFixed(3)}</strong> `
      + `with CV RMSE ${money(b.cv)}. The test split was not consulted.`;
  });

  /* ========================= 2 · PATHS ========================= */
  const pathState = {method: 'ridge', bad: true};
  function drawPaths() {
    const keys = pathState.bad ? Prep.FEATURES.map(f => f.key) : Prep.BASE8;
    const d = Prep.build(keys);
    const grid = Prep.lamGrid(0.002, 160, 44);
    const series = d.labels.map((l, j) => ({name: l, points: [],
      color: Plot.colors[j % Plot.colors.length]}));
    grid.forEach(a => {
      const w = Prep.fit(d, pathState.method, a, 0.5);
      w.forEach((v, j) => series[j].points.push([a, v]));
    });
    const best = tune(keys, pathState.method, 0.5);
    Plot.line($('pathChart'), series, {height: 360, logX: true, dots: false, zeroLine: true,
      xLabel: 'penalty α (log scale)', yLabel: 'coefficient (₹ per s.d.)',
      vLine: best.alpha, vLabel: 'CV α = ' + best.alpha.toFixed(3)});
    Plot.legend($('pathLeg'), series.map(s => ({name: s.name, color: s.color})));

    const w0 = Prep.fit(d, pathState.method, grid[0], 0.5);
    const order = [];
    grid.forEach(a => {
      const w = Prep.fit(d, pathState.method, a, 0.5);
      w.forEach((v, j) => { if (Math.abs(v) < 1e-8 && !order.includes(j)) order.push(j); });
    });
    $('pathNote').innerHTML = pathState.method === 'ridge'
      ? 'Ridge paths approach zero without reaching it, so every coefficient survives at every α. '
        + 'Look instead at where <code>is_peak</code> and <code>is_rain</code> converge: that is the '
        + 'equal-split behaviour from <a href="theory.html#ridge-group">Theory §4.2</a>.'
      : order.length
        ? 'Elimination order as α grows: ' + order.map(j => '<code>' + d.labels[j] + '</code>').join(' → ')
          + '. The first to go should be the feature with no true effect; check it against '
          + '<a href="dataset.html#corr">the correlation matrix</a>. '
          + (pathState.method === 'enet'
              ? 'The correlated weather group tends to leave together rather than one at a time.'
              : 'Among correlated features Lasso keeps roughly one and discards its partners.')
        : 'No coefficient has hit zero across this α range.';
  }
  $('pathMethod').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    [...e.currentTarget.children].forEach(c => c.classList.toggle('on', c === b));
    pathState.method = b.dataset.m; drawPaths();
  });
  $('pathBad').addEventListener('change', e => { pathState.bad = e.target.checked; drawPaths(); });

  /* ========================= 3 · CV CURVES ========================= */
  let cvMethod = 'ridge';
  function drawCV() {
    const nUse = +$('cvN').value;
    $('cvNV').textContent = nUse;
    const d = Prep.build(Prep.BASE8);
    // Re-standardize inside the subsample: the estimator only ever sees these rows.
    const Xsub = d.Xtr.slice(0, nUse);
    const sc = ML.standardize(Xsub);
    const ysub = d.ytr.slice(0, nUse);
    const c = ML.center(ysub);
    const Zte = ML.applyScale(d.Xte, sc.mean, sc.std);

    const grid = Prep.lamGrid(0.002, 160, 34);
    const tr = [], te = [], cv = [];
    let best = null;
    grid.forEach(a => {
      const w = cvMethod === 'ridge' ? ML.ridge(sc.Z, c.yc, a * nUse).w
              : ML.elasticNet(sc.Z, c.yc, a, cvMethod === 'lasso' ? 1 : 0.5).w;
      tr.push([a, ML.rmse(ysub, ML.predict(sc.Z, w, c.mean))]);
      const teR = ML.rmse(d.yte, ML.predict(Zte, w, c.mean));
      te.push([a, teR]);
      const k = Math.min(5, nUse);
      const cvR = ML.kFoldCV(sc.Z, ysub, k, (X, y) => {
        const n = X.length;
        return cvMethod === 'ridge' ? ML.ridge(X, y, a * n).w
             : ML.elasticNet(X, y, a, cvMethod === 'lasso' ? 1 : 0.5).w;
      });
      cv.push([a, cvR]);
      if (!best || cvR < best.cv) best = {alpha: a, cv: cvR, test: teR};
    });
    const series = [
      {name: 'Training RMSE', color: 'var(--c-ols)', points: tr, dashed: true},
      {name: '5-fold CV RMSE', color: 'var(--accent)', points: cv, width: 2.6},
      {name: 'Held-out test RMSE', color: 'var(--rose)', points: te}
    ];
    const allY = series.flatMap(x => x.points.map(p => p[1])).filter(isFinite);
    Plot.line($('cvChart'), series, {height: 320, logX: true, dots: false,
      xLabel: 'penalty α (log scale)', yLabel: 'RMSE (₹)',
      yDomain: [Math.min(...allY) * 0.9, Math.min(Math.max(...allY), Math.min(...allY) * 4.5)],
      vLine: best.alpha, vLabel: 'CV minimum'});
    Plot.legend($('cvLeg'), series.map(x => ({name: x.name, color: x.color})));
    $('cvBest').textContent = best.alpha.toFixed(3);
    $('cvBestR').textContent = money(best.cv);
    $('cvTestR').textContent = money(best.test);

    const atEdge = best.alpha <= grid[1];
    const cvFlat = Math.max(...cv.slice(0, 12).map(p => p[1])) - best.cv;
    $('cvNote').innerHTML = atEdge
      ? `With <strong>${nUse} training trips</strong> the CV minimum sits at the smallest α on the `
        + `grid: cross-validation is correctly reporting that this problem does not need `
        + `regularization. Reduce the sample size and the minimum moves inward.`
      : `With <strong>${nUse} training trips</strong> the CV curve has an interior minimum at `
        + `α = ${best.alpha.toFixed(3)}. There is now less data than the model wants, so the penalty `
        + `earns its place. The training curve still prefers α → 0, which is why you cannot select `
        + `on it.`
      + (cvFlat < 0.5 ? ' The bottom is nearly flat, so quote α to one significant figure at most.' : '');
  }
  $('cvMethod').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    [...e.currentTarget.children].forEach(c => c.classList.toggle('on', c === b));
    cvMethod = b.dataset.m; drawCV();
  });
  $('cvN').addEventListener('input', drawCV);

  /* ========================= 4 · COMPARISON TABLE ========================= */
  function drawCompare() {
    const keys = Prep.FEATURES.map(f => f.key);
    const d = Prep.build(keys);
    const rows = ['ols', 'ridge', 'lasso', 'enet'].map(m => {
      const t = tune(keys, m, 0.5);
      const w = Prep.fit(d, m, t.alpha, 0.5);
      const s = Prep.score(d, w);
      const dropped = d.labels.filter((l, i) => Math.abs(w[i]) < 1e-8);
      return {m, alpha: t.alpha, s, dropped};
    });
    const bestTest = Math.min(...rows.map(r => r.s.testRMSE));
    $('cmpTable').querySelector('tbody').innerHTML = rows.map(r => `
      <tr>
        <td><span class="badge ${r.m === 'ols' ? 'mle' : r.m}">${NAME[r.m]}</span></td>
        <td class="num">${r.m === 'ols' ? '—' : r.alpha.toFixed(3)}</td>
        <td class="num">${r.s.trainRMSE.toFixed(2)}</td>
        <td class="num"${r.s.testRMSE === bestTest ? ' style="color:var(--accent);font-weight:700"' : ''}>${r.s.testRMSE.toFixed(2)}</td>
        <td class="num">${r.s.testR2.toFixed(3)}</td>
        <td class="num">${r.s.nonzero} / ${d.p}</td>
        <td class="small">${r.dropped.length ? r.dropped.map(x => '<code>' + x + '</code>').join(' ') : '<span class="muted">none</span>'}</td>
      </tr>`).join('');
    const spread = Math.max(...rows.map(r => r.s.testRMSE)) - bestTest;
    $('cmpNote').innerHTML = `The best and worst held-out RMSE differ by <strong>₹${spread.toFixed(2)}</strong>`
      + (spread < 2
        ? ', on fares of a few hundred. When the statistical difference is this small the deployment '
          + 'decision is operational: interpretability, stability across refits, and how many '
          + 'coefficients the team has to defend. Say so rather than declaring a winner on the third '
          + 'decimal.'
        : '. That is a real gap. Identify which model earns it and whether it survives refitting on '
          + 'a different split.');
  }

  /* ========================= 5 · COLLINEARITY ========================= */
  (function initPairs() {
    const opts = Prep.FEATURES.map(f =>
      `<option value="${f.key}">${f.label} — ${f.pretty}</option>`).join('');
    $('colA').innerHTML = opts; $('colB').innerHTML = opts;
    $('colA').value = 'is_rain'; $('colB').value = 'traffic_speed_kmph';
  })();

  function drawCollinear() {
    const ka = $('colA').value, kb = $('colB').value;
    if (ka === kb) { $('colB').value = Prep.FEATURES.find(f => f.key !== ka).key; }
    const keys = [$('colA').value, $('colB').value];
    const labels = keys.map(k => Prep.FEATURES.find(f => f.key === k).label);
    const n = +$('colN').value;
    const lam = Math.pow(10, +$('colLam').value);
    $('colNV').textContent = n;
    $('colLamV').textContent = lam < 1 ? lam.toFixed(3) : lam.toFixed(2);

    const d = Prep.build(keys);

    /* Fit on the first k rows, standardizing WITHIN that subsample — otherwise the
       conditioning you display is not the conditioning the estimator experiences. */
    function fitAt(k) {
      const X = d.Xtr.slice(0, k), y = d.ytr.slice(0, k);
      const sc = ML.standardize(X), c = ML.center(y);
      const XtX = ML.matmul(ML.T(sc.Z), sc.Z);
      const o = ML.ols(sc.Z, c.yc);
      return {
        ols: o.singular ? [NaN, NaN] : o.w,
        ridge: ML.ridge(sc.Z, c.yc, lam * k).w,
        cond: ML.condition(XtX),
        r: ML.corr(X.map(r0 => r0[0]), X.map(r0 => r0[1]))
      };
    }

    const olsA = [], olsB = [], ridA = [], ridB = [];
    for (let k = 3; k <= 150; k++) {
      const f = fitAt(k);
      if (isFinite(f.ols[0])) { olsA.push([k, f.ols[0]]); olsB.push([k, f.ols[1]]); }
      ridA.push([k, f.ridge[0]]); ridB.push([k, f.ridge[1]]);
    }
    const cur = fitAt(n);

    $('colOlsPeak').textContent = isFinite(cur.ols[0]) ? cur.ols[0].toFixed(1) : 'singular';
    $('colOlsRain').textContent = isFinite(cur.ols[1]) ? cur.ols[1].toFixed(1) : 'singular';
    $('colR').textContent = cur.r.toFixed(2);
    $('colCond').textContent = isFinite(cur.cond) ? cur.cond.toFixed(1) : '∞';

    /* Both panels share a y-range so the visual contrast is not an artefact of scaling. */
    const all = olsA.concat(olsB).map(p => p[1]).filter(isFinite);
    const lo = Math.min(...all, 0), hi = Math.max(...all, 0);
    const padv = (hi - lo) * 0.08 + 1;
    const yD = [lo - padv, hi + padv];

    const sO = [{name: labels[0], points: olsA, color: 'var(--c-ols)'},
                {name: labels[1], points: olsB, color: 'var(--rose)'}];
    const sR = [{name: labels[0], points: ridA, color: 'var(--c-ridge)'},
                {name: labels[1], points: ridB, color: 'var(--accent)'}];
    Plot.line($('colOlsBox'), sO, {height: 250, dots: false, zeroLine: true, yDomain: yD,
      xLabel: 'training trips n', yLabel: 'coefficient (₹ per s.d.)', vLine: n});
    Plot.line($('colRidgeBox'), sR, {height: 250, dots: false, zeroLine: true, yDomain: yD,
      xLabel: 'training trips n', yLabel: 'coefficient (₹ per s.d.)', vLine: n});
    Plot.legend($('colOlsLeg'), sO.map(x => ({name: x.name, color: x.color})));
    Plot.legend($('colRidgeLeg'), sR.map(x => ({name: x.name, color: x.color})));

    /* How much does ONE extra trip move the estimate?  That is the instability, quantified. */
    const nxt = fitAt(Math.min(n + 1, 150));
    const jumpO = isFinite(cur.ols[0]) && isFinite(nxt.ols[0])
      ? Math.max(Math.abs(nxt.ols[0] - cur.ols[0]), Math.abs(nxt.ols[1] - cur.ols[1])) : NaN;
    const jumpR = Math.max(Math.abs(nxt.ridge[0] - cur.ridge[0]), Math.abs(nxt.ridge[1] - cur.ridge[1]));

    $('colNote').innerHTML = !isFinite(cur.ols[0])
      ? '<strong>XᵀX is singular.</strong> With this few trips the two columns are linearly '
        + 'dependent and the OLS estimator does not exist: there is no unique answer. Ridge still '
        + 'returns one, because adding λI makes every eigenvalue strictly positive.'
      : `Subsample correlation <strong>r = ${cur.r.toFixed(2)}</strong>, condition number `
        + `<strong>${cur.cond.toFixed(1)}</strong>. Adding one more trip moves the OLS estimate by `
        + `<strong>₹${isFinite(jumpO) ? jumpO.toFixed(1) : '—'}</strong> and the Ridge estimate by `
        + `<strong>₹${jumpR.toFixed(1)}</strong>. `
        + (jumpO > jumpR * 2.5
            ? 'That ratio is the argument for regularization: same data, same information, very '
              + 'different sensitivity to one observation.'
            : 'At this sample size both are stable. Slide n back toward 5 to break OLS, or choose a '
              + 'more strongly correlated pair such as <code>is_rain</code> and '
              + '<code>is_bad_weather</code>.');
  }
  ['colN', 'colLam'].forEach(id => $(id).addEventListener('input', drawCollinear));
  ['colA', 'colB'].forEach(id => $(id).addEventListener('change', drawCollinear));

  /* ========================= 6 · p >> n ========================= */
  let ncMethod = 'lasso';
  const NC = window.NEWCITY_DATA;
  function drawNewCity() {
    const a = Math.pow(10, +$('ncAlpha').value);
    $('ncAlphaV').textContent = a.toFixed(2);
    const yi = NC.columns.indexOf('surge_additive_inr');
    const fcols = NC.columns.filter(c => c !== 'surge_additive_inr');
    const X = NC.rows.map(r => fcols.map(c => r[NC.columns.indexOf(c)]));
    const y = NC.rows.map(r => r[yi]);
    const sc = ML.standardize(X), c = ML.center(y);
    const n = X.length;
    const w = ncMethod === 'ridge' ? ML.ridge(sc.Z, c.yc, a * n).w
            : ML.elasticNet(sc.Z, c.yc, a, ncMethod === 'lasso' ? 1 : 0.5).w;
    const sel = w.map((v, j) => ({j, v})).filter(o => Math.abs(o.v) > 1e-8);
    const truth = NC.truth;
    const truthNames = Object.keys(truth);
    const hit = sel.filter(o => truthNames.includes(fcols[o.j])).length;
    const rm = ML.rmse(y, ML.predict(sc.Z, w, c.mean));

    $('ncSel').textContent = sel.length;
    $('ncHit').textContent = hit + ' / 6';
    $('ncFP').textContent = sel.length - hit;
    $('ncRmse').textContent = money(rm);

    const top = sel.slice().sort((p, q) => Math.abs(q.v) - Math.abs(p.v)).slice(0, 16);
    Plot.barsH($('ncChart'), top.map(o => ({
      label: fcols[o.j], value: o.v,
      color: truthNames.includes(fcols[o.j]) ? 'var(--accent)' : 'var(--rose)'
    })), {labelWidth: 168, height: Math.max(160, top.length * 25 + 44)});

    const missed = truthNames.filter(t => !sel.some(o => fcols[o.j] === t));
    $('ncNote').innerHTML = ncMethod === 'ridge'
      ? '<strong>Ridge keeps all 120 features.</strong> Nothing is zero, so there is no shortlist '
        + 'for the operations team: every coefficient is small and the true drivers are buried among '
        + '114 noise features. This is why the new-city problem needs an ℓ₁ penalty.'
      : `Green bars are the six real drivers; red bars are noise or correlated decoys. `
        + (missed.length
            ? `Missed: <strong>${missed.map(m => '<code>' + m + '</code>').join(', ')}</strong>. `
              + `Each has a planted partner correlated at r ≈ 0.9, and the penalty has no reason to `
              + `pay for both.`
            : `All six recovered. Raise α to see which go first.`)
        + ` Note: with n = 60, Lasso can select at most 60 features no matter how small α is.`;
  }
  $('ncAlpha').addEventListener('input', drawNewCity);
  $('ncMethod').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    [...e.currentTarget.children].forEach(x => x.classList.toggle('on', x === b));
    ncMethod = b.dataset.m; drawNewCity();
  });

  /* ========================= export ========================= */
  $('exportBtn').addEventListener('click', () => {
    if (!wb.last) return;
    const {d, w, s} = wb.last;
    const lines = ['feature,coefficient_scaled,coefficient_raw_per_unit'];
    const raw = Prep.unscale(d, w);
    d.keys.forEach((k, j) => lines.push(`${k},${w[j].toFixed(6)},${raw[j].toFixed(6)}`));
    lines.push('', 'metric,value');
    lines.push(`method,${wb.method}`, `alpha,${wb.alpha}`,
      `train_rmse,${s.trainRMSE.toFixed(4)}`, `test_rmse,${s.testRMSE.toFixed(4)}`,
      `test_r2,${s.testR2.toFixed(4)}`, `nonzero,${s.nonzero}`);
    UI.downloadText('workbench_fit.csv', lines.join('\n'), 'text/csv');
  });

  /* ========================= boot ========================= */
  UI.onDraw(() => {
    drawWB(); drawPaths(); drawCV(); drawCompare(); drawCollinear(); drawNewCity();
  });
})();
