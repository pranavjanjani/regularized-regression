/* =============================================================================
   algorithms.js — step-by-step, visual executions of every estimator.

   All five walkthroughs share one two-feature problem drawn from the Mumbai
   data.  Both X and y are standardized here, purely so the parameter space has
   a readable scale: a coefficient of 1 means "one standard deviation of surge
   per standard deviation of the feature".  Multiply by sd(y) to get rupees.
============================================================================= */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const F = (v, d = 3) => (v == null || !isFinite(v)) ? '—' : v.toFixed(d);

  /* ======================================================================
     Shared problem state
  ====================================================================== */
  const S = {keys: ['is_peak', 'traffic_speed_kmph'], n: 60};

  function rebuild() {
    const d = Prep.build(S.keys);
    const X = d.Xtr.slice(0, S.n);
    const yRaw = d.ytr.slice(0, S.n);
    const sc = ML.standardize(X);
    const ym = ML.mean(yRaw);
    const ysd = Math.sqrt(ML.mean(yRaw.map(v => (v - ym) ** 2))) || 1;
    const y = yRaw.map(v => (v - ym) / ysd);

    const n = X.length;
    const Zt = ML.T(sc.Z);
    const A = ML.matmul(Zt, sc.Z).map(r => r.map(v => v / n));   // = correlation matrix
    const b = ML.matvec(Zt, y).map(v => v / n);                  // = corr(x_j, y)
    const Ainv = ML.inv(A);
    const singular = !Ainv;
    const wOls = Ainv ? ML.matvec(Ainv, b) : null;

    /* When XᵀX is singular the loss is a flat-bottomed trough rather than a bowl:
       every point along one direction fits the data exactly as well. We still need
       somewhere to centre the picture, so use the minimum-norm solution, and widen
       the smallest eigenvalue very slightly so the contours are drawable. */
    const Areg = [[A[0][0] + 1e-3, A[0][1]], [A[1][0], A[1][1] + 1e-3]];
    const wMin = ML.matvec(ML.inv(Areg), b);
    const centre = wOls || wMin;
    const eg = ML.eigSym(A);
    const nullDir = [eg.vectors[0][1], eg.vectors[1][1]];   // smallest-eigenvalue direction

    Object.assign(S, {
      d, Z: sc.Z, y, yRaw, ym, ysd, n, A, b, Ainv, wOls, singular,
      Acontour: singular ? Areg : A, centre, wMin, nullDir,
      r: A[0][1], det: A[0][0] * A[1][1] - A[0][1] * A[1][0],
      labels: S.keys.map(k => Prep.FEATURES.find(f => f.key === k).label),
      lim: Math.max(1.5, Math.min(3.2,
        Math.max(Math.abs(centre[0]), Math.abs(centre[1])) * 1.55))
    });
    S.levels = [0.015, 0.05, 0.12, 0.24, 0.42, 0.66, 0.98, 1.4];
    return S;
  }

  /* loss  f(w) = ½(1 − 2wᵀb + wᵀAw) */
  const loss = w => 0.5 * (1 - 2 * (w[0] * S.b[0] + w[1] * S.b[1])
    + w[0] * (S.A[0][0] * w[0] + S.A[0][1] * w[1])
    + w[1] * (S.A[1][0] * w[0] + S.A[1][1] * w[1]));
  const l1 = w => Math.abs(w[0]) + Math.abs(w[1]);
  const l2sq = w => w[0] * w[0] + w[1] * w[1];

  /* ======================================================================
     Small HTML builders
  ====================================================================== */
  function mat(M, o = {}) {
    const hi = o.highlight || [];
    const cells = M.map((row, i) => row.map((v, j) => {
      const on = hi.some(h => h[0] === i && h[1] === j);
      const dim = o.dim && o.dim.some(h => h[0] === i && h[1] === j);
      return `<div class="cell${on ? ' on' : ''}${dim ? ' dim' : ''}">${F(v, o.dp == null ? 3 : o.dp)}</div>`;
    }).join('')).join('');
    return `<div class="mat" style="grid-template-columns:repeat(${M[0].length},auto)">${cells}</div>`;
  }
  const vec = (v, o = {}) => mat(v.map(x => [x]), o);
  const row = (label, html, cap) =>
    `<div class="mat-row"><span class="lbl">${label}</span>${html}</div>`
    + (cap ? `<p class="mat-cap">${cap}</p>` : '');
  const kv = items => items.map(([k, v, cls]) =>
    `<div class="kv"><span class="k">${k}</span><span class="v ${cls || ''}">${v}</span></div>`).join('');

  /* ======================================================================
     Step player
  ====================================================================== */
  function makePlayer(root, spec) {
    const bar = root.querySelector('.player');
    bar.innerHTML = `
      <div class="player-bar">
        <button class="pbtn" data-a="reset" title="Back to the start">⟲ Reset</button>
        <button class="pbtn" data-a="prev">◀ Back</button>
        <button class="pbtn go" data-a="next">Step ▶</button>
        <button class="pbtn" data-a="play">▶ Play</button>
        <button class="pbtn" data-a="end" title="Jump to the final state">Skip to end ⏭</button>
        <div class="pstat"><b class="ptitle">—</b><br><span class="pcount"></span></div>
      </div>
      <div class="ptrack"><div class="pfill" style="width:0%"></div></div>`;
    const st = {i: 0, steps: [], timer: null};

    function render() {
      const s = st.steps[st.i];
      if (!s) return;
      root.querySelector('.math-pane').innerHTML =
        `<div class="step-title"><span class="sn">STEP ${st.i}</span>${s.title}</div>` + (s.math || '');
      root.querySelector('.narration').innerHTML = s.narration || '';
      bar.querySelector('.ptitle').textContent = s.short || s.title;
      bar.querySelector('.pcount').textContent = `step ${st.i} of ${st.steps.length - 1}`;
      bar.querySelector('.pfill').style.width =
        (100 * st.i / Math.max(st.steps.length - 1, 1)) + '%';
      bar.querySelector('[data-a="prev"]').disabled = st.i === 0;
      bar.querySelector('[data-a="next"]').disabled = st.i >= st.steps.length - 1;
      if (spec.viz) spec.viz(root, s, st.i);
      if (window.renderMathInElement) {
        [root.querySelector('.math-pane'), root.querySelector('.narration')].forEach(el => {
          try {
            window.renderMathInElement(el, {delimiters: [
              {left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}],
              throwOnError: false});
          } catch (e) {}
        });
      }
    }
    function stop() {
      clearInterval(st.timer); st.timer = null;
      bar.querySelector('[data-a="play"]').textContent = '▶ Play';
    }
    bar.addEventListener('click', e => {
      const b = e.target.closest('.pbtn'); if (!b) return;
      const a = b.dataset.a;
      if (a === 'next') { stop(); st.i = Math.min(st.i + 1, st.steps.length - 1); render(); }
      if (a === 'prev') { stop(); st.i = Math.max(st.i - 1, 0); render(); }
      if (a === 'reset') { stop(); st.i = 0; render(); }
      if (a === 'end') { stop(); st.i = st.steps.length - 1; render(); }
      if (a === 'play') {
        if (st.timer) { stop(); return; }
        if (st.i >= st.steps.length - 1) st.i = 0;
        b.textContent = '❙❙ Pause';
        st.timer = setInterval(() => {
          if (st.i >= st.steps.length - 1) { stop(); return; }
          st.i++; render();
        }, spec.speed || 1150);
      }
    });
    return {
      rebuild(keepPos) {
        const was = st.i;
        st.steps = spec.steps();
        st.i = keepPos ? Math.min(was, st.steps.length - 1) : 0;
        render();
      },
      /** Say so on the page instead of leaving stale numbers on screen. */
      showError(err) {
        stop();
        root.querySelector('.math-pane').innerHTML =
          '<div class="step-title">Not available for this pair</div>'
          + '<p class="small">This estimator could not be built for the features and sample size '
          + 'currently selected above. Pick a different pair, or raise the number of trips.</p>'
          + '<p class="muted small" style="font-family:var(--mono);font-size:.78rem">'
          + String(err && err.message || err) + '</p>';
        root.querySelector('.viz-pane').innerHTML = '';
        root.querySelector('.narration').innerHTML = '';
        ['.slice-pane', '.obj-pane', '.soft-pane', '.cmp-pane', '.tau-pane'].forEach(sel => {
          const el2 = root.querySelector(sel);
          if (el2) el2.innerHTML = '';
        });
      },
      render, stop
    };
  }

  /* ======================================================================
     Shared visual: the parameter plane
  ====================================================================== */
  function plane(host, o) {
    Plot.param2d(host, Object.assign({
      lim: S.lim, labels: S.labels.map((l, i) => `w${i + 1} · ${l}`),
      maxSide: 420
    }, o));
  }

  /* ======================================================================
     0 · SETUP PANEL
  ====================================================================== */
  (function setup() {
    const opts = Prep.FEATURES.map(f =>
      `<option value="${f.key}">${f.label} — ${f.pretty}</option>`).join('');
    $('fA').innerHTML = opts; $('fB').innerHTML = opts;
    $('fA').value = S.keys[0]; $('fB').value = S.keys[1];

    function drawSetup() {
      const head = `<table class="data-mini"><thead><tr><th style="text-align:left">trip</th>
        <th>z₁ ${S.labels[0]}</th><th>z₂ ${S.labels[1]}</th><th>y (std)</th><th>y (₹)</th></tr></thead><tbody>`;
      const body = S.Z.slice(0, 6).map((r, i) =>
        `<tr><td style="text-align:left">${i + 1}</td><td>${F(r[0], 2)}</td><td>${F(r[1], 2)}</td>
         <td>${F(S.y[i], 2)}</td><td>${F(S.yRaw[i], 0)}</td></tr>`).join('');
      $('setupZ').innerHTML = '<div class="scroll-x">' + head + body + '</tbody></table></div>';

      Plot.scatter($('setupScatter'), S.Z.map((r, i) => ({
        x: r[0], y: r[1],
        color: S.y[i] > 0 ? 'var(--rose)' : 'var(--c-ridge)',
        r: 3 + Math.min(Math.abs(S.y[i]) * 2.4, 6),
        tip: `trip ${i + 1}\nz₁ = ${F(r[0], 2)}  z₂ = ${F(r[1], 2)}\nsurge ₹${F(S.yRaw[i], 0)}`
      })), {height: 300, xLabel: 'z₁ · ' + S.labels[0], yLabel: 'z₂ · ' + S.labels[1], opacity: .75});

      const ar = Math.abs(S.r);
      $('setupCorr').innerHTML =
        `Correlation between the two columns: <strong>r = ${F(S.r, 3)}</strong>, so
         det(XᵀX/n) = 1 − r² = <strong>${F(S.det, 3)}</strong>.
         Marker size is |surge|; red is above-average surge, blue below. `
        + (ar > 0.7
            ? 'The cloud lies along a line — the two features are close to carrying the same '
              + 'information, and every estimator below will struggle to tell them apart.'
            : ar > 0.35
              ? 'A visible tilt: the features overlap somewhat but are still distinguishable.'
              : 'A round cloud — these two are nearly independent, which is the easy case.');
    }

    function refresh(keep) {
      if ($('fA').value === $('fB').value) {
        $('fB').value = Prep.FEATURES.find(f => f.key !== $('fA').value).key;
      }
      S.keys = [$('fA').value, $('fB').value];
      S.n = +$('nObs').value;
      $('nObsV').textContent = S.n;
      rebuild();
      drawSetup();
      /* Rebuild every player independently. Without this, one throwing estimator
         aborted the loop and left the remaining panels displaying numbers from the
         previous feature pair — a silent wrong answer, worse than a visible error. */
      ALL.forEach(p => {
        try {
          p.rebuild(keep);
        } catch (err) {
          console.error('walkthrough failed to rebuild', err);
          p.showError(err);
        }
      });
    }
    S.refresh = refresh;
    ['fA', 'fB'].forEach(id => $(id).addEventListener('change', () => refresh(false)));
    $('nObs').addEventListener('input', () => refresh(true));
    $('setupPreset').addEventListener('click', () => {
      // The first ten trips make is_rain and is_bad_weather identical columns,
      // so XᵀX is exactly singular and least squares has no answer at all.
      $('fA').value = 'is_rain'; $('fB').value = 'is_bad_weather';
      $('nObs').value = 10; refresh(false);
    });
    S.drawSetup = drawSetup;
  })();

  /* ======================================================================
     1 · OLS — the normal equations, entry by entry
  ====================================================================== */
  const olsPlayer = makePlayer($('wOls'), {
    speed: 1250,
    steps() {
      const {A, b, det, Ainv, wOls, labels, n, r} = S;
      const st = [];
      const zero = [[0, 0], [0, 0]];

      st.push({
        short: 'Standardize', title: 'Centre and standardize',
        math: row('Z =', `<span class="muted small">n = ${n} rows × 2 columns — see the table above</span>`)
          + row('ȳ =', `<span class="muted small">removed, so no intercept is fitted</span>`)
          + `<div style="margin-top:14px">${kv([
              ['rows used', n], ['mean of each column', '0.000'],
              ['sd of each column', '1.000'], ['mean of y', '0.000']])}</div>`,
        narration: `<p>Every column is rescaled to mean 0 and standard deviation 1, and the target
          is centred. Two consequences follow. No intercept is needed, since it would be exactly
          zero. And the Gram matrix $X^\\top X / n$ becomes the <strong>correlation matrix</strong>,
          which is why the next step is so readable.</p>`,
        viz: {stage: 0}
      });

      const entryNarr = [
        `<p>The top-left entry is $\\frac{1}{n}\\sum_i z_{i1}^2$. Column 1 was standardized to unit
          variance, so this is <span class="hl">exactly 1</span>.</p>`,
        `<p>The off-diagonal is $\\frac{1}{n}\\sum_i z_{i1} z_{i2}$, which for standardized columns
          is the correlation coefficient: <span class="hl">r = ${F(r, 3)}</span>.</p>
          <p>Almost everything that follows depends on this number. Collinearity, in this course,
          means this entry is close to ±1.</p>`,
        `<p>Bottom-right is 1 again, so the Gram matrix is
          $\\begin{pmatrix}1 & r\\\\ r & 1\\end{pmatrix}$: a two-parameter problem reduced to one
          number.</p>`
      ];
      [[0, 0], [0, 1], [1, 1]].forEach((h, k) => {
        const shown = [[k >= 0 ? A[0][0] : 0, k >= 1 ? A[0][1] : 0],
                       [k >= 1 ? A[1][0] : 0, k >= 2 ? A[1][1] : 0]];
        const dim = [];
        if (k < 1) { dim.push([0, 1], [1, 0]); }
        if (k < 2) dim.push([1, 1]);
        st.push({
          short: 'Gram entry ' + (k + 1),
          title: `Build $X^\\top X / n$ — entry (${h[0] + 1},${h[1] + 1})`,
          math: row('XᵀX / n =', mat(shown, {highlight: k === 1 ? [[0, 1], [1, 0]] : [h], dim}))
            + `<p class="mat-cap">after standardizing, this is the correlation matrix of the two features</p>`,
          narration: entryNarr[k],
          viz: {stage: k >= 2 ? 1 : 0}
        });
      });

      st.push({
        short: 'Xᵀy', title: 'Build $X^\\top y / n$',
        math: row('XᵀX / n =', mat(A)) + row('Xᵀy / n =', vec(b, {highlight: [[0, 0], [1, 0]]}))
          + `<p class="mat-cap">each entry is the correlation of that feature with the target</p>`,
        narration: `<p>With everything standardized, $X^\\top y/n$ holds each feature's correlation
          with surge: <span class="hl">${F(b[0], 3)}</span> and
          <span class="hl">${F(b[1], 3)}</span>.</p>
          <p>The loss surface is now fully determined, so it can be drawn. It is a bowl; the task is
          to find its lowest point.</p>`,
        viz: {stage: 2}
      });

      st.push({
        short: 'Determinant', title: 'The determinant decides everything',
        math: row('det =', `<span class="lbl">1 · 1 − (${F(r, 3)})² = <strong>${F(det, 4)}</strong></span>`)
          + `<div style="margin-top:12px">${kv([
              ['correlation r', F(r, 3)],
              ['determinant 1 − r²', F(det, 4), det < 0.15 ? 'warn' : 'good'],
              ['condition number', F((1 + Math.abs(r)) / (1 - Math.abs(r)), 1),
               Math.abs(r) > 0.8 ? 'warn' : ''],
              ['variance inflation 1/(1−r²)', F(1 / det, 2), det < 0.15 ? 'warn' : '']])}</div>`,
        narration: `<p>For a $2\\times2$ matrix the determinant is $1 - r^2$, here
          <span class="hl">${F(det, 4)}</span>. The inverse divides by it, so every coefficient and
          every standard error is multiplied by $1/(1-r^2) =$
          <span class="hl">${F(1 / det, 2)}</span>.</p>
          ${det < 0.15
            ? `<p><strong>That is small.</strong> The two features are nearly the same column, the
               data barely distinguishes them, and least squares will amplify noise by a factor of
               ${F(1 / det, 1)}.</p>`
            : `<p>Far enough from zero that the inverse is well behaved. Choose a more correlated
               pair, or reduce the sample size, to make it collapse.</p>`}`,
        viz: {stage: 2}
      });

      st.push({
        short: 'Invert', title: 'Invert the Gram matrix',
        math: S.singular
          ? `<div class="mat-row"><span class="lbl">(XᵀX/n)⁻¹ =</span>
               <span class="lbl" style="color:var(--rose);font-weight:700">does not exist</span></div>
             <p class="mat-cap">dividing by a determinant of ${F(det, 6)} is not an operation</p>`
          : row('(XᵀX/n)⁻¹ =', mat(Ainv)) + `<p class="mat-cap">= (1/${F(det, 4)}) × [[1, −r], [−r, 1]]</p>`,
        narration: S.singular
          ? `<p><strong>There is no inverse.</strong> The two columns are linearly dependent on
             these rows, so $X^\\top X$ has a zero eigenvalue and cannot be inverted in exact
             arithmetic.</p>
             <p>The contours are now parallel lines rather than ellipses: the loss surface is a
             flat-bottomed trough and the red dashed line is its floor. Every point on that line
             fits identically well, so there is no unique maximum-likelihood estimate.</p>`
          : `<p>For a symmetric $2\\times2$ matrix the inverse is
             $\\frac{1}{1-r^2}\\begin{pmatrix}1 & -r\\\\ -r & 1\\end{pmatrix}$.</p>
             <p>Note the negative off-diagonal: raise one coefficient and the other must fall. When
             $r$ is large, that trade-off happens at a large scale.</p>`,
        viz: {stage: 2}
      });

      st.push({
        short: 'Solve', title: 'Multiply to get $\\hat w$',
        math: S.singular ? `<p style="color:var(--rose);font-weight:650;margin:14px 0 6px">
            No solution exists — there is nothing to multiply by.</p>
            <p class="small">Any point along the trough is a least-squares solution. Conventionally
            one reports the <em>minimum-norm</em> one,
            (${F(S.wMin[0], 3)}, ${F(S.wMin[1], 3)}), but that is a tie-breaking convention, not an
            estimate the data supports. Ridge, in the next section, makes the tie-breaking explicit
            and principled instead of arbitrary.</p>`
          : row('ŵ =', mat(Ainv)) + row('×', vec(b))
          + row('=', vec(wOls, {highlight: [[0, 0], [1, 0]]}))
          + `<div style="margin-top:12px">${kv([
              [`w₁ · ${labels[0]}`, F(wOls[0], 4)],
              [`w₂ · ${labels[1]}`, F(wOls[1], 4)],
              ['in rupees per s.d.', `₹${F(wOls[0] * S.ysd, 1)} and ₹${F(wOls[1] * S.ysd, 1)}`]])}</div>`,
        narration: S.singular
          ? `<p>This is the situation Task 1b asks you to describe precisely. "The MLE does not
             exist" is a statement about the likelihood surface — it has no unique maximum — not
             about NumPy raising an exception.</p>
             <p>Add more trips with the slider above and watch the trough close into a bowl the
             moment a row arrives where the two features finally disagree.</p>`
          : `<p>Written out, $\\hat w_1 = (b_1 - r\\,b_2)/(1-r^2)$, and symmetrically for
          $\\hat w_2$. Each coefficient is that feature's correlation with the target, minus the part
          the other feature already explains, divided by the determinant.</p>
          <p>That subtraction is why adding a correlated feature can flip a coefficient's sign.</p>`,
        viz: {stage: 3}
      });

      const wShow = wOls || S.wMin;
      const yh = ML.predict(S.Z, wShow, 0);
      st.push({
        short: 'Check the fit', title: 'Check: residuals and fit',
        math: `<div>${kv([
            ['loss  f(ŵ)', F(loss(wShow), 4), 'good'],
            ['R² on these rows', F(ML.r2(S.y, yh), 3)],
            ['RMSE (standardized)', F(ML.rmse(S.y, yh), 3)],
            ['RMSE in rupees', '₹' + F(ML.rmse(S.y, yh) * S.ysd, 1)],
            ['‖ŵ‖₂', F(Math.sqrt(l2sq(wShow)), 3)],
            ['‖ŵ‖₁', F(l1(wShow), 3)]])}</div>
          <p class="muted small" style="margin-top:12px">Nothing was penalized, so this is the lowest
          loss any linear model can reach on these rows — and, for exactly that reason, not
          necessarily the lowest on rows it has not seen.</p>`,
        narration: `<p>The solution sits at the bottom of the bowl, where the gradient is zero and
          the residual is orthogonal to both columns: the maximum-likelihood estimate under Gaussian
          noise.</p>
          <p>Note the norms. <span class="hl">‖ŵ‖₂ = ${F(Math.sqrt(l2sq(wShow)), 3)}</span> and
          <span class="hl">‖ŵ‖₁ = ${F(l1(wShow), 3)}</span> are what Ridge and Lasso charge for.</p>`,
        viz: {stage: 3}
      });
      return st;
    },
    viz(root, s) {
      const host = root.querySelector('.viz-pane');
      const stage = s.viz.stage;
      plane(host, {
        contours: stage >= 2 ? [{A: S.Acontour, center: S.centre, levels: S.levels}]
                : stage >= 1 ? [{A: S.Acontour, center: S.centre, levels: S.levels, opacity: .22, dashed: true}]
                : [],
        points: stage >= 3 && S.wOls ? [{w: S.wOls, label: 'ŵ OLS', color: 'var(--c-ols)'}] : [],
        path: S.singular && stage >= 2 ? [{pts: [
          [S.centre[0] - 9 * S.nullDir[0], S.centre[1] - 9 * S.nullDir[1]],
          [S.centre[0] + 9 * S.nullDir[0], S.centre[1] + 9 * S.nullDir[1]]],
          color: 'var(--rose)', dashed: true, dots: false, width: 2}] : []
      });
      host.insertAdjacentHTML('beforeend',
        `<p class="muted small" style="margin:8px 0 0">${
          stage < 1 ? 'Nothing can be drawn yet — the loss surface needs XᵀX.'
          : stage < 2 ? 'The <em>shape</em> of the bowl is fixed by XᵀX: narrow and tilted when the features are correlated. Its position still needs Xᵀy.'
          : stage < 3 ? 'The full loss surface. Contours are ellipses; their elongation is the collinearity you measured as r.'
          : 'The solution sits at the exact centre of the ellipses.'}</p>`);
    }
  });

  /* ======================================================================
     2 · MAP — building the penalty out of a prior

     The other walkthroughs take a penalized objective as given. This one
     constructs it: likelihood × prior, take logs, and the penalty appears,
     with λ = σ²/τ² falling out of the algebra rather than being asserted.
  ====================================================================== */
  /* τ defaults to 0.03 rather than 1: with 60 trips a prior of width 1 is so vague
     that it moves the estimate by less than a rupee, and the walkthrough would open
     on a contest that is not happening. 0.03 is where the two sides are comparable. */
  let mapTau = 0.03, mapSigma = 10, mapPrior = 'gauss';

  const mapPlayer = makePlayer($('wMap'), {
    speed: 1400,
    steps() {
      const n = S.n, Z = S.Z, y = S.y;
      const sigZ = mapSigma / S.ysd;                 // noise on the standardized target
      const s2 = sigZ * sigZ, t2 = mapTau * mapTau;
      const bLap = mapTau / Math.SQRT2;              // Laplace scale at equal variance
      const gauss = mapPrior === 'gauss';

      /* The two halves of the negative log posterior, as written on the page. */
      const dataTerm = w => n * loss(w) / s2;        // ‖y − Zw‖²/(2σ²)
      const priorTerm = w => gauss ? l2sq(w) / (2 * t2) : l1(w) / bLap;
      const J = w => dataTerm(w) + priorTerm(w);

      const lam = s2 / t2;                           // ℓ₂ penalty weight
      const lamL1 = 2 * s2 / bLap;                   // ℓ₁ penalty weight
      const wMap = gauss
        ? ML.ridge(Z, y, lam).w
        : ML.elasticNet(Z, y, s2 / (n * bLap), 1.0).w;
      const wOther = gauss
        ? ML.elasticNet(Z, y, s2 / (n * bLap), 1.0).w
        : ML.ridge(Z, y, lam).w;
      const wMle = S.wOls;
      const ref = wMle || S.wMin;

      const Alik = S.A.map(r => r.map(v => v * n / s2));
      const Apri = gauss ? [[1 / t2, 0], [0, 1 / t2]] : null;
      const Apost = gauss
        ? [[Alik[0][0] + 1 / t2, Alik[0][1]], [Alik[1][0], Alik[1][1] + 1 / t2]]
        : Alik;
      const shrink = Math.sqrt(l2sq(wMap) / (l2sq(ref) || 1e-12));
      const pack = {wMap, wOther, wMle, ref, dataTerm, priorTerm, J, gauss,
                    Alik, Apri, Apost, lam, lamL1, sigZ, bLap, t2, s2, n};

      const st = [];

      st.push({
        short: 'Likelihood only', title: 'Start with the likelihood alone',
        math: `<div class="narration" style="margin:0 0 12px;border-left-color:var(--c-ols)">
            $$p(y \\mid X, w) \;\\propto\; \\exp\\!\\left(-\\frac{\\lVert y - Xw\\rVert^2}{2\\sigma^2}\\right)$$
          </div>
          <div>${kv([
            ['noise σ', '₹' + F(mapSigma, 0) + '  (= ' + F(sigZ, 3) + ' standardized)'],
            ['trips n', n],
            ['−log likelihood at ŵ MLE', wMle ? F(dataTerm(wMle), 1) : 'undefined', 'good'],
            ['−log likelihood at w = 0', F(dataTerm([0, 0]), 1), 'warn'],
            ['ŵ MLE', wMle ? `(${F(wMle[0], 3)}, ${F(wMle[1], 3)})` : 'does not exist', wMle ? '' : 'warn']])}</div>`,
        narration: `<p>With no prior — or, equivalently, a perfectly flat one — maximising the
          posterior is maximising the likelihood. The mode is the MLE, and the penalty term does
          not exist yet.</p>
          <p>Note the two numbers above: the negative log likelihood is
          <span class="hl">${F(dataTerm([0, 0]), 0)}</span> at the origin and
          <span class="hl">${wMle ? F(dataTerm(wMle), 0) : '—'}</span> at the MLE. That gap is how
          hard the data will pull, and it is what any prior has to argue against.</p>`,
        viz: Object.assign({stage: 0}, pack)
      });

      st.push({
        short: 'Add a prior', title: gauss ? 'Choose a prior: Gaussian' : 'Choose a prior: Laplace',
        math: `<div class="narration" style="margin:0 0 12px;border-left-color:var(--c-ridge)">
            ${gauss
              ? `$$p(w) \;\\propto\; \\exp\\!\\left(-\\frac{\\lVert w\\rVert_2^2}{2\\tau^2}\\right)$$`
              : `$$p(w) \;\\propto\; \\exp\\!\\left(-\\frac{\\lVert w\\rVert_1}{b}\\right)$$`}
          </div>
          <div>${kv([
            ['prior width τ', F(mapTau, 3)],
            gauss ? ['prior variance τ²', F(t2, 4)]
                  : ['Laplace scale b = τ/√2', F(bLap, 4)],
            gauss ? ['−log prior', '‖w‖₂² / 2τ²'] : ['−log prior', '‖w‖₁ / b'],
            ['−log prior at w = 0', '0.0', 'good'],
            ['−log prior at ŵ MLE', wMle ? F(priorTerm(wMle), 1) : '—', 'warn']])}</div>`,
        narration: `<p>The prior is a statement made before seeing data: each coefficient is
          probably near zero, with spread τ = <span class="hl">${F(mapTau, 3)}</span>.
          ${gauss
            ? 'On the plane it is a circle centred on the origin.'
            : 'Its contours are diamonds rather than circles, and that single fact is what will '
              + 'eventually produce exact zeros.'}</p>
          <p>Both priors are shown at the <em>same variance</em>, so the comparison is fair:
          a Laplace with scale $b$ has variance $2b^2$, hence $b = \\tau/\\sqrt{2}$.</p>
          <p>The prior costs <span class="hl">0</span> at the origin and
          <span class="hl">${wMle ? F(priorTerm(wMle), 1) : '—'}</span> at the MLE. The data wants
          one place, the prior wants another.</p>`,
        viz: Object.assign({stage: 1}, pack)
      });

      st.push({
        short: 'Multiply', title: 'Bayes: multiply them',
        math: `<div class="narration" style="margin:0 0 12px;border-left-color:var(--accent)">
            $$p(w \\mid y) \;\\propto\; \\underbrace{p(y \\mid X,w)}_{\\text{likelihood}}
              \\times \\underbrace{p(w)}_{\\text{prior}}$$
          </div>
          <p class="mat-cap">a product is awkward to maximise, so take the logarithm</p>
          <div class="narration" style="margin:10px 0 0;border-left-color:var(--accent)">
            $$\\log p(w \\mid y) = \\log p(y\\mid X,w) + \\log p(w) + \\text{const}$$
          </div>`,
        narration: `<p>Posterior ∝ likelihood × prior. Taking the logarithm turns that product into
          a <strong>sum</strong>, and the normalising constant drops out because it does not depend
          on $w$.</p>
          <p>This is the step that matters. Everything after it is bookkeeping: we now have two
          terms to add up instead of two densities to multiply.</p>`,
        viz: Object.assign({stage: 2}, pack)
      });

      st.push({
        short: 'Negate', title: 'Negate: the objective appears',
        math: `<div class="narration" style="margin:0 0 12px;border-left-color:var(--accent)">
            $$J(w) \;=\; \\underbrace{\\frac{\\lVert y - Xw\\rVert^2}{2\\sigma^2}}_{\\text{data fit}}
              \;+\; \\underbrace{${gauss ? '\\frac{\\lVert w\\rVert_2^2}{2\\tau^2}' : '\\frac{\\lVert w\\rVert_1}{b}'}}_{\\text{penalty}}$$
          </div>
          <div>${kv([
            ['J at w = 0', F(J([0, 0]), 1)],
            ['J at ŵ MLE', wMle ? F(J(wMle), 1) : '—'],
            ['J at ŵ MAP', F(J(wMap), 1), 'good'],
            ['— of which data fit', F(dataTerm(wMap), 1)],
            ['— of which penalty', F(priorTerm(wMap), 1)]])}</div>`,
        narration: `<p>Maximising the log posterior is minimising its negative, and
          <strong>$-\\log(\\text{prior})$ is the penalty</strong>. Choose a prior, negate its log,
          and you have derived a regularizer. Nothing was assumed about "shrinkage".</p>
          <p>Look at the three values of $J$. The MAP beats both the origin and the MLE, because
          the origin fits the data badly and the MLE pays too much penalty. The chart below shows
          the whole trade-off.</p>`,
        viz: Object.assign({stage: 3}, pack)
      });

      st.push({
        short: 'λ = σ²/τ²', title: 'The penalty weight is not a free parameter',
        math: `<p class="mat-cap">multiply through by 2σ² — it does not move the minimum</p>
          <div class="narration" style="margin:6px 0 12px;border-left-color:var(--amber)">
            ${gauss
              ? `$$2\\sigma^2 J(w) = \\lVert y - Xw\\rVert^2 + \\underbrace{\\frac{\\sigma^2}{\\tau^2}}_{\\lambda}\\lVert w\\rVert_2^2$$`
              : `$$2\\sigma^2 J(w) = \\lVert y - Xw\\rVert^2 + \\underbrace{\\frac{2\\sigma^2}{b}}_{\\lambda_1}\\lVert w\\rVert_1$$`}
          </div>
          <div>${kv([
            ['σ² (standardized)', F(s2, 4)],
            gauss ? ['τ²', F(t2, 4)] : ['b', F(bLap, 4)],
            gauss ? ['λ = σ²/τ²', F(lam, 3), 'good'] : ['λ₁ = 2σ²/b', F(lamL1, 3), 'good'],
            ['in rupee units', gauss ? F(lam * S.ysd * S.ysd, 1) : F(lamL1 * S.ysd, 1)]])}</div>`,
        narration: `<p>${gauss
            ? 'And there it is: <span class="hl">λ = σ²/τ²</span>, the ratio of noise variance to '
              + 'prior variance. The penalty weight was never a knob to tune — it is a statement '
              + 'about how noisy the data is relative to how confident the prior is.'
            : 'For the Laplace prior the same manipulation gives <span class="hl">λ₁ = 2σ²/b</span>.'}</p>
          <p>Noisy data (large σ) or a confident prior (small ${gauss ? 'τ' : 'b'}) both mean more
          shrinkage, and for the same reason. Drag the two sliders above and watch λ move:
          it is currently <span class="hl">${F(gauss ? lam : lamL1, 3)}</span>.</p>
          <p>The assignment fixes σ = 10 and τ = 1, giving λ = 100 in raw rupee units. Cross-validation
          disagrees with that choice, which is the argument you are asked to make in
          <a href="assignments.html#t2">Task 2</a>.</p>`,
        viz: Object.assign({stage: 3}, pack)
      });

      st.push({
        short: 'Find the mode', title: 'Solve for the mode',
        math: `<div class="narration" style="margin:0 0 12px;border-left-color:var(--accent)">
            ${gauss
              ? `$$\\nabla J = 0 \;\\Rightarrow\; \\hat w_{\\text{MAP}} = (X^\\top X + \\lambda I)^{-1}X^\\top y$$`
              : `$$\\text{no closed form; coordinate descent as in §4}$$`}
          </div>
          ${row('ŵ MAP =', vec(wMap, {highlight: [[0, 0], [1, 0]]}))}
          ${wMle ? row('ŵ MLE =', vec(wMle, {dim: [[0, 0], [1, 0]]})) : ''}
          <div style="margin-top:12px">${kv([
            ['‖ŵ‖₂ retained vs MLE', (100 * shrink).toFixed(0) + '%'],
            ['coefficients at exactly zero', String(wMap.filter(v => Math.abs(v) < 1e-9).length),
             wMap.some(v => Math.abs(v) < 1e-9) ? 'warn' : 'zero'],
            ['distance from MLE', F(Math.hypot(wMap[0] - ref[0], wMap[1] - ref[1]), 4)],
            ['distance from origin', F(Math.sqrt(l2sq(wMap)), 4)]])}</div>`,
        narration: `<p>${gauss
            ? 'Setting the gradient to zero gives the ridge formula — but notice that we did not '
              + 'start from ridge. We started from a prior, and ridge is what came out.'
            : 'The ℓ₁ term is not differentiable at zero, so there is no closed form and we fall '
              + 'back on coordinate descent. The mode can now sit exactly on an axis.'}</p>
          <p>The mode sits between the two things pulling on it: the MLE and the origin. On the
          slice chart below, that is the point where the falling data term and the rising penalty
          sum to their minimum.</p>
          ${shrink > 0.97
            ? `<p><strong>Note how little happened.</strong> The estimate kept
               ${(100 * shrink).toFixed(0)}% of its length, so at τ = ${F(mapTau, 3)} this prior is
               too vague to matter against ${n} trips. That is not a flaw in the method — it is the
               finding <a href="assignments.html#t2">Task 2</a> asks you to report, and the reason
               cross-validation rejects the assignment's τ = 1. Pull τ down to about 0.03 to make
               it a real contest.</p>`
            : ''}
          ${wMap.some(v => Math.abs(v) < 1e-9)
            ? '<p><strong>One coefficient is exactly zero.</strong> A Gaussian prior at this same '
              + 'variance could not do that, however small you made τ.</p>' : ''}`,
        viz: Object.assign({stage: 4}, pack)
      });

      st.push({
        short: 'Dial τ', title: 'What τ controls, end to end',
        math: `<div>${kv([
            ['τ → ∞  (no prior)', wMle ? `(${F(wMle[0], 2)}, ${F(wMle[1], 2)})  = MLE` : 'MLE undefined'],
            [`τ = ${F(mapTau, 2)}  (here)`, `(${F(wMap[0], 2)}, ${F(wMap[1], 2)})`, 'good'],
            ['τ → 0  (certain of zero)', '(0.00, 0.00)  = the prior wins'],
            [gauss ? 'λ = σ²/τ² here' : 'λ₁ = 2σ²/b here', F(gauss ? lam : lamL1, 3)]])}</div>
          <p class="muted small" style="margin-top:12px">The dashed curve on the right is every MAP
          estimate as τ sweeps across its whole range, and the chart below it plots the same path
          coefficient by coefficient.</p>`,
        narration: `<p>τ interpolates continuously between two extremes. A vague prior
          (large τ) lets the likelihood win and the MAP converges on the MLE; a dogmatic prior
          (small τ) crushes everything to zero regardless of the data.</p>
          <p>Every penalized estimator in this course is a point on a curve like this one. Ridge
          is the Gaussian version, Lasso the Laplace version, and choosing λ by cross-validation
          is choosing where on the curve to stand when you have no honest prior to quote.</p>`,
        viz: Object.assign({stage: 5}, pack)
      });

      st.push({
        short: 'Both priors', title: 'The same construction, the other prior',
        math: `${row(gauss ? 'ŵ MAP Gaussian =' : 'ŵ MAP Laplace =', vec(wMap, {highlight: [[0, 0], [1, 0]]}))}
          ${row(gauss ? 'ŵ MAP Laplace =' : 'ŵ MAP Gaussian =', vec(wOther))}
          <div style="margin-top:12px">${kv([
            ['same prior variance?', 'yes — b = τ/√2', 'good'],
            ['Gaussian zeros', String((gauss ? wMap : wOther).filter(v => Math.abs(v) < 1e-9).length), 'zero'],
            ['Laplace zeros', String((gauss ? wOther : wMap).filter(v => Math.abs(v) < 1e-9).length),
             (gauss ? wOther : wMap).some(v => Math.abs(v) < 1e-9) ? 'warn' : 'zero'],
            ['J(Gaussian mode)', F(J(gauss ? wMap : wOther), 2)],
            ['J(Laplace mode)', F(J(gauss ? wOther : wMap), 2)]])}</div>
          <p class="muted small" style="margin-top:10px">Both J values are computed under the
          <em>currently selected</em> prior, so the selected one is necessarily lower. Switch the
          prior to see the comparison reverse.</p>`,
        narration: `<p>Two priors, one construction, two different penalties. The Gaussian gives
          $\\lambda\\lVert w\\rVert_2^2$ and never reaches zero; the Laplace gives
          $\\lambda_1\\lVert w\\rVert_1$ and can. Nothing else changed — not the data, not the
          likelihood, not the algebra.</p>
          <p>That is the whole reason this course treats Ridge and Lasso as the same idea with a
          different assumption, rather than as two unrelated tricks. Go on to
          <a href="#ridge">§3</a> for the Gaussian case solved in closed form, and
          <a href="#lasso">§4</a> for the Laplace case solved by coordinate descent.</p>
          ${!(gauss ? wOther : wMap).some(x => Math.abs(x) < 1e-9)
            ? `<p><strong>No zero yet, and that is worth understanding.</strong> Deriving λ from a
               stated prior — rather than picking it — means an exact zero needs a prior tight
               enough to outvote the data. Against ${n} trips that is roughly τ below 0.01. Two ways
               to see one: raise the noise σ toward ₹40, or pick a feature the data barely supports,
               such as <code>is_weekend</code>, in the setup panel at the top. Its true coefficient
               is zero, so the likelihood puts up almost no fight. The zeros in
               <a href="#lasso">§4</a> come easily because α is chosen directly there, not derived
               from a prior.</p>`
            : ''}`,
        viz: Object.assign({stage: 6}, pack)
      });
      return st;
    },

    viz(root, s) {
      const v = s.viz, stage = v.stage;
      const host = root.querySelector('.viz-pane');

      /* Contours are drawn at 1σ, 2σ and 3σ of each distribution: levels ½d². */
      const SIG = [0.5, 2, 4.5];
      const contours = [];
      if (stage >= 0) contours.push({A: v.Alik, center: v.ref, levels: SIG,
        color: 'var(--c-ols)', dashed: true, opacity: .85});
      if (stage >= 1 && v.Apri) contours.push({A: v.Apri, center: [0, 0], levels: SIG,
        color: 'var(--c-ridge)', dashed: true, opacity: .8});
      if (stage >= 4) contours.push({A: v.Apost, center: v.wMap, levels: SIG,
        color: 'var(--accent)', width: 2, opacity: 1});

      const path = [];
      if (stage >= 5) {
        const pts = [];
        for (let i = 0; i <= 70; i++) {
          const tt = Math.pow(10, -1.6 + 3.4 * i / 70);
          const w = v.gauss
            ? ML.ridge(S.Z, S.y, v.s2 / (tt * tt)).w
            : ML.elasticNet(S.Z, S.y, v.s2 / (S.n * (tt / Math.SQRT2)), 1.0).w;
          pts.push(w);
        }
        path.push({pts, color: 'var(--c-enet)', dashed: true, dots: false, width: 1.8});
      }

      /* These four points bunch together in the middle of the plane, so each label
         gets its own vertical offset rather than overprinting its neighbours. */
      const points = [];
      if (v.wMle) points.push({w: v.wMle, label: 'MLE', color: 'var(--c-ols)', r: 5, dy: -4});
      if (stage >= 1) points.push({w: [0, 0], label: 'prior', color: 'var(--c-ridge)', r: 4.5, dy: -4});
      if (stage >= 4) points.push({w: v.wMap, label: 'MAP', color: 'var(--accent)', r: 6.5, dy: 22});
      if (stage >= 6) points.push({w: v.wOther,
        label: v.gauss ? 'Laplace' : 'Gaussian', color: 'var(--c-lasso)', r: 5, dy: 38});

      plane(host, {contours, path, points,
        constraint: stage >= 1 && !v.gauss ? {type: 'l1', t: l1(v.wMap)} : null});

      host.insertAdjacentHTML('beforeend',
        `<div class="legend" style="margin-top:8px">
           <span><i style="background:var(--c-ols)"></i>likelihood</span>
           ${stage >= 1 ? '<span><i style="background:var(--c-ridge)"></i>prior</span>' : ''}
           ${stage >= 4 ? '<span><i style="background:var(--accent)"></i>posterior</span>' : ''}
         </div>
         <p class="muted small" style="margin:6px 0 0">${
           stage === 0 ? 'Contours at 1σ, 2σ and 3σ of the likelihood. Its centre is the MLE.'
           : stage === 1 ? 'The prior is centred on the origin and favours no direction. Note how much wider or narrower it is than the likelihood — that ratio decides the outcome.'
           : stage < 4 ? 'Two beliefs, about to be combined. Nothing has moved yet.'
           : stage < 5 ? 'The posterior (green) sits between the two, closer to whichever is more concentrated.'
           : stage < 6 ? 'The purple dashed curve is every MAP estimate as τ runs from 0.025 to 60.'
           : 'Both modes at the same prior variance. The Laplace mode is the one that can land on an axis.'}</p>`);

      /* ---- slice: the objective along the ray through the MAP ---- */
      const sp = root.querySelector('.slice-pane');
      if (sp) {
        if (stage < 3) {
          sp.innerHTML = '<p class="muted small" style="margin:0;padding:22px 0">'
            + 'The objective does not exist yet — it appears once the prior is negated and added '
            + 'to the data term (step 3).</p>';
        } else {
          const T = [];
          for (let i = 0; i <= 120; i++) T.push(1.6 * i / 120);
          const at = t => [v.wMap[0] * t, v.wMap[1] * t];
          const series = [
            {name: 'data fit', color: 'var(--c-ols)', dashed: true,
             points: T.map(t => [t, v.dataTerm(at(t))])},
            {name: 'penalty', color: 'var(--c-ridge)', dashed: true,
             points: T.map(t => [t, v.priorTerm(at(t))])},
            {name: 'J = data + penalty', color: 'var(--accent)', width: 2.6,
             points: T.map(t => [t, v.J(at(t))])}
          ];
          Plot.line(sp, series, {height: 260, dots: false,
            xLabel: 't   (0 = prior mode, 1 = MAP)', yLabel: 'negative log posterior',
            vLine: 1, vLabel: 'MAP'});
          sp.insertAdjacentHTML('beforeend',
            `<p class="muted small" style="margin:6px 0 0">Walking out from the origin toward the
             MAP, the data term falls and the penalty rises. Their sum bottoms out at
             <strong>t = 1</strong>, which is the definition of the mode. Past that, extra fit
             costs more penalty than it is worth.</p>`);
        }
      }

      /* ---- coefficients against τ ---- */
      const tp = root.querySelector('.tau-pane');
      if (tp) {
        const grid = [];
        for (let i = 0; i <= 60; i++) grid.push(Math.pow(10, -1.6 + 3.4 * i / 60));
        const w1 = [], w2 = [];
        grid.forEach(tt => {
          const w = v.gauss
            ? ML.ridge(S.Z, S.y, v.s2 / (tt * tt)).w
            : ML.elasticNet(S.Z, S.y, v.s2 / (S.n * (tt / Math.SQRT2)), 1.0).w;
          w1.push([tt, w[0]]); w2.push([tt, w[1]]);
        });
        Plot.line(tp, [
          {name: S.labels[0], points: w1, color: 'var(--c-ridge)'},
          {name: S.labels[1], points: w2, color: 'var(--rose)'}
        ], {height: 260, dots: false, logX: true, zeroLine: true, inlineLabels: true,
            xLabel: 'prior width τ (log scale)', yLabel: 'coefficient',
            vLine: mapTau, vLabel: 'τ = ' + F(mapTau, 2)});
        tp.insertAdjacentHTML('beforeend',
          `<p class="muted small" style="margin:6px 0 0">Left edge: the prior wins and both
           coefficients are crushed. Right edge: the likelihood wins and they settle on the MLE.
           ${v.gauss ? 'With a Gaussian prior they approach zero without reaching it.'
                     : 'With a Laplace prior they hit zero exactly, at a finite τ.'}</p>`);
      }
    }
  });

  $('mapTau').addEventListener('input', e => {
    mapTau = Math.pow(10, +e.target.value);
    $('mapTauV').textContent = F(mapTau, mapTau < 1 ? 3 : 2);
    mapPlayer.rebuild(true);
  });
  $('mapSigma').addEventListener('input', e => {
    mapSigma = +e.target.value;
    $('mapSigmaV').textContent = mapSigma;
    mapPlayer.rebuild(true);
  });
  $('mapPrior').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    [...e.currentTarget.children].forEach(c => c.classList.toggle('on', c === b));
    mapPrior = b.dataset.p;
    mapPlayer.rebuild(true);
  });

  /* ======================================================================
     3 · RIDGE
  ====================================================================== */
  let ridgeAlpha = 0.5;
  const ridgePlayer = makePlayer($('wRidge'), {
    speed: 1300,
    steps() {
      const a = ridgeAlpha;
      const {A, b, r, wOls, labels} = S;
      const Aa = [[A[0][0] + a, A[0][1]], [A[1][0], A[1][1] + a]];
      const detA = S.det, detAa = (1 + a) * (1 + a) - r * r;
      const AaInv = ML.inv(Aa);
      const wR = AaInv ? ML.matvec(AaInv, b) : [0, 0];
      const eg0 = ML.eigSym(A), eg1 = ML.eigSym(Aa);
      const wRef = wOls || S.wMin;
      const shrink = Math.sqrt(l2sq(wR) / (l2sq(wRef) || 1e-9));

      return [
        {short: 'Same start', title: 'Start from the same Gram matrix',
         math: row('XᵀX/n =', mat(A)) + row('Xᵀy/n =', vec(b))
           + `<p class="mat-cap">identical to step 4 of the least-squares walkthrough</p>`,
         narration: `<p>Ridge changes one line of the computation. Standardizing, the Gram matrix
           and $X^\\top y$ are all untouched.</p>`,
         viz: {stage: 0, wR}},

        {short: 'Add αI', title: 'Add $\\alpha I$ to the diagonal',
         math: row('XᵀX/n + αI =', mat(Aa, {highlight: [[0, 0], [1, 1]]}))
           + `<p class="mat-cap">α = ${F(a, 3)} added to each diagonal entry, nothing else changed</p>`
           + `<div style="margin-top:12px">${kv([
               ['det before', F(detA, 4), detA < 0.15 ? 'warn' : ''],
               ['det after', F(detAa, 4), 'good'],
               ['multiplied by', '×' + F(detAa / detA, 2)]])}</div>`,
         narration: `<p>That is the whole algorithmic difference between maximum likelihood and MAP
           with a Gaussian prior: <span class="hl">+ α on the diagonal</span>.</p>
           <p>The determinant rose from ${F(detA, 4)} to <span class="hl">${F(detAa, 4)}</span>. The
           inverse divides by it, so the coefficients shrink and the variance inflation falls with
           them. The matrix can no longer be singular, however correlated the features are.</p>`,
         viz: {stage: 1, wR}},

        {short: 'Eigenvalues', title: 'What it does to the eigenvalues',
         math: `<div>${kv([
             ['λ₁ before', F(eg0.values[0], 3)], ['λ₁ after', F(eg1.values[0], 3)],
             ['λ₂ before', F(eg0.values[1], 4), eg0.values[1] < 0.15 ? 'warn' : ''],
             ['λ₂ after', F(eg1.values[1], 4), 'good'],
             ['shrinkage on direction 1', F(eg0.values[0] / (eg0.values[0] + a), 3)],
             ['shrinkage on direction 2', F(eg0.values[1] / (eg0.values[1] + a), 3),
              eg0.values[1] / (eg0.values[1] + a) < 0.5 ? 'warn' : '']])}</div>`,
         narration: `<p>Adding $\\alpha I$ adds $\\alpha$ to every eigenvalue, but the relative
           effect is uneven. The well-determined direction keeps
           ${(100 * eg0.values[0] / (eg0.values[0] + a)).toFixed(0)}% of its length; the poorly
           determined one keeps
           <span class="hl">${(100 * eg0.values[1] / (eg0.values[1] + a)).toFixed(0)}%</span>.</p>
           <p>Ridge shrinks the directions the data is least sure about, which is what a prior
           should do.</p>`,
         viz: {stage: 1, wR}},

        {short: 'Solve', title: 'Invert and multiply',
         math: row('(XᵀX/n + αI)⁻¹ =', mat(AaInv || [[0, 0], [0, 0]]))
           + row('ŵ ridge =', vec(wR, {highlight: [[0, 0], [1, 0]]}))
           + (wOls ? row('ŵ OLS =', vec(wOls, {dim: [[0, 0], [1, 0]]}))
                   : `<p class="mat-cap" style="color:var(--rose)">no OLS solution exists on these rows — Ridge still returns one</p>`)
           + `<div style="margin-top:12px">${kv([
               [`w₁ · ${labels[0]}`, `${wOls ? F(wOls[0], 3) + ' → ' : ''}${F(wR[0], 3)}`],
               [`w₂ · ${labels[1]}`, `${wOls ? F(wOls[1], 3) + ' → ' : ''}${F(wR[1], 3)}`],
               ['‖w‖₂ retained', (100 * shrink).toFixed(0) + '%'],
               ['coefficients at exactly zero', '0', 'zero']])}</div>`,
         narration: `<p>Both coefficients moved toward the origin, and toward each other: the
           equal-split behaviour. Neither is zero, and neither will be, because a quadratic penalty
           has zero gradient at the origin.</p>
           <p>Compare the loss: ${F(loss(wRef), 4)} at the unpenalized point, ${F(loss(wR), 4)} here. We gave
           up some fit. The question the rest of the course answers is whether that trade buys
           anything on data we have not seen.</p>`,
         viz: {stage: 2, wR}},

        {short: 'The whole path', title: 'Sweep α from 0 to ∞',
         math: `<div>${kv([
             ['α = 0', wOls ? `(${F(wOls[0], 2)}, ${F(wOls[1], 2)})  — OLS`
                                    : 'undefined — XᵀX is singular'],
             [`α = ${F(a, 2)}`, `(${F(wR[0], 2)}, ${F(wR[1], 2)})  — here`],
             ['α → ∞', '(0.00, 0.00)  — the prior wins'],
             ['‖w‖₂ budget used', F(Math.sqrt(l2sq(wR)), 3)]])}</div>
           <p class="muted small" style="margin-top:12px">The green curve on the right is every ridge
           solution at every α. Notice it is a smooth curve that only reaches the origin in the
           limit — it never touches an axis on the way.</p>`,
         narration: `<p>This is the ridge path, the same object plotted against a log α axis in
           <a href="lab.html#paths">Lab §2</a>, drawn in the parameter plane instead.</p>
           <p>The shaded circle is the $\\ell_2$ ball the solution sits on. Ridge minimises the loss
           subject to $\\lVert w\\rVert_2 \\le t$, and the smoothness of that circle is why no
           coefficient reaches zero. Compare the diamond in the next section.</p>`,
         viz: {stage: 3, wR}}
      ];
    },
    viz(root, s) {
      const host = root.querySelector('.viz-pane');
      const {stage, wR} = s.viz;
      const path = [];
      if (stage >= 3) {
        const pts = [];
        for (let i = 0; i <= 60; i++) {
          const aa = i === 0 ? 0 : Math.pow(10, -3 + 5 * i / 60);
          const M = [[S.A[0][0] + aa, S.A[0][1]], [S.A[1][0], S.A[1][1] + aa]];
          const Mi = ML.inv(M);
          if (Mi) pts.push(ML.matvec(Mi, S.b));
        }
        path.push({pts, color: 'var(--c-ridge)', dashed: true, dots: false, width: 1.8});
      }
      plane(host, {
        contours: [{A: S.Acontour, center: S.centre, levels: S.levels}],
        constraint: stage >= 3 ? {type: 'l2', t: Math.sqrt(l2sq(wR))} : null,
        path,
        points: (S.wOls ? [{w: S.wOls, label: 'OLS', color: 'var(--c-ols)', r: 5}] : [])
          .concat(stage >= 2 ? [{w: wR, label: 'ridge', color: 'var(--c-ridge)'}] : [])
      });
      host.insertAdjacentHTML('beforeend',
        `<p class="muted small" style="margin:8px 0 0">${
          stage < 2 ? 'The loss surface is unchanged — Ridge does not alter the data, only what we ask of the answer.'
          : stage < 3 ? 'The solution has been pulled toward the origin, off the bottom of the bowl.'
          : 'Dashed line: every ridge solution as α runs from 0 to ∞. Shaded: the ℓ₂ ball the current solution sits on.'}</p>`);
    }
  });
  $('ridgeAlpha').addEventListener('input', e => {
    ridgeAlpha = Math.pow(10, +e.target.value);
    $('ridgeAlphaV').textContent = F(ridgeAlpha, ridgeAlpha < 1 ? 3 : 2);
    ridgePlayer.rebuild(true);
  });

  /* ======================================================================
     4 & 5 · COORDINATE DESCENT (Lasso and Elastic Net)
  ====================================================================== */
  /** Run cyclic coordinate descent, recording every single update. */
  function cdTrace(alpha, rho, start) {
    const A = S.A, b = S.b;
    const l1p = alpha * rho, l2p = alpha * (1 - rho);
    let w = start.slice();
    const trace = [{init: true, w: w.slice(), obj: loss(w) + l1p * l1(w) + 0.5 * l2p * l2sq(w)}];
    const MAXSWEEP = 14;
    for (let sweep = 0; sweep < MAXSWEEP; sweep++) {
      let maxDelta = 0;
      for (let j = 0; j < 2; j++) {
        const k = 1 - j;
        // ρ_j = (1/n) x_jᵀ(y − Σ_{k≠j} x_k w_k) = b_j − A_jk w_k
        const rhoJ = b[j] - A[j][k] * w[k];
        const before = w[j];
        const num = ML.softThreshold(rhoJ, l1p);
        const denom = A[j][j] + l2p;                    // = 1 + α(1−ρ) for standardized columns
        const after = num / denom;
        const wBefore = w.slice();
        w[j] = after;
        maxDelta = Math.max(maxDelta, Math.abs(after - before));
        trace.push({
          sweep: sweep + 1, j, rhoJ, l1p, l2p, denom, num, before, after,
          wBefore, w: w.slice(),
          obj: loss(w) + l1p * l1(w) + 0.5 * l2p * l2sq(w),
          thresholded: Math.abs(rhoJ) <= l1p
        });
      }
      if (maxDelta < 1e-6) { trace.converged = sweep + 1; break; }
      if (trace.length > 26) break;
    }
    return trace;
  }

  function makeCD(rootId, cfg) {
    const root = $(rootId);
    let state = {alpha: cfg.alpha0, rho: cfg.rho0, start: 'zero'};

    function startPoint() {
      if (state.start === 'ols') return S.wOls.slice();
      if (state.start === 'far') return [2, -2];
      return [0, 0];
    }

    const player = makePlayer(root, {
      speed: 1000,
      steps() {
        const tr = cdTrace(state.alpha, state.rho, startPoint());
        state.trace = tr;
        const l1p = state.alpha * state.rho, l2p = state.alpha * (1 - state.rho);
        const isLasso = state.rho >= 0.999;
        const steps = [];

        steps.push({
          short: 'Initialise', title: 'Initialise',
          math: row('w⁰ =', vec(tr[0].w)) + row('α =', `<span class="lbl">${F(state.alpha, 3)}</span>`)
            + (cfg.showRho ? row('ρ =', `<span class="lbl">${F(state.rho, 2)}</span>`) : '')
            + `<div style="margin-top:12px">${kv([
                ['ℓ₁ weight  α·ρ', F(l1p, 4)],
                ['ℓ₂ weight  α(1−ρ)', F(l2p, 4)],
                ['update denominator  1 + α(1−ρ)', F(1 + l2p, 4)],
                ['objective', F(tr[0].obj, 4)]])}</div>`,
          narration: `<p>Any starting point will do, since the objective is convex with a single
            minimum. We begin at
            <span class="hl">(${F(tr[0].w[0], 2)}, ${F(tr[0].w[1], 2)})</span>.</p>
            <p>The two penalty weights are computed once. The ℓ₁ weight
            <span class="hl">${F(l1p, 4)}</span> becomes a threshold; the ℓ₂ weight
            <span class="hl">${F(l2p, 4)}</span> becomes a divisor.</p>`,
          idx: 0
        });

        tr.slice(1).forEach((t, i) => {
          const name = S.labels[t.j], other = S.labels[1 - t.j];
          const killed = t.thresholded;
          steps.push({
            short: `Sweep ${t.sweep} · w${t.j + 1}`,
            title: `Sweep ${t.sweep}: optimise $w_{${t.j + 1}}$ (${name})`,
            math: `<div>${kv([
                [`freeze w${2 - t.j} (${other})`, F(t.wBefore[1 - t.j], 4)],
                [`ρ${t.j + 1} = b${t.j + 1} − r·w${2 - t.j}`, F(t.rhoJ, 4)],
                ['threshold  α·ρ', F(t.l1p, 4)],
                ['|ρ| vs threshold', `${F(Math.abs(t.rhoJ), 4)} ${killed ? '≤' : '>'} ${F(t.l1p, 4)}`,
                 killed ? 'warn' : 'good'],
                ['S(ρ, α·ρ)', F(t.num, 4), killed ? 'zero' : ''],
                ['÷ denominator', F(t.denom, 4)],
                [`w${t.j + 1} : before → after`,
                 `${F(t.before, 4)} → ${F(t.after, 4)}`, killed ? 'zero' : 'good'],
                ['objective', F(t.obj, 5)]])}</div>
              ${row('w =', vec(t.w, {highlight: [[t.j, 0]]}))}`,
            narration: killed
              ? `<p>The partial correlation $\\rho_{${t.j + 1}} =$
                 <span class="hl">${F(t.rhoJ, 4)}</span> is smaller in magnitude than the threshold
                 <span class="hl">${F(t.l1p, 4)}</span>, so soft-thresholding returns
                 <strong>exactly zero</strong> — an exact arithmetic result, not a small number.</p>
                 <p>On the chart below, this step landed in the flat dead zone.
                 <code>${name}</code> is now out of the model and stays out unless the other
                 coefficient moves enough to push $\\rho$ back past the threshold.</p>`
              : `<p>Holding <code>${other}</code> fixed at ${F(t.wBefore[1 - t.j], 3)}, the best value
                 for <code>${name}</code> along its own axis is
                 $\\rho_{${t.j + 1}} = ${F(t.rhoJ, 4)}$ before the penalty.</p>
                 <p>The penalty then ${isLasso
                    ? `subtracts <span class="hl">${F(t.l1p, 4)}</span> from its magnitude`
                    : `subtracts <span class="hl">${F(t.l1p, 4)}</span> and divides by
                       <span class="hl">${F(t.denom, 3)}</span>`},
                 giving <span class="hl">${F(t.after, 4)}</span>, and the objective falls to
                 ${F(t.obj, 5)}. On the plane the move is a straight line along one axis, since only
                 one coordinate changed.</p>`,
            idx: i + 1
          });
        });

        const fin = tr[tr.length - 1];
        const nz = fin.w.filter(v => Math.abs(v) > 1e-9).length;
        steps.push({
          short: 'Converged', title: tr.converged ? `Converged after ${tr.converged} sweeps` : 'Stopped',
          math: `${row('ŵ =', vec(fin.w, {highlight: [[0, 0], [1, 0]]}))}
            <div style="margin-top:12px">${kv([
              ['coefficients kept', `${nz} of 2`, nz < 2 ? 'warn' : 'good'],
              ['objective', F(fin.obj, 5)],
              ['loss alone', F(loss(fin.w), 5)],
              ['penalty paid', F(fin.obj - loss(fin.w), 5)],
              ['‖ŵ‖₁', F(l1(fin.w), 4)],
              ['‖ŵ‖₂', F(Math.sqrt(l2sq(fin.w)), 4)],
              [S.wOls ? 'distance from OLS' : 'distance from min-norm fit',
               (rf => F(Math.hypot(fin.w[0] - rf[0], fin.w[1] - rf[1]), 4))(S.wOls || S.wMin)]])}</div>`,
          narration: `<p>No coordinate moved by more than $10^{-6}$, so each is optimal given the
            others. For a convex objective that means the global minimum.</p>
            ${nz < 2
              ? `<p><strong>One coefficient is exactly zero.</strong> Ridge could not produce this at
                 any α. The difference between shrinkage and selection comes from a single
                 <code>max(|ρ| − γ, 0)</code>.</p>`
              : `<p>Both coefficients survived at this α. Raise α and step through again to see which
                 dies first, then check it against the correlations in the setup panel.</p>`}
            <p>Path length: ${tr.length - 1} coordinate updates, each a right-angled move.</p>`,
          idx: tr.length - 1
        });
        return steps;
      },
      viz(root2, s) {
        const tr = state.trace;
        const t = tr[s.idx];
        const w = t.w;
        const pts = tr.slice(0, s.idx + 1).map(x => x.w);
        const l1p = state.alpha * state.rho;
        const fin = tr[tr.length - 1];

        plane(root2.querySelector('.viz-pane'), {
          contours: [{A: S.Acontour, center: S.centre, levels: S.levels}],
          constraint: state.rho > 0.02 ? {type: state.rho >= 0.999 ? 'l1' : 'en',
            t: state.rho >= 0.999 ? l1(fin.w) : state.rho * l1(fin.w) + (1 - state.rho) / 2 * l2sq(fin.w),
            rho: state.rho} : {type: 'l2', t: Math.sqrt(l2sq(fin.w))},
          path: [{pts, color: 'var(--c-lasso)', width: 2.4}],
          activeLine: t.init ? null : {axis: t.j, w: t.wBefore},
          points: (S.wOls ? [{w: S.wOls, label: 'OLS', color: 'var(--c-ols)', r: 5}] : [])
            .concat([{w, label: 'w now', color: 'var(--c-lasso)'}])
        });
        root2.querySelector('.viz-pane').insertAdjacentHTML('beforeend',
          `<p class="muted small" style="margin:8px 0 0">${t.init
            ? 'Starting point. The shaded region is the penalty budget the solution will end up on.'
            : `Amber dashed line: the axis this step searches along — every other coefficient is
               frozen, so the solution can only move horizontally or vertically. Corners in the
               orange path are the algorithm switching coordinates.`}</p>`);

        /* soft-threshold panel */
        const sp = root2.querySelector('.soft-pane');
        if (sp) {
          const denom = t.init ? 1 + state.alpha * (1 - state.rho) : t.denom;
          const xs = [];
          for (let i = 0; i <= 200; i++) xs.push(-1.2 + 2.4 * i / 200);
          const series = [
            {name: 'no penalty (ρⱼ)', color: 'var(--c-ols)', dashed: true,
             points: xs.map(x => [x, x])},
            {name: 'this update', color: 'var(--c-lasso)', width: 2.6,
             points: xs.map(x => [x, ML.softThreshold(x, l1p) / denom])}
          ];
          Plot.line(sp, series, {height: 250, dots: false, zeroLine: true,
            xLabel: 'ρⱼ  (partial correlation with the residual)', yLabel: 'updated wⱼ',
            vLine: t.init ? null : t.rhoJ,
            vLabel: t.init ? null : 'ρ = ' + F(t.rhoJ, 3)});
          sp.insertAdjacentHTML('beforeend',
            `<p class="muted small" style="margin:6px 0 0">Flat section is the dead zone
             |ρ| ≤ ${F(l1p, 3)}: any feature in there is set to exactly zero.
             ${state.rho < 0.999 ? `The ℓ₂ term also tilts the live section down by a factor
             ${F(1 / denom, 3)} — threshold <em>and</em> shrink.` : ''}</p>`);
        }

        /* objective panel */
        const op = root2.querySelector('.obj-pane');
        if (op) {
          Plot.line(op, [{name: 'objective', color: 'var(--accent)',
            points: tr.map((x, i) => [i, x.obj])}],
            {height: 250, xLabel: 'coordinate update', yLabel: 'objective  f(w) + penalty',
             vLine: s.idx, dotR: 2.6});
        }

        /* three-estimator comparison (elastic net section only) */
        const cp = root2.querySelector('.cmp-pane');
        if (cp) {
          const wl = cdTrace(state.alpha, 1, [0, 0]).slice(-1)[0].w;
          const wr = cdTrace(state.alpha, 0, [0, 0]).slice(-1)[0].w;
          Plot.param2d(cp, {
            lim: S.lim, maxSide: 330, labels: S.labels.map((l, i) => `w${i + 1}`),
            contours: [{A: S.Acontour, center: S.centre, levels: S.levels}],
            points: (S.wOls ? [{w: S.wOls, label: 'OLS', color: 'var(--c-ols)', r: 5}] : [])
              .concat([{w: wl, label: 'Lasso', color: 'var(--c-lasso)', r: 5},
               {w: wr, label: 'Ridge', color: 'var(--c-ridge)', r: 5},
               {w, label: 'Elastic Net', color: 'var(--c-enet)', r: 6}])
          });
          Plot.legend($('enetCmpLeg'), [
            {name: 'OLS', color: 'var(--c-ols)'}, {name: 'Lasso (ρ=1)', color: 'var(--c-lasso)'},
            {name: 'Ridge (ρ=0)', color: 'var(--c-ridge)'},
            {name: 'Elastic Net (ρ=' + F(state.rho, 2) + ')', color: 'var(--c-enet)'}]);
        }
      }
    });

    cfg.bind(state, player);
    return player;
  }

  const lassoPlayer = makeCD('wLasso', {
    alpha0: 0.3, rho0: 1, showRho: false,
    bind(state, player) {
      $('lassoAlpha').addEventListener('input', e => {
        state.alpha = Math.pow(10, +e.target.value);
        $('lassoAlphaV').textContent = F(state.alpha, 3);
        player.rebuild(false);
      });
      $('lassoStart').addEventListener('change', e => {
        state.start = e.target.value; player.rebuild(false);
      });
    }
  });

  const enetPlayer = makeCD('wEnet', {
    alpha0: 0.3, rho0: 0.5, showRho: true,
    bind(state, player) {
      $('enetAlpha').addEventListener('input', e => {
        state.alpha = Math.pow(10, +e.target.value);
        $('enetAlphaV').textContent = F(state.alpha, 3);
        player.rebuild(true);
      });
      $('enetRho').addEventListener('input', e => {
        state.rho = +e.target.value;
        $('enetRhoV').textContent = F(state.rho, 2);
        player.rebuild(true);
      });
    }
  });

  /* ======================================================================
     6 · BAYESIAN UPDATING
  ====================================================================== */
  let bayesTau = 1, bayesSigma = 10;
  const bayesPlayer = makePlayer($('wBayes'), {
    speed: 1300,
    steps() {
      const tau2 = bayesTau * bayesTau;
      const sigmaZ = bayesSigma / S.ysd;              // noise on the standardized target
      const s2 = sigmaZ * sigmaZ;
      const ks = [0, 2, 5, 10, 20, 40, 80, S.n].filter((v, i, a) => v <= S.n && a.indexOf(v) === i);
      const eqAlpha = s2 / tau2;

      return ks.map((k, i) => {
        const Zk = S.Z.slice(0, k), yk = S.y.slice(0, k);
        let post, wMl = null, Alik = null;
        if (k === 0) {
          post = {mu: [0, 0], cov: [[tau2, 0], [0, tau2]], sd: [bayesTau, bayesTau]};
        } else {
          post = ML.bayesLinear(Zk, yk, tau2, s2);
          const Ak = ML.matmul(ML.T(Zk), Zk);
          Alik = Ak.map(r => r.map(v => v / s2));
          const Ai = ML.inv(Ak);
          if (Ai) wMl = ML.matvec(Ai, ML.matvec(ML.T(Zk), yk));
        }
        const prec = ML.inv(post.cov);
        const ridgeW = (() => {
          if (k === 0) return [0, 0];
          const Ak = ML.matmul(ML.T(Zk), Zk);
          const M = [[Ak[0][0] + eqAlpha, Ak[0][1]], [Ak[1][0], Ak[1][1] + eqAlpha]];
          const Mi = ML.inv(M);
          return Mi ? ML.matvec(Mi, ML.matvec(ML.T(Zk), yk)) : [0, 0];
        })();

        return {
          short: k === 0 ? 'Prior only' : `${k} trips`,
          title: k === 0 ? 'Before any data: the prior'
                         : `After ${k} trip${k > 1 ? 's' : ''}`,
          math: `${row('μ  posterior mean =', vec(post.mu, {highlight: [[0, 0], [1, 0]]}))}
            ${row('Σ  posterior covariance =', mat(post.cov, {dp: 4}))}
            <div style="margin-top:12px">${kv([
              ['sd(w₁)', '± ' + F(post.sd[0], 3), post.sd[0] > 0.3 ? 'warn' : 'good'],
              ['sd(w₂)', '± ' + F(post.sd[1], 3), post.sd[1] > 0.3 ? 'warn' : 'good'],
              ['95% interval for w₁',
               `[${F(post.mu[0] - 1.96 * post.sd[0], 2)}, ${F(post.mu[0] + 1.96 * post.sd[0], 2)}]`,
               (post.mu[0] - 1.96 * post.sd[0]) * (post.mu[0] + 1.96 * post.sd[0]) < 0 ? 'warn' : 'good'],
              ['95% interval for w₂',
               `[${F(post.mu[1] - 1.96 * post.sd[1], 2)}, ${F(post.mu[1] + 1.96 * post.sd[1], 2)}]`,
               (post.mu[1] - 1.96 * post.sd[1]) * (post.mu[1] + 1.96 * post.sd[1]) < 0 ? 'warn' : 'good'],
              ['equivalent ridge α = σ²/τ²', F(eqAlpha, 3)],
              ['ridge solution at that α',
               k === 0 ? '(0.00, 0.00)' : `(${F(ridgeW[0], 3)}, ${F(ridgeW[1], 3)})`, 'good'],
              ['MLE on these rows',
               wMl ? `(${F(wMl[0], 3)}, ${F(wMl[1], 3)})` : 'undefined — not enough data',
               wMl ? '' : 'warn']])}</div>`,
          narration: k === 0
            ? `<p>Before any data, the prior says each coefficient is near zero with standard
               deviation τ = <span class="hl">${F(bayesTau, 2)}</span> — a circle centred on the
               origin, favouring no direction.</p>
               <p>This assumption does real work: a tight prior that is wrong biases every estimate
               that follows. Raise τ to widen the circle toward indifference.</p>`
            : `<p>The likelihood from ${k} trip${k > 1 ? 's' : ''} (amber) multiplies the prior (blue)
               to give the posterior (green). Precision adds: $\\Sigma^{-1} = X^\\top X/\\sigma^2 +
               I/\\tau^2$, so each trip tightens the green contours and never loosens them.</p>
               <p>The posterior mean
               <span class="hl">(${F(post.mu[0], 3)}, ${F(post.mu[1], 3)})</span> equals the ridge
               solution at $\\alpha = \\sigma^2/\\tau^2 = ${F(eqAlpha, 2)}$, listed on the left.
               Ridge is this distribution with the error bars removed.</p>
               ${k < 12
                 ? `<p>With ${k} trips the intervals are wide${wMl ? '' : ' and the MLE does not yet exist'},
                    and the prior still carries most of the answer.</p>`
                 : `<p>The likelihood now dominates: the posterior mean has moved most of the way from
                    the origin to the least-squares point, and the intervals have tightened to
                    ±${F(post.sd[0], 2)}.</p>`}`,
          viz: {k, post, prec, Alik, wMl, tau2}
        };
      });
    },
    viz(root, s) {
      const host = root.querySelector('.viz-pane');
      const {k, post, prec, Alik, wMl, tau2} = s.viz;
      const contours = [{A: [[1 / tau2, 0], [0, 1 / tau2]], center: [0, 0],
        levels: [0.3, 1.2, 2.8], color: 'var(--c-ridge)', dashed: true, opacity: .7}];
      if (Alik && wMl) contours.push({A: Alik, center: wMl,
        levels: [0.3, 1.2, 2.8], color: 'var(--amber)', dashed: true, opacity: .7});
      contours.push({A: prec, center: post.mu, levels: [0.3, 1.2, 2.8],
        color: 'var(--accent)', width: 2, fill: 'var(--accent)', opacity: 1});

      plane(host, {
        contours,
        points: (S.wOls ? [{w: S.wOls, label: 'OLS', color: 'var(--c-ols)', r: 4.5}] : [])
          .concat([{w: post.mu, label: 'μ', color: 'var(--accent)'}])
      });
      host.insertAdjacentHTML('beforeend',
        `<div class="legend" style="margin-top:8px">
           <span><i style="background:var(--c-ridge)"></i>prior</span>
           <span><i style="background:var(--amber)"></i>likelihood (${k} trips)</span>
           <span><i style="background:var(--accent)"></i>posterior</span>
         </div>
         <p class="muted small" style="margin:6px 0 0">${k === 0
           ? 'Only the prior exists. Its circle is centred on zero because the prior is isotropic.'
           : k < 12
             ? 'The likelihood is still a wide, tilted ellipse — few trips, and correlated features, mean the data constrains one direction far better than the other.'
             : 'The posterior (solid green) has contracted to a small blob — that shrinking area IS the growing certainty. Its centre μ is simultaneously the posterior mean, the MAP estimate and the ridge solution.'}</p>`);
    }
  });
  $('bayesTau').addEventListener('input', e => {
    bayesTau = Math.pow(10, +e.target.value);
    $('bayesTauV').textContent = F(bayesTau, 2);
    bayesPlayer.rebuild(true);
  });
  $('bayesSigma').addEventListener('input', e => {
    bayesSigma = +e.target.value;
    $('bayesSigmaV').textContent = bayesSigma;
    bayesPlayer.rebuild(true);
  });

  /* ======================================================================
     boot
  ====================================================================== */
  const ALL = [olsPlayer, mapPlayer, ridgePlayer, lassoPlayer, enetPlayer, bayesPlayer];
  UI.onDraw(() => {
    rebuild();
    S.drawSetup();
    ALL.forEach(p => p.rebuild(true));
  });
})();
