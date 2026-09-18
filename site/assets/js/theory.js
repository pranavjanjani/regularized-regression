/* =============================================================================
   theory.js — the six live figures in the lecture notes.
============================================================================= */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);

  /* ---------- Fig 1: variance inflation vs smallest singular value ---------- */
  (function figCond() {
    const host = $('figCond'); if (!host) return;
    const d = Prep.build(Prep.FEATURES.map(f => f.key));          // all 9, incl. derived
    const XtX = ML.matmul(ML.T(d.Ztr), d.Ztr);
    const ev = ML.eigSym(XtX).values.map(v => Math.max(v, 1e-9));
    const dmin = Math.sqrt(ev[ev.length - 1] / d.n);              // per-observation scale
    const pts = [];
    for (let i = 0; i <= 120; i++) {
      const s = 0.02 + (1.2 - 0.02) * i / 120;
      pts.push([s, 1 / (s * s)]);
    }
    UI.onDraw(() => {
      Plot.line(host, [{name: 'Var(ŵ) ∝ 1/d²', points: pts, color: 'var(--rose)'}], {
        height: 260, dots: false, logX: false,
        xLabel: 'smallest singular value  d_min (scaled)', yLabel: 'variance multiplier',
        yDomain: [0, 260], vLine: dmin,
        vLabel: 'Mumbai data: d_min = ' + dmin.toFixed(3)
      });
    });
  })();

  /* ---------- Fig 2: Gaussian vs Laplace prior ---------- */
  (function figPriors() {
    const host = $('figPriors'); if (!host) return;
    const curves = [
      {name: 'Gaussian  N(0, 1)  → Ridge', color: 'var(--c-ridge)', fn: x => ML.gaussPdf(x, 0, 1)},
      {name: 'Laplace  b = 1/√2  → Lasso', color: 'var(--c-lasso)', fn: x => ML.laplacePdf(x, 0, 1 / Math.SQRT2)}
    ];
    UI.onDraw(() => {
      Plot.density(host, curves, {height: 250, xDomain: [-4, 4],
        xLabel: 'coefficient  wⱼ', yLabel: 'prior density  p(wⱼ)'});
      Plot.legend($('figPriorsLeg'), curves.map(c => ({name: c.name, color: c.color})));
    });
  })();

  /* ---------- Fig 3: ridge shrinkage factor ---------- */
  (function figShrink() {
    const host = $('figShrink'); if (!host) return;
    const lams = [1, 10, 100];
    const cols = ['var(--accent)', 'var(--c-ridge)', 'var(--violet)'];
    const series = lams.map((l, i) => ({
      name: 'λ = ' + l, color: cols[i],
      points: Array.from({length: 121}, (_, k) => {
        const dd = 0.1 + (30 - 0.1) * k / 120;
        return [dd, dd * dd / (dd * dd + l)];
      })
    }));
    UI.onDraw(() => {
      Plot.line(host, series, {height: 250, dots: false, yDomain: [0, 1.04],
        xLabel: 'singular value  d', yLabel: 'shrinkage factor  d²/(d²+λ)'});
      Plot.legend($('figShrinkLeg'), series.map(s => ({name: s.name, color: s.color})));
    });
  })();

  /* ---------- Fig 4: soft-thresholding ---------- */
  (function figSoft() {
    const host = $('figSoft'); if (!host) return;
    const sl = $('softLam'), lv = $('softLamV');
    const draw = () => {
      const g = +sl.value; lv.textContent = g.toFixed(2);
      const xs = Array.from({length: 161}, (_, i) => -4 + 8 * i / 160);
      const series = [
        {name: 'OLS (identity)', color: 'var(--c-ols)', dashed: true,
         points: xs.map(x => [x, x])},
        {name: 'Soft threshold (Lasso)', color: 'var(--c-lasso)', width: 2.6,
         points: xs.map(x => [x, ML.softThreshold(x, g)])},
        {name: 'Hard threshold', color: 'var(--rose)', width: 1.6,
         points: xs.map(x => [x, Math.abs(x) > g ? x : 0])},
        {name: 'Ridge (λ=1)', color: 'var(--c-ridge)', width: 1.6,
         points: xs.map(x => [x, x / (1 + 1)])}
      ];
      Plot.line(host, series, {height: 280, dots: false, zeroLine: true,
        xLabel: 'ρⱼ  =  correlation of feature j with the partial residual',
        yLabel: 'fitted coefficient  ŵⱼ'});
      Plot.legend($('figSoftLeg'), series.map(s => ({name: s.name, color: s.color})));
    };
    sl.addEventListener('input', draw);
    UI.onDraw(draw);
  })();

  /* ---------- Fig 5: constraint geometry ---------- */
  (function figGeo() {
    const host = $('figGeo'); if (!host) return;
    const NS = 'http://www.w3.org/2000/svg';
    const E = (n, a) => { const e = document.createElementNS(NS, n);
      for (const k in a) if (a[k] != null) e.setAttribute(k, a[k]); return e; };
    let mode = 'l1';
    const tEl = $('geoT'), tV = $('geoTV'), note = $('geoNote');

    // A small 2-feature least-squares problem with correlated columns.
    const A = [[1, 0.78], [0.78, 1]];           // XᵀX for standardized, r = 0.78
    const wOls = [1.45, 0.55];
    const b = ML.matvec(A, wOls);               // Xᵀy consistent with that optimum
    const rss = w => {
      const dv = [w[0] - wOls[0], w[1] - wOls[1]];
      return dv[0] * (A[0][0] * dv[0] + A[0][1] * dv[1]) + dv[1] * (A[1][0] * dv[0] + A[1][1] * dv[1]);
    };
    const eig = ML.eigSym(A);

    const inside = (w, t) => {
      const l1 = Math.abs(w[0]) + Math.abs(w[1]), l2 = w[0] * w[0] + w[1] * w[1];
      if (mode === 'l1') return l1 <= t;
      if (mode === 'l2') return Math.sqrt(l2) <= t;
      return 0.5 * l1 + 0.5 * l2 <= t;
    };
    const radiusAt = (ux, uy, t) => {           // boundary radius along unit direction
      const l1 = Math.abs(ux) + Math.abs(uy);
      if (mode === 'l1') return t / l1;
      if (mode === 'l2') return t;
      // 0.5·r·l1 + 0.5·r² = t  (unit direction ⇒ ‖u‖²=1)
      const A2 = 0.5, B2 = 0.5 * l1, C2 = -t;
      return (-B2 + Math.sqrt(B2 * B2 - 4 * A2 * C2)) / (2 * A2);
    };

    function draw() {
      const t = +tEl.value; tV.textContent = t.toFixed(2);
      const W = Math.max(host.clientWidth || 560, 260), pad = 30;
      const lim = 2.6;
      // Equal scale on both axes, or the L2 ball would not look like a ball.
      // The square is sized by the column, and the SVG height follows from it.
      const side = Math.min(Math.max(W - 2 * pad, 150), 400);
      const H = side + 2 * pad + 10;
      const ox = (W - side) / 2, oy = pad;
      host.innerHTML = '';
      const svg = E('svg', {viewBox: `0 0 ${W} ${H}`, width: '100%', height: H,
        preserveAspectRatio: 'xMidYMid meet'});
      svg.style.display = 'block';
      svg.style.overflow = 'hidden';
      svg.style.maxWidth = '100%';
      const sx = v => ox + (v + lim) / (2 * lim) * side;
      const sy = v => oy + side - (v + lim) / (2 * lim) * side;
      const cid = 'geoclip';
      const defs = E('defs', {});
      const cp = E('clipPath', {id: cid});
      cp.appendChild(E('rect', {x: ox, y: oy, width: side, height: side}));
      defs.appendChild(cp); svg.appendChild(defs);
      svg.appendChild(E('rect', {x: ox, y: oy, width: side, height: side, rx: 6,
        fill: 'var(--bg)', stroke: 'var(--line)', 'stroke-width': 1}));
      const clip = E('g', {'clip-path': `url(#${cid})`});
      svg.appendChild(clip);
      host.appendChild(svg);

      // axes
      clip.appendChild(E('line', {x1: sx(-lim), y1: sy(0), x2: sx(lim), y2: sy(0),
        stroke: 'var(--line-2)', 'stroke-width': 1.2}));
      clip.appendChild(E('line', {x1: sx(0), y1: sy(-lim), x2: sx(0), y2: sy(lim),
        stroke: 'var(--line-2)', 'stroke-width': 1.2}));
      [['w₁', ox + side - 6, oy + side + 14, 'end'], ['w₂', ox, oy - 7, 'start']]
        .forEach(([tx, x, y, anc]) => {
          const e = E('text', {x, y, fill: 'var(--fg-3)', 'font-size': 11,
            'text-anchor': anc, 'font-family': 'var(--mono)'});
          e.textContent = tx; svg.appendChild(e);
        });

      // constraint region boundary
      const bpts = [];
      const STEPS = 1440;
      for (let i = 0; i <= STEPS; i++) {
        const th = i * 2 * Math.PI / STEPS, ux = Math.cos(th), uy = Math.sin(th);
        const r = radiusAt(ux, uy, t);
        bpts.push([r * ux, r * uy]);
      }
      // The corners are the whole point, so make sure they are exact candidates
      // rather than whatever the angular sweep happened to land near.
      const corners = mode === 'l2' ? [] : [[1, 0], [-1, 0], [0, 1], [0, -1]]
        .map(([ux, uy]) => { const r = radiusAt(ux, uy, t); return [r * ux, r * uy]; });
      clip.appendChild(E('path', {
        d: bpts.map((p, i) => (i ? 'L' : 'M') + sx(p[0]) + ' ' + sy(p[1])).join(' ') + ' Z',
        fill: mode === 'l1' ? 'var(--c-lasso)' : mode === 'l2' ? 'var(--c-ridge)' : 'var(--c-enet)',
        'fill-opacity': .15,
        stroke: mode === 'l1' ? 'var(--c-lasso)' : mode === 'l2' ? 'var(--c-ridge)' : 'var(--c-enet)',
        'stroke-width': 2
      }));

      // constrained solution: unconstrained point if feasible, else search the boundary
      let sol = wOls, onBoundary = false;
      if (!inside(wOls, t)) {
        onBoundary = true;
        let best = Infinity;
        bpts.concat(corners).forEach(p => { const v = rss(p); if (v < best) { best = v; sol = p; } });
      }

      // RSS contours, including the one through the solution
      const levels = [0.08, 0.3, 0.75, 1.5, 2.6];
      if (onBoundary) levels.push(rss(sol));
      levels.sort((a, b2) => a - b2).forEach(c => {
        const pts = [];
        for (let i = 0; i <= 180; i++) {
          const th = i * 2 * Math.PI / 180;
          const a1 = Math.sqrt(c / eig.values[0]) * Math.cos(th);
          const a2 = Math.sqrt(c / eig.values[1]) * Math.sin(th);
          const x = wOls[0] + eig.vectors[0][0] * a1 + eig.vectors[0][1] * a2;
          const y = wOls[1] + eig.vectors[1][0] * a1 + eig.vectors[1][1] * a2;
          pts.push([x, y]);
        }
        const isSol = onBoundary && Math.abs(c - rss(sol)) < 1e-9;
        clip.appendChild(E('path', {
          d: pts.map((p, i) => (i ? 'L' : 'M') + sx(p[0]) + ' ' + sy(p[1])).join(' ') + ' Z',
          fill: 'none', stroke: isSol ? 'var(--rose)' : 'var(--line-2)',
          'stroke-width': isSol ? 2 : 1, 'stroke-dasharray': isSol ? null : '3 3',
          opacity: isSol ? 1 : .85
        }));
      });

      // the two special points
      clip.appendChild(E('circle', {cx: sx(wOls[0]), cy: sy(wOls[1]), r: 5,
        fill: 'var(--c-ols)', stroke: 'var(--bg-2)', 'stroke-width': 1.5}));
      // Labels are drawn outside the clip and flipped to whichever side has room,
      // so they are never sliced off at the panel edge.
      const room = (x, w) => x + w < ox + side - 4;
      const oRight = room(sx(wOls[0]) + 9, 42);
      const lo = E('text', {x: sx(wOls[0]) + (oRight ? 9 : -9), y: sy(wOls[1]) - 7,
        'text-anchor': oRight ? 'start' : 'end', fill: 'var(--fg-2)',
        'font-size': 11, 'font-family': 'var(--mono)'});
      lo.textContent = 'ŵ OLS'; svg.appendChild(lo);
      clip.appendChild(E('circle', {cx: sx(sol[0]), cy: sy(sol[1]), r: 6,
        fill: 'var(--accent)', stroke: 'var(--bg-2)', 'stroke-width': 2}));
      const solRight = room(sx(sol[0]) + 11, 86);
      const ls = E('text', {x: sx(sol[0]) + (solRight ? 11 : -11), y: sy(sol[1]) + 17,
        'text-anchor': solRight ? 'start' : 'end', fill: 'var(--accent)',
        'font-size': 11, 'font-family': 'var(--mono)', 'font-weight': 600});
      ls.textContent = `(${sol[0].toFixed(2)}, ${sol[1].toFixed(2)})`;
      svg.appendChild(ls);

      const zeroed = Math.abs(sol[0]) < 1e-9 || Math.abs(sol[1]) < 1e-9;
      note.innerHTML = !onBoundary
        ? 'The budget is large enough to contain the least-squares solution, so the penalty is '
          + 'inactive and the answer is plain OLS.'
        : mode === 'l1'
          ? (zeroed ? '<strong>A corner.</strong> One coefficient is exactly zero — the feature has '
              + 'been deleted from the model. Shrink the budget further and the other one follows.'
              : 'The ellipse is touching an <em>edge</em> of the diamond, so both coefficients survive '
              + '— but keep shrinking the budget and contact will slide to a corner.')
          : mode === 'l2'
            ? 'Contact with a smooth circle: both coefficients shrink toward zero together, and '
              + 'neither ever reaches it.'
            : 'The Elastic Net region has corners <em>and</em> curved edges — it can zero a '
              + 'coefficient, but it rounds the diamond enough to keep correlated partners together.';
    }

    $('geoMode').addEventListener('click', e => {
      const b2 = e.target.closest('button'); if (!b2) return;
      [...e.currentTarget.children].forEach(c => c.classList.toggle('on', c === b2));
      mode = b2.dataset.m; draw();
    });
    tEl.addEventListener('input', draw);
    UI.onDraw(draw);
  })();

  /* ---------- Fig 6: posterior intervals vs sample size ---------- */
  (function figBayes() {
    const host = $('figBayes'); if (!host) return;
    const nEl = $('bayesN'), nV = $('bayesNV');
    const full = Prep.build(Prep.BASE8);
    const draw = () => {
      const n = +nEl.value; nV.textContent = n;
      const Z = full.Ztr.slice(0, n), y = full.ytr.slice(0, n);
      const c = ML.center(y);
      const post = ML.bayesLinear(Z, c.yc, 1e4, 100);   // τ²=10⁴ (weak), σ²=100
      if (!post) return;
      const items = [];
      full.labels.forEach((l, i) => {
        items.push({label: l, value: post.mu[i], color: 'var(--c-ridge)',
          lo: post.mu[i] - 1.96 * post.sd[i], hi: post.mu[i] + 1.96 * post.sd[i]});
      });
      const mx = Math.max(...items.map(d => Math.abs(d.hi)), ...items.map(d => Math.abs(d.lo)));
      const f = Plot.barsH(host, items.map(d => ({label: d.label, value: d.value,
        color: Math.sign(d.lo) === Math.sign(d.hi) ? 'var(--c-ridge)' : 'var(--fg-3)'})),
        {labelWidth: 120, height: 300, xDomain: [-mx * 1.1, mx * 1.1]});
      // overlay the credible intervals as whiskers
      const svg = host.querySelector('svg');
      const g = svg.querySelector('g');
      const band = (300 - 40) / items.length;
      const sx = v => 145 + (v + mx * 1.1) / (2 * mx * 1.1) * (f.iw);
      items.forEach((d, i) => {
        const y0 = i * band + band * .5;
        const NSx = 'http://www.w3.org/2000/svg';
        const ln = document.createElementNS(NSx, 'line');
        ln.setAttribute('x1', sx(d.lo) - f.m.l); ln.setAttribute('x2', sx(d.hi) - f.m.l);
        ln.setAttribute('y1', y0); ln.setAttribute('y2', y0);
        ln.setAttribute('stroke', 'var(--fg)'); ln.setAttribute('stroke-width', 1.6);
        ln.setAttribute('opacity', .75);
        g.appendChild(ln);
        [d.lo, d.hi].forEach(v => {
          const cap = document.createElementNS(NSx, 'line');
          cap.setAttribute('x1', sx(v) - f.m.l); cap.setAttribute('x2', sx(v) - f.m.l);
          cap.setAttribute('y1', y0 - 4); cap.setAttribute('y2', y0 + 4);
          cap.setAttribute('stroke', 'var(--fg)'); cap.setAttribute('stroke-width', 1.6);
          cap.setAttribute('opacity', .75);
          g.appendChild(cap);
        });
      });
    };
    nEl.addEventListener('input', draw);
    UI.onDraw(draw);
  })();
})();
