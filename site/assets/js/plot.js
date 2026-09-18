/* =============================================================================
   plot.js — dependency-free SVG charts (line / bar / scatter / histogram /
   heatmap / density). Everything is theme-aware through CSS variables, so the
   figures flip with the light/dark toggle without being redrawn.
============================================================================= */
(function (global) {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const el = (n, a = {}) => {
    const e = document.createElementNS(NS, n);
    for (const k in a) if (a[k] != null) e.setAttribute(k, a[k]);
    return e;
  };
  const C = {
    axis: 'var(--line-2)', grid: 'var(--line)', text: 'var(--fg-3)', strong: 'var(--fg-2)',
    series: ['var(--c-ridge)', 'var(--c-lasso)', 'var(--c-enet)', 'var(--accent)',
             'var(--rose)', 'var(--c-ols)', 'var(--blue)', 'var(--amber)', 'var(--violet)']
  };
  const fmt = (v, d = 2) => {
    if (!isFinite(v)) return '—';
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1);
    return v.toFixed(d).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
  };

  let uid = 0;

  /**
   * Set up an SVG sized to its container. The viewBox matches the pixel width
   * exactly, so nothing is scaled and text stays crisp; a clip path keeps data
   * marks inside the plot area instead of bleeding over the surrounding page.
   */
  function frame(host, opt) {
    host.innerHTML = '';
    host.style.position = 'relative';
    const W = Math.max(opt.width || host.clientWidth || 640, 200);
    const H = opt.height || 320;
    const m = Object.assign({t: 16, r: 16, b: 42, l: 54}, opt.margin);
    const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, width: '100%', height: H,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img', 'aria-label': opt.title || 'chart'});
    svg.style.display = 'block';
    svg.style.overflow = 'hidden';
    svg.style.maxWidth = '100%';
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const clipId = 'clip' + (++uid);
    const defs = el('defs');
    const cp = el('clipPath', {id: clipId});
    cp.appendChild(el('rect', {x: -2, y: -2, width: iw + 4, height: ih + 4}));
    defs.appendChild(cp);
    svg.appendChild(defs);
    host.appendChild(svg);
    let tip = host.querySelector('.tip-layer');
    if (!tip) { tip = document.createElement('div'); tip.className = 'tip-layer'; host.appendChild(tip); }
    return {svg, W, H, m, iw, ih, tip, host, clipId};
  }

  /** A group inside the axes that clips its contents to the plot rectangle. */
  function clipped(f, parent) {
    const g = el('g', {'clip-path': `url(#${f.clipId})`});
    (parent || f.svg).appendChild(g);
    return g;
  }
  function showTip(f, x, y, text) {
    f.tip.textContent = text; f.tip.style.opacity = 1;
    const sc = f.host.clientWidth / f.W;
    f.tip.style.left = (x * sc + 12) + 'px';
    f.tip.style.top = (y * sc - 6) + 'px';
  }
  const hideTip = f => { f.tip.style.opacity = 0; };

  function axes(f, sx, sy, o) {
    const {svg, m, iw, ih} = f;
    const g = el('g', {transform: `translate(${m.l},${m.t})`});
    svg.appendChild(g);
    (o.yTicks || sy.ticks(5)).forEach(t => {
      const y = sy(t);
      g.appendChild(el('line', {x1: 0, x2: iw, y1: y, y2: y, stroke: C.grid, 'stroke-width': 1}));
      const lb = el('text', {x: -9, y: y + 4, 'text-anchor': 'end', fill: C.text,
        'font-size': 11, 'font-family': 'var(--mono)'});
      lb.textContent = o.yFmt ? o.yFmt(t) : fmt(t);
      g.appendChild(lb);
    });
    (o.xTicks || sx.ticks(6)).forEach(t => {
      const x = sx(t);
      if (o.xGrid !== false)
        g.appendChild(el('line', {x1: x, x2: x, y1: 0, y2: ih, stroke: C.grid, 'stroke-width': 1,
          'stroke-dasharray': '2 3'}));
      const lb = el('text', {x, y: ih + 17, 'text-anchor': 'middle', fill: C.text,
        'font-size': 11, 'font-family': 'var(--mono)'});
      lb.textContent = o.xFmt ? o.xFmt(t) : fmt(t);
      g.appendChild(lb);
    });
    g.appendChild(el('line', {x1: 0, x2: iw, y1: ih, y2: ih, stroke: C.axis, 'stroke-width': 1.2}));
    g.appendChild(el('line', {x1: 0, x2: 0, y1: 0, y2: ih, stroke: C.axis, 'stroke-width': 1.2}));
    if (o.xLabel) {
      const t = el('text', {x: iw / 2, y: ih + 37, 'text-anchor': 'middle', fill: C.strong, 'font-size': 12});
      t.textContent = o.xLabel; g.appendChild(t);
    }
    if (o.yLabel) {
      const t = el('text', {x: -(ih / 2), y: -40, 'text-anchor': 'middle', fill: C.strong,
        'font-size': 12, transform: `rotate(-90)`});
      t.setAttribute('transform', `rotate(-90) translate(${-ih / 2 - 0} ${0})`);
      t.setAttribute('x', 0); t.setAttribute('y', -40);
      t.textContent = o.yLabel; g.appendChild(t);
    }
    return g;
  }

  function linScale(d0, d1, r0, r1) {
    if (d0 === d1) { d1 = d0 + 1; }
    const s = v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
    s.invert = q => d0 + (q - r0) / (r1 - r0) * (d1 - d0);
    s.domain = [d0, d1];
    s.ticks = n => niceTicks(d0, d1, n);
    return s;
  }
  function logScale(d0, d1, r0, r1) {
    const a = Math.log10(Math.max(d0, 1e-12)), b = Math.log10(Math.max(d1, 1e-11));
    const s = v => r0 + (Math.log10(Math.max(v, 1e-12)) - a) / (b - a) * (r1 - r0);
    s.invert = q => Math.pow(10, a + (q - r0) / (r1 - r0) * (b - a));
    s.domain = [d0, d1];
    s.ticks = () => {
      const out = [];
      for (let e = Math.floor(a); e <= Math.ceil(b); e++) {
        const v = Math.pow(10, e);
        if (v >= d0 * 0.999 && v <= d1 * 1.001) out.push(v);
      }
      return out.length ? out : [d0, d1];
    };
    return s;
  }
  function niceTicks(a, b, n = 5) {
    const span = b - a || 1;
    const step0 = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-6; v += step)
      out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  const extent = arr => arr.reduce((a, v) => [Math.min(a[0], v), Math.max(a[1], v)], [Infinity, -Infinity]);
  function pad(lo, hi, f = 0.08) {
    if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
    if (lo === hi) return [lo - 1, hi + 1];
    const d = (hi - lo) * f; return [lo - d, hi + d];
  }

  /* ---------------- line chart ---------------- */
  /* series: [{name, color, points:[[x,y],...], dashed, width}] */
  function line(host, series, o = {}) {
    const f = frame(host, o);
    const xs = series.flatMap(s => s.points.map(p => p[0]));
    const ys = series.flatMap(s => s.points.map(p => p[1]));
    let [x0, x1] = o.xDomain || extent(xs);
    let [y0, y1] = o.yDomain || pad(...extent(ys));
    const sx = (o.logX ? logScale : linScale)(x0, x1, 0, f.iw);
    const sy = linScale(y0, y1, f.ih, 0);
    const g = axes(f, sx, sy, o);
    const gc = clipped(f, g);
    if (o.zeroLine && y0 < 0 && y1 > 0)
      g.appendChild(el('line', {x1: 0, x2: f.iw, y1: sy(0), y2: sy(0),
        stroke: C.axis, 'stroke-width': 1.4, 'stroke-dasharray': '4 3'}));
    if (o.vLine != null && o.vLine >= x0 && o.vLine <= x1) {
      g.appendChild(el('line', {x1: sx(o.vLine), x2: sx(o.vLine), y1: 0, y2: f.ih,
        stroke: 'var(--accent)', 'stroke-width': 1.6, 'stroke-dasharray': '5 3', opacity: .8}));
      if (o.vLabel) {
        const t = el('text', {x: sx(o.vLine) + 5, y: 12, fill: 'var(--accent)', 'font-size': 10.5,
          'font-family': 'var(--mono)'});
        t.textContent = o.vLabel; g.appendChild(t);
      }
    }
    series.forEach((s, i) => {
      const col = s.color || C.series[i % C.series.length];
      const d = s.points.map((p, k) => (k ? 'L' : 'M') + sx(p[0]).toFixed(2) + ' ' + sy(p[1]).toFixed(2)).join(' ');
      gc.appendChild(el('path', {d, fill: 'none', stroke: col, 'stroke-width': s.width || 2,
        'stroke-linejoin': 'round', 'stroke-linecap': 'round',
        'stroke-dasharray': s.dashed ? '5 4' : null, opacity: s.opacity || 1}));
      if (o.dots !== false) s.points.forEach(p => {
        const c = el('circle', {cx: sx(p[0]), cy: sy(p[1]), r: o.dotR || 3, fill: col,
          stroke: 'var(--bg-2)', 'stroke-width': 1.2});
        c.style.cursor = 'crosshair';
        c.addEventListener('mouseenter', () => showTip(f, sx(p[0]) + f.m.l, sy(p[1]) + f.m.t,
          `${s.name}\n${o.xLabel || 'x'}=${fmt(p[0], 3)}\n${o.yLabel || 'y'}=${fmt(p[1], 3)}`));
        c.addEventListener('mouseleave', () => hideTip(f));
        gc.appendChild(c);
      });
      if (s.label !== false && o.inlineLabels && s.points.length) {
        const last = s.points[s.points.length - 1];
        const t = el('text', {x: sx(last[0]) - 6, y: sy(last[1]) - 7, fill: col,
          'font-size': 10.5, 'font-weight': 600, 'text-anchor': 'end'});
        t.textContent = s.name; gc.appendChild(t);
      }
    });
    return f;
  }

  /* ---------------- horizontal bar chart (coefficients) ---------------- */
  /* items: [{label, value, color}] */
  function barsH(host, items, o = {}) {
    const H = o.height || Math.max(150, items.length * 26 + 46);
    const f = frame(host, Object.assign({}, o, {height: H,
      margin: Object.assign({t: 10, r: 56, b: 30, l: o.labelWidth || 145}, o.margin)}));
    let lo, hi;
    if (o.xDomain) { [lo, hi] = o.xDomain; }
    else {
      const vals = items.map(d => d.value);
      [lo, hi] = extent(vals);
      lo = Math.min(0, lo); hi = Math.max(0, hi);
      if (o.symmetric) { const m2 = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m2; hi = m2; }
      [lo, hi] = pad(lo, hi, .06);
    }
    const sx = linScale(lo, hi, 0, f.iw);
    const band = f.ih / items.length, iw = f.iw;
    const g = el('g', {transform: `translate(${f.m.l},${f.m.t})`});
    f.svg.appendChild(g);
    const gc = clipped(f, g);
    sx.ticks(5).forEach(t => {
      g.appendChild(el('line', {x1: sx(t), x2: sx(t), y1: 0, y2: f.ih, stroke: C.grid}));
      const lb = el('text', {x: sx(t), y: f.ih + 16, 'text-anchor': 'middle', fill: C.text,
        'font-size': 10.5, 'font-family': 'var(--mono)'});
      lb.textContent = fmt(t, 1); g.appendChild(lb);
    });
    g.appendChild(el('line', {x1: sx(0), x2: sx(0), y1: 0, y2: f.ih, stroke: C.axis, 'stroke-width': 1.3}));
    items.forEach((d, i) => {
      const y = i * band + band * .17, h = band * .66;
      const x = Math.min(sx(0), sx(d.value)), w = Math.abs(sx(d.value) - sx(0));
      const zero = Math.abs(d.value) < 1e-8;
      const r = el('rect', {x, y, width: Math.max(w, zero ? 0 : 1.2), height: h, rx: 3,
        fill: d.color || (d.value >= 0 ? 'var(--accent)' : 'var(--rose)'), opacity: zero ? .25 : .92});
      r.addEventListener('mouseenter', () => showTip(f, sx(d.value) + f.m.l, y + f.m.t + h / 2,
        `${d.label}\n${fmt(d.value, 3)}`));
      r.addEventListener('mouseleave', () => hideTip(f));
      gc.appendChild(r);
      const lb = el('text', {x: -9, y: y + h / 2 + 4, 'text-anchor': 'end',
        fill: zero ? C.text : C.strong, 'font-size': 11.5, 'font-family': 'var(--mono)'});
      lb.textContent = d.label; g.appendChild(lb);
      // Values live in a fixed right-hand column so they never collide with a
      // long feature name when the bar points left.
      const vl = el('text', {x: iw + 7, y: y + h / 2 + 4, 'text-anchor': 'start',
        fill: zero ? C.text : C.strong, 'font-size': 10.5, 'font-family': 'var(--mono)'});
      vl.textContent = zero ? '0' : fmt(d.value, 2); g.appendChild(vl);
    });
    return f;
  }

  /* ---------------- scatter ---------------- */
  function scatter(host, pts, o = {}) {
    const f = frame(host, o);
    const [x0, x1] = o.xDomain || pad(...extent(pts.map(p => p.x)));
    const [y0, y1] = o.yDomain || pad(...extent(pts.map(p => p.y)));
    const sx = linScale(x0, x1, 0, f.iw), sy = linScale(y0, y1, f.ih, 0);
    const g = axes(f, sx, sy, o);
    const gc = clipped(f, g);
    pts.forEach(p => {
      const c = el('circle', {cx: sx(p.x), cy: sy(p.y), r: p.r || o.r || 3.6,
        fill: p.color || 'var(--accent)', opacity: o.opacity || .62,
        stroke: 'var(--bg-2)', 'stroke-width': .8});
      if (p.tip) {
        c.style.cursor = 'crosshair';
        c.addEventListener('mouseenter', () => showTip(f, sx(p.x) + f.m.l, sy(p.y) + f.m.t, p.tip));
        c.addEventListener('mouseleave', () => hideTip(f));
      }
      gc.appendChild(c);
    });
    if (o.fit) {   // {slope, intercept} drawn across the x-domain
      const y_a = o.fit.intercept + o.fit.slope * x0, y_b = o.fit.intercept + o.fit.slope * x1;
      gc.appendChild(el('line', {x1: sx(x0), y1: sy(y_a), x2: sx(x1), y2: sy(y_b),
        stroke: 'var(--rose)', 'stroke-width': 2, 'stroke-dasharray': '6 4'}));
    }
    if (o.diagonal) {
      const lo = Math.max(x0, y0), hi = Math.min(x1, y1);
      gc.appendChild(el('line', {x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi),
        stroke: C.axis, 'stroke-width': 1.4, 'stroke-dasharray': '5 4'}));
    }
    return f;
  }

  /* ---------------- histogram ---------------- */
  function histogram(host, values, o = {}) {
    const f = frame(host, o);
    const bins = o.bins || 18;
    const [lo, hi] = extent(values);
    const w = (hi - lo) / bins || 1;
    const counts = new Array(bins).fill(0);
    values.forEach(v => { let b = Math.floor((v - lo) / w); if (b >= bins) b = bins - 1; if (b < 0) b = 0; counts[b]++; });
    const sx = linScale(lo, hi, 0, f.iw), sy = linScale(0, Math.max(...counts) * 1.08, f.ih, 0);
    const g = axes(f, sx, sy, Object.assign({xGrid: false}, o));
    const gc = clipped(f, g);
    counts.forEach((c, i) => {
      const x = sx(lo + i * w), x2 = sx(lo + (i + 1) * w);
      const r = el('rect', {x: x + .6, y: sy(c), width: Math.max(x2 - x - 1.2, 1), height: f.ih - sy(c),
        fill: o.color || 'var(--accent)', opacity: .8, rx: 2});
      r.addEventListener('mouseenter', () => showTip(f, x + f.m.l, sy(c) + f.m.t,
        `[${fmt(lo + i * w, 1)}, ${fmt(lo + (i + 1) * w, 1)})\nn = ${c}`));
      r.addEventListener('mouseleave', () => hideTip(f));
      gc.appendChild(r);
    });
    return f;
  }

  /* ---------------- correlation heatmap ---------------- */
  function heatmap(host, labels, M, o = {}) {
    const n = labels.length;
    const cell = o.cell || Math.min(46, Math.max(26, 420 / n));
    const L = o.labelWidth || 132, Tm = o.topLabel || 96;
    const W = L + n * cell + 14, H = Tm + n * cell + 14;
    const f = frame(host, {width: W, height: H, margin: {t: 0, r: 0, b: 0, l: 0}});
    f.svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const g = el('g'); f.svg.appendChild(g);
    const col = v => {
      const a = Math.min(Math.abs(v), 1);
      return v >= 0 ? `color-mix(in srgb, var(--c-ridge) ${(a * 88).toFixed(0)}%, var(--bg-2))`
                    : `color-mix(in srgb, var(--rose) ${(a * 88).toFixed(0)}%, var(--bg-2))`;
    };
    labels.forEach((lb, i) => {
      const t = el('text', {x: L - 8, y: Tm + i * cell + cell / 2 + 4, 'text-anchor': 'end',
        fill: C.strong, 'font-size': 10.5, 'font-family': 'var(--mono)'});
      t.textContent = lb; g.appendChild(t);
      const t2 = el('text', {x: 0, y: 0, fill: C.strong, 'font-size': 10.5, 'font-family': 'var(--mono)',
        'text-anchor': 'start', transform: `translate(${L + i * cell + cell / 2 + 4},${Tm - 8}) rotate(-90)`});
      t2.textContent = lb; g.appendChild(t2);
    });
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = M[i][j];
      const x = L + j * cell, y = Tm + i * cell;
      const r = el('rect', {x, y, width: cell - 1.5, height: cell - 1.5, rx: 3, fill: col(v)});
      r.addEventListener('mouseenter', () => showTip(f, x, y, `${labels[i]} × ${labels[j]}\nr = ${v.toFixed(3)}`));
      r.addEventListener('mouseleave', () => hideTip(f));
      g.appendChild(r);
      if (cell >= 30) {
        const t = el('text', {x: x + cell / 2 - .75, y: y + cell / 2 + 3.5, 'text-anchor': 'middle',
          'font-size': 9.5, 'font-family': 'var(--mono)',
          fill: Math.abs(v) > .55 ? 'var(--bg-2)' : C.text});
        t.textContent = v.toFixed(2).replace('0.', '.'); g.appendChild(t);
      }
    }
    return f;
  }

  /* ---------------- filled density curves ---------------- */
  /* curves: [{name,color,fn(x)}] over [x0,x1] */
  function density(host, curves, o = {}) {
    const f = frame(host, o);
    const [x0, x1] = o.xDomain || [-4, 4], N = o.samples || 240;
    const pts = curves.map(c => {
      const arr = [];
      for (let i = 0; i <= N; i++) { const x = x0 + (x1 - x0) * i / N; arr.push([x, c.fn(x)]); }
      return arr;
    });
    const ymax = o.yMax || Math.max(...pts.flat().map(p => p[1])) * 1.08;
    const sx = linScale(x0, x1, 0, f.iw), sy = linScale(0, ymax, f.ih, 0);
    const g = axes(f, sx, sy, o);
    const gc = clipped(f, g);
    curves.forEach((c, i) => {
      const col = c.color || C.series[i % C.series.length];
      const d = pts[i].map((p, k) => (k ? 'L' : 'M') + sx(p[0]).toFixed(2) + ' ' + sy(p[1]).toFixed(2)).join(' ');
      gc.appendChild(el('path', {d: d + ` L${sx(x1)} ${sy(0)} L${sx(x0)} ${sy(0)} Z`, fill: col, opacity: .13}));
      gc.appendChild(el('path', {d, fill: 'none', stroke: col, 'stroke-width': 2.2}));
    });
    if (o.marks) o.marks.forEach(mk => {
      g.appendChild(el('line', {x1: sx(mk.x), x2: sx(mk.x), y1: 0, y2: f.ih,
        stroke: mk.color || 'var(--fg-3)', 'stroke-width': 1.5, 'stroke-dasharray': '4 3'}));
      if (mk.label) {
        const t = el('text', {x: sx(mk.x) + 5, y: 13, fill: mk.color || 'var(--fg-3)',
          'font-size': 10.5, 'font-family': 'var(--mono)'});
        t.textContent = mk.label; g.appendChild(t);
      }
    });
    return f;
  }

  /* ---------------- 2-D parameter space ----------------
     Draws the loss surface of a two-feature problem as contours, plus any of:
     a penalty constraint region, the optimizer's trajectory, the line a
     coordinate-descent step searches along, and labelled points.

     opts = {A, center, levels, constraint:{type,t,rho}, path:[{pts,color}],
             points:[{w,label,color,r}], activeLine:{axis,w}, lim, labels}
  ------------------------------------------------------------------- */
  function param2d(host, o) {
    host.innerHTML = '';
    host.style.position = 'relative';
    const W = Math.max(o.width || host.clientWidth || 440, 220);
    const pad = 30;

    /* Both axes share one scale, or an L2 ball would render as an ellipse and the
       geometric argument would be wrong. The panel is a square sized to the column
       (up to maxSide), and the SVG height follows from it rather than the reverse. */
    const side = Math.min(Math.max(W - 2 * pad, 140), o.maxSide || 400);
    const H = side + 2 * pad + 14;
    const ox = (W - side) / 2, oy = pad;
    const lim = o.lim || 2.5;
    const sx = v => ox + (v + lim) / (2 * lim) * side;
    const sy = v => oy + side - (v + lim) / (2 * lim) * side;

    const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, width: '100%', height: H,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img', 'aria-label': o.title || 'parameter space'});
    svg.style.display = 'block';
    svg.style.overflow = 'hidden';
    svg.style.maxWidth = '100%';

    // Contours of a shallow loss surface are enormous; clip them to the square.
    const clipId = 'pclip' + (++uid);
    const defs = el('defs');
    const cp = el('clipPath', {id: clipId});
    cp.appendChild(el('rect', {x: ox, y: oy, width: side, height: side}));
    defs.appendChild(cp);
    svg.appendChild(defs);
    host.appendChild(svg);

    let tip = host.querySelector('.tip-layer');
    if (!tip) { tip = document.createElement('div'); tip.className = 'tip-layer'; host.appendChild(tip); }

    // plot background, so the clipped square reads as a panel
    svg.appendChild(el('rect', {x: ox, y: oy, width: side, height: side, rx: 6,
      fill: 'var(--bg)', stroke: 'var(--line)', 'stroke-width': 1}));

    const g = el('g', {'clip-path': `url(#${clipId})`});
    svg.appendChild(g);

    /* gridlines + axes */
    const ticks = niceTicks(-lim, lim, 5);
    ticks.forEach(t => {
      g.appendChild(el('line', {x1: sx(t), x2: sx(t), y1: sy(-lim), y2: sy(lim),
        stroke: C.grid, 'stroke-width': 1}));
      g.appendChild(el('line', {x1: sx(-lim), x2: sx(lim), y1: sy(t), y2: sy(t),
        stroke: C.grid, 'stroke-width': 1}));
    });
    g.appendChild(el('line', {x1: sx(-lim), x2: sx(lim), y1: sy(0), y2: sy(0),
      stroke: C.axis, 'stroke-width': 1.3}));
    g.appendChild(el('line', {x1: sx(0), x2: sx(0), y1: sy(-lim), y2: sy(lim),
      stroke: C.axis, 'stroke-width': 1.3}));

    /* constraint region */
    if (o.constraint) {
      const {type, t, rho} = o.constraint;
      const radiusAt = (ux, uy) => {
        const l1 = Math.abs(ux) + Math.abs(uy);
        if (type === 'l1') return t / l1;
        if (type === 'l2') return t;
        const r = rho == null ? 0.5 : rho;
        const A2 = (1 - r) / 2, B2 = r * l1, C2 = -t;
        return A2 < 1e-9 ? t / B2 : (-B2 + Math.sqrt(B2 * B2 - 4 * A2 * C2)) / (2 * A2);
      };
      const pts = [];
      for (let i = 0; i <= 720; i++) {
        const th = i * Math.PI / 360, ux = Math.cos(th), uy = Math.sin(th);
        const r = radiusAt(ux, uy);
        if (isFinite(r)) pts.push([r * ux, r * uy]);
      }
      const col = type === 'l1' ? 'var(--c-lasso)' : type === 'l2' ? 'var(--c-ridge)' : 'var(--c-enet)';
      if (pts.length) g.appendChild(el('path', {
        d: pts.map((p, i) => (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ' ' + sy(p[1]).toFixed(1)).join(' ') + ' Z',
        fill: col, 'fill-opacity': .13, stroke: col, 'stroke-width': 1.8}));
    }

    /* quadratic contour families: f(w) = f* + ½(w−c)ᵀA(w−c) */
    const families = o.contours || (o.A ? [{A: o.A, center: o.center, levels: o.levels,
      color: o.contourColor}] : []);
    families.forEach(fam => {
      const eg = eigSymLocal(fam.A);
      const c = fam.center || [0, 0];
      const levels = fam.levels || [0.05, 0.2, 0.5, 1.0, 1.8, 3.0, 4.6];
      levels.forEach((lv, li) => {
        const pts = [];
        for (let i = 0; i <= 160; i++) {
          const th = i * 2 * Math.PI / 160;
          const a1 = Math.sqrt(2 * lv / Math.max(eg.values[0], 1e-9)) * Math.cos(th);
          const a2 = Math.sqrt(2 * lv / Math.max(eg.values[1], 1e-9)) * Math.sin(th);
          const x = c[0] + eg.vectors[0][0] * a1 + eg.vectors[0][1] * a2;
          const y = c[1] + eg.vectors[1][0] * a1 + eg.vectors[1][1] * a2;
          if (!isFinite(x) || !isFinite(y)) return;
          // Clamp far-flung points so the path data stays small; the clip does the rest.
          pts.push([Math.max(-9 * lim, Math.min(9 * lim, x)),
                    Math.max(-9 * lim, Math.min(9 * lim, y))]);
        }
        if (pts.length < 3) return;
        g.appendChild(el('path', {
          d: pts.map((p, i) => (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ' ' + sy(p[1]).toFixed(1)).join(' ') + ' Z',
          fill: fam.fill || 'none', 'fill-opacity': fam.fill ? 0.07 : 0,
          stroke: fam.color || 'var(--fg-3)', 'stroke-dasharray': fam.dashed ? '4 3' : null,
          'stroke-width': fam.width || 1.1, opacity: (fam.opacity || 0.7) - li * 0.05}));
      });
    });

    /* the line a coordinate step searches along */
    if (o.activeLine) {
      const {axis, w} = o.activeLine;
      const a = axis === 0 ? [[-lim, w[1]], [lim, w[1]]] : [[w[0], -lim], [w[0], lim]];
      g.appendChild(el('line', {x1: sx(a[0][0]), y1: sy(a[0][1]), x2: sx(a[1][0]), y2: sy(a[1][1]),
        stroke: 'var(--amber)', 'stroke-width': 1.8, 'stroke-dasharray': '6 4', opacity: .95}));
    }

    /* trajectories */
    (o.path || []).forEach(pp => {
      if (!pp.pts || pp.pts.length < 2) return;
      g.appendChild(el('path', {
        d: pp.pts.map((p, i) => (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ' ' + sy(p[1]).toFixed(1)).join(' '),
        fill: 'none', stroke: pp.color || 'var(--accent)', 'stroke-width': pp.width || 2,
        'stroke-linejoin': 'round', 'stroke-dasharray': pp.dashed ? '4 3' : null,
        opacity: pp.opacity || 1}));
      if (pp.dots !== false) pp.pts.forEach((p, i) => {
        if (i % (pp.every || 1)) return;
        g.appendChild(el('circle', {cx: sx(p[0]), cy: sy(p[1]), r: 2.2,
          fill: pp.color || 'var(--accent)', opacity: .65}));
      });
    });

    /* points, with labels kept inside the square */
    (o.points || []).forEach(pt => {
      const px = sx(pt.w[0]), py = sy(pt.w[1]);
      if (!isFinite(px) || !isFinite(py)) return;
      g.appendChild(el('circle', {cx: px, cy: py, r: pt.r || 6,
        fill: pt.color || 'var(--accent)', stroke: 'var(--bg-2)', 'stroke-width': 2}));
      if (pt.label) {
        const right = px < ox + side * 0.72;
        const t = el('text', {
          x: px + (right ? 10 : -10), y: Math.max(py - 9, oy + 12),
          'text-anchor': right ? 'start' : 'end',
          fill: pt.color || 'var(--accent)', 'font-size': 10.5, 'font-weight': 650,
          'font-family': 'var(--mono)'});
        t.textContent = pt.label;
        g.appendChild(t);
      }
    });

    /* axis captions, outside the clipped square */
    const lx = el('text', {x: ox + side, y: oy + side + 15, 'text-anchor': 'end',
      fill: C.strong, 'font-size': 10.5, 'font-family': 'var(--mono)'});
    lx.textContent = (o.labels && o.labels[0]) || 'w₁';
    svg.appendChild(lx);
    const ly = el('text', {x: ox, y: oy - 8, fill: C.strong,
      'font-size': 10.5, 'font-family': 'var(--mono)'});
    ly.textContent = (o.labels && o.labels[1]) || 'w₂';
    svg.appendChild(ly);
    [-lim, 0, lim].forEach(t => {
      const e = el('text', {x: sx(t), y: oy + side + 15, 'text-anchor': 'middle',
        fill: C.text, 'font-size': 9.5, 'font-family': 'var(--mono)'});
      e.textContent = fmt(t, 1);
      if (t !== lim) svg.appendChild(e);
    });
    return {svg, sx, sy};
  }

  /* Local 2x2 symmetric eigen-decomposition (plot.js must not depend on mlcore). */
  function eigSymLocal(A) {
    const a = A[0][0], b = A[0][1], d = A[1][1];
    const tr = a + d, det = a * d - b * b;
    const disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0));
    const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    const v1 = Math.abs(b) > 1e-12 ? [l1 - d, b] : [1, 0];
    const v2 = Math.abs(b) > 1e-12 ? [l2 - d, b] : [0, 1];
    const n1 = Math.hypot(...v1) || 1, n2 = Math.hypot(...v2) || 1;
    return {values: [l1, l2],
            vectors: [[v1[0] / n1, v2[0] / n2], [v1[1] / n1, v2[1] / n2]]};
  }

  function legend(host, items) {
    host.innerHTML = items.map(i =>
      `<span><i style="background:${i.color}"></i>${i.name}</span>`).join('');
  }

  global.Plot = {line, barsH, scatter, histogram, heatmap, density, param2d, legend, fmt,
                 linScale, logScale, colors: C.series};
})(window);
