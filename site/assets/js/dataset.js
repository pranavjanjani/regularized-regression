/* =============================================================================
   dataset.js — the data explorer: summary stats, correlation heatmap,
   distributions and a sortable / filterable raw table.
============================================================================= */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const D = window.SURGE_DATA;
  const COLS = D.columns;
  const idx = c => COLS.indexOf(c);
  const colVals = c => D.rows.map(r => r[idx(c)]);
  const NUMERIC = COLS.filter(c => c !== 'zone' && c !== 'trip_id');

  /* ---------- 1 · summary stats ---------- */
  (function stats() {
    const surge = colVals('surge_additive_inr');
    const capped = surge.filter(v => v >= 150).length;
    const nRain = colVals('is_rain').reduce((a, b) => a + b, 0);
    const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
    const sd = a => { const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) ** 2))); };
    const cards = [
      ['Trips', D.rows.length, '150 train / 50 test'],
      ['Mean surge', '₹' + mean(surge).toFixed(1), 'sd ₹' + sd(surge).toFixed(1)],
      ['Rainy trips', nRain, (100 * nRain / D.rows.length).toFixed(0) + '% of the data'],
      ['At the ₹150 cap', capped, 'censored observations']
    ];
    $('dsStats').innerHTML = cards.map(([k, v, s]) =>
      `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="sub">${s}</div></div>`).join('');
    $('dsCap').textContent = capped;
  })();

  /* ---------- 3 · correlation heatmap ---------- */
  const HEAT = ['is_peak', 'is_rain', 'is_bad_weather', 'traffic_speed_kmph',
                'drivers_available_500m', 'open_requests_500m', 'historical_demand',
                'is_event_nearby', 'is_airport_pickup', 'is_weekend', 'surge_additive_inr'];
  const SHORT = {is_peak: 'peak', is_rain: 'rain', is_bad_weather: 'bad_weather',
    traffic_speed_kmph: 'traffic', drivers_available_500m: 'drivers',
    open_requests_500m: 'requests', historical_demand: 'hist_demand',
    is_event_nearby: 'event', is_airport_pickup: 'airport', is_weekend: 'weekend',
    surge_additive_inr: 'SURGE'};
  function drawHeat() {
    const series = HEAT.map(colVals);
    const M = series.map(a => series.map(b => ML.corr(a, b)));
    Plot.heatmap($('dsHeat'), HEAT.map(c => SHORT[c]), M, {cell: 42});
  }

  /* ---------- 4 · distributions ---------- */
  (function initSelects() {
    const opts = NUMERIC.map(c => `<option value="${c}">${c}</option>`).join('');
    $('dsCol').innerHTML = opts; $('dsX').innerHTML = opts;
    $('dsCol').value = 'surge_additive_inr';
    $('dsX').value = 'traffic_speed_kmph';
  })();
  let colorBy = 'none';

  function drawDists() {
    const yc = $('dsCol').value, xc = $('dsX').value;
    const yv = colVals(yc), xv = colVals(xc);

    Plot.histogram($('dsHist'), yv, {height: 280, bins: 20, xLabel: yc, yLabel: 'count',
      color: 'var(--accent)'});

    const rain = colVals('is_rain'), peak = colVals('is_peak');
    const pts = yv.map((v, i) => {
      let color = 'var(--accent)';
      if (colorBy === 'is_rain') color = rain[i] ? 'var(--c-ridge)' : 'var(--fg-3)';
      if (colorBy === 'is_peak') color = peak[i] ? 'var(--c-lasso)' : 'var(--fg-3)';
      if (colorBy === 'split') color = i < 150 ? 'var(--accent)' : 'var(--rose)';
      return {x: xv[i], y: v, color,
        tip: `trip ${i + 1} · ${D.rows[i][idx('zone')]}\n${xc} = ${xv[i]}\n${yc} = ${v}`};
    });
    Plot.scatter($('dsScatter'), pts, {height: 280, xLabel: xc, yLabel: yc, opacity: .7});

    const legends = {
      none: [], split: [{name: 'train (1–150)', color: 'var(--accent)'}, {name: 'test (151–200)', color: 'var(--rose)'}],
      is_rain: [{name: 'rain', color: 'var(--c-ridge)'}, {name: 'dry', color: 'var(--fg-3)'}],
      is_peak: [{name: 'peak', color: 'var(--c-lasso)'}, {name: 'off-peak', color: 'var(--fg-3)'}]
    };
    Plot.legend($('dsScatterLeg'), legends[colorBy]);

    const r = ML.corr(xv, yv);
    $('dsNote').innerHTML = `Correlation between <code>${xc}</code> and <code>${yc}</code> is `
      + `<strong>r = ${r.toFixed(3)}</strong>. `
      + (Math.abs(r) > 0.7 ? 'Strong enough that including both as predictors will make the '
          + 'least-squares solution unstable — this is exactly the pair to try in '
          + '<a href="lab.html#collinear">Lab §5</a>.'
        : Math.abs(r) > 0.35 ? 'Moderate — enough for the coefficients to trade off against each '
          + 'other, not enough to break the inverse.'
        : 'Weak. These two carry largely independent information.');
  }
  ['dsCol', 'dsX'].forEach(id => $(id).addEventListener('change', drawDists));
  $('dsColor').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    [...e.currentTarget.children].forEach(c => c.classList.toggle('on', c === b));
    colorBy = b.dataset.c; drawDists();
  });

  /* ---------- 5 · sortable, filterable table ---------- */
  const SHOW = ['trip_id', 'zone', 'hour', 'is_peak', 'is_rain', 'is_bad_weather',
    'traffic_speed_kmph', 'drivers_available_500m', 'open_requests_500m',
    'demand_supply_ratio', 'base_fare_inr', 'surge_additive_inr', 'final_fare_inr'];
  let sortCol = 'trip_id', sortDir = 1, filter = 'all', search = '';

  function rows() {
    let out = D.rows.map((r, i) => ({r, i}));
    if (filter === 'train') out = out.filter(o => o.i < 150);
    if (filter === 'test') out = out.filter(o => o.i >= 150);
    if (filter === 'capped') out = out.filter(o => o.r[idx('surge_additive_inr')] >= 150);
    if (filter === 'surge') out = out.filter(o => o.r[idx('surge_additive_inr')] > 100);
    if (search) {
      const q = search.toLowerCase();
      out = out.filter(o => String(o.r[idx('zone')]).toLowerCase().includes(q));
    }
    const si = idx(sortCol);
    out.sort((a, b) => {
      const x = a.r[si], y = b.r[si];
      return (typeof x === 'string' ? x.localeCompare(y) : x - y) * sortDir;
    });
    return out;
  }

  function drawTable() {
    const data = rows();
    $('dsTable').querySelector('thead').innerHTML = '<tr>' + SHOW.map(c => {
      const num = c !== 'zone';
      const arrow = c === sortCol ? (sortDir > 0 ? ' ↑' : ' ↓') : '';
      return `<th class="sortable${num ? ' num' : ''}" data-c="${c}">${c}${arrow}</th>`;
    }).join('') + '</tr>';
    $('dsTable').querySelector('tbody').innerHTML = data.map(o => '<tr>' + SHOW.map(c => {
      const v = o.r[idx(c)];
      const capped = c === 'surge_additive_inr' && v >= 150;
      return `<td class="${c === 'zone' ? '' : 'num'}"${capped ? ' style="color:var(--rose);font-weight:600"' : ''}>${v}</td>`;
    }).join('') + '</tr>').join('');
    $('tblCount').innerHTML = `Showing <strong>${data.length}</strong> of 200 trips`
      + (filter === 'capped' ? ' — every one of these is a censored observation.' : '.');
    return data;
  }

  $('dsTable').querySelector('thead').addEventListener('click', e => {
    const th = e.target.closest('th'); if (!th) return;
    const c = th.dataset.c;
    if (c === sortCol) sortDir = -sortDir; else { sortCol = c; sortDir = 1; }
    drawTable();
  });
  $('tblFilter').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    [...e.currentTarget.children].forEach(c => c.classList.toggle('on', c === b));
    filter = b.dataset.f; drawTable();
  });
  $('tblSearch').addEventListener('input', e => { search = e.target.value.trim(); drawTable(); });
  $('tblCsv').addEventListener('click', () => {
    const data = rows();
    const lines = [COLS.join(',')].concat(data.map(o => o.r.join(',')));
    UI.downloadText('mumbai_surge_filtered.csv', lines.join('\n'), 'text/csv');
  });

  UI.onDraw(() => { drawHeat(); drawDists(); drawTable(); });
})();
