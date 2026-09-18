/* =============================================================================
   uber.js — the live surge calculator in §7 of the case study.
   Fits an Elastic Net on the training split, then prices whatever scenario the
   student dials in, applying the assignment's policy layer on top.
============================================================================= */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  if (!$('surgeCard')) return;

  const CAP = 150, MIN_RATIO = 0.8, SURGE_RATIO = 1.5;
  const keys = Prep.FEATURES.map(f => f.key);           // all 9, incl. is_bad_weather
  const d = Prep.build(keys);
  const w = Prep.fit(d, 'enet', 0.5, 0.5);

  const flags = {is_rain: 1, is_event_nearby: 0, is_airport_pickup: 0, is_weekend: 0};

  const sliders = {
    uHour:    {el: $('uHour'),    out: $('uHourV'),    fmt: v => String(v).padStart(2, '0') + ':00'},
    uDrivers: {el: $('uDrivers'), out: $('uDriversV'), fmt: v => v},
    uReq:     {el: $('uReq'),     out: $('uReqV'),     fmt: v => v},
    uTraffic: {el: $('uTraffic'), out: $('uTrafficV'), fmt: v => v},
    uDist:    {el: $('uDist'),    out: $('uDistV'),    fmt: v => (+v).toFixed(1)}
  };

  function scenario() {
    const hour = +sliders.uHour.el.value;
    const traffic = +sliders.uTraffic.el.value;
    const isPeak = ((hour >= 8 && hour <= 10) || (hour >= 18 && hour <= 21)) ? 1 : 0;
    return {
      is_peak: isPeak,
      is_rain: flags.is_rain,
      traffic_speed_kmph: traffic,
      drivers_available_500m: +sliders.uDrivers.el.value,
      is_event_nearby: flags.is_event_nearby,
      is_airport_pickup: flags.is_airport_pickup,
      is_weekend: flags.is_weekend,
      open_requests_500m: +sliders.uReq.el.value,
      is_bad_weather: (flags.is_rain === 1 && traffic < 25) ? 1 : 0,
      _hour: hour, _dist: +sliders.uDist.el.value
    };
  }

  function draw() {
    for (const k in sliders) sliders[k].out.textContent = sliders[k].fmt(sliders[k].el.value);
    const row = scenario();

    const raw = Prep.predictRow(d, w, row);
    const ratio = row.open_requests_500m / row.drivers_available_500m;

    // Policy layer — deliberately kept OUTSIDE the model, as in the architecture diagram.
    let surge = raw, reason = 'model output, uncapped';
    if (ratio < MIN_RATIO) { surge = 0; reason = 'suppressed: supply exceeds demand'; }
    else if (ratio < SURGE_RATIO) { surge = 0; reason = 'suppressed: ratio below 1.5 threshold'; }
    else if (surge < 0) { surge = 0; reason = 'model went negative — floored at zero'; }
    else if (surge > CAP) { surge = CAP; reason = 'clipped at the ₹150 ethics cap'; }
    else { reason = 'within cap and above threshold'; }

    const timeMin = row._dist * 2 + (row.is_peak ? 12 : 0);
    const base = 40 + 12 * row._dist + 2 * timeMin;

    $('uRatio').textContent = ratio.toFixed(2);
    $('uRatioNote').textContent = ratio >= SURGE_RATIO ? 'above the 1.5 surge threshold'
      : ratio < MIN_RATIO ? 'soft market — no surge' : 'below threshold — no surge';
    $('uRaw').textContent = '₹' + raw.toFixed(0);
    $('uSurge').textContent = '₹' + surge.toFixed(0);
    $('uSurgeNote').textContent = reason;
    $('uFare').textContent = '₹' + (base + surge).toFixed(0);

    // Per-feature contribution in the additive model.
    const items = d.keys.map((k, j) => ({
      label: d.labels[j],
      value: w[j] * ((row[k] - d.mean[j]) / d.std[j]),
      color: 'var(--c-enet)'
    })).filter(it => Math.abs(it.value) > 0.01)
       .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    items.forEach(it => { it.color = it.value >= 0 ? 'var(--rose)' : 'var(--accent)'; });
    Plot.barsH($('uContrib'), items, {labelWidth: 132, height: Math.max(140, items.length * 26 + 40)});

    const move = surge > 50;
    $('uVerdict').innerHTML = surge === 0
      ? '<strong>No surge.</strong> ' + reason.charAt(0).toUpperCase() + reason.slice(1)
        + '. The policy layer discards the model output of ₹' + raw.toFixed(0) + '.'
      : '<strong>₹' + surge.toFixed(0) + ' surge.</strong> A driver 3 km away in Andheri '
        + (move ? '<span style="color:var(--accent);font-weight:600">should reposition</span>: the '
                + 'incentive clears the ₹50 threshold from Task 5.'
                : '<span style="color:var(--fg-3)">will probably stay put</span>, since ₹'
                + surge.toFixed(0) + ' does not cover a 3 km deadhead against the ₹50 threshold.')
        + (surge >= CAP ? ' The cap is binding, so this row would be censored if it entered '
            + 'tomorrow\'s training set.' : '');
  }

  Object.values(sliders).forEach(s => s.el.addEventListener('input', draw));
  document.querySelector('#surgeCard .chips').addEventListener('click', e => {
    const c = e.target.closest('.chip'); if (!c) return;
    c.classList.toggle('on');
    flags[c.dataset.f] = c.classList.contains('on') ? 1 : 0;
    draw();
  });
  UI.onDraw(draw);
})();
