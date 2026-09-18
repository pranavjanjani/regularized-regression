/* home.js — the "whole course in one slider" teaser on the landing page.

   The slider is the SHRINKAGE BUDGET s = ‖w‖₁/‖w_OLS‖₁, not a raw α. Both
   estimators are placed at the same total coefficient mass, so the only thing
   left to see is how each one chooses to spend it. */
(function () {
  'use strict';
  const d = Prep.build(Prep.BASE8);
  const sEl = document.getElementById('teaseLam');
  const sV  = document.getElementById('teaseLamV');
  const note = document.getElementById('teaseNote');
  const COLR = 'var(--c-ridge)', COLL = 'var(--c-lasso)';
  const wOls = Prep.fit(d, 'ols');
  const l1 = w => w.reduce((a, v) => a + Math.abs(v), 0);

  function draw() {
    const s = +sEl.value;
    sV.textContent = (s * 100).toFixed(0) + '%';

    const rr = Prep.fitAtBudget(d, 'ridge', s);
    const ll = Prep.fitAtBudget(d, 'lasso', s);
    const wR = rr.w || rr, wL = ll.w || ll;
    const sR = Prep.score(d, wR), sL = Prep.score(d, wL);
    const mx = Math.max(...wOls.map(Math.abs));

    const dom = [-mx * 1.12, mx * 1.12];
    const items = (w, c) => d.labels.map((l, i) => ({label: l, value: w[i], color: c}));
    Plot.barsH(document.getElementById('teaseRidge'), items(wR, COLR),
      {labelWidth: 108, height: 250, xDomain: dom});
    Plot.barsH(document.getElementById('teaseLasso'), items(wL, COLL),
      {labelWidth: 108, height: 250, xDomain: dom});

    document.getElementById('teaseRidgeK').textContent = sR.nonzero;
    document.getElementById('teaseLassoK').textContent = sL.nonzero;
    document.getElementById('teaseRidgeR').textContent = '₹' + sR.testRMSE.toFixed(1);
    document.getElementById('teaseLassoR').textContent = '₹' + sL.testRMSE.toFixed(1);

    const dropped = d.labels.filter((x, i) => Math.abs(wL[i]) < 1e-8);
    note.innerHTML = s > 0.97
      ? 'At full budget both estimators <em>are</em> ordinary least squares. Start pulling the '
        + 'budget down.'
      : dropped.length === 0
        ? 'Both models have shrunk by the same total amount, and neither has deleted anything yet.'
        : dropped.length >= 8
          ? 'Lasso has deleted <strong>every</strong> feature — the model is now just the average '
            + 'fare. Ridge still holds eight non-zero coefficients, all minuscule. Same budget, '
            + 'completely different answer.'
          : 'Same budget, different spending. Lasso has set <strong>'
            + dropped.map(x => '<code>' + x + '</code>').join(', ')
            + '</strong> to exactly zero and put the mass into the survivors. Ridge has zeroed '
            + 'nothing — it just scales everything down together. Note which feature Lasso kills '
            + 'first, and check it against the <a href="dataset.html#corr">correlation matrix</a>.';
  }

  sEl.addEventListener('input', draw);
  document.getElementById('teaseReset').addEventListener('click', () => { sEl.value = 1; draw(); });
  UI.onDraw(draw);
})();
