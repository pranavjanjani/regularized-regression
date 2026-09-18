/* =============================================================================
   site.js — shared chrome: navigation, theme, KaTeX, table of contents,
   scroll-spy and a few formatting helpers. Loaded by every page.
============================================================================= */
(function () {
  'use strict';

  const NAV = [
    ['index.html',        'Home'],
    ['syllabus.html',     'Syllabus'],
    ['theory.html',       'Theory'],
    ['applications.html', 'Applications'],
    ['uber.html',         'Case Study'],
    ['algorithms.html',   'Algorithms'],
    ['lab.html',          'Lab'],
    ['dataset.html',      'Data'],
    ['assignments.html',  'Assignments'],
    ['notebooks.html',    'Notebooks'],
    ['resources.html',    'Resources'],
  ];

  const page = location.pathname.split('/').pop() || 'index.html';

  /* ---------- theme ---------- */
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  };
  const saved = store.get('rr-theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);

  function toggleTheme() {
    // Light is the default for this site, so an unset theme toggles to dark.
    const cur = document.documentElement.getAttribute('data-theme') || 'light';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    store.set('rr-theme', next);
    document.dispatchEvent(new CustomEvent('themechange', {detail: next}));
  }

  /* ---------- chrome ---------- */
  function chrome() {
    const header = document.createElement('header');
    header.className = 'nav';
    header.innerHTML = `
      <div class="nav-in">
        <a class="brand" href="index.html"><span class="mark">λ</span>
          <span>Regularized&nbsp;Regression</span></a>
        <button class="icon-btn nav-toggle" id="navToggle" aria-label="Menu">☰</button>
        <nav class="nav-links" id="navLinks">
          ${NAV.map(([h, t]) =>
            `<a href="${h}"${h === page ? ' class="active" aria-current="page"' : ''}>${t}</a>`).join('')}
          <button class="icon-btn" id="themeBtn" title="Toggle light / dark" aria-label="Toggle theme">◐</button>
        </nav>
      </div>`;
    document.body.prepend(header);

    const foot = document.createElement('footer');
    foot.className = 'site';
    foot.innerHTML = `
      <div class="wrap foot-grid">
        <div>
          <h5>Course</h5>
          <ul>
            <li><a href="syllabus.html">Syllabus &amp; schedule</a></li>
            <li><a href="theory.html">Lecture notes</a></li>
            <li><a href="algorithms.html">Algorithm walkthroughs</a></li>
          </ul>
        </div>
        <div>
          <h5>Practice</h5>
          <ul>
            <li><a href="lab.html">Interactive lab</a></li>
            <li><a href="dataset.html">Dataset explorer</a></li>
            <li><a href="notebooks.html">Jupyter notebooks</a></li>
          </ul>
        </div>
        <div>
          <h5>Assessment</h5>
          <ul>
            <li><a href="assignments.html">Assignment brief</a></li>
            <li><a href="assignments/Regression_Assignment.pdf">PDF hand-out</a></li>
            <li><a href="assignments/Regression_Assignment.docx">Word hand-out</a></li>
          </ul>
        </div>
        <div>
          <h5>About</h5>
          <p class="small" style="margin:0">Graduate module on maximum likelihood,
          MAP estimation and regularized regression, taught through a single
          production case study: Uber surge pricing in Mumbai.</p>
        </div>
      </div>
      <div class="wrap" style="margin-top:24px;padding-top:16px;border-top:1px solid var(--line);
           display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap">
        <span>Course materials for classroom use. Datasets are synthetic.</span>
        <span>Built as a static site — no server required.</span>
      </div>`;
    document.body.appendChild(foot);

    document.getElementById('themeBtn').addEventListener('click', toggleTheme);
    const tg = document.getElementById('navToggle'), links = document.getElementById('navLinks');
    tg.addEventListener('click', () => links.classList.toggle('open'));
    links.addEventListener('click', e => { if (e.target.tagName === 'A') links.classList.remove('open'); });
  }

  /* ---------- table of contents + scroll-spy ---------- */
  function buildTOC() {
    const host = document.querySelector('[data-toc]');
    if (!host) return;
    const scope = document.querySelector('main');
    const hs = [...scope.querySelectorAll('h2[id], h3[id]')];
    if (!hs.length) return;
    host.innerHTML = '<div class="tt">On this page</div>' + hs.map(h =>
      `<a href="#${h.id}" class="${h.tagName === 'H3' ? 'sub' : ''}">${h.dataset.short || h.textContent}</a>`
    ).join('');
    const anchors = [...host.querySelectorAll('a')];
    const spy = () => {
      let cur = hs[0];
      const y = window.scrollY + 110;
      hs.forEach(h => { if (h.offsetTop <= y) cur = h; });
      anchors.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + cur.id));
    };
    window.addEventListener('scroll', spy, {passive: true});
    spy();
  }

  /* ---------- math ---------- */
  function renderMath() {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(document.body, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '\\[', right: '\\]', display: true},
        {left: '$', right: '$', display: false},
        {left: '\\(', right: '\\)', display: false}
      ],
      throwOnError: false,
      macros: {'\\argmin': '\\operatorname*{arg\\,min}', '\\argmax': '\\operatorname*{arg\\,max}'}
    });
  }

  /* ---------- helpers exposed to page scripts ---------- */
  window.UI = {
    fmt(v, d = 2) { return (v == null || !isFinite(v)) ? '—' : Number(v).toFixed(d); },
    downloadText(name, text, mime = 'text/plain') {
      const a = document.createElement('a');
      const blob = new Blob([text], {type: mime});
      a.href = URL.createObjectURL(blob); a.download = name;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    },
    /** Re-run every registered draw function (used on theme change / resize). */
    redraws: [],
    onDraw(fn) { this.redraws.push(fn); fn(); },
    fireDraw() { this.redraws.forEach(f => { try { f(); } catch (e) { console.error(e); } }); },
    toggleTheme
  };

  document.addEventListener('DOMContentLoaded', () => { chrome(); buildTOC(); renderMath(); });
  window.addEventListener('load', renderMath);
  document.addEventListener('themechange', () => setTimeout(() => window.UI.fireDraw(), 30));
  let rt;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => window.UI.fireDraw(), 180); });
})();
