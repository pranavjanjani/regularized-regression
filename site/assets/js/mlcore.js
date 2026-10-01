/* =============================================================================
   mlcore.js — linear algebra + penalized regression, implemented from scratch
   so students can read every estimator the lab uses. No dependencies.

   Estimators
     ols(X,y)                      w = (XᵀX)⁻¹Xᵀy                 [MLE]
     ridge(X,y,alpha)              w = (XᵀX + αI)⁻¹Xᵀy            [MAP, Gaussian prior]
     lasso(X,y,lambda)             coordinate descent + soft-threshold [MAP, Laplace prior]
     elasticNet(X,y,alpha,l1Ratio) coordinate descent, L1+L2
     bayesLinear(X,y,alpha,sigma2) full Gaussian posterior (mean + covariance)

   Convention: X is standardized and y is centred, so no intercept is fitted
   inside the solvers; the intercept is restored by the caller (mean of y).
============================================================================= */
(function (global) {
  'use strict';

  /* ---------------- basic linear algebra ---------------- */
  const T = A => A[0].map((_, j) => A.map(r => r[j]));
  const matmul = (A, B) => {
    const n = A.length, k = B.length, m = B[0].length, C = [];
    for (let i = 0; i < n; i++) {
      const row = new Array(m).fill(0), Ai = A[i];
      for (let t = 0; t < k; t++) {
        const a = Ai[t]; if (a === 0) continue;
        const Bt = B[t];
        for (let j = 0; j < m; j++) row[j] += a * Bt[j];
      }
      C.push(row);
    }
    return C;
  };
  const matvec = (A, v) => A.map(r => r.reduce((s, a, j) => s + a * v[j], 0));
  const eye = (n, s = 1) => Array.from({length: n}, (_, i) =>
    Array.from({length: n}, (_, j) => (i === j ? s : 0)));
  const addM = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));

  /** Gauss-Jordan inverse with partial pivoting. Returns null if singular. */
  function inv(M) {
    const n = M.length;
    const A = M.map((r, i) => r.concat(eye(n)[i]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      if (Math.abs(A[p][c]) < 1e-12) return null;            // singular → MLE undefined
      [A[c], A[p]] = [A[p], A[c]];
      const piv = A[c][c];
      for (let j = 0; j < 2 * n; j++) A[c][j] /= piv;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = A[r][c]; if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j];
      }
    }
    return A.map(r => r.slice(n));
  }

  /** Condition number proxy: ratio of largest/smallest eigenvalue of XᵀX
      via power iteration on M and on its inverse. Used to *show* collinearity. */
  function condition(M) {
    const n = M.length;
    const power = A => {
      let v = new Array(n).fill(1 / Math.sqrt(n)), lam = 0;
      for (let it = 0; it < 220; it++) {
        const w = matvec(A, v);
        const nr = Math.hypot(...w); if (nr < 1e-14) return 0;
        v = w.map(x => x / nr); lam = nr;
      }
      return lam;
    };
    const Mi = inv(M);
    if (!Mi) return Infinity;
    return power(M) * power(Mi);
  }

  /** Symmetric eigen-decomposition by cyclic Jacobi rotations.
      Returns {values (descending), vectors (columns)}. Used for the SVD/shrinkage
      figures and for reading off how ill-conditioned XᵀX really is. */
  function eigSym(Min, sweeps = 60) {
    const n = Min.length;
    const A = Min.map(r => r.slice());
    let V = eye(n);
    for (let s = 0; s < sweeps; s++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] ** 2;
      if (off < 1e-20) break;
      for (let p0 = 0; p0 < n - 1; p0++) for (let q = p0 + 1; q < n; q++) {
        if (Math.abs(A[p0][q]) < 1e-14) continue;
        const theta = (A[q][q] - A[p0][p0]) / (2 * A[p0][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p0], akq = A[k][q];
          A[k][p0] = c * akp - sn * akq; A[k][q] = sn * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p0][k], aqk = A[q][k];
          A[p0][k] = c * apk - sn * aqk; A[q][k] = sn * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p0], vkq = V[k][q];
          V[k][p0] = c * vkp - sn * vkq; V[k][q] = sn * vkp + c * vkq;
        }
      }
    }
    const order = Array.from({length: n}, (_, i) => i).sort((a, b) => A[b][b] - A[a][a]);
    return {
      values: order.map(i => A[i][i]),
      vectors: V.map(row => order.map(i => row[i]))
    };
  }

  /* ---------------- preprocessing ---------------- */
  function standardize(X) {
    const n = X.length, p = X[0].length, mean = [], std = [];
    for (let j = 0; j < p; j++) {
      let m = 0; for (let i = 0; i < n; i++) m += X[i][j]; m /= n;
      let v = 0; for (let i = 0; i < n; i++) v += (X[i][j] - m) ** 2; v = Math.sqrt(v / n) || 1e-8;
      mean.push(m); std.push(v);
    }
    const Z = X.map(r => r.map((v, j) => (v - mean[j]) / std[j]));
    return {Z, mean, std};
  }
  const applyScale = (X, mean, std) => X.map(r => r.map((v, j) => (v - mean[j]) / std[j]));
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const center = y => { const m = mean(y); return {yc: y.map(v => v - m), mean: m}; };

  /* ---------------- estimators ---------------- */
  function ols(X, y) {
    const Xt = T(X), XtX = matmul(Xt, X), Xty = matvec(Xt, y);
    const Inv = inv(XtX);
    if (!Inv) return {w: new Array(X[0].length).fill(NaN), singular: true};
    return {w: matvec(Inv, Xty), singular: false};
  }

  function ridge(X, y, alpha) {
    const p = X[0].length, Xt = T(X);
    const A = addM(matmul(Xt, X), eye(p, alpha));
    const Inv = inv(A);
    if (!Inv) return {w: new Array(p).fill(0), singular: true};
    return {w: matvec(Inv, matvec(Xt, y)), singular: false};
  }

  const softThreshold = (rho, lam) =>
    rho > lam ? rho - lam : (rho < -lam ? rho + lam : 0);

  /**
   * Elastic Net by cyclic coordinate descent, minimising
   *   (1/2n)‖y − Xw‖² + alpha·l1Ratio·‖w‖₁ + (alpha(1−l1Ratio)/2)·‖w‖²
   * i.e. exactly scikit-learn's ElasticNet objective, so notebook and browser agree.
   * l1Ratio = 1 → Lasso; l1Ratio = 0 → Ridge (in scaled form).
   */
  function elasticNet(X, y, alpha, l1Ratio, opts = {}) {
    const maxIter = opts.maxIter || 3000, tol = opts.tol || 1e-7;
    const n = X.length, p = X[0].length;
    const w = (opts.warm && opts.warm.slice()) || new Array(p).fill(0);
    const l1 = alpha * l1Ratio, l2 = alpha * (1 - l1Ratio);
    // Column norms ‖x_j‖²/n, precomputed.
    const nrm = new Array(p).fill(0);
    for (let j = 0; j < p; j++) { let s = 0; for (let i = 0; i < n; i++) s += X[i][j] ** 2; nrm[j] = s / n; }
    const r = y.slice();                                  // residual y − Xw
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) r[i] -= X[i][j] * w[j];
    let iter = 0;
    for (; iter < maxIter; iter++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        const wj = w[j];
        let rho = 0;                                      // (1/n)·x_jᵀ(r + x_j w_j)
        for (let i = 0; i < n; i++) rho += X[i][j] * (r[i] + X[i][j] * wj);
        rho /= n;
        const wNew = softThreshold(rho, l1) / (nrm[j] + l2);
        const d = wNew - wj;
        if (d !== 0) {
          for (let i = 0; i < n; i++) r[i] -= X[i][j] * d;
          w[j] = wNew;
          if (Math.abs(d) > maxDelta) maxDelta = Math.abs(d);
        }
      }
      if (maxDelta < tol) break;
    }
    return {w, iters: iter};
  }

  const lasso = (X, y, lambda, opts) => elasticNet(X, y, lambda, 1.0, opts);

  /**
   * Bayesian linear regression with prior w ~ N(0, τ²I) and noise N(0, σ²).
   * Posterior: N(mu, S) with S = (XᵀX/σ² + I/τ²)⁻¹, mu = S Xᵀy/σ².
   * The posterior MEAN equals the Ridge/MAP solution with α = σ²/τ².
   */
  function bayesLinear(X, y, tau2, sigma2) {
    const p = X[0].length, Xt = T(X);
    const prec = addM(matmul(Xt, X).map(r => r.map(v => v / sigma2)), eye(p, 1 / tau2));
    const S = inv(prec);
    if (!S) return null;
    const mu = matvec(S, matvec(Xt, y).map(v => v / sigma2));
    return {mu, cov: S, sd: S.map((r, i) => Math.sqrt(Math.max(r[i], 0)))};
  }

  /* ---------------- penalized logistic regression ----------------
     Minimises the average log-loss plus the same penalty family as above:

       (1/n) Σ [ −y log σ(z) − (1−y) log(1−σ(z)) ]
         + α·ρ·‖w‖₁ + (α(1−ρ)/2)·‖w‖²        with z = Xw + b

     The intercept b is never penalized. There is no closed form, so this is
     proximal gradient descent with FISTA acceleration: a gradient step on the
     smooth part, then a soft-threshold for the ℓ₁ part. Equivalent to
     sklearn's LogisticRegression at C = 1/(n·α).
  ------------------------------------------------------------------- */
  const sigmoid = z => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));

  function logLoss(y, p, eps = 1e-12) {
    let s = 0;
    for (let i = 0; i < y.length; i++) {
      const q = Math.min(Math.max(p[i], eps), 1 - eps);
      s += -(y[i] * Math.log(q) + (1 - y[i]) * Math.log(1 - q));
    }
    return s / y.length;
  }

  function logistic(X, y, opt = {}) {
    const alpha = opt.alpha == null ? 0 : opt.alpha;
    const rho = opt.l1Ratio == null ? 0 : opt.l1Ratio;
    const maxIter = opt.maxIter || 4000, tol = opt.tol || 1e-9;
    const n = X.length, p = X[0].length;
    const l1 = alpha * rho, l2 = alpha * (1 - rho);

    /* Step size from the Lipschitz constant of the log-loss gradient. The
       Hessian of σ is bounded by 1/4, and the intercept contributes a column
       of ones, hence the +1 on the row norms. */
    let colMax = 0;
    for (let i = 0; i < n; i++) {
      let r = 1;
      for (let j = 0; j < p; j++) r += X[i][j] * X[i][j];
      if (r > colMax) colMax = r;
    }
    const L = 0.25 * colMax + l2 + 1e-9;
    const step = 1 / L;

    let w = (opt.warm && opt.warm.slice()) || new Array(p).fill(0);
    let b = opt.warmB || 0;
    let wy = w.slice(), by = b, tk = 1;          // FISTA momentum terms
    let iters = 0, converged = false;

    const obj = (ww, bb) => {
      const pr = new Array(n);
      for (let i = 0; i < n; i++) {
        let z = bb;
        for (let j = 0; j < p; j++) z += X[i][j] * ww[j];
        pr[i] = sigmoid(z);
      }
      let pen = 0;
      for (let j = 0; j < p; j++) pen += l1 * Math.abs(ww[j]) + l2 / 2 * ww[j] * ww[j];
      return logLoss(y, pr) + pen;
    };

    let prev = obj(w, b);
    for (; iters < maxIter; iters++) {
      /* gradient of the smooth part at the momentum point */
      const g = new Array(p).fill(0);
      let gb = 0;
      for (let i = 0; i < n; i++) {
        let z = by;
        for (let j = 0; j < p; j++) z += X[i][j] * wy[j];
        const d = sigmoid(z) - y[i];
        gb += d;
        for (let j = 0; j < p; j++) g[j] += X[i][j] * d;
      }
      gb /= n;
      for (let j = 0; j < p; j++) g[j] = g[j] / n + l2 * wy[j];

      /* proximal step: gradient, then soft-threshold */
      const wNew = new Array(p);
      for (let j = 0; j < p; j++) wNew[j] = softThreshold(wy[j] - step * g[j], step * l1);
      const bNew = by - step * gb;

      /* FISTA extrapolation */
      const tNext = (1 + Math.sqrt(1 + 4 * tk * tk)) / 2;
      const mom = (tk - 1) / tNext;
      for (let j = 0; j < p; j++) wy[j] = wNew[j] + mom * (wNew[j] - w[j]);
      by = bNew + mom * (bNew - b);
      w = wNew; b = bNew; tk = tNext;

      if (iters % 10 === 9) {
        const cur = obj(w, b);
        if (Math.abs(prev - cur) < tol * Math.max(1, Math.abs(prev))) { converged = true; iters++; break; }
        /* A non-monotone step means the momentum overshot; restart it. */
        if (cur > prev) { wy = w.slice(); by = b; tk = 1; }
        prev = cur;
      }
    }
    return {w, b, iters, converged, objective: obj(w, b)};
  }

  /** Predicted probabilities for a fitted logistic model. */
  const predictProba = (X, w, b) =>
    X.map(r => sigmoid(b + r.reduce((s, v, j) => s + v * w[j], 0)));

  /** Confusion matrix and the usual rates at a given decision threshold. */
  function classMetrics(y, prob, threshold = 0.5) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < y.length; i++) {
      const hat = prob[i] >= threshold ? 1 : 0;
      if (y[i] === 1 && hat === 1) tp++;
      else if (y[i] === 0 && hat === 1) fp++;
      else if (y[i] === 0 && hat === 0) tn++;
      else fn++;
    }
    const prec = tp + fp ? tp / (tp + fp) : NaN;
    const rec = tp + fn ? tp / (tp + fn) : NaN;
    return {tp, fp, tn, fn,
      accuracy: (tp + tn) / y.length,
      precision: prec, recall: rec,
      specificity: tn + fp ? tn / (tn + fp) : NaN,
      f1: (prec && rec) ? 2 * prec * rec / (prec + rec) : NaN};
  }

  /** ROC curve and AUC. AUC is computed by the rank (Mann-Whitney) identity,
      which is exact and avoids trapezoid error on ties. */
  function rocCurve(y, prob) {
    const idx = y.map((_, i) => i).sort((a, b) => prob[b] - prob[a]);
    const P = y.reduce((s, v) => s + (v === 1), 0), N = y.length - P;
    const pts = [[0, 0]];
    let tp = 0, fp = 0;
    idx.forEach(i => {
      if (y[i] === 1) tp++; else fp++;
      pts.push([N ? fp / N : 0, P ? tp / P : 0]);
    });
    /* AUC from average ranks of the positives. */
    const order = y.map((_, i) => i).sort((a, b) => prob[a] - prob[b]);
    const rank = new Array(y.length);
    let i2 = 0;
    while (i2 < order.length) {
      let j = i2;
      while (j + 1 < order.length && prob[order[j + 1]] === prob[order[i2]]) j++;
      const avg = (i2 + j) / 2 + 1;
      for (let k = i2; k <= j; k++) rank[order[k]] = avg;
      i2 = j + 1;
    }
    let sumRankPos = 0;
    for (let i = 0; i < y.length; i++) if (y[i] === 1) sumRankPos += rank[i];
    const auc = (P && N) ? (sumRankPos - P * (P + 1) / 2) / (P * N) : NaN;
    return {points: pts, auc};
  }

  /* ---------------- metrics & model selection ---------------- */
  const predict = (X, w, b = 0) => X.map(r => b + r.reduce((s, v, j) => s + v * w[j], 0));
  const rmse = (y, yh) => Math.sqrt(y.reduce((s, v, i) => s + (v - yh[i]) ** 2, 0) / y.length);
  const mae = (y, yh) => y.reduce((s, v, i) => s + Math.abs(v - yh[i]), 0) / y.length;
  function r2(y, yh) {
    const m = mean(y);
    const ss = y.reduce((s, v, i) => s + (v - yh[i]) ** 2, 0);
    const tt = y.reduce((s, v) => s + (v - m) ** 2, 0);
    return 1 - ss / tt;
  }

  /** k-fold CV RMSE for a fitting function fit(Xtr,ytr) -> w (scaled space). */
  function kFoldCV(X, y, k, fit) {
    const n = X.length, idx = Array.from({length: n}, (_, i) => i);
    const folds = Array.from({length: k}, () => []);
    idx.forEach((v, i) => folds[i % k].push(v));
    let tot = 0;
    for (let f = 0; f < k; f++) {
      const te = new Set(folds[f]);
      const Xtr = [], ytr = [], Xte = [], yte = [];
      for (let i = 0; i < n; i++) (te.has(i) ? (Xte.push(X[i]), yte.push(y[i])) : (Xtr.push(X[i]), ytr.push(y[i])));
      const {yc, mean: ym} = center(ytr);
      const w = fit(Xtr, yc);
      tot += rmse(yte, predict(Xte, w, ym)) ** 2;
    }
    return Math.sqrt(tot / k);
  }

  /* ---------------- sampling helpers (for the Bayesian demo) ---------------- */
  let spare = null;
  function randn() {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u, v, s2;
    do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; s2 = u * u + v * v; }
    while (s2 >= 1 || s2 === 0);
    const f = Math.sqrt(-2 * Math.log(s2) / s2);
    spare = v * f; return u * f;
  }
  const gaussPdf = (x, mu, sd) =>
    Math.exp(-((x - mu) ** 2) / (2 * sd * sd)) / (sd * Math.sqrt(2 * Math.PI));
  const laplacePdf = (x, mu, b) => Math.exp(-Math.abs(x - mu) / b) / (2 * b);

  function corr(a, b) {
    const ma = mean(a), mb = mean(b);
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < a.length; i++) {
      const x = a[i] - ma, y = b[i] - mb;
      num += x * y; da += x * x; db += y * y;
    }
    return num / (Math.sqrt(da * db) || 1e-12);
  }

  global.ML = {
    T, matmul, matvec, inv, eye, addM, condition, eigSym,
    standardize, applyScale, center, mean,
    ols, ridge, lasso, elasticNet, softThreshold, bayesLinear,
    predict, rmse, mae, r2, kFoldCV, randn, gaussPdf, laplacePdf, corr,
    sigmoid, logistic, logLoss, predictProba, classMetrics, rocCurve
  };
})(window);
