"""Notebook 05 - logistic regression (Task 6). Kept in its own module because
its code cells contain triple-quoted docstrings, which do not nest inside the
other builders' string literals."""
from nbbuild import code, md

TODO = '# TODO: your code here\nraise NotImplementedError'


def cells(preamble):
    C = preamble(
        'Notebook 05 · Logistic Regression',
        'Covers **Task 6**: a Bernoulli likelihood in place of a Gaussian one, the same '
        'ℓ₁ and ℓ₂ priors, ROC and the threshold as a policy choice, and a measurement of '
        'what dichotomising the target cost.')

    C += [md(r"""
## 6a — The label, and why least squares will not do

There is no binary column in the data, so we build one:
`high_surge = surge_additive_inr > 100`. That is a real operational trigger, and also a
deliberate loss of information. Section 6e measures the cost.
""")]

    C += [code(r"""
CUT = 100.0
df['is_bad_weather'] = ((df.is_rain == 1) & (df.traffic_speed_kmph < 25)).astype(int)
FEATS = BASE8 + ['is_bad_weather']

Xtr, Xte, ytr_rupees, yte_rupees = split(df, FEATS)
Ztr, Zte, MU, SD = standardize(Xtr, Xte)
ytr = (ytr_rupees > CUT).astype(int)
yte = (yte_rupees > CUT).astype(int)

print(f'positives: {ytr.sum()}/{len(ytr)} train, {yte.sum()}/{len(yte)} test')
print(f'base rate (always "no alert"): {max(ytr.mean(), 1-ytr.mean()):.1%} train, '
      f'{max(yte.mean(), 1-yte.mean()):.1%} test')
print()
print('Every accuracy below must be read against that base rate.')
""", r"""
CUT = 100.0
# TODO: build the binary label, then report positives in train/test and the base rate.
""" + TODO)]

    C += [code(r"""
# Least squares on a 0/1 label will leave [0, 1].
w_bad = np.linalg.solve(Ztr.T @ Ztr, Ztr.T @ (ytr - ytr.mean()))
pred_bad = Ztr @ w_bad + ytr.mean()
outside = int(((pred_bad < 0) | (pred_bad > 1)).sum())
print(f'OLS on the 0/1 label predicts outside [0,1] for {outside} of {len(ytr)} trips')
print(f'  range: {pred_bad.min():.2f} to {pred_bad.max():.2f}')

fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.hist(pred_bad, bins=30, color=C['ols'], alpha=.85)
ax.axvspan(min(-0.5, pred_bad.min()-.1), 0, color=C['rose'], alpha=.12)
ax.axvspan(1, max(1.5, pred_bad.max()+.1), color=C['rose'], alpha=.12)
ax.axvline(0, color=C['rose'], lw=1.4); ax.axvline(1, color=C['rose'], lw=1.4)
ax.set(xlabel='fitted value from least squares on the 0/1 label', ylabel='trips',
       title='Shaded regions are impossible probabilities')
plt.tight_layout(); plt.show()
""", r"""
# TODO: fit OLS to the 0/1 label and count how many fitted values fall outside [0, 1].
""" + TODO)]

    C += [md(r"""
> **Write-up prompt (6a step 3).** The Bernoulli log-likelihood is
> $\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$ with $p_i = \sigma(x_i^\top w + b)$.
> Maximising it is minimising the average of its negative, which is the log-loss. Same
> manoeuvre as Task 1 with a different density: the loss comes from the noise model.

## 6b — Implement penalized logistic regression

With the intercept **unpenalized**:

$$\frac{1}{n}\sum_i \big[-y_i\log p_i-(1-y_i)\log(1-p_i)\big]
 + \alpha\rho\lVert w\rVert_1 + \frac{\alpha(1-\rho)}{2}\lVert w\rVert_2^2$$

There is no closed form. This is proximal gradient descent with FISTA acceleration: a
gradient step on the smooth part, then the same soft-threshold as Task 3.
""")]

    C += [code(r"""
def sigmoid(z):
    out = np.empty_like(np.asarray(z, dtype=float))
    z = np.asarray(z, dtype=float)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def soft_threshold(rho, gamma):
    return np.sign(rho) * np.maximum(np.abs(rho) - gamma, 0.0)


def logistic_fit(Z, y, alpha=0.0, l1_ratio=0.0, max_iter=4000, tol=1e-10):
    # Proximal gradient + FISTA. Returns (w, b, iterations).
    n, p = Z.shape
    l1, l2 = alpha * l1_ratio, alpha * (1 - l1_ratio)
    # step from the Lipschitz constant: |sigma''| <= 1/4, and +1 for the intercept
    L = 0.25 * np.max((Z ** 2).sum(1) + 1) + l2 + 1e-9
    step = 1 / L

    w = np.zeros(p); b = 0.0
    wy, by, tk = w.copy(), b, 1.0

    def obj(ww, bb):
        pr = sigmoid(Z @ ww + bb)
        return log_loss(y, pr) + l1 * np.abs(ww).sum() + l2 / 2 * (ww ** 2).sum()

    prev = obj(w, b)
    for it in range(max_iter):
        d = sigmoid(Z @ wy + by) - y
        g = Z.T @ d / n + l2 * wy
        gb = d.mean()
        w_new = soft_threshold(wy - step * g, step * l1)
        b_new = by - step * gb
        t_next = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        mom = (tk - 1) / t_next
        wy = w_new + mom * (w_new - w)
        by = b_new + mom * (b_new - b)
        w, b, tk = w_new, b_new, t_next
        if it % 10 == 9:
            cur = obj(w, b)
            if abs(prev - cur) < tol * max(1.0, abs(prev)):
                return w, b, it + 1
            if cur > prev:                 # momentum overshot: restart it
                wy, by, tk = w.copy(), b, 1.0
            prev = cur
    return w, b, max_iter


w_log, b_log, iters = logistic_fit(Ztr, ytr, alpha=0.01, l1_ratio=0.0)
print(f'converged in {iters} iterations,  intercept {b_log:+.4f}')
print(pd.Series({SHORT[f]: round(float(v), 3) for f, v in zip(FEATS, w_log)},
                name='log-odds per s.d.').to_string())
""", r"""
def sigmoid(z):
    pass

def log_loss(y, p):
    pass

def logistic_fit(Z, y, alpha=0.0, l1_ratio=0.0, max_iter=4000, tol=1e-10):
    # Minimise mean log-loss + alpha*rho*|w|_1 + alpha*(1-rho)/2*|w|^2.
    #
    # Hints:
    #   smooth gradient : Z.T @ (sigmoid(Zw+b) - y)/n + alpha*(1-rho)*w
    #   intercept       : mean(sigmoid(Zw+b) - y)   -- and do NOT penalize b
    #   step size       : 1/L with L = 0.25*max_i(||z_i||^2 + 1) + alpha*(1-rho)
    #   after the step  : soft-threshold w by step*alpha*rho
""" + '    ' + TODO.replace('\n', '\n    '))]

    C += [md(r"""
### Verify against scikit-learn

With this objective `LogisticRegression` matches at $C = 1/(n\alpha)$.

> **Read this before you spend an hour debugging.** `solver='liblinear'` **penalizes the
> intercept** and our objective does not, so it disagrees by far more than solver
> tolerance and no tuning will close the gap. Use `lbfgs` for ℓ₂ or `saga` for ℓ₁ and
> elastic net. Setting `intercept_scaling` very large makes liblinear agree, which is how
> you confirm that is the cause.
""")]

    C += [code(r"""
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression

n = len(ytr)
print(f"{'alpha':>7} {'rho':>5} {'max|w diff|':>13} {'|b diff|':>10}  solver")
for alpha, rho in [(0.01, 0.0), (0.1, 0.0), (0.02, 1.0), (0.05, 0.5)]:
    mine_w, mine_b, _ = logistic_fit(Ztr, ytr, alpha, rho, max_iter=80000, tol=1e-14)
    Cpar = 1.0 / (n * alpha)
    if rho == 0.0:
        sk = LogisticRegression(C=Cpar, solver='lbfgs', max_iter=50000, tol=1e-12)
        name = 'lbfgs'
    else:
        sk = LogisticRegression(C=Cpar, penalty='elasticnet', l1_ratio=rho,
                                solver='saga', max_iter=500000, tol=1e-12)
        name = 'saga'
    sk.fit(Ztr, ytr)
    dw = float(np.max(np.abs(mine_w - sk.coef_[0])))
    db = float(abs(mine_b - sk.intercept_[0]))
    print(f'{alpha:>7} {rho:>5} {dw:>13.2e} {db:>10.2e}  {name}')

lib = LogisticRegression(C=1/(n*0.02), penalty='l1', solver='liblinear',
                         max_iter=50000, tol=1e-12).fit(Ztr, ytr)
lib2 = LogisticRegression(C=1/(n*0.02), penalty='l1', solver='liblinear',
                          intercept_scaling=1e4, max_iter=50000, tol=1e-12).fit(Ztr, ytr)
mine_w, mine_b, _ = logistic_fit(Ztr, ytr, 0.02, 1.0, max_iter=80000, tol=1e-14)
print()
print(f'liblinear default      b = {lib.intercept_[0]:+.4f}   <- intercept penalized')
print(f'liblinear scaling=1e4  b = {lib2.intercept_[0]:+.4f}')
print(f'ours                   b = {mine_b:+.4f}   <- matches the rescaled version')
""")]

    C += [md('## 6c — Coefficient paths and the elimination order')]

    C += [code(r"""
alphas = np.logspace(-3, 0.6, 40)
paths = {rho: np.array([logistic_fit(Ztr, ytr, a, rho)[0] for a in alphas])
         for rho in (0.0, 1.0)}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
titles = ['$\\ell_2$ — shrinks, never reaches zero', '$\\ell_1$ — reaches zero exactly']
for ax, rho, title in zip(axes, (0.0, 1.0), titles):
    for j, f in enumerate(FEATS):
        ax.plot(alphas, paths[rho][:, j], lw=1.8, label=SHORT[f])
    ax.set_xscale('log'); ax.axhline(0, color='#c9cbc3', lw=1)
    ax.set(xlabel=r'$\alpha$ (log scale)', title=title)
axes[0].set_ylabel('coefficient (log-odds per s.d.)')
axes[1].legend(ncol=2, fontsize=8)
plt.tight_layout(); plt.show()

order, seen = [], set()
for k, a in enumerate(alphas):
    for j in np.where(np.abs(paths[1.0][k]) < 1e-8)[0]:
        if j not in seen:
            seen.add(j); order.append((SHORT[FEATS[j]], round(float(a), 4)))
print('l1 elimination order (feature, alpha at which it died):')
for f, a in order:
    print(f'  {f:<16} {a}')
""")]

    C += [md(r"""
> **Write-up prompt (6c step 2).** Compare this with your Task 3c prediction. On this data
> the redundant weather features tend to go *before* `is_weekend`, even though
> `is_weekend` has no real effect at all — because `traffic_speed` already carries the
> weather information and ℓ₁ will not pay for it twice.
>
> So an ℓ₁ ordering ranks **marginal usefulness given the other features**, not true
> importance. That is the selection-instability caveat from Theory §5.3 in a new guise,
> and a good reason not to read a Lasso's survivors as a causal story.
""")]

    C += [code(r"""
def cv_log_loss(Z, y, k, fit):
    n = len(y); idx = np.arange(n); tot = cnt = 0.0
    for f in range(k):
        te = np.zeros(n, bool); te[idx[f::k]] = True
        w, b, _ = fit(Z[~te], y[~te])
        p = sigmoid(Z[te] @ w + b)
        tot += log_loss(y[te], p) * te.sum(); cnt += te.sum()
    return tot / cnt


def auc(y, score):
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), float); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j+1]] == score[order[i]]:
            j += 1
        ranks[order[i:j+1]] = (i + j) / 2 + 1
        i = j + 1
    P, N = int(y.sum()), int((1 - y).sum())
    return float((ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N))


grid = np.logspace(-3, 0.3, 22)
rows = []
for name, rho in [('no penalty', None), ('l2', 0.0), ('l1', 1.0), ('elastic', 0.5)]:
    if rho is None:
        alpha, rr = 1e-6, 0.0
    else:
        rr = rho
        scores = [cv_log_loss(Ztr, ytr, 5, lambda Z, t, a=a: logistic_fit(Z, t, a, rr))
                  for a in grid]
        alpha = float(grid[int(np.argmin(scores))])
    w, b, _ = logistic_fit(Ztr, ytr, alpha, rr)
    p = sigmoid(Zte @ w + b)
    rows.append({'model': name, 'alpha': None if rho is None else round(alpha, 5),
                 'test_logloss': round(log_loss(yte, p), 4),
                 'test_auc': round(auc(yte, p), 4),
                 'test_acc': round(float(((p >= .5).astype(int) == yte).mean()), 4),
                 'non_zero': int((np.abs(w) > 1e-8).sum()),
                 'dropped': ', '.join(SHORT[f] for f, v in zip(FEATS, w)
                                      if abs(v) < 1e-8) or '-'})
cls_table = pd.DataFrame(rows)
print(cls_table.to_string(index=False))
print()
print(f'base rate on the test split: {max(yte.mean(), 1-yte.mean()):.1%}')
""", r"""
# TODO: cross-validate on LOG-LOSS (not accuracy) for no-penalty / l2 / l1 / elastic,
#       then report held-out log-loss, AUC, accuracy and non-zero count for each.
""" + TODO)]

    C += [md('## 6d — ROC, and choosing a threshold')]

    C += [code(r"""
best = cls_table.loc[cls_table.test_logloss.idxmin()]
rho_best = {'no penalty': 0.0, 'l2': 0.0, 'l1': 1.0, 'elastic': 0.5}[best.model]
w_b, b_b, _ = logistic_fit(Ztr, ytr, best.alpha if best.alpha else 1e-6, rho_best)
p_te = sigmoid(Zte @ w_b + b_b)
print(f'best by CV log-loss: {best.model} (alpha={best.alpha})')


def confusion(y, p, t):
    hat = (p >= t).astype(int)
    return (int(((hat == 1) & (y == 1)).sum()), int(((hat == 1) & (y == 0)).sum()),
            int(((hat == 0) & (y == 0)).sum()), int(((hat == 0) & (y == 1)).sum()))


MISS_COST, FALSE_COST = 4, 1     # dispatch: a missed alert costs 4x a false alarm
print()
print(f"{'thresh':>7}{'TP':>5}{'FP':>5}{'TN':>5}{'FN':>5}{'recall':>9}{'precision':>11}{'cost':>8}")
best_t, best_cost = None, np.inf
for t in np.arange(0.05, 0.96, 0.05):
    tp, fp, tn, fn = confusion(yte, p_te, t)
    cost = MISS_COST * fn + FALSE_COST * fp
    if cost < best_cost:
        best_cost, best_t = cost, t
    rec = tp / (tp + fn) if tp + fn else float('nan')
    prec = tp / (tp + fp) if tp + fp else float('nan')
    print(f'{t:>7.2f}{tp:>5}{fp:>5}{tn:>5}{fn:>5}{rec:>9.3f}{prec:>11.3f}{cost:>8}')
print()
print(f'minimum expected cost at threshold {best_t:.2f}  (cost {best_cost})')
print('Note it is BELOW 0.5: when a miss costs 4x a false alarm, alert more readily.')

fpr, tpr = [], []
for t in np.r_[1.01, np.sort(p_te)[::-1], -0.01]:
    tp, fp, tn, fn = confusion(yte, p_te, t)
    tpr.append(tp / (tp + fn) if tp + fn else 0.0)
    fpr.append(fp / (fp + tn) if fp + tn else 0.0)

fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.plot(fpr, tpr, lw=2.6, color=C['accent'], label=f'ROC (AUC = {auc(yte, p_te):.3f})')
ax.plot([0, 1], [0, 1], '--', color=C['ols'], label='chance')
tp, fp, tn, fn = confusion(yte, p_te, best_t)
ax.plot(fp / (fp + tn), tp / (tp + fn), 'o', ms=10, color=C['rose'],
        label=f'cost-optimal t = {best_t:.2f}')
ax.set(xlabel='false positive rate', ylabel='true positive rate',
       title='Every point on the curve is a threshold you could choose')
ax.legend(loc='lower right')
plt.tight_layout(); plt.show()
""")]

    C += [md(r"""
> **Write-up prompt (6d step 3).** The cost-minimising threshold is below 0.5, because a
> missed alert is four times as expensive as a false alarm. Note what did **not** change
> when you moved it: not a single coefficient. The threshold is a policy parameter and
> belongs beside the ₹150 cap from Task 5, not inside the model.

## 6e — Was the label worth making?
""")]

    C += [code(r"""
def ridge_norm(Z, y, a):
    n_, p_ = Z.shape
    return np.linalg.solve(Z.T @ Z + n_ * a * np.eye(p_), Z.T @ y)


def kfold_rmse(Z, y, k, fit):
    # k-fold CV on the CONTINUOUS target; re-centre y inside each fold.
    n_ = len(y); idx = np.arange(n_); tot = 0.0
    for f in range(k):
        te = np.zeros(n_, bool); te[idx[f::k]] = True
        ybar = y[~te].mean()
        w = fit(Z[~te], y[~te] - ybar)
        tot += float(np.sum((y[te] - (Z[te] @ w + ybar)) ** 2))
    return float(np.sqrt(tot / n_))


grid_lin = np.logspace(-3, 1.7, 24)
cv = [kfold_rmse(Ztr, ytr_rupees, 5, lambda Z, t, a=a: ridge_norm(Z, t, a))
      for a in grid_lin]
a_lin = float(grid_lin[int(np.argmin(cv))])
w_lin = ridge_norm(Ztr, ytr_rupees - ytr_rupees.mean(), a_lin)
yhat_rupees = Zte @ w_lin + ytr_rupees.mean()
hard = (yhat_rupees > CUT).astype(int)

cls_acc = float(((p_te >= 0.5).astype(int) == yte).mean())
cls_auc = auc(yte, p_te)
lin_acc = float((hard == yte).mean())
lin_auc = auc(yte, yhat_rupees)

print(f'{"":<36}{"accuracy":>10}{"AUC":>8}')
print(f'{"purpose-built classifier":<36}{cls_acc:>10.3f}{cls_auc:>8.3f}')
print(f'{"Ridge in rupees, then cut at " + str(int(CUT)):<36}{lin_acc:>10.3f}{lin_auc:>8.3f}')
print()
print(f'base rate: {max(yte.mean(), 1-yte.mean()):.3f}')
print()
if lin_auc >= cls_auc - 0.005:
    print('The regression matches or beats the classifier, and it also tells you HOW FAR')
    print('above the cut each cell sits, which the classifier discarded. Dichotomising the')
    print('target cost resolution and bought nothing.')
else:
    print('The classifier wins here. Say what it exploits that the regression does not.')
""", r"""
# TODO: take your Ridge model from Task 2, predict rupees on the held-out trips,
#       threshold the prediction at CUT, and compare accuracy and AUC with your best
#       classifier. Then conclude.
""" + TODO)]

    C += [code(r"""
fig, ax = plt.subplots(figsize=(9, 4.8))
ok = yte == 1
ax.scatter(yhat_rupees[~ok], p_te[~ok], s=38, color=C['ridge'], alpha=.75, label='no alert')
ax.scatter(yhat_rupees[ok], p_te[ok], s=38, color=C['rose'], alpha=.75, label='alert')
ax.axvline(CUT, color=C['ols'], ls='--', lw=1.4)
ax.axhline(0.5, color=C['ols'], ls=':', lw=1.4)
ax.set(xlabel='Ridge prediction (Rs)', ylabel='classifier P(alert)',
       title='Two routes to the same yes/no — opposite corners are disagreements')
ax.legend()
plt.tight_layout(); plt.show()

dis = int(((yhat_rupees > CUT) != (p_te >= 0.5)).sum())
print(f'the two approaches disagree on {dis} of {len(yte)} held-out trips')
""")]

    C += [md(r"""
---

### Checklist

- [ ] Base rate quoted beside every accuracy
- [ ] OLS-on-0/1 shown leaving [0, 1], with a reason
- [ ] `logistic_fit` written from scratch, matching sklearn via **lbfgs / saga**
- [ ] The liblinear intercept trap identified
- [ ] Coefficient paths for ℓ₂ and ℓ₁; elimination order vs your Task 3c guess
- [ ] Model selection on **log-loss**, not accuracy
- [ ] ROC, AUC, and a threshold justified by the 4:1 cost ratio (**Deliverable 4**)
- [ ] 6e answered: did dichotomising the target cost anything?

Nothing in Task 6 needed a new idea about priors — only a new likelihood.
""")]
    return C
