"""Build the eight course notebooks (four starters, four solutions)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nbformat as nbf
from nbbuild import CONVENTIONS, HEADER, SETUP, build, code, md   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f'{ROOT}/notebooks'
COURSE = 'Regularized Regression — From Likelihood to Lasso'

TODO = '# TODO: your code here\nraise NotImplementedError'


def preamble(title, blurb):
    return [md(HEADER.format(title=title, course=COURSE, blurb=blurb)),
            code(SETUP), code(CONVENTIONS)]


# =========================================================================== #
# 01 · MLE and Ridge
# =========================================================================== #
def nb01():
    C = preamble(
        'Notebook 01 · Least Squares and Ridge',
        'Covers **Task 1** (maximum likelihood by hand, and two ways for it to fail) and '
        '**Task 2** (Ridge as MAP with a Gaussian prior).')

    C += [md('''
## Task 1a — An ill-conditioned pair

`is_rain` and `traffic_speed_kmph` on the **first six trips**. The two columns are
strongly correlated, and six rows leave barely enough independent variation to tell them
apart.
''')]

    C += [code('''
A_FEATS = ['is_rain', 'traffic_speed_kmph']
n6 = 6

X = df[A_FEATS].to_numpy(float)[:n6]
y = df['surge_additive_inr'].to_numpy(float)[:n6]

# Standardize using ONLY these six rows: the estimator sees nothing else.
mu, sd = X.mean(0), X.std(0)
Z = (X - mu) / sd
yc = y - y.mean()

print('standardized design matrix Z:')
print(Z)
print('\\ncentred target:', np.round(yc, 2))
print('\\nsample correlation r =', round(float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]), 4))
''')]

    C += [md('''
### Step 1 — build $X^\\top X$ and $X^\\top y$ by hand

Each entry of $X^\\top X$ is a dot product of two columns. Since the columns are
standardized, the diagonal entries are exactly $n$ and the off-diagonal is $n\\,r$, which
is why the correlation coefficient is the number that matters here.
''')]

    C += [code('''
XtX = Z.T @ Z
Xty = Z.T @ yc

print('XtX =\\n', XtX)
print('\\ncheck by hand:')
print('  (1,1) = sum(z1^2)   =', round(float((Z[:, 0] ** 2).sum()), 4))
print('  (1,2) = sum(z1*z2)  =', round(float((Z[:, 0] * Z[:, 1]).sum()), 4))
print('  (2,2) = sum(z2^2)   =', round(float((Z[:, 1] ** 2).sum()), 4))
print('\\nXty =', np.round(Xty, 4))
''', '''
XtX = ...   # TODO
Xty = ...   # TODO
''' + TODO)]

    C += [md('''
### Step 2 — determinant and the explicit inverse

For a $2\\times2$ matrix,
$\\begin{pmatrix}a&b\\\\b&d\\end{pmatrix}^{-1} = \\frac{1}{ad-b^2}\\begin{pmatrix}d&-b\\\\-b&a\\end{pmatrix}$.

The inverse divides by the determinant, so everything the estimator produces —
coefficients and their standard errors — is scaled by $1/\\det$.
''')]

    C += [code('''
a, b, d = XtX[0, 0], XtX[0, 1], XtX[1, 1]
det = a * d - b * b
XtX_inv = np.array([[d, -b], [-b, a]]) / det

print(f'det(XtX)        = {det:.4f}')
print(f'1/det           = {1/det:.4f}   <- every coefficient is scaled by this')
print(f'condition number= {np.linalg.cond(XtX):.1f}')
print('\\ninverse =\\n', XtX_inv)
print('\\nagrees with np.linalg.inv:', np.allclose(XtX_inv, np.linalg.inv(XtX)))

w6 = XtX_inv @ Xty
print(f'\\nw_MLE = ({w6[0]:.3f}, {w6[1]:.3f})   [Rs per standard deviation]')
''', '''
a, b, d = ...
det = ...
XtX_inv = ...
w6 = ...
''' + TODO)]

    C += [code('''
# Prediction for trip 2 (19:00, peak and rain, 6 drivers)
pred2 = float(Z[1] @ w6 + y.mean())
print(f'predicted surge for trip 2: Rs {pred2:.2f}')
print(f'actually charged:           Rs {y[1]:.0f}')
print('\\nClose — but the model saw six points, two of them at the price cap.')
''')]

    C += [md('''
### Step 3 — add trips one at a time

Refit on the first $n = 6, 7, \\dots, 15$ trips and watch the coefficients. In a
well-posed problem, adding one observation out of six would nudge the answer.
''')]

    C += [code('''
rows, prev = [], None
for n in range(6, 16):
    Xn = df[A_FEATS].to_numpy(float)[:n]
    yn = df['surge_additive_inr'].to_numpy(float)[:n]
    Zn = (Xn - Xn.mean(0)) / Xn.std(0)
    An = Zn.T @ Zn
    wn = np.linalg.solve(An, Zn.T @ (yn - yn.mean()))
    jump = None if prev is None else float(np.max(np.abs(wn - prev)))
    rows.append({'n': n,
                 'r': round(float(np.corrcoef(Xn[:, 0], Xn[:, 1])[0, 1]), 3),
                 'det': round(float(np.linalg.det(An)), 2),
                 'w_is_rain': round(float(wn[0]), 2),
                 'w_traffic': round(float(wn[1]), 2),
                 'max_move': None if jump is None else round(jump, 2)})
    prev = wn

inst = pd.DataFrame(rows)
print(inst.to_string(index=False))
print(f"\\nLargest single-observation move: Rs {inst.max_move.max():.2f} per s.d.")
''', '''
rows, prev = [], None
for n in range(6, 16):
    # TODO: refit on the first n trips, record r, det, both coefficients,
    #       and how far the estimate moved from the previous n.
    pass
''' + TODO)]

    C += [code('''
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(inst.n, inst.w_is_rain, 'o-', color=C['ridge'], label='is_rain')
ax[0].plot(inst.n, inst.w_traffic, 's-', color=C['rose'], label='traffic_speed')
ax[0].axhline(0, color='#c9cbc3', lw=1)
ax[0].set(xlabel='training trips n', ylabel='coefficient (Rs per s.d.)',
          title='Least squares, one trip at a time')
ax[0].legend()
ax[1].plot(inst.n, inst.det, 'o-', color=C['accent'])
ax[1].set(xlabel='training trips n', ylabel='det(X$^T$X)',
          title='The determinant recovering as data arrives')
plt.tight_layout(); plt.show()
''')]

    C += [md('''
> **Write-up prompt (Task 1a).** Reporting that the coefficients move is not enough.
> Explain why, in terms of the determinant and the $-r$ off-diagonal of the inverse. One
> useful extra check: is the **sum** of the two coefficients more stable than either
> alone? If so, what does that say about which quantity the data identifies?
''')]

    C += [md('''
## Task 1b — A pair where OLS does not exist

Now `is_rain` and `is_bad_weather` on the **first ten trips**.
''')]

    C += [code('''
df['is_bad_weather'] = ((df.is_rain == 1) & (df.traffic_speed_kmph < 25)).astype(int)

sub = df[['is_rain', 'traffic_speed_kmph', 'is_bad_weather']].head(10)
print(sub.to_string())
print('\\ncolumns identical on these rows:',
      bool((sub.is_rain == sub.is_bad_weather).all()))
''')]

    C += [code('''
Xb = df[['is_rain', 'is_bad_weather']].to_numpy(float)[:10]
yb = df['surge_additive_inr'].to_numpy(float)[:10]
Zb = (Xb - Xb.mean(0)) / Xb.std(0)
Ab = Zb.T @ Zb

print('XtX =\\n', Ab)
print(f'\\ndet = {np.linalg.det(Ab):.10f}')
try:
    np.linalg.inv(Ab)
except np.linalg.LinAlgError as e:
    print('np.linalg.inv raised:', type(e).__name__, '-', e)

# When does it become solvable?
for n in range(4, 30):
    Xn = df[['is_rain', 'is_bad_weather']].to_numpy(float)[:n]
    if Xn.std(0).min() == 0:
        continue
    Zn = (Xn - Xn.mean(0)) / Xn.std(0)
    if abs(np.linalg.det(Zn.T @ Zn)) > 1e-8:
        print(f'\\nfirst solvable at n = {n}')
        break
''', '''
# TODO: build XtX on the first ten trips, report its determinant,
#       try to invert it, and find the smallest n at which it becomes solvable.
''' + TODO)]

    C += [md('''
> **Write-up prompt (Task 1b, step 3).** "The MLE does not exist" is a claim about the
> likelihood surface, not about NumPy. Adding $c$ to one coefficient and subtracting $c$
> from the other changes no prediction, so the likelihood has a flat direction and no
> unique maximum. Say that in one sentence, in your own words.
''')]

    C += [md('## Task 1c — The real thing: eight features, 150 trips')]

    C += [code('''
Xtr, Xte, ytr, yte = split(df, BASE8)
Ztr, Zte, mu8, sd8 = standardize(Xtr, Xte)
ybar = ytr.mean()

w_ols = np.linalg.solve(Ztr.T @ Ztr, Ztr.T @ (ytr - ybar))

print(pd.Series({SHORT[f]: round(float(v), 2) for f, v in zip(BASE8, w_ols)},
                name='coef (Rs per s.d.)').to_string())
print(f'\\ncondition number of XtX : {np.linalg.cond(Ztr.T @ Ztr):.1f}')
print(f'train RMSE              : Rs {rmse(ytr, Ztr @ w_ols + ybar):.2f}')
print(f'test  RMSE              : Rs {rmse(yte, Zte @ w_ols + ybar):.2f}')
print(f'test  R2                : {r2(yte, Zte @ w_ols + ybar):.3f}')
''', '''
Xtr, Xte, ytr, yte = split(df, BASE8)
Ztr, Zte, mu8, sd8 = standardize(Xtr, Xte)
ybar = ytr.mean()
w_ols = ...   # TODO: solve the normal equations
''' + TODO)]

    C += [md('''
> **Note the condition number.** An order of magnitude better than the six-row problem.
> With 150 trips and 8 features this dataset is not ill-conditioned, which is worth
> remembering when cross-validation picks a tiny penalty in Task 2.
''')]

    C += [md('''
---

## Task 2 — Ridge regression as MAP

Gaussian prior $w \\sim \\mathcal{N}(0, \\tau^2)$ gives
$\\hat w = (X^\\top X + \\lambda I)^{-1}X^\\top y$ with $\\lambda = \\sigma^2/\\tau^2$.

Use `np.linalg.solve`, not `np.linalg.inv`: the explicit inverse is slower and
numerically worse. Task 1 formed one only because we wanted to look at it.
''')]

    C += [code('''
def ridge_fit(Z, y, alpha_sklearn):
    """Closed-form ridge on the sklearn alpha scale (no 1/n in the loss)."""
    p = Z.shape[1]
    return np.linalg.solve(Z.T @ Z + alpha_sklearn * np.eye(p), Z.T @ y)

yc_tr = ytr - ybar
grid = [0.1, 1, 10, 100, 1000]
rows = []
for a in grid:
    w = ridge_fit(Ztr, yc_tr, a)
    rows.append({'alpha': a,
                 **{SHORT[f]: round(float(v), 2) for f, v in zip(BASE8, w)},
                 'gap_peak_rain': round(float(abs(w[0] - w[1])), 2),
                 'train_rmse': round(rmse(ytr, Ztr @ w + ybar), 2),
                 'test_rmse': round(rmse(yte, Zte @ w + ybar), 2)})
ridge_table = pd.DataFrame(rows)
print(ridge_table.to_string(index=False))
''', '''
def ridge_fit(Z, y, alpha_sklearn):
    """Closed-form ridge. Use np.linalg.solve."""
''' + '    ' + TODO.replace('\n', '\n    ') + '''

yc_tr = ytr - ybar
# TODO: loop over alpha in [0.1, 1, 10, 100, 1000] and build the table
''')]

    C += [code('''
# Verify against scikit-learn (allowed here: we are CHECKING, not computing)
from sklearn.linear_model import Ridge
for a in grid:
    mine = ridge_fit(Ztr, yc_tr, a)
    theirs = Ridge(alpha=a, fit_intercept=False).fit(Ztr, yc_tr).coef_
    print(f'alpha={a:>6}:  max |difference| = {np.max(np.abs(mine - theirs)):.2e}')
''')]

    C += [md('### The coefficient path (Deliverable 1, panel 1)')]

    C += [code('''
alphas = np.logspace(-2, 4, 80)
paths = np.array([ridge_fit(Ztr, yc_tr, a) for a in alphas])

fig, ax = plt.subplots(figsize=(10, 5))
for j, f in enumerate(BASE8):
    ax.plot(alphas, paths[:, j], lw=1.9, label=SHORT[f])
ax.set_xscale('log')
ax.axhline(0, color='#c9cbc3', lw=1)
ax.set(xlabel=r'penalty $\\alpha$  (log scale)', ylabel='coefficient (Rs per s.d.)',
       title='Ridge path — everything shrinks, nothing reaches zero')
ax.legend(ncol=2, loc='upper right')
plt.tight_layout(); plt.show()

gaps = np.abs(paths[:, 0] - paths[:, 1])
print('gap between is_peak and is_rain:')
for a, g in zip(alphas[::16], gaps[::16]):
    print(f'  alpha={a:9.2f}   |gap| = {g:5.2f}')
print('\\nThe gap falls monotonically: the two correlated features converge toward')
print('each other rather than meeting at some particular alpha.')
''')]

    C += [md('''
### Choosing $\\alpha$ honestly

Cross-validate **on the training split only**. The test set is for reporting, once.
''')]

    C += [code('''
def kfold_cv(Z, y, k, fit):
    n = len(y)
    idx = np.arange(n)
    total = 0.0
    for f in range(k):
        te = np.zeros(n, bool); te[idx[f::k]] = True
        ybar_f = y[~te].mean()
        w = fit(Z[~te], y[~te] - ybar_f)
        total += np.sum((y[te] - (Z[te] @ w + ybar_f)) ** 2)
    return float(np.sqrt(total / n))

cv_grid = np.logspace(-2, 4, 60)
cv_scores = [kfold_cv(Ztr, ytr, 5, lambda Z, t, a=a: ridge_fit(Z, t, a)) for a in cv_grid]
best_i = int(np.argmin(cv_scores))
best_alpha = float(cv_grid[best_i])

train_e = [rmse(ytr, Ztr @ ridge_fit(Ztr, yc_tr, a) + ybar) for a in cv_grid]
test_e = [rmse(yte, Zte @ ridge_fit(Ztr, yc_tr, a) + ybar) for a in cv_grid]

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(cv_grid, train_e, '--', color=C['ols'], label='training RMSE')
ax.plot(cv_grid, cv_scores, lw=2.4, color=C['accent'], label='5-fold CV RMSE')
ax.plot(cv_grid, test_e, color=C['rose'], label='held-out test RMSE')
ax.axvline(best_alpha, color=C['accent'], ls=':', lw=1.6)
ax.set_xscale('log')
ax.set(xlabel=r'$\\alpha$ (log scale)', ylabel='RMSE (Rs)',
       title='Training error is not evidence')
ax.legend()
plt.tight_layout(); plt.show()

print(f'CV-optimal alpha        : {best_alpha:.3f}')
print(f'CV RMSE there           : Rs {cv_scores[best_i]:.2f}')
print(f'test RMSE there         : Rs {test_e[best_i]:.2f}')
print(f'prior says lambda       : {SIGMA**2/TAU**2:.0f}')
print(f'test RMSE at lambda=100 : Rs {rmse(yte, Zte @ ridge_fit(Ztr, yc_tr, 100) + ybar):.2f}')
''', '''
def kfold_cv(Z, y, k, fit):
    """k-fold CV RMSE. Re-centre y inside each fold."""
''' + '    ' + TODO.replace('\n', '\n    ') + '''

# TODO: sweep alpha, record train / CV / test RMSE, plot all three, report the minimum
''')]

    C += [md('''
> **Write-up prompt (Task 2, step 6).** Cross-validation and the stated prior disagree by
> a wide margin. The prior $\\tau = 1$ asserts coefficients of about $\\pm1$ rupee per
> standard deviation; the data says about $\\pm10$. Which do you trust with 150
> observations, and what does the disagreement say about the prior?
''')]

    C += [md('''
---

### Checklist before you move on

- [ ] `XtX` computed by hand and cross-checked against NumPy
- [ ] Instability table produced, with the largest one-trip move quoted
- [ ] Singular case reproduced, and explained as a property of the likelihood
- [ ] Ridge implemented with `solve`, matching sklearn to ~1e-14
- [ ] Coefficient path plotted with a log x-axis (**Deliverable 1**)
- [ ] CV run on the training split only, and the CV-vs-prior disagreement discussed

Next: **Notebook 02 — Lasso by coordinate descent**.
''')]
    return C


# =========================================================================== #
# 02 · Lasso
# =========================================================================== #
def nb02():
    C = preamble(
        'Notebook 02 · Lasso and Coordinate Descent',
        'Covers **Task 3**: soft-thresholding derived and implemented, cyclic coordinate '
        'descent, verification against scikit-learn, and the $p \\gg n$ new-city '
        'selection problem where the decoys bite.')

    C += [md('''
## 3a — Soft-thresholding

Minimising $\\tfrac{1}{2n}\\lVert y - Xw\\rVert^2 + \\alpha\\lVert w\\rVert_1$ one
coordinate at a time gives

$$w_j \\leftarrow \\frac{S(\\rho_j,\\ \\alpha)}{\\lVert x_j\\rVert^2/n},
\\qquad S(\\rho,\\gamma) = \\operatorname{sign}(\\rho)\\max(|\\rho| - \\gamma,\\ 0)$$

where $\\rho_j$ is the correlation of feature $j$ with the partial residual: the residual
with feature $j$'s own contribution added back in.
''')]

    C += [code('''
def soft_threshold(rho, gamma):
    """Shrink rho toward zero by gamma, and clamp at exactly zero."""
    return np.sign(rho) * max(abs(rho) - gamma, 0.0)

xs = np.linspace(-3, 3, 400)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(xs, xs, '--', color=C['ols'], label='no penalty (identity)')
for g, ls in [(0.5, '-'), (1.0, '-'), (1.8, '-')]:
    ax.plot(xs, [soft_threshold(x, g) for x in xs], ls, lw=2, label=f'$\\\\gamma$ = {g}')
ax.axhline(0, color='#c9cbc3', lw=1); ax.axvline(0, color='#c9cbc3', lw=1)
ax.set(xlabel=r'$\\rho_j$', ylabel=r'updated $w_j$',
       title='Soft-thresholding: the flat region is where features die')
ax.legend()
plt.tight_layout(); plt.show()
''', '''
def soft_threshold(rho, gamma):
    """Shrink rho toward zero by gamma, and clamp at exactly zero."""
''' + '    ' + TODO.replace('\n', '\n    '))]

    C += [md('''
### Cyclic coordinate descent

Sweep over coordinates, updating each to its exact conditional optimum until nothing
moves. Update the residual incrementally; recomputing `y - Z @ w` from scratch on every
coordinate is the usual reason an implementation is slow.
''')]

    C += [code('''
def coordinate_descent(Z, y, alpha, l1_ratio=1.0, tol=1e-7, max_iter=3000, verbose=False):
    """Minimise (1/2n)||y - Zw||^2 + a*r*||w||_1 + (a*(1-r)/2)*||w||^2.

    Identical to sklearn's ElasticNet objective. l1_ratio=1 is the Lasso."""
    n, p = Z.shape
    w = np.zeros(p)
    l1, l2 = alpha * l1_ratio, alpha * (1 - l1_ratio)
    col_sq = (Z ** 2).sum(0) / n
    r = y - Z @ w
    for sweep in range(max_iter):
        max_delta = 0.0
        for j in range(p):
            rho = Z[:, j] @ (r + Z[:, j] * w[j]) / n
            new = soft_threshold(rho, l1) / (col_sq[j] + l2)
            delta = new - w[j]
            if delta != 0.0:
                r -= Z[:, j] * delta          # incremental residual update
                w[j] = new
                max_delta = max(max_delta, abs(delta))
        if verbose and sweep < 4:
            obj = np.sum((y - Z @ w) ** 2) / (2 * n) + l1 * np.abs(w).sum() \\
                  + l2 / 2 * (w ** 2).sum()
            print(f'  sweep {sweep + 1}: objective = {obj:.6f}, max move = {max_delta:.5f}')
        if max_delta < tol:
            break
    return w, sweep + 1

df['is_bad_weather'] = ((df.is_rain == 1) & (df.traffic_speed_kmph < 25)).astype(int)
Xtr, Xte, ytr, yte = split(df, BASE8)
Ztr, Zte, _, _ = standardize(Xtr, Xte)
ybar = ytr.mean(); yc_tr = ytr - ybar

print('first sweeps at alpha = 2:')
w_demo, iters = coordinate_descent(Ztr, yc_tr, 2.0, verbose=True)
print(f'\\nconverged in {iters} sweeps')
print(pd.Series({SHORT[f]: round(float(v), 3) for f, v in zip(BASE8, w_demo)}).to_string())
''', '''
def coordinate_descent(Z, y, alpha, l1_ratio=1.0, tol=1e-7, max_iter=3000):
    """Minimise (1/2n)||y - Zw||^2 + a*r*||w||_1 + (a*(1-r)/2)*||w||^2.

    Hints:
      - keep a running residual r = y - Z@w and update it incrementally
      - rho_j = Z[:, j] @ (r + Z[:, j] * w[j]) / n
      - denominator is ||x_j||^2/n + alpha*(1 - l1_ratio)
      - stop when the largest coefficient change is below tol
    """
''' + '    ' + TODO.replace('\n', '\n    '))]

    C += [md('''
### Verify against scikit-learn

If these disagree by more than the solver tolerance, the objective differs: usually a
missing $1/2n$ or a stray factor of 2. Find the factor rather than tuning until the
numbers line up.
''')]

    C += [code('''
from sklearn.linear_model import Lasso
print(f"{'alpha':>7} {'sweeps':>7} {'max |diff|':>12} {'nnz mine':>9} {'nnz sklearn':>12}")
for a in [0.5, 2.0, 8.0]:
    mine, it = coordinate_descent(Ztr, yc_tr, a, 1.0)
    theirs = Lasso(alpha=a, fit_intercept=False, max_iter=100000, tol=1e-10).fit(
        Ztr, yc_tr).coef_
    print(f'{a:>7} {it:>7} {np.max(np.abs(mine - theirs)):>12.2e} '
          f'{int((np.abs(mine) > 1e-8).sum()):>9} {int((np.abs(theirs) > 1e-8).sum()):>12}')
''')]

    C += [md('### The Lasso path and the order of elimination (Deliverable 1, panel 2)')]

    C += [code('''
alphas = np.logspace(-2, 2, 90)
paths = np.array([coordinate_descent(Ztr, yc_tr, a, 1.0)[0] for a in alphas])

fig, ax = plt.subplots(figsize=(10, 5))
for j, f in enumerate(BASE8):
    ax.plot(alphas, paths[:, j], lw=1.9, label=SHORT[f])
ax.set_xscale('log'); ax.axhline(0, color='#c9cbc3', lw=1)
ax.set(xlabel=r'penalty $\\alpha$ (log scale)', ylabel='coefficient (Rs per s.d.)',
       title='Lasso path — coefficients hit exactly zero and stay there')
ax.legend(ncol=2)
plt.tight_layout(); plt.show()

order, seen = [], set()
for k, a in enumerate(alphas):
    for j in np.where(np.abs(paths[k]) < 1e-8)[0]:
        if j not in seen:
            seen.add(j); order.append((SHORT[BASE8[j]], round(float(a), 3)))
print('elimination order (feature, alpha at which it died):')
for f, a in order:
    print(f'  {f:<16} {a}')
''')]

    C += [md('''
> **Expected finding.** `is_weekend` should go first: its true coefficient is exactly
> zero, so any weight it carries is fitted noise, which is what the penalty removes. If
> something else goes first in your run, investigate before writing anything else.
''')]

    C += [md('''
---

## 3b — The new-city launch: 60 trips, 120 features

OLS is undefined here: $X^\\top X$ is $120\\times120$ with rank at most 60. Six features
drive surge, and **each has a decoy planted beside it** at $r \\approx 0.9$. Do not open
the truth file until step 3.
''')]

    C += [code('''
nc = load('new_city_features.csv')
fcols = [c for c in nc.columns if c not in ('trip_id', 'surge_additive_inr')]
Xn = nc[fcols].to_numpy(float)
yn = nc['surge_additive_inr'].to_numpy(float)
Zn = (Xn - Xn.mean(0)) / np.where(Xn.std(0) == 0, 1e-8, Xn.std(0))
ync = yn - yn.mean()

print(f'{Xn.shape[0]} trips, {Xn.shape[1]} candidate features')
print(f'rank of X          : {np.linalg.matrix_rank(Xn)}')
print(f'so X^T X is {Xn.shape[1]}x{Xn.shape[1]} with rank at most {Xn.shape[0]}'
      ' -> singular, and OLS has no unique solution.')
''')]

    C += [code('''
sweep = []
for a in [1, 2, 4, 6, 8, 10, 12, 15, 20, 25]:
    w, _ = coordinate_descent(Zn, ync, a, 1.0)
    sel = [fcols[j] for j in np.where(np.abs(w) > 1e-8)[0]]
    sweep.append({'alpha': a, 'n_selected': len(sel), 'selected': sel})
print(pd.DataFrame(sweep)[['alpha', 'n_selected']].to_string(index=False))
''', '''
# TODO: sweep alpha over [1, 2, 4, 6, 8, 10, 12, 15, 20, 25] and record how many
#       features survive at each. Do NOT look at the truth file yet.
''' + TODO)]

    C += [md('### Now open the truth file')]

    C += [code('''
truth = load('new_city_truth.csv')
real = set(truth.loc[truth.true_coefficient != 0, 'feature'])
print('the six real drivers:')
print(truth[truth.true_coefficient != 0].to_string(index=False))

for row in sweep:
    sel = set(row['selected'])
    row['recall'] = len(sel & real) / len(real)
    row['precision'] = len(sel & real) / max(len(sel), 1)
    row['missed'] = sorted(real - sel)

res = pd.DataFrame(sweep)
print('\\n', res[['alpha', 'n_selected', 'recall', 'precision', 'missed']].to_string(index=False))

fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(res.alpha, res.recall, 'o-', color=C['accent'], label='recall (of 6 real)')
ax.plot(res.alpha, res.precision, 's-', color=C['lasso'], label='precision')
ax.set(xlabel=r'$\\alpha$', ylabel='rate', ylim=(-0.03, 1.05),
       title='Recall stays perfect for a while; precision is what improves')
ax.legend()
plt.tight_layout(); plt.show()
''')]

    C += [md('''
### The interesting part: which driver dies first, and what replaces it
''')]

    C += [code('''
first_bad = next((r for r in sweep if r['missed']), None)
if first_bad:
    lost = first_bad['missed'][0]
    li = fcols.index(lost)
    cors = [(fcols[j], float(np.corrcoef(Xn[:, li], Xn[:, j])[0, 1]))
            for j in range(len(fcols)) if j != li]
    decoy, rv = max(cors, key=lambda c: abs(c[1]))
    kept_fakes = [f for f in first_bad['selected'] if f not in real]
    print(f"At alpha = {first_bad['alpha']}, Lasso drops '{lost}' — a REAL driver.")
    print(f"Its strongest correlate is '{decoy}' at r = {rv:.3f}.")
    print(f"Features kept that are not real: {kept_fakes}")
    print(f"\\n'{decoy}' is still in the model: "
          f"{decoy in first_bad['selected']}")
''')]

    C += [md('''
> **Write-up prompt (Task 3b, step 4).** Lasso kept a pure-noise feature and discarded a
> genuine driver. Explain why this is expected behaviour for an $\\ell_1$ penalty rather
> than a bug: the two columns carry nearly the same information, the penalty charges per
> non-zero coefficient, and nothing in the objective prefers the real one.
''')]

    C += [code('''
# Does Elastic Net rescue us here? Check before assuming.
comp = []
for a in [4, 6, 10, 20]:
    for rho, name in [(1.0, 'lasso'), (0.9, 'enet rho=0.9'), (0.5, 'enet rho=0.5')]:
        w, _ = coordinate_descent(Zn, ync, a, rho)
        sel = {fcols[j] for j in np.where(np.abs(w) > 1e-8)[0]}
        comp.append({'alpha': a, 'method': name, 'n_selected': len(sel),
                     'recall': round(len(sel & real) / 6, 2),
                     'precision': round(len(sel & real) / max(len(sel), 1), 3)})
print(pd.DataFrame(comp).to_string(index=False))
''')]

    C += [md('''
> **The counter-intuitive result.** Elastic Net at $\\rho = 0.5$ has perfect recall and
> poor precision: the $\\ell_2$ term resists setting coefficients to zero, so it returns
> dozens of features instead of a shortlist.
>
> The grouping effect is a virtue when the correlated partner is real (Task 4) and a
> liability when it is a decoy. Report this rather than concluding that the more elaborate
> method must be better.
>
> With $n = 60$, Lasso can select at most 60 features however small $\\alpha$ gets. Check
> whether you came near that bound.
''')]

    C += [md('''
---

### Checklist

- [ ] `soft_threshold` and `coordinate_descent` written from scratch
- [ ] Matches `sklearn.Lasso` to ~1e-7 at three different $\\alpha$
- [ ] Lasso path plotted; elimination order recorded; `is_weekend` dies first
- [ ] Recall and precision curves for the new city
- [ ] The real driver that gets dropped, and its decoy, both identified
- [ ] Elastic Net compared — including the result that contradicts expectation

Next: **Notebook 03 — Elastic Net and the grouping effect**.
''')]
    return C


# =========================================================================== #
# 03 · Elastic Net
# =========================================================================== #
def nb03():
    C = preamble(
        'Notebook 03 · Elastic Net and the Grouping Effect',
        'Covers **Task 4**: build a correlated group, show that the three estimators '
        'handle it in three distinguishable ways, then *measure* selection stability with '
        'a bootstrap instead of asserting it.')

    C += [code('''
# Re-create the estimators from notebook 02 (kept short here on purpose).
def soft_threshold(rho, gamma):
    return np.sign(rho) * max(abs(rho) - gamma, 0.0)

def coordinate_descent(Z, y, alpha, l1_ratio=1.0, tol=1e-7, max_iter=3000):
    n, p = Z.shape
    w = np.zeros(p)
    l1, l2 = alpha * l1_ratio, alpha * (1 - l1_ratio)
    col_sq = (Z ** 2).sum(0) / n
    r = y - Z @ w
    for sweep in range(max_iter):
        md_ = 0.0
        for j in range(p):
            rho = Z[:, j] @ (r + Z[:, j] * w[j]) / n
            new = soft_threshold(rho, l1) / (col_sq[j] + l2)
            d = new - w[j]
            if d != 0.0:
                r -= Z[:, j] * d; w[j] = new; md_ = max(md_, abs(d))
        if md_ < tol:
            break
    return w, sweep + 1

def ridge_norm(Z, y, alpha):
    """Ridge on the SAME normalised scale as coordinate_descent (needs n*alpha)."""
    n, p = Z.shape
    return np.linalg.solve(Z.T @ Z + n * alpha * np.eye(p), Z.T @ y)

print('estimators ready')
''')]

    C += [md('''
## Step 1 — build the correlated group

`is_bad_weather` is derived from its two parents, so all three describe one situation:
heavy rain with traffic already crawling.
''')]

    C += [code('''
df['is_bad_weather'] = ((df.is_rain == 1) & (df.traffic_speed_kmph < 25)).astype(int)
FEATS = BASE8 + ['is_bad_weather']
GROUP = ['is_rain', 'traffic_speed_kmph', 'is_bad_weather']

print(f"corr(is_bad_weather, is_rain)       = {df.is_bad_weather.corr(df.is_rain):+.3f}")
print(f"corr(is_bad_weather, traffic_speed) = "
      f"{df.is_bad_weather.corr(df.traffic_speed_kmph):+.3f}")
print(f"\\n{df.is_bad_weather.sum()} of {len(df)} trips flagged")

Xtr, Xte, ytr, yte = split(df, FEATS)
Ztr, Zte, _, _ = standardize(Xtr, Xte)
ybar = ytr.mean(); yc_tr = ytr - ybar
''')]

    C += [md('## Steps 2–3 — the three estimators on one correlated group')]

    C += [code('''
ALPHA = 2.0   # a common alpha, so the comparison is like for like
fits = {'ridge': ridge_norm(Ztr, yc_tr, ALPHA),
        'enet (rho=0.5)': coordinate_descent(Ztr, yc_tr, ALPHA, 0.5)[0],
        'lasso': coordinate_descent(Ztr, yc_tr, ALPHA, 1.0)[0]}

rows = []
for name, w in fits.items():
    rows.append({'estimator': name,
                 **{SHORT[g]: round(float(w[FEATS.index(g)]), 2) for g in GROUP},
                 'non_zero': int((np.abs(w) > 1e-8).sum()),
                 'zeroed': ', '.join(SHORT[f] for f, v in zip(FEATS, w)
                                     if abs(v) < 1e-8) or '-',
                 'test_rmse': round(rmse(yte, Zte @ w + ybar), 2)})
print(pd.DataFrame(rows).to_string(index=False))
''', '''
ALPHA = 2.0
# TODO: fit ridge, elastic net (rho=0.5) and lasso at this common alpha, then
#       tabulate the three group coefficients, the non-zero count and what was zeroed.
''' + TODO)]

    C += [code('''
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
for ax, (name, w) in zip(axes, fits.items()):
    vals = [w[FEATS.index(g)] for g in GROUP]
    col = C['ridge'] if 'ridge' in name else (C['enet'] if 'enet' in name else C['lasso'])
    bars = ax.bar([SHORT[g] for g in GROUP], vals, color=col, alpha=.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (1 if v >= 0 else -2.2),
                '0 (dropped)' if abs(v) < 1e-8 else f'{v:.1f}',
                ha='center', fontsize=8.5,
                color='#767d7a' if abs(v) < 1e-8 else '#1b1d1c')
    ax.axhline(0, color='#c9cbc3', lw=1)
    ax.set_title(name)
    ax.tick_params(axis='x', rotation=18)
axes[0].set_ylabel('coefficient (Rs per s.d.)')
fig.suptitle('The same correlated group, three ways', y=1.02, fontsize=12.5)
plt.tight_layout(); plt.show()
''')]

    C += [md('''
> **What to look for.** Ridge spreads the weather signal across all three members. Lasso
> zeroes `is_bad_weather` and loads its weight onto `traffic_speed`. Elastic Net keeps all
> three at moderate values and still zeroes `is_weekend`: the thing Ridge cannot do and
> Lasso does too aggressively.
>
> If your run disagrees, say so and investigate. An honest negative result is worth more
> than a forced positive one.
''')]

    C += [md('## Step 4 — sweep the mixing ratio $\\rho$')]

    C += [code('''
rhos = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
sweep = []
for rho in rhos:
    w = ridge_norm(Ztr, yc_tr, ALPHA) if rho == 0 else \\
        coordinate_descent(Ztr, yc_tr, ALPHA, rho)[0]
    sweep.append({'rho': rho,
                  **{SHORT[g]: round(float(w[FEATS.index(g)]), 2) for g in GROUP},
                  'non_zero': int((np.abs(w) > 1e-8).sum())})
sw = pd.DataFrame(sweep)
print(sw.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.4))
for g, col in zip(GROUP, [C['ridge'], C['rose'], C['enet']]):
    ax.plot(sw.rho, sw[SHORT[g]], 'o-', color=col, lw=2, label=SHORT[g])
ax.axhline(0, color='#c9cbc3', lw=1)
ax.set(xlabel=r'$\\rho$   (0 = Ridge, 1 = Lasso)', ylabel='coefficient (Rs per s.d.)',
       title='Where the correlated group breaks apart')
ax.legend()
plt.tight_layout(); plt.show()
''')]

    C += [md('''
## Step 5 — the bootstrap: measuring selection stability

"Lasso picks one of a correlated group at random" is a claim about variability across
resamples, so it cannot be checked on a single fit. Resample the training set with
replacement, refit, and count how often each feature survives.
''')]

    C += [code('''
rng = np.random.default_rng(42)
N_BOOT = 200
counts = {'lasso': np.zeros(len(FEATS)), 'enet': np.zeros(len(FEATS))}

for _ in range(N_BOOT):
    idx = rng.integers(0, len(yc_tr), len(yc_tr))
    Zb, yb = Ztr[idx], yc_tr[idx]
    yb = yb - yb.mean()
    for name, rho in [('lasso', 1.0), ('enet', 0.5)]:
        w, _ = coordinate_descent(Zb, yb, ALPHA, rho)
        counts[name] += (np.abs(w) > 1e-8)

stab = pd.DataFrame({
    'feature': [SHORT[f] for f in FEATS],
    'lasso_%': (100 * counts['lasso'] / N_BOOT).round(1),
    'enet_%': (100 * counts['enet'] / N_BOOT).round(1),
}).sort_values('lasso_%')
print(stab.to_string(index=False))
''', '''
rng = np.random.default_rng(42)
N_BOOT = 200
# TODO: resample the training rows with replacement N_BOOT times, refit Lasso and
#       Elastic Net each time, and count how often each feature is selected.
''' + TODO)]

    C += [code('''
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ypos = np.arange(len(stab))
ax.barh(ypos - 0.2, stab['lasso_%'], height=0.38, color=C['lasso'], label='Lasso')
ax.barh(ypos + 0.2, stab['enet_%'], height=0.38, color=C['enet'], label='Elastic Net')
ax.set_yticks(ypos); ax.set_yticklabels(stab.feature)
ax.set(xlabel='% of 200 bootstrap resamples in which the feature was selected',
       xlim=(0, 105), title='Selection stability — the grouping effect, measured')
ax.legend(loc='lower right')
plt.tight_layout(); plt.show()

bw = stab[stab.feature == 'is_bad_weather'].iloc[0]
wk = stab[stab.feature == 'is_weekend'].iloc[0]
print(f"is_bad_weather: Lasso keeps it {bw['lasso_%']:.0f}% of the time, "
      f"Elastic Net {bw['enet_%']:.0f}%.")
print(f"is_weekend    : Lasso keeps it {wk['lasso_%']:.0f}% of the time, "
      f"Elastic Net {wk['enet_%']:.0f}%.")
''')]

    C += [md('''
> **Read the whole table.** Elastic Net keeps the real group together, as expected. But
> `is_weekend` has a true coefficient of exactly zero, and Elastic Net keeps it far more
> often than Lasso does: its reluctance to zero anything is a cost as well as a benefit.
> A write-up that notices this has read the evidence.
''')]

    C += [md('''
### Cross-validated comparison, for Deliverable 2
''')]

    C += [code('''
def kfold_cv(Z, y, k, fit):
    n = len(y); idx = np.arange(n); tot = 0.0
    for f in range(k):
        te = np.zeros(n, bool); te[idx[f::k]] = True
        yb = y[~te].mean()
        w = fit(Z[~te], y[~te] - yb)
        tot += np.sum((y[te] - (Z[te] @ w + yb)) ** 2)
    return float(np.sqrt(tot / n))

grid = np.logspace(-3, 2.2, 40)
rows = []
for name, fit_at in [
        ('OLS', None),
        ('Ridge', lambda Z, t, a: ridge_norm(Z, t, a)),
        ('Lasso', lambda Z, t, a: coordinate_descent(Z, t, a, 1.0)[0]),
        ('Elastic Net', lambda Z, t, a: coordinate_descent(Z, t, a, 0.5)[0])]:
    if fit_at is None:
        a_best, w = None, np.linalg.solve(Ztr.T @ Ztr, Ztr.T @ yc_tr)
    else:
        scores = [kfold_cv(Ztr, ytr, 5, lambda Z, t, a=a: fit_at(Z, t, a)) for a in grid]
        a_best = float(grid[int(np.argmin(scores))])
        w = fit_at(Ztr, yc_tr, a_best)
    rows.append({'model': name, 'alpha': None if a_best is None else round(a_best, 4),
                 'train_rmse': round(rmse(ytr, Ztr @ w + ybar), 2),
                 'test_rmse': round(rmse(yte, Zte @ w + ybar), 2),
                 'test_r2': round(r2(yte, Zte @ w + ybar), 3),
                 'non_zero': int((np.abs(w) > 1e-8).sum()),
                 'dropped': ', '.join(SHORT[f] for f, v in zip(FEATS, w)
                                      if abs(v) < 1e-8) or '-'})
cmp_table = pd.DataFrame(rows)
print(cmp_table.to_string(index=False))
spread = cmp_table.test_rmse.max() - cmp_table.test_rmse.min()
print(f'\\nBest and worst test RMSE differ by Rs {spread:.2f}.')
''')]

    C += [md('''
> **Deliverable 3 starts here.** If the spread is under a rupee or two on fares of several
> hundred, there is no statistically meaningful winner. Say so, then argue on operational
> grounds: a short fare breakdown, stability across refits, one hyperparameter versus two,
> and whether a feature vanishing after an overnight retrain is acceptable to the team that
> owns the model. Do not decide on the third decimal of RMSE.

---

### Checklist

- [ ] `is_bad_weather` built; both parent correlations reported
- [ ] Three estimators compared at a common $\\alpha$, and CV-tuned
- [ ] $\\rho$ sweep plotted; the break-up point identified
- [ ] **Bootstrap stability table and figure** (the graded step)
- [ ] Comparison table for Deliverable 2, with the RMSE spread quoted

Next: **Notebook 04 — the surge engine and the posterior**.
''')]
    return C


# =========================================================================== #
# 04 · Engine + Bayes
# =========================================================================== #
def nb04():
    C = preamble(
        'Notebook 04 · The Surge Engine and the Posterior',
        'Covers **Task 5** (assemble a deployable pricing function with its policy layer) '
        'and the **Bonus** (the full Gaussian posterior, credible intervals and a '
        'predictive band).')

    C += [code('''
def soft_threshold(rho, gamma):
    return np.sign(rho) * max(abs(rho) - gamma, 0.0)

def coordinate_descent(Z, y, alpha, l1_ratio=1.0, tol=1e-7, max_iter=3000):
    n, p = Z.shape
    w = np.zeros(p); l1, l2 = alpha * l1_ratio, alpha * (1 - l1_ratio)
    col_sq = (Z ** 2).sum(0) / n; r = y - Z @ w
    for sweep in range(max_iter):
        md_ = 0.0
        for j in range(p):
            rho = Z[:, j] @ (r + Z[:, j] * w[j]) / n
            new = soft_threshold(rho, l1) / (col_sq[j] + l2)
            d = new - w[j]
            if d != 0.0:
                r -= Z[:, j] * d; w[j] = new; md_ = max(md_, abs(d))
        if md_ < tol:
            break
    return w, sweep + 1

def kfold_cv(Z, y, k, fit):
    n = len(y); idx = np.arange(n); tot = 0.0
    for f in range(k):
        te = np.zeros(n, bool); te[idx[f::k]] = True
        yb = y[~te].mean()
        w = fit(Z[~te], y[~te] - yb)
        tot += np.sum((y[te] - (Z[te] @ w + yb)) ** 2)
    return float(np.sqrt(tot / n))

df['is_bad_weather'] = ((df.is_rain == 1) & (df.traffic_speed_kmph < 25)).astype(int)
FEATS = BASE8 + ['is_bad_weather']
Xtr, Xte, ytr, yte = split(df, FEATS)
Ztr, Zte, MU, SD = standardize(Xtr, Xte)
YBAR = ytr.mean()

# Tune by 5-fold CV on the training split, exactly as in notebook 03.
grid = np.logspace(-3, 2.2, 40)
scores = [kfold_cv(Ztr, ytr, 5, lambda Z, t, a=a: coordinate_descent(Z, t, a, 0.5)[0])
          for a in grid]
ALPHA = float(grid[int(np.argmin(scores))])
W, _ = coordinate_descent(Ztr, ytr - YBAR, ALPHA, 0.5)
print(f'Elastic Net, CV-tuned alpha = {ALPHA:.4f}')
print('test RMSE = Rs', round(rmse(yte, Zte @ W + YBAR), 2))
''')]

    C += [md('''
## Task 5 — the pricing function

Note where the policy lives: **outside** the model. The regression estimates what the
market would bear; the cap and threshold are business decisions about what to charge.
Separate layers let you change the cap without retraining, and audit the two
independently.
''')]

    C += [code('''
CAP, THRESHOLD, MOVE_INCENTIVE = 150.0, 1.5, 50.0

def price(hour, is_rain, traffic, drivers, requests, distance_km,
          is_event=0, is_airport=0, is_weekend=0, cap=CAP, threshold=THRESHOLD):
    """Price one trip. Returns every intermediate quantity, not just the answer."""
    is_peak = int((8 <= hour <= 10) or (18 <= hour <= 21))
    bad_weather = int(is_rain == 1 and traffic < 25)
    x = np.array([is_peak, is_rain, traffic, drivers, is_event,
                  is_airport, is_weekend, requests, bad_weather], float)
    z = (x - MU) / SD
    raw = float(z @ W + YBAR)                      # ---- model layer ----
    ratio = requests / max(drivers, 1)             # ---- policy layer ----
    if ratio < threshold:
        surge, reason = 0.0, f'suppressed: demand/supply {ratio:.2f} < {threshold}'
    elif raw > cap:
        surge, reason = cap, f'clipped at the Rs {cap:.0f} cap'
    elif raw < 0:
        surge, reason = 0.0, 'model went negative, floored at zero'
    else:
        surge, reason = raw, 'within cap and above threshold'
    time_min = distance_km * 2 + (12 if is_peak else 0)
    base = 40 + 12 * distance_km + 2 * time_min
    return {'is_peak': is_peak, 'demand_supply_ratio': round(ratio, 3),
            'raw_model_output': round(raw, 2), 'surge': round(surge, 2),
            'policy': reason, 'base_fare': round(base, 2),
            'final_fare': round(base + surge, 2),
            'driver_should_move': surge > MOVE_INCENTIVE}

bkc = price(hour=19, is_rain=1, traffic=16, drivers=5, requests=25, distance_km=8.0)
for k, v in bkc.items():
    print(f'  {k:<22} {v}')
''', '''
CAP, THRESHOLD, MOVE_INCENTIVE = 150.0, 1.5, 50.0

def price(hour, is_rain, traffic, drivers, requests, distance_km,
          is_event=0, is_airport=0, is_weekend=0, cap=CAP, threshold=THRESHOLD):
    """Price one trip. Keep the policy layer OUTSIDE the model."""
''' + '    ' + TODO.replace('\n', '\n    '))]

    C += [md('''
> **Step 3 — what the ₹50 rule assumes.** "Reposition if the surge exceeds ₹50" assumes
> the driver's cost of a 3 km deadhead (fuel, time, the option value of staying put) is
> below ₹50, and that they will still be matched on arrival. Neither is modelled. The
> second matters: the probability of getting the trip falls as every other driver responds
> to the same signal.
''')]

    C += [md('## Step 4 — sweep the supply and look at the discontinuity')]

    C += [code('''
drivers_range = np.arange(1, 41)
sw = pd.DataFrame([price(19, 1, 16, int(d), 25, 8.0) for d in drivers_range])
sw['drivers'] = drivers_range

fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.plot(sw.drivers, sw.raw_model_output, '--', color=C['ols'],
        label='raw model output')
ax.plot(sw.drivers, sw.surge, lw=2.6, color=C['accent'], label='surge actually charged')
ax.axhline(CAP, color=C['rose'], ls=':', lw=1.5)
ax.text(30, CAP + 3, f'Rs {CAP:.0f} cap', color=C['rose'], fontsize=9)
cross = sw.loc[sw.surge == 0, 'drivers'].min()
ax.axvline(cross, color=C['lasso'], ls=':', lw=1.5)
ax.text(cross + 0.4, 95, f'ratio drops below {THRESHOLD}\\nat {cross} drivers',
        color=C['lasso'], fontsize=8.6)
ax.set(xlabel='drivers available within 500 m', ylabel='Rs',
       title='Surge against supply: a cap at one end, a cliff at the other')
ax.legend()
plt.tight_layout(); plt.show()

print(sw.loc[sw.drivers.between(cross - 2, cross + 1),
             ['drivers', 'demand_supply_ratio', 'raw_model_output', 'surge',
              'final_fare']].to_string(index=False))
''')]

    C += [md('''
> **Write-up prompt.** One extra driver on the street moves the fare by the full surge
> amount. Two riders requesting the same trip thirty seconds apart can pay very different
> prices because of something neither of them can see. Is a step function the right
> policy? What would a smooth ramp cost you, and what would it buy? And note that the
> ratio is itself an *estimate* with uncertainty — which a hard threshold ignores entirely.
>
> **Look harder at the dashed line.** The raw model output is almost flat across the whole
> supply range: going from 1 driver to 40 barely moves it. That is a *modelling* failure,
> not a policy one. The data generating process contains a $50/(S+1)$ term — sharply
> non-linear in supply — and a model linear in `drivers_available` simply cannot represent
> it. Adding `1/(drivers+1)` as a feature is a one-line change; try it, report what happens
> to the sweep and to test RMSE, and note that no amount of regularization would have found
> this for you. Choosing the right features is still your job.
''')]

    C += [md('''
> **Step 5 — elasticity.** The data cannot answer it. Prices were set by a policy that
> responded to demand, so price and demand are simultaneously determined and any regression
> of demand on price is confounded. Identifying elasticity needs experimental variation:
> hold out a fraction of cell-buckets from the surge policy, or randomise the surge within
> a narrow band. Name the endogeneity; "we need more data" is not the answer.
''')]

    C += [md('''
---

## Bonus — the full Gaussian posterior

$$\\Sigma = \\left(\\frac{X^\\top X}{\\sigma^2} + \\frac{I}{\\tau^2}\\right)^{-1},
\\qquad \\mu = \\frac{1}{\\sigma^2}\\Sigma X^\\top y$$

with $\\sigma = 10$, $\\tau = 1$. The posterior *mean* should equal the ridge solution at
$\\lambda = \\sigma^2/\\tau^2 = 100$ — because the mode of a Gaussian is its mean.
''')]

    C += [code('''
Xtr8, Xte8, ytr8, yte8 = split(df, BASE8)
Z8, Zt8, MU8, SD8 = standardize(Xtr8, Xte8)
yb8 = ytr8.mean(); yc8 = ytr8 - yb8
s2, t2 = SIGMA ** 2, TAU ** 2

Sigma = np.linalg.inv(Z8.T @ Z8 / s2 + np.eye(8) / t2)
mu_post = Sigma @ Z8.T @ yc8 / s2
sd_post = np.sqrt(np.diag(Sigma))

w_ridge100 = np.linalg.solve(Z8.T @ Z8 + (s2 / t2) * np.eye(8), Z8.T @ yc8)
print(f'max |posterior mean - ridge(lambda=100)| = '
      f'{np.max(np.abs(mu_post - w_ridge100)):.2e}   <- an identity, not an approximation')

post = pd.DataFrame({
    'feature': [SHORT[f] for f in BASE8],
    'mean': mu_post.round(2), 'sd': sd_post.round(2),
    'lo95': (mu_post - 1.96 * sd_post).round(2),
    'hi95': (mu_post + 1.96 * sd_post).round(2)})
post['covers_zero'] = post.lo95 * post.hi95 < 0
print('\\n', post.to_string(index=False))
''', '''
s2, t2 = SIGMA ** 2, TAU ** 2
Sigma = ...      # TODO
mu_post = ...    # TODO
sd_post = ...    # TODO
''' + TODO)]

    C += [code('''
fig, ax = plt.subplots(figsize=(8.8, 4.8))
ypos = np.arange(len(post))[::-1]
colors = [C['rose'] if z else C['ridge'] for z in post.covers_zero]
ax.errorbar(post['mean'], ypos,
            xerr=1.96 * post['sd'], fmt='o', capsize=4, lw=1.6,
            ecolor='#4c514f', mfc='white', mec='#1b1d1c', ms=6)
for y_, m, lo, hi, col in zip(ypos, post['mean'], post.lo95, post.hi95, colors):
    ax.plot([lo, hi], [y_, y_], lw=3.5, color=col, alpha=.35, zorder=0)
ax.axvline(0, color='#1b1d1c', lw=1.2, ls='--')
ax.set_yticks(ypos); ax.set_yticklabels(post.feature)
ax.set(xlabel='coefficient (Rs per s.d.)  with 95% credible interval',
       title='Red intervals cross zero — no finding, just a coefficient')
plt.tight_layout(); plt.show()

print('intervals covering zero:',
      ', '.join(post.loc[post.covers_zero, 'feature']) or 'none')
''')]

    C += [md('## Predictive uncertainty for the BKC trip')]

    C += [code('''
x_star = np.array([1, 1, 16, 5, 0, 0, 0, 25], float)   # peak, rain, traffic 16, 5 drivers
z_star = (x_star - MU8) / SD8

pred = float(z_star @ mu_post + yb8)
var_w = float(z_star @ Sigma @ z_star)     # uncertainty about w
var_total = var_w + s2                     # + irreducible observation noise
sd_total = np.sqrt(var_total)

print(f'point estimate            : Rs {pred:.1f}')
print(f'variance from w           : {var_w:.2f}   ({100*var_w/var_total:.0f}% of total)')
print(f'irreducible noise         : {s2:.0f}   ({100*s2/var_total:.0f}% of total)')
print(f'\\nquote to the rider        : Rs {pred:.0f} +/- Rs {sd_total:.0f}')
print(f'95% predictive interval   : [{pred-1.96*sd_total:.0f}, {pred+1.96*sd_total:.0f}]')
print(f'\\nNote the upper end against the Rs {CAP:.0f} cap.')
''')]

    C += [md('''
> **Step 5 — which term dominates?** Almost all the predictive variance is the irreducible
> $\\sigma^2$, not uncertainty about $w$. Ten times more data would shrink the first term
> and leave the second untouched. A tighter quote needs a better model — the non-linear
> $50/(S+1)$ supply term, for instance — not more rows.
>
> **Step 6 — the sentence to refuse.** The upper end of that interval exceeds the ₹150 cap,
> so an honest interval promises a price the policy layer forbids. Refuse any phrasing that
> implies the rider could be charged above the cap, or that presents the point estimate as
> a guarantee. A subtler problem: showing a range invites riders to wait and re-request,
> which changes the demand the model was fitted on.

---

### Checklist

- [ ] `price()` returns every intermediate quantity, with policy outside the model
- [ ] BKC scenario reported in full; reposition decision justified
- [ ] Supply sweep plotted, with cap and threshold marked, and the cliff discussed
- [ ] Endogeneity named in the elasticity answer
- [ ] Posterior mean confirmed equal to ridge at $\\lambda = 100$
- [ ] Credible intervals plotted; the interval covering zero identified
- [ ] Predictive variance decomposed, and the cap conflict noticed

**You now have everything for Deliverables 1–3.** Good luck.
''')]
    return C


# =========================================================================== #
def main():
    os.makedirs(OUT, exist_ok=True)
    specs = [('01_mle_and_ridge', nb01()),
             ('02_lasso_coordinate_descent', nb02()),
             ('03_elastic_net', nb03()),
             ('04_surge_engine_and_bayes', nb04())]
    for name, cells in specs:
        for suffix, is_starter in [('starter', True), ('solution', False)]:
            nb = build(cells, starter=is_starter)
            path = f'{OUT}/{name}_{suffix}.ipynb'
            nbf.write(nb, path)
            print(f'  {os.path.basename(path):<48} {len(nb.cells):>3} cells')


if __name__ == '__main__':
    main()
