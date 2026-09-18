"""
Reference solutions for all five tasks, computed once so that the Word/PDF
hand-outs, the solution notebooks and the website quote the *same* numbers.

Run directly to print a summary:  python scripts/analysis.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE8 = ['is_peak', 'is_rain', 'traffic_speed_kmph', 'drivers_available_500m',
         'is_event_nearby', 'is_airport_pickup', 'is_weekend', 'open_requests_500m']
SHORT = {'is_peak': 'is_peak', 'is_rain': 'is_rain', 'traffic_speed_kmph': 'traffic_speed',
         'drivers_available_500m': 'drivers_avail', 'is_event_nearby': 'is_event',
         'is_airport_pickup': 'is_airport', 'is_weekend': 'is_weekend',
         'open_requests_500m': 'open_requests', 'is_bad_weather': 'is_bad_weather'}
N_TRAIN = 150
SIGMA, TAU = 10.0, 1.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load():
    return pd.read_csv(f'{ROOT}/data/uber_surge_mumbai.csv')


def split(df, features):
    X = df[features].to_numpy(float)
    y = df['surge_additive_inr'].to_numpy(float)
    return X[:N_TRAIN], X[N_TRAIN:], y[:N_TRAIN], y[N_TRAIN:]


def standardize(Xtr, Xte):
    """Fit the scaler on TRAINING rows only, then apply it to both."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd == 0, 1e-8, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def soft_threshold(rho, gamma):
    return np.sign(rho) * max(abs(rho) - gamma, 0.0)


def coordinate_descent(Z, y, alpha, l1_ratio=1.0, tol=1e-7, max_iter=3000):
    """Minimise (1/2n)||y - Zw||^2 + a*r*||w||_1 + a(1-r)/2*||w||^2.

    Exactly scikit-learn's ElasticNet objective, so the two agree to ~1e-8."""
    n, p = Z.shape
    w = np.zeros(p)
    l1, l2 = alpha * l1_ratio, alpha * (1 - l1_ratio)
    col_sq = (Z ** 2).sum(0) / n
    r = y - Z @ w
    for it in range(max_iter):
        max_delta = 0.0
        for j in range(p):
            rho = Z[:, j] @ (r + Z[:, j] * w[j]) / n
            new = soft_threshold(rho, l1) / (col_sq[j] + l2)
            d = new - w[j]
            if d != 0.0:
                r -= Z[:, j] * d
                w[j] = new
                max_delta = max(max_delta, abs(d))
        if max_delta < tol:
            break
    return w, it + 1


def ridge_closed_form(Z, y, alpha_norm):
    """alpha_norm is on the (1/2n) scale, so the closed form needs n*alpha."""
    n, p = Z.shape
    return np.linalg.solve(Z.T @ Z + n * alpha_norm * np.eye(p), Z.T @ y)


def kfold_cv(Z, y, k, fit):
    n = len(y)
    idx = np.arange(n)
    folds = [idx[i::k] for i in range(k)]
    tot = 0.0
    for f in folds:
        te = np.zeros(n, bool)
        te[f] = True
        ybar = y[~te].mean()
        w = fit(Z[~te], y[~te] - ybar)
        tot += np.sum((y[te] - (Z[te] @ w + ybar)) ** 2)
    return float(np.sqrt(tot / n))


def tune(Z, y, method, l1_ratio=0.5, grid=None):
    grid = grid if grid is not None else np.logspace(-3, 2.2, 40)
    best = None
    for a in grid:
        if method == 'ridge':
            fit = lambda X, t, a=a: ridge_closed_form(X, t, a)
        else:
            rr = 1.0 if method == 'lasso' else l1_ratio
            fit = lambda X, t, a=a, rr=rr: coordinate_descent(X, t, a, rr)[0]
        cv = kfold_cv(Z, y, 5, fit)
        if best is None or cv < best[1]:
            best = (float(a), cv)
    return best


# --------------------------------------------------------------------------- #
# Task 1 — MLE / OLS by hand
# --------------------------------------------------------------------------- #
def task1(df):
    """Two features that really are hard to tell apart, on a handful of trips.

    Pair A (is_rain, traffic_speed) is severely ill-conditioned at n = 6.
    Pair B (is_rain, is_bad_weather) is EXACTLY singular on the first 10 trips,
    because in that stretch every rainy trip also had traffic below 25 km/h —
    the two columns are literally identical, and OLS does not exist."""
    out = {}
    d = df.copy()
    d['is_bad_weather'] = ((d.is_rain == 1) & (d.traffic_speed_kmph < 25)).astype(int)

    def fit_first(feats, n, frame=None):
        frame = d if frame is None else frame
        X = frame[feats].to_numpy(float)[:n]
        y = frame['surge_additive_inr'].to_numpy(float)[:n]
        sd = X.std(0)
        if (sd == 0).any():
            return {'n': n, 'degenerate': 'a column is constant on these rows'}
        Z = (X - X.mean(0)) / sd
        yc = y - y.mean()
        XtX, Xty = Z.T @ Z, Z.T @ yc
        det = float(np.linalg.det(XtX))
        r = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
        res = {'n': n, 'XtX': np.round(XtX, 3).tolist(), 'Xty': np.round(Xty, 3).tolist(),
               'det': round(det, 4), 'corr': round(r, 4),
               'cond': round(float(np.linalg.cond(XtX)), 1)}
        if abs(det) < 1e-8:
            res['w'] = None
            res['singular'] = True
        else:
            w = np.linalg.solve(XtX, Xty)
            res['w'] = [round(float(v), 3) for v in w]
            res['singular'] = False
            res['inv'] = np.round(np.linalg.inv(XtX), 4).tolist()
        return res

    # ---- Pair A: ill-conditioned, but solvable ----
    A_FEATS = ['is_rain', 'traffic_speed_kmph']
    out['pairA_features'] = A_FEATS
    out['pairA_n6'] = fit_first(A_FEATS, 6)
    tbl = []
    for n in range(6, 16):
        f = fit_first(A_FEATS, n)
        prev = tbl[-1]['w'] if tbl and tbl[-1].get('w') else None
        jump = (None if (prev is None or f.get('w') is None)
                else round(max(abs(a - b) for a, b in zip(f['w'], prev)), 2))
        tbl.append({'n': n, 'corr': f['corr'], 'det': f['det'], 'cond': f['cond'],
                    'w': f.get('w'), 'jump_from_previous': jump})
    out['pairA_instability'] = tbl
    out['pairA_max_jump'] = max((t['jump_from_previous'] or 0) for t in tbl)

    # ---- Pair B: exactly singular ----
    B_FEATS = ['is_rain', 'is_bad_weather']
    out['pairB_features'] = B_FEATS
    out['pairB_n10'] = fit_first(B_FEATS, 10)
    first_ok = next((n for n in range(3, 60) if not fit_first(B_FEATS, n).get('singular', True)
                     and 'degenerate' not in fit_first(B_FEATS, n)), None)
    out['pairB_first_solvable_n'] = first_ok
    sub = d.head(10)[['is_rain', 'is_bad_weather']]
    out['pairB_rows'] = sub.to_dict('records')
    out['pairB_identical'] = bool((sub.is_rain == sub.is_bad_weather).all())

    # ---- prediction for trip 2, from pair A at n = 6 ----
    X = d[A_FEATS].to_numpy(float)[:6]
    y = d['surge_additive_inr'].to_numpy(float)[:6]
    Z = (X - X.mean(0)) / X.std(0)
    w6 = np.linalg.solve(Z.T @ Z, Z.T @ (y - y.mean()))
    out['pred_trip2'] = round(float(Z[1] @ w6 + y.mean()), 2)
    out['actual_trip2'] = float(y[1])

    # ---- full 8-feature OLS on the real training split ----
    Xtr, Xte, ytr, yte = split(df, BASE8)
    Ztr, Zte, _, _ = standardize(Xtr, Xte)
    ybar = ytr.mean()
    w = np.linalg.solve(Ztr.T @ Ztr, Ztr.T @ (ytr - ybar))
    out['w_full'] = {SHORT[f]: round(float(v), 3) for f, v in zip(BASE8, w)}
    out['train_rmse'] = round(rmse(ytr, Ztr @ w + ybar), 3)
    out['test_rmse'] = round(rmse(yte, Zte @ w + ybar), 3)
    out['test_r2'] = round(r2(yte, Zte @ w + ybar), 4)
    out['full_cond'] = round(float(np.linalg.cond(Ztr.T @ Ztr)), 1)
    return out


# --------------------------------------------------------------------------- #
# Task 2 — Ridge
# --------------------------------------------------------------------------- #
def task2(df):
    out = {}
    Xtr, Xte, ytr, yte = split(df, BASE8)
    Ztr, Zte, _, _ = standardize(Xtr, Xte)
    ybar = ytr.mean()
    yc = ytr - ybar
    n = len(ytr)

    rows = []
    for a_sk in [0.1, 1, 10, 100, 1000]:
        w = np.linalg.solve(Ztr.T @ Ztr + a_sk * np.eye(8), Ztr.T @ yc)
        sk = Ridge(alpha=a_sk, fit_intercept=False).fit(Ztr, yc)
        rows.append({
            'alpha': a_sk,
            'coefs': {SHORT[f]: round(float(v), 3) for f, v in zip(BASE8, w)},
            'train_rmse': round(rmse(ytr, Ztr @ w + ybar), 3),
            'test_rmse': round(rmse(yte, Zte @ w + ybar), 3),
            'max_abs_diff_vs_sklearn': float(np.max(np.abs(w - sk.coef_))),
            'gap_peak_rain': round(float(abs(w[0] - w[1])), 3),
        })
    out['grid'] = rows

    a_cv, cv = tune(Ztr, ytr, 'ridge')
    w_cv = ridge_closed_form(Ztr, yc, a_cv)
    out['cv_alpha_normalised'] = round(a_cv, 5)
    out['cv_alpha_sklearn_equiv'] = round(a_cv * n, 3)
    out['cv_rmse'] = round(cv, 3)
    out['cv_test_rmse'] = round(rmse(yte, Zte @ w_cv + ybar), 3)
    out['prior_lambda'] = SIGMA ** 2 / TAU ** 2
    w_prior = np.linalg.solve(Ztr.T @ Ztr + out['prior_lambda'] * np.eye(8), Ztr.T @ yc)
    out['prior_test_rmse'] = round(rmse(yte, Zte @ w_prior + ybar), 3)
    out['prior_coefs'] = {SHORT[f]: round(float(v), 3) for f, v in zip(BASE8, w_prior)}
    return out


# --------------------------------------------------------------------------- #
# Task 3 — Lasso
# --------------------------------------------------------------------------- #
def task3(df):
    out = {}
    Xtr, Xte, ytr, yte = split(df, BASE8)
    Ztr, Zte, _, _ = standardize(Xtr, Xte)
    ybar = ytr.mean()
    yc = ytr - ybar

    checks = []
    for a in [0.5, 2.0, 8.0]:
        w_mine, iters = coordinate_descent(Ztr, yc, a, 1.0)
        sk = Lasso(alpha=a, fit_intercept=False, max_iter=100000, tol=1e-10).fit(Ztr, yc)
        checks.append({'alpha': a, 'iters': iters,
                       'max_abs_diff': float(np.max(np.abs(w_mine - sk.coef_))),
                       'nonzero_mine': int((np.abs(w_mine) > 1e-8).sum()),
                       'nonzero_sklearn': int((np.abs(sk.coef_) > 1e-8).sum())})
    out['sklearn_check'] = checks

    order, seen = [], set()
    for a in np.logspace(-2, 2, 120):
        w, _ = coordinate_descent(Ztr, yc, a, 1.0)
        for j, v in enumerate(w):
            if abs(v) < 1e-8 and j not in seen:
                seen.add(j)
                order.append(SHORT[BASE8[j]])
    out['elimination_order'] = order

    a_cv, cv = tune(Ztr, ytr, 'lasso')
    w_cv, _ = coordinate_descent(Ztr, yc, a_cv, 1.0)
    out['cv_alpha'] = round(a_cv, 5)
    out['cv_rmse'] = round(cv, 3)
    out['cv_test_rmse'] = round(rmse(yte, Zte @ w_cv + ybar), 3)
    out['cv_coefs'] = {SHORT[f]: round(float(v), 3) for f, v in zip(BASE8, w_cv)}
    out['cv_nonzero'] = int((np.abs(w_cv) > 1e-8).sum())

    # ---- 3b: new city, p >> n ----
    nc = pd.read_csv(f'{ROOT}/data/new_city_features.csv')
    truth = pd.read_csv(f'{ROOT}/data/new_city_truth.csv')
    real = set(truth.loc[truth.true_coefficient != 0, 'feature'])
    fc = [c for c in nc.columns if c not in ('trip_id', 'surge_additive_inr')]
    Xn = nc[fc].to_numpy(float)
    yn = nc['surge_additive_inr'].to_numpy(float)
    Zn = (Xn - Xn.mean(0)) / np.where(Xn.std(0) == 0, 1e-8, Xn.std(0))
    ync = yn - yn.mean()

    res = []
    for a in [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0]:
        for rr, name in [(1.0, 'lasso'), (0.5, 'enet'), (0.9, 'enet90')]:
            w, _ = coordinate_descent(Zn, ync, a, rr)
            sel = {fc[j] for j in np.where(np.abs(w) > 1e-8)[0]}
            hit = len(sel & real)
            res.append({'alpha': a, 'method': name, 'selected': len(sel),
                        'true_found': hit, 'false_pos': len(sel - real),
                        'precision': round(hit / max(len(sel), 1), 3),
                        'missed': sorted(real - sel)})
    out['new_city'] = res
    las = [r for r in res if r['method'] == 'lasso']
    best = max(las, key=lambda r: (r['true_found'], r['precision']))
    out['new_city_best_lasso'] = best
    first_missed = next((r for r in las if r['missed']), None)
    out['new_city_first_missed'] = first_missed
    # Which decoy is each true driver paired with?
    decoys = {}
    for t in sorted(real):
        ti = fc.index(t)
        cors = [(fc[j], float(np.corrcoef(Xn[:, ti], Xn[:, j])[0, 1]))
                for j in range(len(fc)) if j != ti]
        nm, cv = max(cors, key=lambda c: abs(c[1]))
        decoys[t] = {'decoy': nm, 'r': round(cv, 3)}
    out['new_city_decoys'] = decoys
    # The false positive Lasso keeps at its best alpha
    wb, _ = coordinate_descent(Zn, ync, best['alpha'], 1.0)
    selb = [fc[j] for j in np.where(np.abs(wb) > 1e-8)[0]]
    out['new_city_best_selection'] = [
        {'feature': f, 'coef': round(float(wb[fc.index(f)]), 2), 'is_real': f in real}
        for f in sorted(selb, key=lambda f: -abs(wb[fc.index(f)]))]
    out['new_city_shape'] = list(nc.shape)
    out['new_city_real'] = sorted(real)
    out['lasso_bound'] = len(yn)
    return out


# --------------------------------------------------------------------------- #
# Task 4 — Elastic Net and the grouping effect
# --------------------------------------------------------------------------- #
def task4(df, n_boot=200, seed=42):
    out = {}
    d = df.copy()
    d['is_bad_weather'] = ((d.is_rain == 1) & (d.traffic_speed_kmph < 25)).astype(int)
    feats = BASE8 + ['is_bad_weather']
    out['corr_with_rain'] = round(float(d.is_bad_weather.corr(d.is_rain)), 3)
    out['corr_with_traffic'] = round(float(d.is_bad_weather.corr(d.traffic_speed_kmph)), 3)

    Xtr, Xte, ytr, yte = split(d, feats)
    Ztr, Zte, _, _ = standardize(Xtr, Xte)
    ybar = ytr.mean()
    yc = ytr - ybar
    group = ['is_rain', 'traffic_speed', 'is_bad_weather']

    table = {}
    tuned = {}
    for m in ['ridge', 'lasso', 'enet']:
        a, cv = tune(Ztr, ytr, m)
        w = (ridge_closed_form(Ztr, yc, a) if m == 'ridge'
             else coordinate_descent(Ztr, yc, a, 1.0 if m == 'lasso' else 0.5)[0])
        tuned[m] = (a, w)
        table[m] = {
            'alpha': round(a, 5), 'cv_rmse': round(cv, 3),
            'test_rmse': round(rmse(yte, Zte @ w + ybar), 3),
            'test_r2': round(r2(yte, Zte @ w + ybar), 4),
            'nonzero': int((np.abs(w) > 1e-8).sum()),
            'coefs': {SHORT[f]: round(float(v), 3) for f, v in zip(feats, w)},
            'group': {SHORT[f]: round(float(w[feats.index(f)]), 3)
                      for f in ['is_rain', 'traffic_speed_kmph', 'is_bad_weather']},
        }
    out['tuned'] = table

    # A fixed alpha where all three are directly comparable.
    A_FIX = 2.0
    fixed = {}
    for m, rr in [('ridge', 0.0), ('enet', 0.5), ('lasso', 1.0)]:
        w = (ridge_closed_form(Ztr, yc, A_FIX) if m == 'ridge'
             else coordinate_descent(Ztr, yc, A_FIX, rr)[0])
        fixed[m] = {'group': {SHORT[f]: round(float(w[feats.index(f)]), 3)
                              for f in ['is_rain', 'traffic_speed_kmph', 'is_bad_weather']},
                    'nonzero': int((np.abs(w) > 1e-8).sum()),
                    'zeroed': [SHORT[f] for f, v in zip(feats, w) if abs(v) < 1e-8]}
    out['fixed_alpha'] = A_FIX
    out['at_fixed_alpha'] = fixed

    # rho sweep
    sweep = []
    for rr in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        w = (ridge_closed_form(Ztr, yc, A_FIX) if rr == 0
             else coordinate_descent(Ztr, yc, A_FIX, rr)[0])
        sweep.append({'l1_ratio': rr,
                      **{SHORT[f]: round(float(w[feats.index(f)]), 3)
                         for f in ['is_rain', 'traffic_speed_kmph', 'is_bad_weather']},
                      'nonzero': int((np.abs(w) > 1e-8).sum())})
    out['rho_sweep'] = sweep

    # bootstrap selection stability — the heart of the task
    rng = np.random.default_rng(seed)
    counts = {'lasso': np.zeros(len(feats)), 'enet': np.zeros(len(feats))}
    n = len(ytr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Zb, yb = Ztr[idx], yc[idx]
        for m, rr in [('lasso', 1.0), ('enet', 0.5)]:
            w, _ = coordinate_descent(Zb, yb - yb.mean(), A_FIX, rr)
            counts[m] += (np.abs(w) > 1e-8)
    out['n_boot'] = n_boot
    out['bootstrap'] = {m: {SHORT[f]: round(float(100 * c / n_boot), 1)
                            for f, c in zip(feats, counts[m])} for m in counts}
    return out


# --------------------------------------------------------------------------- #
# Task 5 — the surge engine
# --------------------------------------------------------------------------- #
def build_engine(df):
    d = df.copy()
    d['is_bad_weather'] = ((d.is_rain == 1) & (d.traffic_speed_kmph < 25)).astype(int)
    feats = BASE8 + ['is_bad_weather']
    Xtr, Xte, ytr, yte = split(d, feats)
    Ztr, Zte, mu, sd = standardize(Xtr, Xte)
    ybar = ytr.mean()
    a, _ = tune(Ztr, ytr, 'enet')
    w, _ = coordinate_descent(Ztr, ytr - ybar, a, 0.5)

    def price(hour, is_rain, traffic, drivers, requests, distance_km,
              is_event=0, is_airport=0, is_weekend=0, cap=150.0, threshold=1.5):
        is_peak = int((8 <= hour <= 10) or (18 <= hour <= 21))
        bad = int(is_rain == 1 and traffic < 25)
        x = np.array([is_peak, is_rain, traffic, drivers, is_event,
                      is_airport, is_weekend, requests, bad], float)
        z = (x - mu) / sd
        raw = float(z @ w + ybar)
        ratio = requests / max(drivers, 1)
        surge = 0.0 if ratio < threshold else min(max(raw, 0.0), cap)
        time_min = distance_km * 2 + (12 if is_peak else 0)
        base = 40 + 12 * distance_km + 2 * time_min
        return {'raw_model_output': round(raw, 2), 'demand_supply_ratio': round(ratio, 3),
                'surge': round(surge, 2), 'base_fare': round(base, 2),
                'final_fare': round(base + surge, 2), 'is_peak': is_peak,
                'capped': surge >= cap, 'suppressed': ratio < threshold}
    return price, (a, w, feats, mu, sd, ybar, Ztr, Zte, ytr, yte)


def task5(df):
    price, _ = build_engine(df)
    out = {'bkc': price(hour=19, is_rain=1, traffic=16, drivers=5, requests=25, distance_km=8.0)}
    out['driver_should_move'] = out['bkc']['surge'] > 50
    sweep = []
    for dr in [1, 2, 3, 5, 8, 12, 16, 17, 20, 25, 30, 40]:
        p = price(hour=19, is_rain=1, traffic=16, drivers=dr, requests=25, distance_km=8.0)
        sweep.append({'drivers': dr, 'ratio': p['demand_supply_ratio'],
                      'raw': p['raw_model_output'], 'surge': p['surge'],
                      'final': p['final_fare']})
    out['supply_sweep'] = sweep
    out['threshold_crossing'] = next((s['drivers'] for s in sweep if s['surge'] == 0), None)
    out['dry_comparison'] = price(hour=19, is_rain=0, traffic=38, drivers=5,
                                  requests=25, distance_km=8.0)
    return out


# --------------------------------------------------------------------------- #
# Bonus — the full posterior
# --------------------------------------------------------------------------- #
def bonus(df):
    out = {}
    Xtr, Xte, ytr, yte = split(df, BASE8)
    Ztr, Zte, mu, sd = standardize(Xtr, Xte)
    ybar = ytr.mean()
    yc = ytr - ybar
    s2, t2 = SIGMA ** 2, TAU ** 2

    Sigma = np.linalg.inv(Ztr.T @ Ztr / s2 + np.eye(8) / t2)
    mu_post = Sigma @ Ztr.T @ yc / s2
    sd_post = np.sqrt(np.diag(Sigma))
    w_ridge = np.linalg.solve(Ztr.T @ Ztr + (s2 / t2) * np.eye(8), Ztr.T @ yc)

    out['lambda'] = s2 / t2
    out['max_abs_diff_map_vs_ridge'] = float(np.max(np.abs(mu_post - w_ridge)))
    out['coefficients'] = [
        {'feature': SHORT[f], 'mean': round(float(m), 3), 'sd': round(float(s), 3),
         'lo': round(float(m - 1.96 * s), 3), 'hi': round(float(m + 1.96 * s), 3),
         'covers_zero': bool((m - 1.96 * s) * (m + 1.96 * s) < 0)}
        for f, m, s in zip(BASE8, mu_post, sd_post)]
    out['n_covering_zero'] = sum(c['covers_zero'] for c in out['coefficients'])

    # predictive interval for the BKC scenario (8 base features, no derived one)
    x = np.array([1, 1, 16, 5, 0, 0, 0, 25], float)
    z = (x - mu) / sd
    pred = float(z @ mu_post + ybar)
    var_w = float(z @ Sigma @ z)
    out['bkc_pred'] = round(pred, 2)
    out['bkc_var_from_w'] = round(var_w, 3)
    out['bkc_var_noise'] = s2
    out['bkc_sd_total'] = round(float(np.sqrt(var_w + s2)), 2)
    out['bkc_interval'] = [round(pred - 1.96 * np.sqrt(var_w + s2), 1),
                           round(pred + 1.96 * np.sqrt(var_w + s2), 1)]
    return out


# --------------------------------------------------------------------------- #
# comparison table (Deliverable 2)
# --------------------------------------------------------------------------- #
def comparison(df):
    d = df.copy()
    d['is_bad_weather'] = ((d.is_rain == 1) & (d.traffic_speed_kmph < 25)).astype(int)
    feats = BASE8 + ['is_bad_weather']
    Xtr, Xte, ytr, yte = split(d, feats)
    Ztr, Zte, _, _ = standardize(Xtr, Xte)
    ybar = ytr.mean()
    yc = ytr - ybar
    rows = []
    for m in ['ols', 'ridge', 'lasso', 'enet']:
        if m == 'ols':
            a, w = None, np.linalg.solve(Ztr.T @ Ztr, Ztr.T @ yc)
        else:
            a, _ = tune(Ztr, ytr, m)
            w = (ridge_closed_form(Ztr, yc, a) if m == 'ridge'
                 else coordinate_descent(Ztr, yc, a, 1.0 if m == 'lasso' else 0.5)[0])
        rows.append({'model': m, 'alpha': None if a is None else round(a, 4),
                     'train_rmse': round(rmse(ytr, Ztr @ w + ybar), 2),
                     'test_rmse': round(rmse(yte, Zte @ w + ybar), 2),
                     'test_r2': round(r2(yte, Zte @ w + ybar), 3),
                     'nonzero': int((np.abs(w) > 1e-8).sum()),
                     'dropped': [SHORT[f] for f, v in zip(feats, w) if abs(v) < 1e-8]})
    spread = max(r['test_rmse'] for r in rows) - min(r['test_rmse'] for r in rows)
    return {'rows': rows, 'n_features': len(feats), 'test_rmse_spread': round(spread, 2)}


def run_all():
    df = load()
    res = {
        'dataset': {'n': len(df), 'n_train': N_TRAIN, 'n_test': len(df) - N_TRAIN,
                    'capped': int((df.surge_additive_inr >= 150).sum()),
                    'mean_surge': round(float(df.surge_additive_inr.mean()), 2),
                    'sd_surge': round(float(df.surge_additive_inr.std()), 2)},
        'correlations': {f'{a}|{b}': round(float(df[a].corr(df[b])), 3) for a, b in [
            ('is_peak', 'traffic_speed_kmph'), ('is_rain', 'traffic_speed_kmph'),
            ('is_peak', 'is_rain'), ('is_peak', 'drivers_available_500m'),
            ('open_requests_500m', 'historical_demand')]},
        'task1': task1(df), 'task2': task2(df), 'task3': task3(df),
        'task4': task4(df), 'task5': task5(df), 'bonus': bonus(df),
        'comparison': comparison(df),
    }
    return res


if __name__ == '__main__':
    r = run_all()
    with open(f'{ROOT}/build/results.json', 'w') as f:
        json.dump(r, f, indent=1)
    print(json.dumps(r, indent=1)[:4000])
    print('\n... written to build/results.json')
