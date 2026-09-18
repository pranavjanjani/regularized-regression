"""Build the assignment brief and the model solutions as .docx and .pdf."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docgen import to_docx, to_pdf          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f'{ROOT}/assignments'
R = json.load(open(f'{ROOT}/build/results.json'))

COURSE = 'Regularized Regression — From Likelihood to Lasso'
FOOT = 'Regularized Regression · Assignment · graduate module'


def n(x, d=2):
    return f'{x:.{d}f}' if isinstance(x, (int, float)) else str(x)


# =========================================================================== #
# THE BRIEF
# =========================================================================== #
def brief():
    t1, t3 = R['task1'], R['task3']
    a6 = t1['pairA_n6']
    inst = t1['pairA_instability']
    B = []

    B += [('title', 'Building a Surge Pricing Engine',
           f'{COURSE} · Assignment 1 of 1 · 100% of the module mark')]

    B += [('p', 'You will build one system end to end: a model that takes the state of a 500 m '
                'grid cell in Mumbai (hour, weather, traffic, drivers, open requests) and returns '
                'an additive surge in rupees, with a price cap, a no-surge rule and an uncertainty '
                'band. Along the way you implement four estimators from scratch, and see why the '
                'choice between them is usually decided by something other than accuracy.')]

    B += [('table',
           ['Task', 'Topic', 'Marks', 'Time'],
           [['1', 'Maximum likelihood by hand; two ways for OLS to fail', '10', '30 min'],
            ['2', 'Ridge as MAP with a Gaussian prior', '20', '45 min'],
            ['3', 'Lasso, soft-thresholding and coordinate descent', '20', '45 min'],
            ['4', 'Elastic Net and the grouping effect', '20', '60 min'],
            ['5', 'Assembling the surge engine', '20', '30 min'],
            ['Bonus', 'The full Gaussian posterior', '+10', '30 min']],
           [0.08, 0.56, 0.16, 0.20])]

    B += [('h1', 'Data and setup')]
    B += [('p', f"**uber_surge_mumbai.csv** — {R['dataset']['n']} synthetic trips across ten "
                f"Mumbai pickup zones. Trips 1–{R['dataset']['n_train']} are the training split; "
                f"{R['dataset']['n_test']} held-out trips follow. Rows are in time order: "
                '**do not shuffle**.')]
    B += [('p', '**new_city_features.csv** — 60 trips, 120 candidate features, for the '
                'p ≫ n exercise in Task 3b. Ground truth is in **new_city_truth.csv**; do not '
                'open it until the task tells you to.')]
    B += [('p', 'The correlations that make this dataset worth using:')]
    B += [('table', ['Pair', 'r', 'Why they move together'],
           [['`is_peak` × `traffic_speed`', n(R['correlations']['is_peak|traffic_speed_kmph'], 3),
             'The evening peak is exactly when the roads seize up.'],
            ['`is_rain` × `traffic_speed`', n(R['correlations']['is_rain|traffic_speed_kmph'], 3),
             'Rain slows everything, and Mumbai rain slows everything a lot.'],
            ['`is_peak` × `drivers_available`',
             n(R['correlations']['is_peak|drivers_available_500m'], 3),
             'Drivers are already occupied when demand peaks.'],
            ['`is_peak` × `is_rain`', n(R['correlations']['is_peak|is_rain'], 3),
             'Monsoon downpours cluster in the late afternoon.'],
            ['`open_requests` × `historical_demand`',
             n(R['correlations']['open_requests_500m|historical_demand'], 3),
             'Near-duplicate features — a textbook Lasso tie-break.']],
           [0.30, 0.10, 0.60])]

    B += [('h2', 'Assumptions you are given')]
    B += [('table', ['#', 'Assumption', 'Consequence'],
           [['1', 'City gridded into 500 m cells, time into 5-minute buckets.',
             'Each row is one cell-bucket. Spatial correlation between neighbouring cells is '
             'ignored, a simplification worth one sentence in your write-up.'],
            ['2', 'Additive pricing: final = base + surge, with base = 40 + 12·d_km + 2·t_min.',
             'You model the surge term only. The base is deterministic.'],
            ['3', 'Gaussian likelihood, y ~ N(Xw, σ²) with σ = ₹10.',
             'Least squares is the MLE. σ is given, not estimated, so λ can be computed rather '
             'than only tuned.'],
            ['4', 'Ridge prior w ~ N(0, τ²) with τ = 1.',
             'λ = σ²/τ² = 100. Compare with what cross-validation picks: they will not agree, and '
             'explaining that is part of Task 2.'],
            ['5', 'Lasso prior w ~ Laplace(0, b) with b = 5.',
             'λ = 2σ²/b = 40 in the ‖y−Xw‖² + λ‖w‖₁ parameterisation. Mind the scaling when '
             'comparing with scikit-learn.'],
            ['6', 'Policy: surge capped at ₹150; surge = 0 when R/S < 1.5.',
             f"The cap censors {R['dataset']['capped']} of the {R['dataset']['n']} targets. "
             'Coefficients on the drivers of extreme surge are biased toward zero.'],
            ['7', 'Time-based split: trips 1–150 train, 151–200 test.',
             'A random split would leak information across time.'],
            ['8', 'Standardize features before any penalized fit.',
             'Penalties are not scale-invariant. Fit the scaler on training rows only.']],
           [0.04, 0.38, 0.58])]

    B += [('callout', 'warn', 'Non-negotiable',
           'Tasks 1–3 must be implemented **from scratch with NumPy**. Call scikit-learn only to '
           'verify, and show both numbers when you do. A submission that calls `Ridge()` for '
           'Task 2 earns zero for that task regardless of the analysis around it.')]

    # ---------------- Task 1 ----------------
    B += [('pagebreak',), ('h1', 'Task 1 · Maximum likelihood, by hand  (10 marks)')]
    B += [('p', 'Derive the OLS estimator numerically, then watch it fail twice: once by becoming '
                'unstable, once by ceasing to exist.')]

    B += [('h2', '1a · An ill-conditioned pair')]
    B += [('p', 'Use `is_rain` and `traffic_speed_kmph` on the **first six trips**, standardized '
                'using those six rows only.')]
    B += [('numbers', [
        'Compute XᵀX (a 2×2 matrix) and Xᵀy **by hand** — show the arithmetic, not just the '
        'NumPy output.',
        'Report the sample correlation between the two columns and the determinant of XᵀX.',
        'Invert analytically using the 2×2 formula and obtain ŵ_MLE.',
        'Predict the surge for **trip 2** (19:00, peak and rain, 6 drivers) and compare with the '
        '₹150 actually charged.',
        '**Now add trips one at a time.** Refit on the first n = 6, 7, …, 15 trips and tabulate, '
        'for each n: the correlation, the determinant, both coefficients, and how far each '
        'coefficient moved from the previous row.'])]
    B += [('math', 'ŵ = (XᵀX)⁻¹ Xᵀy,    with   (a b; b d)⁻¹ = 1/(ad−b²) · (d −b; −b a)')]
    B += [('callout', 'warn', 'What you should find — and must explain',
           f"The two columns are correlated at **{n(a6['corr'], 3)}** on these rows and the "
           f"determinant is **{n(a6['det'], 3)}**. Between n = 6 and n = 7 one coefficient moves "
           f"by roughly **₹{n(inst[1]['jump_from_previous'], 1)} per standard deviation**, and it "
           'keeps swinging for several more rows. One extra observation changes what the model '
           'says about rain by an amount comparable to the effect itself. Marks are for the '
           'explanation, not the table: connect the swing to the determinant and the −r '
           'off-diagonal of the inverse.')]

    B += [('h2', '1b · A pair where OLS does not exist')]
    B += [('p', 'Now use `is_rain` and the derived '
                '`is_bad_weather = (is_rain == 1) & (traffic_speed_kmph < 25)` on the '
                '**first ten trips**.')]
    B += [('numbers', [
        'Print the two columns side by side. What do you notice?',
        'Compute XᵀX and its determinant. Try to invert it and report exactly what happens.',
        'State, in one sentence, what "the maximum likelihood estimate does not exist" means '
        'here — not "NumPy raised an error" but what is true about the *likelihood surface*.',
        'Find the smallest n at which the problem becomes solvable, and say what changed in the '
        'data at that row.',
        'Would Ridge return an answer on the first ten trips? Explain why, in terms of '
        'eigenvalues. (You do not need to compute it yet — Task 2 will.)'])]

    B += [('h2', '1c · The real thing')]
    B += [('numbers', [
        'Fit all eight base features on the full 150-trip training set. Report the coefficients, '
        'the condition number of XᵀX, and train and test RMSE.',
        'Compare that condition number with the ones from 1a. Is the full problem well '
        'conditioned? If so, say plainly that this dataset does not need regularization at this '
        'sample size — and keep that conclusion in mind for Task 2.'])]

    # ---------------- Task 2 ----------------
    B += [('pagebreak',), ('h1', 'Task 2 · Ridge as MAP  (20 marks)')]
    B += [('p', 'Implement the Gaussian-prior MAP estimator and learn to read a coefficient path.')]
    B += [('math', 'ŵ_ridge = (XᵀX + λI)⁻¹ Xᵀy,     λ = σ²/τ²')]
    B += [('numbers', [
        'Implement the closed form with `np.linalg.solve` (not `inv`) — say why in one line.',
        'Fit for α ∈ {0.1, 1, 10, 100, 1000} on the eight base features. Tabulate all eight '
        'coefficients at each α.',
        '**Plot the coefficient path:** α on a log x-axis, coefficients on y, one line per '
        'feature. This is Deliverable 1.',
        'Identify the α at which `is_peak` and `is_rain` come closest together, and explain the '
        'convergence using the equal-split argument.',
        'Compute training and test RMSE at each α. Then run **5-fold cross-validation on the '
        'training split only** over a log grid and report the CV-optimal α.',
        'Assumption 4 says λ = σ²/τ² = 100. Does CV agree? If not, which would you trust, and '
        'what does the disagreement tell you about the prior?',
        'Verify against `sklearn.linear_model.Ridge` and state the match to 3 decimals.'])]
    B += [('callout', 'tip', 'Expected finding, stated in advance',
           'Cross-validation will choose an α far below 100. That is not a bug: with 150 trips and '
           '8 features the data is plentiful relative to the parameters, and the prior τ = 1 is '
           'more confident than the evidence warrants. Say so rather than forcing the two to '
           'agree.')]

    # ---------------- Task 3 ----------------
    B += [('pagebreak',), ('h1', 'Task 3 · Lasso and coordinate descent  (20 marks)')]
    B += [('h2', '3a · The algorithm')]
    B += [('math', 'wⱼ ← S(ρⱼ, λ/2) / ‖xⱼ‖²,     S(ρ, γ) = sign(ρ)·max(|ρ| − γ, 0)')]
    B += [('numbers', [
        'Implement `soft_threshold(rho, gamma)` in three lines.',
        'Implement cyclic coordinate descent for (1/2n)‖y − Xw‖² + α‖w‖₁, stopping when the '
        'largest coefficient change falls below 1e−7.',
        'Verify against `sklearn.linear_model.Lasso` to **3 decimal places** at three different '
        'α. If they disagree, your objective is scaled differently — find the factor rather than '
        'adjusting until it matches.',
        'Plot the Lasso path over a log α grid and record the **order of elimination**.'])]

    B += [('h2', '3b · The new-city launch')]
    B += [('p', 'Load `new_city_features.csv`: 60 trips, 120 candidate features. Six are real '
                'drivers, and **each real driver has a decoy planted beside it, correlated at '
                'r ≈ 0.90–0.96**.')]
    B += [('numbers', [
        'Explain in one sentence why OLS cannot be computed here at all.',
        'Run your Lasso across α ∈ {1, 2, 4, 6, 8, 10, 12, 15, 20, 25}. For each, record how '
        'many features survive.',
        'Now open the truth file. For each α report **recall** (how many of the six you found) '
        'and **precision** (what fraction of your selections are real). Plot both against α. '
        'Which α would you hand to the city-ops team, and why is it not the α with the best '
        'precision?',
        '**The interesting part.** Somewhere in that sweep, Lasso drops a real driver while '
        'keeping a feature that is not real. Identify which real driver goes first, which '
        'feature survives in its place, and the correlation between them. Then explain why this '
        'is the expected behaviour of an ℓ₁ penalty rather than a failure of your code.',
        'Repeat with Elastic Net at ρ = 0.5 and ρ = 0.9. Does recall improve? Does precision? '
        'Compare the number of features selected — and if Elastic Net turns out to be a worse '
        'tool for this particular job, say so and explain what the ℓ₂ term is doing to sparsity.',
        'State the theoretical bound on how many features Lasso can select when n = 60, and say '
        'whether you ever came near it.'])]
    B += [('callout', 'note', 'Do not assume the more elaborate method wins',
           'Step 5 has a counter-intuitive answer here, and reporting it honestly is worth more '
           'than reporting the answer you expected. The grouping effect is what you want when a '
           'correlated group is the *finding* (Task 4), and not what you want when the correlated '
           'partner is a *decoy*.')]

    # ---------------- Task 4 ----------------
    B += [('pagebreak',), ('h1', 'Task 4 · Elastic Net and the grouping effect  (20 marks)')]
    B += [('math', 'wⱼ ← S(ρⱼ, λ₁) / (‖xⱼ‖² + λ₂)')]
    B += [('numbers', [
        'Construct `is_bad_weather = (is_rain == 1) & (traffic_speed_kmph < 25)`. Report its '
        'correlation with each of its two parents.',
        'Fit all three estimators on the nine features, each tuned by 5-fold CV. Report the '
        'coefficients on `is_rain`, `traffic_speed` and `is_bad_weather` in one table.',
        'Verify the three predicted behaviours: **Ridge** keeps all three with comparable '
        'magnitudes; **Lasso** concentrates on roughly one and zeroes the others; '
        '**Elastic Net** keeps the group together while still zeroing `is_weekend`. If any of '
        'the three does *not* behave as predicted, say so plainly and investigate why.',
        'Sweep ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0} at fixed α and plot the three group '
        'coefficients against ρ. At what ρ does the group break apart?',
        '**Test the stability claim directly.** Bootstrap the training set 200 times, refit '
        'Lasso and Elastic Net each time, and report how often each of the three weather '
        'features is selected.'])]
    B += [('callout', 'ok', 'The step that separates grades',
           'Step 5 is the one that matters. "Lasso picks one at random" is a claim about '
           'variability across resamples, and the bootstrap measures it. A selection-frequency '
           'table proves the point rather than asserting it.')]

    # ---------------- Task 5 ----------------
    B += [('h1', 'Task 5 · The surge engine  (20 marks)')]
    B += [('p', 'Assemble a system, not a model.')]
    B += [('numbers', [
        'Wrap your best estimator in a function `price(...) -> dict` returning the raw model '
        'output, the surge after policy, the base fare and the final fare.',
        'Implement the policy layer **outside** the model.',
        '**Simulate:** 19:00, BKC, raining, 5 drivers, 25 open requests, 8 km trip. Report every '
        'intermediate quantity. Should a driver 3 km away in Andheri reposition? Use the rule '
        '"yes if surge > ₹50" and state what that threshold implicitly assumes about the '
        "driver's cost.",
        'Sweep drivers-available from 1 to 40 with everything else fixed and plot the resulting '
        'surge. Mark where the cap binds and where the threshold switches surge off. Comment on '
        'the discontinuity at R/S = 1.5 — is a step function the right policy?',
        'Compute the price elasticity your model implies: if surge rises by ₹50, how much would '
        'demand have to fall for R/S to return to 1? Explain why **your data cannot answer this** '
        'and what experiment would.'])]
    B += [('code', 'ratio = requests / drivers\n'
                   'if ratio < 1.5:   surge = 0\n'
                   'surge = min(max(surge, 0), 150)\n'
                   'final = base + surge')]

    # ---------------- Bonus ----------------
    B += [('h1', 'Bonus · The full posterior  (+10 marks)')]
    B += [('math', 'Σ = (XᵀX/σ² + I/τ²)⁻¹,    μ = Σ Xᵀy / σ²,    Var(y*) = x*ᵀΣx* + σ²')]
    B += [('numbers', [
        'Compute the Gaussian posterior with σ = 10, τ = 1.',
        'Confirm numerically that μ equals your Ridge solution at λ = 100. Explain why in one '
        'sentence.',
        'Report each coefficient as ŵⱼ ± 1.96·√Σⱼⱼ. Which intervals contain zero? How does that '
        'compare with what Lasso decided about the same features?',
        'For the BKC scenario, compute the posterior predictive variance and quote the surge as '
        '"₹X ± ₹Y".',
        'Which of the two variance terms dominates? What would reduce each of them?',
        'Write two sentences a product manager could put in front of a rider. Then write the one '
        'sentence you would **refuse** to let them write, and say why.'])]

    # ---------------- Deliverables & rubric ----------------
    B += [('pagebreak',), ('h1', 'What to hand in')]
    B += [('table', ['#', 'Deliverable', 'What it must show'],
           [['1', 'Coefficient path figure',
             'One page, three panels — Ridge, Lasso, Elastic Net on shared axes, with the '
             'CV-optimal α labelled on each. A reader should be able to tell which estimator '
             'zeroes coefficients without reading your text.'],
            ['2', 'Comparison table',
             'Train RMSE, test RMSE, test R², non-zero count and chosen α for all four '
             'estimators. State explicitly whether the differences exceed resampling noise.'],
            ['3', 'Deployment recommendation',
             '300 words. Which estimator ships for Mumbai and why. Name the constraint that '
             'decides it — interpretability, stability, latency or accuracy — and be honest if '
             'accuracy is not the deciding factor.'],
            ['4', 'Notebook',
             'Runnable top to bottom, seeded, with your from-scratch implementations intact.']],
           [0.04, 0.24, 0.72])]

    B += [('h2', 'Marking rubric')]
    B += [('table', ['Band', 'Looks like'],
           [['Distinction (70+)',
             'Correct implementations, with a write-up that explains mechanisms rather than '
             'restating outputs. Reports at least one result that contradicted expectation and '
             'investigates it. Notices the censoring at ₹150, or the non-linear 50/(S+1) term, '
             'unprompted. Distinguishes "these models differ" from "these models differ by more '
             'than noise".'],
            ['Merit (60–69)',
             'All five tasks complete and correct. Figures labelled and readable. '
             'Cross-validation on the training split only. Conclusions follow from the evidence.'],
            ['Pass (50–59)',
             'Estimators implemented and broadly correct; analysis thin or descriptive. May '
             'confuse training and test error, or quote α to spurious precision.'],
            ['Fail',
             'Library calls substituted for required from-scratch implementations; test set used '
             'to select hyperparameters; conclusions unsupported by the numbers shown.']],
           [0.20, 0.80])]

    B += [('callout', 'warn', 'Three mistakes that cost marks every year',
           '**1.** Cross-validating on all 200 trips and then reporting test RMSE from the same '
           '200 — the test set is now part of model selection and the number means nothing. '
           '**2.** Forgetting to standardize before a penalized fit, so `traffic_speed` '
           '(range ≈ 55) is penalized enormously more than `is_rain` (range 1). '
           '**3.** Reporting that Lasso "selected the important features" without checking '
           'selection stability — see Task 4 step 5.')]

    B += [('h2', 'Collaboration policy')]
    B += [('p', 'Discuss freely; write alone. Comparing numerical answers with a classmate is '
                'fine: if your Ridge path and theirs disagree, one of you has a bug and finding it '
                'is the exercise. Submitted code and prose must be your own. If you use an AI '
                'assistant, say so in one line at the top of the notebook and state what you '
                'asked it.')]
    return B


# =========================================================================== #
# THE SOLUTIONS
# =========================================================================== #
def solutions():
    t1, t2, t3, t4, t5, bn, cm = (R['task1'], R['task2'], R['task3'], R['task4'],
                                  R['task5'], R['bonus'], R['comparison'])
    B = []
    B += [('title', 'Model Solutions and Marking Notes',
           f'{COURSE} · instructor copy')]
    B += [('callout', 'warn', 'Instructor copy',
           'Every number below comes from `scripts/analysis.py`, the single source of truth for '
           'the website, the notebooks and this document. Re-running it after a change to the '
           'data regenerates all three consistently. Student answers may differ in the last '
           'decimal depending on solver tolerances, but not in sign, magnitude or conclusion.')]

    # ---- Task 1 ----
    B += [('h1', 'Task 1 · Maximum likelihood')]
    B += [('h2', '1a · is_rain and traffic_speed on six trips')]
    a6 = t1['pairA_n6']
    B += [('p', f"Correlation **r = {n(a6['corr'], 4)}**, determinant of XᵀX = "
                f"**{n(a6['det'], 4)}**, condition number **{n(a6['cond'], 1)}**. The estimate is "
                f"ŵ = ({n(a6['w'][0], 3)}, {n(a6['w'][1], 3)}).")]
    B += [('table', ['n', 'r', 'det XᵀX', 'w · is_rain', 'w · traffic_speed', 'max move'],
           [[r['n'], n(r['corr'], 3), n(r['det'], 2),
             n(r['w'][0], 2), n(r['w'][1], 2),
             '—' if r['jump_from_previous'] is None else n(r['jump_from_previous'], 2)]
            for r in t1['pairA_instability']],
           [0.08, 0.14, 0.16, 0.22, 0.24, 0.16])]
    B += [('callout', 'ok', 'Marking note',
           f"The largest one-row move is **₹{n(t1['pairA_max_jump'], 2)} per standard deviation**, "
           'and the coefficients are still oscillating at n = 15. Award marks for an explanation '
           'that ties the swing to the determinant and the −r off-diagonal of the inverse. '
           'Observing only that the numbers move does not answer the question. Noting that the '
           '*sum* of the two coefficients is far more stable than either individually goes beyond '
           'what was asked and should be rewarded.')]
    B += [('p', f"Prediction for trip 2 from the six-row fit: **₹{n(t1['pred_trip2'], 2)}** "
                f"against an actual charge of ₹{n(t1['actual_trip2'], 0)}. Close, but that is "
                'luck: the model was fitted on six points, two of which were at the cap.')]

    B += [('h2', '1b · is_rain and is_bad_weather on ten trips')]
    B += [('p', 'On the first ten trips **every rainy trip also had traffic below 25 km/h**, so '
                'the two columns are identical. XᵀX is the all-ones matrix scaled by n, its '
                f"determinant is exactly 0, and `np.linalg.inv` raises `LinAlgError`. The problem "
                f"first becomes solvable at **n = {t1['pairB_first_solvable_n']}**, when a rainy "
                'trip with traffic above 25 km/h finally arrives.')]
    B += [('callout', 'ok', 'The answer being looked for in step 3',
           'The likelihood surface has a flat direction: adding c to one coefficient and '
           'subtracting c from the other changes no prediction and therefore no likelihood. There '
           'is no unique maximum, so "the MLE" is not well defined — a property of the model and '
           'data, not of the software. Ridge returns an answer because adding λI lifts the zero '
           'eigenvalue to λ > 0, breaking the tie in favour of the minimum-norm solution.')]

    B += [('h2', '1c · Full eight-feature OLS')]
    B += [('p', f"Condition number **{n(t1['full_cond'], 1)}** — well conditioned. "
                f"Train RMSE ₹{n(t1['train_rmse'], 2)}, test RMSE ₹{n(t1['test_rmse'], 2)}, "
                f"test R² {n(t1['test_r2'], 3)}.")]
    B += [('table', ['Feature', 'Coefficient (₹ per s.d.)'],
           [[f'`{k}`', n(v, 2)] for k, v in t1['w_full'].items()], [0.5, 0.5])]

    # ---- Task 2 ----
    B += [('pagebreak',), ('h1', 'Task 2 · Ridge')]
    B += [('table', ['α (sklearn scale)', 'w·is_peak', 'w·is_rain', '|gap|',
                     'Train RMSE', 'Test RMSE', 'Max diff vs sklearn'],
           [[g['alpha'], n(g['coefs']['is_peak'], 2), n(g['coefs']['is_rain'], 2),
             n(g['gap_peak_rain'], 2), n(g['train_rmse'], 2), n(g['test_rmse'], 2),
             f"{g['max_abs_diff_vs_sklearn']:.1e}"] for g in t2['grid']],
           [0.16, 0.13, 0.13, 0.10, 0.14, 0.14, 0.20])]
    B += [('p', f"**The equal-split behaviour is visible in the `|gap|` column**: the distance "
                f"between the peak and rain coefficients falls monotonically as α grows. That is "
                'the answer to step 4 — there is no single α at which they "meet", they converge '
                'asymptotically, and a student who says so is more correct than one who reports a '
                'specific value.')]
    B += [('p', f"5-fold CV on the training split chooses **α = "
                f"{n(t2['cv_alpha_sklearn_equiv'], 2)}** on the scikit-learn scale "
                f"(= {t2['cv_alpha_normalised']} on the 1/2n-normalised scale), with CV RMSE "
                f"₹{n(t2['cv_rmse'], 2)} and test RMSE ₹{n(t2['cv_test_rmse'], 2)}.")]
    B += [('callout', 'ok', 'Step 6 — the prior versus the evidence',
           f"The assumed prior gives λ = {t2['prior_lambda']:.0f}, which yields test RMSE "
           f"₹{n(t2['prior_test_rmse'], 2)} — **{n(t2['prior_test_rmse'] - t2['cv_test_rmse'], 2)} "
           'rupees worse** than the CV choice. The prior τ = 1 asserts that coefficients are '
           'around ±1 rupee per standard deviation; the data says they are around ±10. The prior '
           'is simply wrong, and cross-validation is detecting that. Full marks for saying the '
           'prior is mis-specified and that with 150 observations the likelihood should be '
           'allowed to dominate; partial marks for choosing CV without explaining why.')]

    # ---- Task 3 ----
    B += [('pagebreak',), ('h1', 'Task 3 · Lasso')]
    B += [('h2', '3a · Agreement with scikit-learn')]
    B += [('table', ['α', 'Sweeps to converge', 'Max |difference|', 'Non-zero (mine)',
                     'Non-zero (sklearn)'],
           [[c['alpha'], c['iters'], f"{c['max_abs_diff']:.1e}",
             c['nonzero_mine'], c['nonzero_sklearn']] for c in t3['sklearn_check']],
           [0.12, 0.22, 0.24, 0.21, 0.21])]
    B += [('p', 'Agreement to roughly 1e−7, which is the coordinate-descent tolerance rather than '
                'a modelling difference. Students who report a mismatch have almost always used '
                '‖y−Xw‖² instead of (1/2n)‖y−Xw‖², a factor of 2n.')]
    B += [('p', '**Elimination order as α grows:** ' +
                ' → '.join(f'`{x}`' for x in t3['elimination_order']) + '.')]
    B += [('callout', 'ok', 'Marking note',
           '`is_weekend` should die first — its true coefficient is exactly zero by construction. '
           'That `traffic_speed` and `is_peak` survive longest is the expected mirror image: they '
           'have the strongest marginal correlations with surge.')]

    B += [('h2', '3b · New city, p ≫ n')]
    B += [('table', ['α', 'Method', 'Selected', 'True found', 'False pos.', 'Precision', 'Missed'],
           [[r['alpha'], r['method'], r['selected'], f"{r['true_found']}/6", r['false_pos'],
             n(r['precision'], 2),
             ', '.join(r['missed']) if r['missed'] else '—']
            for r in t3['new_city'] if r['method'] in ('lasso', 'enet90')],
           [0.08, 0.13, 0.12, 0.13, 0.12, 0.13, 0.29])]
    best = t3['new_city_best_lasso']
    fm = t3['new_city_first_missed']
    B += [('p', f"**Best Lasso operating point: α = {best['alpha']:.0f}** — "
                f"{best['selected']} features selected, all {best['true_found']} real drivers "
                f"recovered, {best['false_pos']} false positive, precision "
                f"{n(best['precision'], 2)}.")]
    if fm:
        drv = fm['missed'][0]
        dec = t3['new_city_decoys'].get(drv, {})
        B += [('callout', 'ok', 'Step 4 — the answer',
               f"At α = {fm['alpha']:.0f} Lasso drops **`{drv}`**, a genuine driver, while "
               f"retaining **`{dec.get('decoy', '?')}`**, which is pure noise correlated with it "
               f"at **r = {dec.get('r', float('nan')):.3f}**. The ℓ₁ penalty charges per non-zero "
               'coefficient and the two features carry nearly the same information, so paying for '
               'both is wasteful — but nothing in the objective prefers the *real* one. This is '
               'the selection-instability caveat made concrete, and a student who frames it as '
               '"Lasso made a mistake" has missed the point: the estimator did exactly what it '
               'was asked to do.')]
    enet_rows = [r for r in t3['new_city'] if r['method'] == 'enet' and r['alpha'] in (6.0, 20.0)]
    if enet_rows:
        B += [('p', '**Step 5 — the counter-intuitive result.** Elastic Net at ρ = 0.5 selects '
                    + ' and '.join(f"{r['selected']} features at α = {r['alpha']:.0f}"
                                   for r in enet_rows)
                    + ' — recall is perfect but precision collapses, because the ℓ₂ term actively '
                      'resists setting coefficients to zero. For *pure selection* in p ≫ n, Lasso '
                      'or a high ρ is the better tool. Award full marks for reporting this '
                      'honestly; the grouping effect is a virtue only when the correlated partner '
                      'is real, which here it is not.')]
    B += [('p', f"**Step 6:** Lasso can select at most n = {t3['lasso_bound']} features. The "
                'largest selection observed is well below that, so the bound never binds here.')]

    # ---- Task 4 ----
    B += [('pagebreak',), ('h1', 'Task 4 · Elastic Net and the grouping effect')]
    B += [('p', f"`is_bad_weather` correlates with `is_rain` at **{n(t4['corr_with_rain'], 3)}** "
                f"and with `traffic_speed` at **{n(t4['corr_with_traffic'], 3)}** — a genuine "
                'three-member group.')]
    B += [('h2', f"Step 3 · The three estimators at a common α = {t4['fixed_alpha']:.0f}")]
    fx = t4['at_fixed_alpha']
    B += [('table', ['Estimator', 'w·is_rain', 'w·traffic_speed', 'w·is_bad_weather',
                     'Non-zero', 'Zeroed'],
           [[m.capitalize(), n(fx[m]['group']['is_rain'], 2),
             n(fx[m]['group']['traffic_speed'], 2), n(fx[m]['group']['is_bad_weather'], 2),
             f"{fx[m]['nonzero']}/9",
             ', '.join(f'`{z}`' for z in fx[m]['zeroed']) or '—']
            for m in ('ridge', 'enet', 'lasso')],
           [0.15, 0.14, 0.18, 0.19, 0.11, 0.23])]
    B += [('callout', 'ok', 'All three predictions hold on this data',
           '**Ridge** keeps all nine features and spreads the weather signal across the three '
           'group members. **Lasso** zeroes `is_bad_weather` and loads its weight onto '
           '`traffic_speed`, which nearly doubles. **Elastic Net** keeps all three at moderate '
           'values while still zeroing `is_weekend`: the thing Ridge cannot do and Lasso does too '
           'aggressively.')]
    B += [('h2', 'Step 4 · ρ sweep')]
    B += [('table', ['ρ', 'w·is_rain', 'w·traffic_speed', 'w·is_bad_weather', 'Non-zero'],
           [[n(s['l1_ratio'], 1), n(s['is_rain'], 2), n(s['traffic_speed'], 2),
             n(s['is_bad_weather'], 2), f"{s['nonzero']}/9"] for s in t4['rho_sweep']],
           [0.12, 0.20, 0.24, 0.26, 0.18])]
    B += [('h2', f"Step 5 · Bootstrap selection frequency ({t4['n_boot']} resamples)")]
    bs = t4['bootstrap']
    feats = list(bs['lasso'].keys())
    B += [('table', ['Feature', 'Selected by Lasso', 'Selected by Elastic Net'],
           [[f'`{f}`', f"{bs['lasso'][f]:.0f}%", f"{bs['enet'][f]:.0f}%"] for f in feats],
           [0.42, 0.29, 0.29])]
    B += [('callout', 'ok', 'This table is the task',
           f"`is_bad_weather` is selected by Lasso in **{bs['lasso']['is_bad_weather']:.0f}%** of "
           f"bootstrap resamples and by Elastic Net in **{bs['enet']['is_bad_weather']:.0f}%**: the "
           'grouping effect, measured rather than asserted. Note `is_weekend` too — Lasso keeps it '
           f"{bs['lasso']['is_weekend']:.0f}% of the time, Elastic Net {bs['enet']['is_weekend']:.0f}%. "
           "Elastic Net's reluctance to zero things is a cost as well as a benefit, and a student "
           'who points that out has read the table properly.')]

    # ---- Task 5 ----
    B += [('pagebreak',), ('h1', 'Task 5 · The surge engine')]
    bkc = t5['bkc']
    B += [('table', ['Quantity', 'Value'],
           [['Demand / supply ratio', f"{bkc['demand_supply_ratio']}"],
            ['Raw model output', f"₹{n(bkc['raw_model_output'], 2)}"],
            ['Surge after policy', f"₹{n(bkc['surge'], 2)}"],
            ['Cap binding?', 'yes' if bkc['capped'] else 'no'],
            ['Base fare', f"₹{n(bkc['base_fare'], 2)}"],
            ['Final fare', f"₹{n(bkc['final_fare'], 2)}"],
            ['Driver should reposition?',
             'yes — surge exceeds the ₹50 threshold' if t5['driver_should_move'] else 'no']],
           [0.45, 0.55])]
    B += [('callout', 'ok', 'Step 3 — what the threshold assumes',
           'The "move if surge > ₹50" rule assumes the driver\'s cost of a 3 km deadhead — fuel, '
           'time, and the option value of staying where they are — is below ₹50, and that they '
           'will still be matched when they arrive. Neither is modelled. A good answer notices '
           'that the rule ignores the *probability* of getting the trip, which is precisely what '
           'falls as every other driver responds to the same signal.')]
    B += [('p', f"**Step 4 — supply sweep.** Surge switches off at "
                f"**{t5['threshold_crossing']} drivers**, where the ratio first falls below 1.5. "
                'A rider requesting at 16 drivers pays the full capped surge; at 17 drivers they '
                'pay nothing extra. A ₹150 discontinuity produced by one driver moving one block '
                'is very hard to defend, and the best answers propose a smooth ramp — or note '
                'that the estimated ratio has its own uncertainty, which a hard threshold ignores '
                'entirely.')]
    B += [('p', '**Step 5 — elasticity.** The data contains no experimental price variation: '
                'prices were set *by a policy that responded to demand*, so price and demand are '
                'simultaneously determined and the regression coefficient on price would be '
                'hopelessly confounded. Only a randomised experiment — holding out a fraction of '
                'cell-buckets from the surge policy, or randomising the multiplier within a '
                'narrow band — can identify it. Full marks require naming the endogeneity, not '
                'merely saying "we need more data".')]

    # ---- Bonus ----
    B += [('h1', 'Bonus · The full posterior')]
    B += [('p', f"MAP and Ridge agree to **{bn['max_abs_diff_map_vs_ridge']:.1e}** — machine "
                'precision. The posterior mode of a Gaussian is its mean, so this is an identity, '
                'not an approximation.')]
    B += [('table', ['Feature', 'Mean', '± 1.96 sd', '95% interval', 'Covers zero?'],
           [[f"`{c['feature']}`", n(c['mean'], 2), n(1.96 * c['sd'], 2),
             f"[{n(c['lo'], 2)}, {n(c['hi'], 2)}]", 'yes' if c['covers_zero'] else 'no']
            for c in bn['coefficients']],
           [0.24, 0.13, 0.15, 0.28, 0.20])]
    B += [('p', f"**{bn['n_covering_zero']} of 8 intervals contain zero.** Compare with Lasso, "
                'which set exactly that feature to zero at its CV-optimal α — the two methods '
                'agree about which feature carries no signal, but the posterior expresses it as '
                'a degree of belief rather than a hard deletion.')]
    B += [('p', f"BKC predictive quote: **₹{n(bn['bkc_pred'], 1)} ± ₹{n(bn['bkc_sd_total'], 1)}**, "
                f"i.e. a 95% interval of [{n(bn['bkc_interval'][0], 0)}, "
                f"{n(bn['bkc_interval'][1], 0)}]. Of the total variance, "
                f"**{n(bn['bkc_var_from_w'], 2)} comes from uncertainty in w and "
                f"{bn['bkc_var_noise']:.0f} is irreducible noise** — so roughly "
                f"{100 * bn['bkc_var_noise'] / (bn['bkc_var_noise'] + bn['bkc_var_from_w']):.0f}% "
                'of it cannot be removed by collecting more data.')]
    B += [('callout', 'warn', 'The sentence to refuse (step 6)',
           'The upper end of that interval exceeds the ₹150 cap, which means the honest quote '
           'would promise a price the policy layer forbids. Any sentence implying the rider might '
           'be charged above the cap should be rejected — as should any phrasing that presents '
           'the point estimate as a guarantee. A good answer also notices that showing a range '
           'invites the rider to wait and re-request, which changes their behaviour and therefore '
           'the demand the model was fitted on.')]

    # ---- Deliverable 2 ----
    B += [('h1', 'Deliverable 2 · The comparison table')]
    B += [('table', ['Model', 'α', 'Train RMSE', 'Test RMSE', 'Test R²', 'Non-zero', 'Dropped'],
           [[r['model'].upper(), '—' if r['alpha'] is None else n(r['alpha'], 4),
             n(r['train_rmse'], 2), n(r['test_rmse'], 2), n(r['test_r2'], 3),
             f"{r['nonzero']}/{cm['n_features']}",
             ', '.join(f'`{d}`' for d in r['dropped']) or '—'] for r in cm['rows']],
           [0.11, 0.12, 0.15, 0.14, 0.12, 0.13, 0.23])]
    B += [('callout', 'ok', 'The point of Deliverable 3',
           f"Best and worst held-out RMSE differ by **₹{n(cm['test_rmse_spread'], 2)}** on fares of "
           'a few hundred rupees. There is no statistically meaningful winner, and the strongest '
           'submissions say so and argue on operational grounds: Elastic Net for a short fare '
           'breakdown and stable explanations across refits; plain Ridge for one hyperparameter, a '
           'closed form and no feature silently vanishing after an overnight retrain. Penalise any '
           'answer that declares a winner on the third decimal of RMSE. Reward any that proposes '
           're-running the comparison across several resampled splits first.')]
    return B


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [
        (brief(), 'Regression_Assignment', 'Assignment — Building a Surge Pricing Engine', FOOT),
        (solutions(), 'Regression_Assignment_Solutions',
         'Model Solutions — Building a Surge Pricing Engine',
         'Regularized Regression · Model solutions · instructor copy'),
    ]
    for blocks, name, title, foot in jobs:
        d = to_docx(blocks, f'{OUT}/{name}.docx', footer_text=foot)
        p = to_pdf(blocks, f'{OUT}/{name}.pdf', title=title, footer_text=foot)
        print(f'  {os.path.basename(d):<45} {os.path.getsize(d) // 1024:>4} KB')
        print(f'  {os.path.basename(p):<45} {os.path.getsize(p) // 1024:>4} KB')


if __name__ == '__main__':
    main()
