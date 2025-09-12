# Operating Points — Run `2025_09_08_14_07_49`

**Calibrator:** `isotonic`  
**Window:** `full` (unique days: `42`)

**Files:**  
- Operating points (CSV): `baseline_operating_points_isotonic_calibrated_2025_09_08_14_07_49.csv`  
- ROC/PR (calibrated): `../figures/roc_pr_calibrated_run_2025_09_08_14_07_49.png`

## Columns (quick refresher)

- **mode** — how the cut is chosen:
  - `fpr`: meet a false-positive-rate **budget** (percent of negatives you’re allowed to flag).
  - `topk`: take the top **K%** of rows by risk.
- **budget_type / budget_value** — the target (FPR % or K %).
- **threshold_prob** — calibrated probability cutoff **τ**; predict 1 if `p_cal ≥ τ`.
- **precision** = TP / (TP+FP)
- **recall** = TP / (TP+FN)
- **fpr** = FP / (FP+TN)  _(fraction; e.g., 0.005 = 0.5%)_
- **alerts** — number of rows flagged at the cut
- **alerts_per_day** — `alerts / 42` (unique days in this window)

## FPR-budget rows (risk budget view)

You asked for the **largest τ** such that **FPR ≤ budget**.

| budget (FPR %) | threshold (p_cal) | precision | recall | FPR (fraction) | alerts | alerts/day |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1% | 0.818182 | 0.9017 | 0.2143 | 0.000833 | 966 | 23.00 |
| 0.5% | 0.396907 | 0.7256 | 0.3708 | 0.004998 | 2077 | 49.45 |
| 1% | 0.277419 | 0.6120 | 0.4254 | 0.009610 | 2825 | 67.26 |
| 2% | 0.203810 | 0.4916 | 0.4980 | 0.018353 | 4117 | 98.02 |

**Interpretation.** As you **loosen the FPR budget** (e.g., 0.1% → 2%), the **threshold drops**,
**recall increases** (catch more positives), **precision falls** (more false alarms), and **alerts** go up.
All reported FPR values are **fractions** and should be ≤ the chosen budget (in percent) converted to a fraction.

## Top-K% rows (capacity view)

Top **K%** by calibrated probability using a deterministic tie-break (`p_cal ↓, time ↑, id ↑`).

| K% | threshold (p_cal) | precision | recall | FPR (fraction) | alerts | alerts/day |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5% | 0.842742 | 0.9306 | 0.1353 | 0.000360 | 591 | 14.07 |
| 1% | 0.680851 | 0.8883 | 0.2584 | 0.001157 | 1182 | 28.14 |
| 2% | 0.305031 | 0.6780 | 0.3942 | 0.006673 | 2363 | 56.26 |
| 5% | 0.153705 | 0.3957 | 0.5750 | 0.031295 | 5906 | 140.62 |

**Interpretation.** As **K grows** (e.g., 0.5% → 5%), you review more cases: **precision decreases**,
**recall increases**, **FPR** grows, and the **threshold** is the calibrated probability at the Nₖ-th ranked row.

## ROC & PR (calibrated)

![ROC/PR](../figures/roc_pr_calibrated_run_2025_09_08_14_07_49.png)

## Quick sanity checks

- Monotonicity (FPR budgets): recall ↑, threshold ↓, alerts ↑ … ✔️
- Monotonicity (Top-K): precision ↓, recall ↑, FPR ↑, threshold ↓ … ✔️
- Units: budgets are **percent**, `fpr` column is a **fraction**.
- Day math: `alerts_per_day = alerts / 42`.
- Calibration: thresholds are probabilities in [0,1].

## How to use this

- **Risk budgeted** (e.g., “allow ~0.5% FPR”): pick the **0.5** row and set **τ** to its `threshold_prob`.
- **Capacity constrained** (e.g., “review ~X/day”): pick the **Top-K%** row whose `alerts/day ≈ X` and set **τ** to that row’s `threshold_prob`.
