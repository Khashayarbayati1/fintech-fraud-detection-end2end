# Baseline Experiments

This folder tracks model baselines and their evaluation results.  
Each baseline is fully reproducible with pinned preprocessing, parameters, and splits.  

---

## Baseline v0 — LightGBM + Frozen Preprocessor


### 1. Data

- Source: `data/interim/train_clean_baseline.parquet` (md5: `<auto-filled>`)
- Split:  
  - `time_column`: `<dt/TransactionDT>`  
  - `t_cut`: `<YYYY-MM-DD HH:MM:SS>`  
- Validation set prevalence: **3.44%** (fraud share among validation rows)

This is a highly imbalanced dataset — only ~3.4% of transactions in validation are fraud. Any model evaluation must be done with metrics that account for imbalance.

---

### 2. Model

- Algorithm: **LightGBMClassifier**
- Parameters:  
  - `n_estimators = 5000` (large cap, early stopping governs stopping point)  
  - `early_stopping_rounds = 100`  
  - `learning_rate = 0.05`  
  - `num_leaves = 31`  
  - `min_data_in_leaf = 100`  
  - `feature_fraction = 0.9`  
  - `bagging_fraction = 0.8`  
  - `bagging_freq = 1`  
  - `scale_pos_weight = 27.46`  
  - `random_state = 42`

Parameters were chosen for stability and imbalance handling (`scale_pos_weight`). Early stopping ensures we don’t overfit despite a large cap on trees.

---

### 3. Validation Metrics

- **ROC-AUC:** 0.9115  
- **PR-AUC:** 0.5244  
  - No-skill baseline (positive prevalence): **0.0344**

#### Thresholded performance
- At **threshold = 0.5**  
  - Precision = 0.230  
  - Recall = 0.737  
  - F1 = 0.351  
- At **best-F1 threshold = 0.8113**  
  - Precision = 0.551  
  - Recall = 0.463  
  - F1 = 0.503  

- The model provides excellent ranking (ROC-AUC > 0.91) and strong lift over prevalence (PR-AUC = 0.52 vs 0.03 baseline).  
- At the default 0.5 threshold, the model catches most frauds (recall ~74%) but at the cost of many false positives (precision only ~23%).  
- Raising the threshold to ~0.81 balances things: precision ~55% with recall ~46%. This illustrates the **precision–recall tradeoff** and why threshold choice must align with business cost/benefit.

---

### 4. Curves

- ROC Curve: `data/interim/figures/roc_val.png`  
- Precision–Recall Curve: `data/interim/figures/pr_val.png`  
- Precision@K Curve: `data/interim/figures/precision_at_k_val.png`  
- Recall@FPR Curve: `data/interim/figures/recall_at_fpr_val.png`
### Curves

<p float="left">
  <img src="data/interim/figures/roc_val.png" width="45%" />
  <img src="data/interim/figures/pr_val.png" width="45%" />
</p>

<p float="left">
  <img src="data/interim/figures/precision_at_k_val.png" width="45%" />
  <img src="data/interim/figures/recall_at_fpr_val.png" width="45%" />
</p>

#### ROC Curve (AUC ≈ 0.911)
- **What it shows:** Trade-off between True Positive Rate (TPR/recall) and False Positive Rate (FPR) as the threshold moves.
- **How to read:** Curve far above the diagonal → strong ability to rank fraud > non-fraud across thresholds.
- **Why it matters:** Good global separability; however, ROC can look optimistic on highly imbalanced data (like ours), so pair with PR.

#### Precision–Recall (PR) Curve (PR-AUC ≈ 0.524; baseline ≈ 0.034)
- **What it shows:** Precision (purity of alerts) vs. recall (coverage of fraud) across thresholds.
- **Baseline:** The dashed line at **0.034** is the no-skill precision (fraud prevalence). Anything above it is real lift.
- **Our curve:** Stays well above baseline across recalls → the model concentrates frauds near the top of scores.
- **Operational take:** As we push recall higher, precision falls (more false alarms). Choose thresholds based on analyst capacity and tolerance for FPs.

#### Precision@Top-K% (triage view)
- **What it shows:** If we only review the **top K%** highest-risk transactions, what fraction are actual fraud?
- **Our results (approx.):**  
  - **Top 1%:** precision ~**0.9** → most reviewed cases are true fraud.  
  - **Top 2%:** precision ~**0.67**.  
  - **Top 5%:** precision ~**0.39**.
- **Operational take:** Lets ops pick K to match daily review bandwidth. High precision at small K enables a reliable “high-risk queue.”

#### Recall@FPR (customer-impact view)
- **What it shows:** With a cap on **FPR** (share of legit transactions incorrectly flagged), how much **recall** (fraud caught) do we get?
- **Our results (approx.):**  
  - **FPR 0.1%:** recall ~**0.23**  
  - **FPR 0.5%:** recall ~**0.37**  
  - **FPR 1.0%:** recall ~**0.43**  
  - **FPR 2.0%:** recall ~**0.50+**
- **Operational take:** Use when product sets an FP budget (e.g., “≤1% legit users impacted”); we then read off expected fraud coverage.

#### Putting it together
- **Model ranks well** (ROC-AUC ~0.91) **and delivers real lift** over prevalence (PR-AUC ~0.52 vs 0.034 baseline).
- **Threshold selection is key:**  
  - If review capacity is scarce → target **Precision@K** (e.g., top 1–2%).  
  - If customer friction must be minimal → target **Recall@FPR** (e.g., FPR ≤ 1%).  
  - If you want a single balanced point → use **best-F1** (we observed thr ≈ 0.81 → P≈0.55, R≈0.46).
- **Caveat:** Curves are from **validation** only; thresholds selected here should be re-validated on a held-out test or in a shadow deployment.

---

### 5. Time-Based Cross Validation

- Fold 0: ROC-AUC = 0.926, PR-AUC = 0.610, Best Iter = 442  
- Fold 1: ROC-AUC = 0.940, PR-AUC = 0.660, Best Iter = 1098  
- Fold 2: ROC-AUC = 0.925, PR-AUC = 0.554, Best Iter = 1220  

**Mean performance:**  
- ROC-AUC ≈ **0.930 ± 0.007**  
- PR-AUC ≈ **0.608 ± 0.043**  
- Details saved in: `data/interim/reports/cv_time_splits_v0.json`

The baseline model generalizes well across different time splits. ROC is consistently high; PR fluctuates more due to prevalence shifts — expected in fraud detection.

---

### 6. Feature Importance (Top 15 by Gain)

- V258, C13, V294, C8, card1, card2, TransactionAmt, M4, C1, card6, C14, V70, addr1, D2, C11

The model emphasizes card features, transaction amount, address, and time-derived/engineered V* and C* variables. These are plausible fraud signals — no obvious “nonsense” features dominating, which increases trust in the baseline.

---

### 7. Reproducibility

- Artifacts saved:  
  - `models/baseline_lgbm.pkl`  
  - `data/interim/reports/baseline_summary_v0.json`  
  - `data/interim/reports/eval_summary_v0.json`
- Random state pinned (`42`)  
- Preprocessor frozen (no re-fit on val)  
- No data leakage confirmed  

---


## Calibration & Operating Points (probability-space)

We calibrate scores to probabilities and report actionable thresholds.

- **Calibration report:** `data/interim/reports/calibration_report.md`
- **Operating points (per-run):**  
  `data/interim/reports/baseline_operating_points_isotonic_calibrated_2025_09_08_14_07_49.md`
- **Calibrated ROC/PR:**  
  `data/interim/figures/roc_pr_calibrated_run_2025_09_08_14_07_49.png`

<p float="left">
  <img src="data/interim/figures/roc_pr_calibrated_run_2025_09_08_14_07_49.png" width="70%" />
</p>

**Notes**
- Thresholds are now in **probability space** (e.g., τ = 0.92 means 92% fraud risk).
- **Top-K policy:** deterministic ordering `p_cal ↓, time ↑, id ↑`; exactly Nₖ alerts; predict 1 if `p_cal ≥ τ`.
- **FPR budgets:** we pick the **largest τ** with **FPR ≤ budget**; FPR reported as a **fraction** (e.g., 0.005 = 0.5%).
- **Alerts/day:** `alerts / (# unique days in the window)`.

---

## Quickstart: reproduce artifacts

```bash
# 0) setup metadata & run_id
python eval/phase_00_setup.py

# 1) Top-K (pre-calibration, on frozen Phase-0 file)
python eval/phase_01_topk_table.py

# 2) Recall@FPR (pre-calibration)
python eval/phase_02_recall_at_fpr_table.py

# 3) Calibration (fits isotonic + Platt, writes report & figures)
python eval/phase_03_calibration.py

# 4) Curves & operating points AFTER calibration (probability-space thresholds)
python eval/phase_04_curves_after_calibration.py
```

- Artifacts saved:  
  - Reports: data/interim/reports/
  - Figures: data/interim/figures/
  - Models (calibrator): models/
