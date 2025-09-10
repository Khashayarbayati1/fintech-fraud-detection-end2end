# Calibration Report

- Run ID: `2025_09_08_14_07_49`
- Split: 80% cal_train / 20% cal_holdout (by time)
- Brier (holdout): raw=0.07647, isotonic=0.02401, platt=0.02557
- ECE (holdout, 10 uniform bins): raw=0.16712, isotonic=0.00322, platt=0.01027
- Top-K% precision (holdout, K in {0.5,1,2,5}):

  - raw: 0.5%=0.908, 1%=0.932, 2%=0.810, 5%=0.475
  - isotonic: 0.5%=0.908, 1%=0.932, 2%=0.810, 5%=0.476
  - platt: 0.5%=0.908, 1%=0.932, 2%=0.810, 5%=0.475

- Chosen calibrator: **isotonic**
- Figures: `reliability_iso.png`, `reliability_platt.png`, `reliability_overlay.png`
<p style="display:flex;gap:10px;flex-wrap:wrap;">
  <img src="reliability_iso.png" alt="Isotonic" width="320"/>
  <img src="reliability_platt.png" alt="Platt" width="320"/>
  <img src="reliability_overlay.png" alt="Overlay" width="640"/>
</p>
- Notes:
  - Monotone calibration preserves ranking (ROC/PR unchanged up to ties).
  - Use calibrated probabilities for policy thresholds, SLAs, and cost curves.
