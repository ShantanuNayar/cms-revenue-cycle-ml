# Predicting Revenue Leakage Risk in Medicare Claims

An end-to-end ML pipeline predicting high charge-to-payment variance in Medicare
claims, built on 9.8M rows of 2024 CMS Medicare Physician & Other Practitioners
data, loaded into Snowflake and modeled with XGBoost.

## Results
- **ROC-AUC: 0.871** (train/test gap 0.004 — minimal overfitting)
- High-risk recall 0.75 / precision 0.59, tuned toward recall via
  scale_pos_weight; the decision threshold is a business tradeoff discussed
  in notebook 03
- Facility claims are 2.5x more likely to be high-variance than office claims
  (40.6% vs 16.4%)
- Anesthesia/CRNA claims flag at 83–89% — driven by time-unit billing against
  a low Medicare conversion factor (structural, not error)

## Pipeline
1. **Ingestion** — 3GB CSV staged and loaded to Snowflake via SnowSQL (PUT/COPY INTO)
2. **EDA** (`notebooks/01_EDA.ipynb`) — distributions, leakage by provider type,
   place-of-service comparison, correlation analysis
3. **Feature engineering & selection** (`02`, `03`) — proxy target construction;
   feature-vs-target correlation as a leakage tripwire; mutual information
   ranking; lineage-based exclusion of post-adjudication features
4. **Modeling** (`notebooks/03_Model.ipynb`) — stratified split, XGBoost with
   class weighting, ROC/PR evaluation, feature importance

## Honest limitations
- **The label is a proxy.** Public CMS data contains no denial flag; the target
  is the top quartile of charge-to-payment ratio. It captures contractual
  adjustment as well as true denials — anesthesia's dominance reflects
  Medicare's fee schedule working as designed, not recoverable leakage.
- **Target leakage was found and removed.** Early candidates (payment ratio,
  leakage rate) were algebraically derived from the label and excluded via
  data-lineage review after correlation checks proved insufficient against a
  thresholded target.
- **With production data (Epic 837/835),** the same pipeline would model actual
  CARC/RARC denial codes at claim level with pre-submission features, making
  the output operationally actionable.

## Stack
Snowflake · SnowSQL · Python (pandas, scikit-learn, XGBoost) · Jupyter

## Reproducing
Requires a Snowflake account with the CMS dataset loaded (schema in notebook 01).
Credentials are prompted at runtime — never stored.
