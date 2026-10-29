# ModelScout

**ModelScout** is a Streamlit analytics tool that recommends a machine-learning
model family for a tabular dataset — without training every model.  Upload a CSV,
choose a target column, and get:

- A recommended model family with predicted score range and runtime estimate
- An "Is ML worth it?" verdict
- A quick verification run (real cross-validation on the top 2 models)
- Plain-language explanations of the recommendation

---

## How to Run

### 1. Install dependencies

```bash
cd modelscout
pip install -r requirements.txt
```

Optional (install for better coverage):

```bash
pip install xgboost shap
```

### 2. Generate the sample dataset

```bash
python scripts/make_sample.py
```

### 3. Start the app

```bash
streamlit run app/streamlit_app.py
```

Open <http://localhost:8501> in your browser.

---

## How the Prototype Recommender Works

The recommender currently uses a **transparent rule-based heuristic**.
It does NOT train every model on your full dataset.  Instead, it:

1. **Extracts meta-features** — simple statistics about your dataset (rows, columns,
   missing %, cardinality, skewness, pairwise correlation, class balance).

2. **Runs quick landmarkers** — trains five simple models (decision stump,
   Gaussian Naive Bayes, 1-NN, logistic/linear regression, dummy) on a 10%
   stratified subsample (min 200 rows, 3-fold CV).  This takes a few seconds
   and gives a cheap signal of dataset difficulty.

3. **Applies documented rule adjustments** — a fixed table in `src/recommender.py`
   maps meta-feature conditions to score adjustments per model.  Every rule has
   a short rationale string that appears in the "Why" column of the ranked table.

4. **Computes a worth-it verdict** — compares the best expected score against
   `max(dummy_score, linear_score)`.  If the relative improvement is
   below 2% it is "Not worth it"; 2–5% is "Marginal"; above 5% is "Recommended".
   The thresholds are constants in `src/recommender.py → WORTH_IT_MARGINS`.

The UI always shows a label in the top-right corner:
`Recommender: rule-based prototype (learned meta-model pending)`.

---

## How to Swap in the Trained Meta-Model

When a trained meta-model is ready:

1. Save it as `models/meta_model.joblib` (use `joblib.dump`).

2. The model will be detected automatically on next startup.  The UI label
   switches to `Recommender: learned meta-model`.

3. **Expected interface** — `_recommend_learned` in `src/recommender.py` is the
   integration point.  Implement it to call your model like:

   ```python
   # meta is the dict returned by compute_meta_features()
   # It contains the ~30 keys listed in src/meta_features.py FEATURE_CATALOG
   scores_per_model = meta_model.predict(feature_vector)
   ```

   The function must return a `pd.DataFrame` with the same columns as the
   rule-based output:
   `model, est_score, score_low, score_high, est_fit_time_s, rank, reason, confidence`.

---

## File Structure

```
modelscout/
  app/
    streamlit_app.py    # layout and page flow
    theme.py            # CSS injection and Plotly template
  src/
    meta_features.py    # simple, statistical and landmarker features
    recommender.py      # rule-based now, learned model hook
    verify.py           # quick CV run of top-2 models
    explain.py          # permutation importance + plain-language text
    preprocess.py       # shared ColumnTransformer
  data/
    sample_churn.csv    # generated sample dataset (synthetic)
  models/               # put meta_model.joblib here
  scripts/
    make_sample.py      # generates data/sample_churn.csv
  requirements.txt
  README.md
```

---

## Known Limitations

- **No hyperparameter tuning.** All models use default hyperparameters.
  Real-world performance will differ after tuning.
- **Rule-based estimates are approximate.** Score ranges are derived from quick
  landmarkers and heuristic adjustments, not from exhaustive grid search.
- **Landmarkers use a 10% subsample.** For very small datasets this may be
  unreliable.
- **Feature importance uses permutation importance** on a held-out split of the
  best verified model, not SHAP (SHAP requires the trained meta-model path).
- **No time-series or multi-label support.** ModelScout is designed for standard
  tabular classification and regression.
- **The verification time budget is approximate.** Wall-clock times vary with
  hardware and Python environment.

---

## DSML Project Notes

This tool was built as a prototype for demonstrating meta-learning concepts
in a DSML course project.  The learned meta-model training pipeline is a
future work item.  The current rule-based recommender is intentionally
transparent and documented so that its reasoning can be audited.
