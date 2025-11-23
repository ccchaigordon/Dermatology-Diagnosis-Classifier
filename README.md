# Dermatology Diagnosis Classifier

End-to-end **multi-class classification** on the UCI Dermatology dataset with a **reproducible, leak-free** scikit-learn pipeline: data cleaning, feature scaling/selection, hyperparameter tuning, and multi-metric evaluation (accuracy, weighted precision/recall/F1) with confusion matrices and a side-by-side model comparison.

- 📓 Notebook: [dermatology.ipynb](sandbox:/mnt/data/dermatology.ipynb)
- 🧰 Tech: Python, pandas, numpy, scikit-learn, matplotlib, seaborn

## ✨ Highlights

- **Clean ingest & prep**: parses dataset, coerces types, handles `?` missing values; **median-imputes `age`**.
- **Leak-free splits**: Train/Val/Test = **70/10/20** with fixed seed; all transformers fit on **train only**.
- **Feature selection** (2 paths):

  - **RFE** (tree-based) → top-10 features (for KNN).
  - **Tree feature importance** → top-10 features (for Decision Tree).

- **Modeling & tuning**:

  - **KNN** on scaled RFE features (grid over neighbors/weights/distance).
  - **Decision Tree** on importance-selected features (grid over criterion/depth/min samples).
  - **GridSearchCV (5-fold)** for both.

- **Evaluation**: Accuracy + **weighted** precision/recall/F1 on **validation** and **held-out test**; confusion matrices; consolidated results table for easy comparison.
- **Artifacts**: correlation heatmap, selected-feature lists, best estimators & params, tidy tables ready for dashboards.

---

## 📂 Project Structure

```
.
├─ .gitignore
├─ dataset.csv
├─ dermatology.ipynb
└─ README.md
```

---

## 🚀 Quickstart

1. **Install dependencies**

   Run the cell

   ```bash
   %pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. **Run the notebook**

   - Execute cells in order: Load & Clean → Split → Scale/Select → Train/Tune → Evaluate → Compare.

3. **Customize**

   - Tweak CV grids, number of selected features, or random seed to explore bias/variance trade-offs.

---

## 🔁 Reproducible Pipeline (Overview)

1. **Ingest & Clean**

   - Parse CSV, handle `?` → NaN, type coercion.
   - **Impute** `age` with train-fit median.

2. **Split**

   - **70/10/20** (Train/Val/Test, `random_state=42`).
   - No leakage: fit **Imputer/Scaler/Selectors** on **train**; transform val/test.

3. **Scale & Select**

   - **StandardScaler** for scale-sensitive models.
   - **RFE** (tree) and **tree importances** to get compact top-10 feature sets.

4. **Model & Tune**

   - **KNN** grid: `n_neighbors`, `weights`, `metric`.
   - **DecisionTree** grid: `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
   - **GridSearchCV (cv=5)** for robust selection.

5. **Evaluate & Compare**

   - Metrics: **accuracy**, **weighted precision/recall/F1** (class imbalance aware).
   - Confusion matrices for error patterns.
   - Consolidated results DataFrame (val/test) for side-by-side comparison.

---

## 📊 Outputs

- **EDA**: correlation heatmap of features.
- **Feature sets**: top-10 from RFE and from tree importances.
- **Best models**: tuned KNN and Decision Tree with chosen hyperparameters.
- **Reports**: validation/test metrics & confusion matrices; comparison table.

---

## 🧭 Notes & Reproducibility

- Fixed random seed: `random_state=42`.
- All transformations fit on **train only** and applied to val/test.
- Safe divisions & metric guards to handle edge cases.

---

## ➕ Coming Soon

- Logistic Regression with regularization and RandomForest/XGBoost for benchmarking.
- Probability calibration (Platt/Isotonic); per-class PR curves.
- Stratified CV with confidence intervals on test metrics.
- Export results as CSV for BI tools (Tableau).

---
