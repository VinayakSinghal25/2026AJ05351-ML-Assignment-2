# Predicting Student Dropout and Academic Success

**BITS Pilani WILP · M.Tech (AIML) · Machine Learning: Assignment 2**
**Student:** Vinayak Singhal (2026AJ05351)

- **GitHub repository:** https://github.com/VinayakSinghal25/2026AJ05351-ML-Assignment-2
- **Live Streamlit app:** https://2026aj05351-ml-assignment-2.streamlit.app

---

## a. Problem statement

Higher-education institutions lose a large share of students before graduation, and the signs are often visible
early: academic performance in the first two semesters, tuition-fee status, scholarship, age at enrolment, and so on.
The goal of this project is to predict, **from information available at enrolment and after the first year**,
whether a student will end up as a **Dropout**, still **Enrolled**, or a **Graduate**, so that at-risk
students can be identified for timely support.

This is a **multi-class (3-class) classification** problem. Five classifiers are trained on the same data, compared
on six evaluation metrics, and served through an interactive Streamlit app that scores the saved models on
uploaded test data.

## b. Dataset description

| | |
|---|---|
| **Name** | Predict Students' Dropout and Academic Success |
| **Source** | UCI Machine Learning Repository, dataset id 697 (https://archive.ics.uci.edu/dataset/697) |
| **Origin** | Polytechnic Institute of Portalegre, Portugal (undergraduate degrees, 2008/09 – 2018/19) |
| **Instances** | 4,424 students |
| **Features** | 36 (all numeric-coded; no missing values, no duplicate rows) |
| **Target** | `Target`: Dropout / Enrolled / Graduate |

**Class distribution** (imbalanced):

| Class | Count | Share |
|---|---|---|
| Graduate | 2,209 | 49.9 % |
| Dropout | 1,421 | 32.1 % |
| Enrolled | 794 | 17.9 % |

**Feature groups**

| Group | Examples |
|---|---|
| Demographic | marital status, nationality, gender, age at enrolment, displaced, international |
| Socio-economic | mother's / father's qualification and occupation, debtor, tuition fees up to date, scholarship holder |
| Academic path at entry | application mode and order, course, daytime/evening attendance, previous qualification (and grade), admission grade |
| 1st & 2nd semester performance | curricular units credited / enrolled / evaluated / approved / grade / without evaluations |
| Macro-economic | unemployment rate, inflation rate, GDP |

**Preprocessing**
- Column names cleaned (a stray tab in one header; `Nacionality` renamed to `Nationality`).
- 9 *nominal* columns stored as integer codes (marital status, application mode, course, previous qualification,
  nationality, parents' qualification and occupation) are **one-hot encoded**; codes seen fewer than 10 times are
  grouped into an "infrequent" bucket.
- The remaining 27 numeric / binary columns are **standardised** (`StandardScaler`).
- Stratified **80 / 20 train-test split** (`random_state = 42`): 3,539 training rows, 885 test rows.
- Encoder, scaler and classifier are saved together as one scikit-learn `Pipeline` per model, so the app can
  score raw CSV rows directly.

`test_data.csv` in this repository is exactly the 885-row held-out test split (36 features + `Target`).

## c. Models used and evaluation

All five models are trained on the same training split and evaluated on the same held-out test split.

| # | Model | Key settings |
|---|---|---|
| 1 | Logistic Regression | multinomial, `C = 1.0`, `max_iter = 3000` |
| 2 | Decision Tree | `max_depth = 7`, `min_samples_leaf = 10` |
| 3 | K-Nearest Neighbours | `k = 15`, distance-weighted |
| 4 | Naive Bayes (Gaussian) | `var_smoothing = 0.1` (the default `1e-9` collapses on near-constant one-hot columns) |
| 5 | Support Vector Machine | RBF kernel, `C = 1.0`, `probability = True` |

Because the problem has three classes: **AUC** is one-vs-rest, weighted by class size; **Precision, Recall and F1**
are weighted averages over the three classes; **MCC** is the multi-class Matthews correlation coefficient.

### Comparison table (test set, 885 students)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7740 | 0.8996 | 0.7612 | 0.7740 | 0.7636 | 0.6253 |
| Decision Tree | 0.7333 | 0.8685 | 0.7374 | 0.7333 | 0.7333 | 0.5649 |
| KNN | 0.7232 | 0.8548 | 0.7136 | 0.7232 | 0.6997 | 0.5403 |
| Naive Bayes | 0.6893 | 0.8302 | 0.6881 | 0.6893 | 0.6853 | 0.4879 |
| Support Vector Machine | 0.7582 | 0.8988 | 0.7459 | 0.7582 | 0.7460 | 0.5986 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | _to be written_ |
| Decision Tree | _to be written_ |
| KNN | _to be written_ |
| Naive Bayes | _to be written_ |
| Support Vector Machine | _to be written_ |
| **Overall winner for this dataset** | _to be written_ |

---

## Repository structure

```
2026AJ05351-ML-Assignment-2/
├── app.py                  # Streamlit app: loads saved models, scores uploaded test CSV
├── train_models.ipynb      # data loading, preprocessing, training, evaluation, saving
├── requirements.txt        # pinned dependencies for Streamlit Community Cloud
├── README.md
├── test_data.csv           # held-out test split (upload this to the app)
├── data/
│   └── data.csv            # original UCI dataset (semicolon-separated)
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── support_vector_machine.pkl
│   ├── metadata.json       # feature list, class names, model file names
│   └── test_metrics.csv    # the comparison table above
└── .streamlit/config.toml  # app theme
```

## Streamlit app features

- **CSV upload** of test data (comma- or semicolon-separated); the bundled `test_data.csv` is shown by default and can be downloaded from the sidebar.
- **Model selection** dropdown for the five classifiers.
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
- **Confusion matrix** and **classification report** for the selected model.
- **Compare all models** tab with the best score per metric highlighted, plus a metric chart.
- **Predictions** tab with actual vs predicted outcome per student, a misclassified-only filter and CSV download.

The app never retrains: it only loads the pipelines from `model/` and evaluates them on the uploaded rows.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain the models, run `train_models.ipynb` top to bottom. It reads `data/data.csv` (or downloads it from UCI)
and rewrites `model/` and `test_data.csv`. The notebook was executed on the BITS Virtual Lab (Python 3.12,
scikit-learn 1.7.2) and produced the metrics shown above.
