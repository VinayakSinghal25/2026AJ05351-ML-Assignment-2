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
| Logistic Regression | The best model here. It has the highest score on all six metrics (Accuracy 0.774, AUC 0.900, F1 0.764, MCC 0.625). The strongest clues in this data (how many subjects a student passed and their grades in the 1st and 2nd semester) go up or down fairly steadily with the final outcome, and a simple linear model captures that well. It correctly finds 214 of the 284 dropouts. Like every model, it finds the "Enrolled" group hard (only 65 of 159 correct). |
| Decision Tree | Middle of the table (Accuracy 0.733, MCC 0.565). Its big plus is that its rules are easy to read and explain. Limiting the depth to 7 stops it from simply memorising the training data. It is the best of the five at spotting "Enrolled" students (75 of 159), but it gets more Graduates and Dropouts wrong than Logistic Regression. Its AUC (0.869) is lower because a tree gives rough, step-like probabilities. |
| KNN | Accuracy 0.723 but a weaker F1 (0.700) and MCC (0.540). It leans heavily towards the biggest class: it gets 421 of 442 Graduates right but only 41 of 159 Enrolled students, the worst of all models for that group. After one-hot encoding there are many columns, so "distance" between students becomes less meaningful, which hurts KNN. |
| Naive Bayes | The weakest model (Accuracy 0.689, MCC 0.488). It assumes every feature is independent and bell-shaped, which is not true here: 1st- and 2nd-semester results are closely linked, and many columns are just 0/1. It only worked properly after raising `var_smoothing` to 0.1; with the default setting it scored about 28 % accuracy. It is very fast and simple, but the least reliable. |
| Support Vector Machine | A close second (Accuracy 0.758, MCC 0.599). Its AUC (0.899) is almost the same as Logistic Regression, so it ranks students by risk just as well, but its final class choices are slightly less accurate. It handles many features well, but it is slower to train and harder to explain. |
| **Overall winner for this dataset** | **Logistic Regression.** It is the top model on every metric, and it is also the simplest, fastest and easiest to explain. SVM is a close second. Across all models the "Enrolled" class is the hardest to predict, because these students sit between dropping out and graduating and look similar to both groups. |

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
