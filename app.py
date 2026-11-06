"""Streamlit front end for ML Assignment 2: student dropout / academic success classifiers.

The app only *loads* the five pipelines trained in train_models.ipynb (model/*.pkl) and scores them on an
uploaded test CSV. Nothing is retrained here.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_TEST_CSV = APP_DIR / "test_data.csv"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Student Outcome Classifier", page_icon="🎓", layout="wide")


@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = {name: joblib.load(MODEL_DIR / file_name) for name, file_name in meta["models"].items()}
    return meta, models


def tidy_columns(df):
    # accept both the cleaned test_data.csv and a raw slice of the UCI file
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df.rename(columns={"Nacionality": "Nationality"})


def read_upload(uploaded):
    df = pd.read_csv(uploaded, sep=None, engine="python")  # sniff "," vs ";" (the UCI original uses ";")
    return tidy_columns(df)


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=list(range(n_classes)))
    except ValueError:
        auc = np.nan  # AUC is undefined when the upload is missing one of the classes
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def confusion_figure(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Actual outcome")
    fig.tight_layout()
    return fig


meta, models = load_artifacts()
CLASS_NAMES = meta["class_names"]
TARGET = meta["target"]
FEATURES = meta["feature_columns"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    model_name = st.selectbox("Choose a classifier", list(models), index=0)
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"],
                                help=f"Needs the {len(FEATURES)} feature columns and the '{TARGET}' column.")
    st.download_button("⬇️ Download sample test_data.csv", SAMPLE_TEST_CSV.read_bytes(),
                       file_name="test_data.csv", mime="text/csv")
    st.divider()
    st.caption(f"Dataset: {meta['dataset']}  \n"
               f"Models trained once in `train_models.ipynb` (scikit-learn {meta['sklearn_version']}) "
               "and loaded from `model/` — nothing is retrained here.")

# ---------------------------------------------------------------- header
st.title("🎓 Will the student drop out, stay enrolled, or graduate?")
st.write("Score five pre-trained classifiers on held-out student records and compare how they do.")

if uploaded is not None:
    data, source = read_upload(uploaded), f"uploaded file **{uploaded.name}**"
else:
    data, source = tidy_columns(pd.read_csv(SAMPLE_TEST_CSV)), "bundled **test_data.csv**"
    st.info("👈 Showing the bundled test split. Upload your own test CSV in the sidebar to score that instead.")

missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The CSV is missing {len(missing)} required column(s): {', '.join(missing)}")
    st.stop()

label_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
unknown = sorted(set(data[TARGET].astype(str)) - set(label_to_id))
if unknown:
    st.error(f"Unexpected values in '{TARGET}': {unknown}. Expected one of {CLASS_NAMES}.")
    st.stop()

X = data[FEATURES]
y = data[TARGET].astype(str).map(label_to_id).to_numpy()

counts = data[TARGET].value_counts().reindex(CLASS_NAMES, fill_value=0)
st.caption(f"Using the {source}: {len(data):,} rows · "
           + " · ".join(f"{name}: {n}" for name, n in counts.items()))

tab_model, tab_compare, tab_preds = st.tabs(["📊 Selected model", "🏁 Compare all models", "🔍 Predictions"])

# ---------------------------------------------------------------- selected model
model = models[model_name]
y_pred = model.predict(X)
y_proba = model.predict_proba(X)
scores = compute_metrics(y, y_pred, y_proba, len(CLASS_NAMES))

with tab_model:
    st.subheader(model_name)
    for col, metric in zip(st.columns(6), METRIC_NAMES):
        value = scores[metric]
        col.metric(metric, "n/a" if np.isnan(value) else f"{value:.3f}")
    st.caption("AUC is one-vs-rest; Precision, Recall and F1 are weighted by class size.")

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(confusion_figure(y, y_pred, CLASS_NAMES))
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=list(range(len(CLASS_NAMES))),
                                       target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

# ---------------------------------------------------------------- all models
with tab_compare:
    board = pd.DataFrame({
        name: compute_metrics(y, m.predict(X), m.predict_proba(X), len(CLASS_NAMES)) for name, m in models.items()
    }).T[METRIC_NAMES]
    board.index.name = "Model"
    leader = board["F1"].idxmax()
    st.success(f"Best on this data by weighted F1: **{leader}** (F1 {board.loc[leader, 'F1']:.3f}, "
               f"MCC {board.loc[leader, 'MCC']:.3f})")
    st.dataframe(board.style.format("{:.4f}").highlight_max(axis=0, color="#c8ecd9"),
                 width="stretch")
    chosen_metric = st.radio("Chart metric", METRIC_NAMES, horizontal=True, index=4)
    st.bar_chart(board[chosen_metric])

# ---------------------------------------------------------------- row-level predictions
with tab_preds:
    view = data[[TARGET]].rename(columns={TARGET: "Actual"})
    view.insert(1, "Predicted", [CLASS_NAMES[i] for i in y_pred])
    view.insert(2, "Confidence", y_proba.max(axis=1).round(3))
    view["Correct"] = view["Actual"] == view["Predicted"]
    only_wrong = st.checkbox("Show only misclassified rows")
    st.dataframe(view[~view["Correct"]] if only_wrong else view, width="stretch", height=380)
    st.download_button("⬇️ Download predictions", view.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv", mime="text/csv")
