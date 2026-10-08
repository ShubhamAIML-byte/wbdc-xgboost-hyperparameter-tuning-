import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WDBC Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("wdbc_xgboost_best_model.joblib")


@st.cache_data
def load_features():
    with open("feature_names.json", "r") as f:
        return json.load(f)


@st.cache_data
def load_metrics():
    with open("metrics.json", "r") as f:
        return json.load(f)


# Load files
model = load_model()
feature_names = load_features()
metrics = load_metrics()

# ============================================================
# HEADER
# ============================================================

st.title("🩺 WDBC Breast Cancer Prediction")

st.markdown(
    "### XGBoost-based Wisconsin Diagnostic Breast Cancer Classification"
)

st.info(
    "Enter the 30 WDBC diagnostic features below to obtain "
    "a Benign/Malignant prediction."
)

# ============================================================
# SIDEBAR - MODEL INFORMATION
# ============================================================

st.sidebar.header("📊 Model Information")

st.sidebar.metric(
    "Test Accuracy",
    f"{metrics['test_accuracy'] * 100:.2f}%"
)

st.sidebar.metric(
    "Test Precision",
    f"{metrics['test_precision'] * 100:.2f}%"
)

st.sidebar.metric(
    "Test Recall",
    f"{metrics['test_recall'] * 100:.2f}%"
)

st.sidebar.metric(
    "Test F1 Score",
    f"{metrics['test_f1'] * 100:.2f}%"
)

st.sidebar.metric(
    "Test ROC-AUC",
    f"{metrics['test_roc_auc'] * 100:.2f}%"
)

st.sidebar.markdown("---")

st.sidebar.write("**Model:** XGBoost")
st.sidebar.write("**Hyperparameter Tuning:** RandomizedSearchCV")
st.sidebar.write("**Cross Validation:** 5-Fold Stratified CV")

# ============================================================
# PATIENT FEATURE INPUT
# ============================================================

st.header("🔬 Patient Feature Input")

st.write(
    "Enter the 30 numerical WDBC features used by the trained model."
)

# ============================================================
# DEFAULT VALUES
# ============================================================

defaults = {
    "feature_1": 14.0,
    "feature_2": 20.0,
    "feature_3": 90.0,
    "feature_4": 600.0,
    "feature_5": 0.10,
    "feature_6": 0.10,
    "feature_7": 0.08,
    "feature_8": 0.05,
    "feature_9": 0.18,
    "feature_10": 0.06,
    "feature_11": 0.40,
    "feature_12": 1.0,
    "feature_13": 3.0,
    "feature_14": 40.0,
    "feature_15": 0.01,
    "feature_16": 0.02,
    "feature_17": 0.02,
    "feature_18": 0.01,
    "feature_19": 0.02,
    "feature_20": 0.003,
    "feature_21": 15.0,
    "feature_22": 25.0,
    "feature_23": 100.0,
    "feature_24": 700.0,
    "feature_25": 0.14,
    "feature_26": 0.30,
    "feature_27": 0.40,
    "feature_28": 0.15,
    "feature_29": 0.30,
    "feature_30": 0.08
}

# ============================================================
# INPUT FORM
# ============================================================

with st.form("prediction_form"):

    input_values = {}

    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(feature_names):

        if i < 10:
            column = col1

        elif i < 20:
            column = col2

        else:
            column = col3

        with column:

            input_values[feature] = st.number_input(
                feature,
                value=float(defaults.get(feature, 0.0)),
                format="%.6f"
            )

    submitted = st.form_submit_button(
        "🔍 Predict Diagnosis",
        use_container_width=True
    )

# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # Create DataFrame in exactly the same feature order
    input_df = pd.DataFrame(
        [input_values],
        columns=feature_names
    )

    # Prediction
    prediction = model.predict(input_df)[0]

    # Probability
    probabilities = model.predict_proba(input_df)[0]

    benign_probability = probabilities[0]
    malignant_probability = probabilities[1]

    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("---")

    st.header("🧠 Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ Prediction: MALIGNANT"
        )

    else:

        st.success(
            "✅ Prediction: BENIGN"
        )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Benign Probability",
            f"{benign_probability * 100:.2f}%"
        )

        st.progress(
            float(benign_probability)
        )

    with col2:

        st.metric(
            "Malignant Probability",
            f"{malignant_probability * 100:.2f}%"
        )

        st.progress(
            float(malignant_probability)
        )

    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================

    st.markdown("---")

    st.subheader("📋 Prediction Summary")

    result_df = pd.DataFrame({
        "Prediction": [
            "Malignant" if prediction == 1 else "Benign"
        ],
        "Benign Probability": [
            f"{benign_probability * 100:.2f}%"
        ],
        "Malignant Probability": [
            f"{malignant_probability * 100:.2f}%"
        ]
    })

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.header("📈 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Accuracy",
        f"{metrics['test_accuracy'] * 100:.2f}%"
    )

with col2:

    st.metric(
        "Precision",
        f"{metrics['test_precision'] * 100:.2f}%"
    )

with col3:

    st.metric(
        "Recall",
        f"{metrics['test_recall'] * 100:.2f}%"
    )

with col4:

    st.metric(
        "F1 Score",
        f"{metrics['test_f1'] * 100:.2f}%"
    )

with col5:

    st.metric(
        "ROC-AUC",
        f"{metrics['test_roc_auc'] * 100:.2f}%"
    )

# ============================================================
# BEST HYPERPARAMETERS
# ============================================================

st.markdown("---")

st.header("⚙️ Best XGBoost Hyperparameters")

best_params = metrics.get(
    "best_parameters",
    {}
)

if best_params:

    params_df = pd.DataFrame(
        list(best_params.items()),
        columns=[
            "Parameter",
            "Value"
        ]
    )

    st.dataframe(
        params_df,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

st.markdown("---")

st.header("🔄 5-Fold Cross Validation")

cv_summary = metrics.get(
    "cv_summary",
    {}
)

if cv_summary:

    rows = []

    for metric, values in cv_summary.items():

        rows.append({
            "Metric": metric.upper(),
            "Mean": values["mean"],
            "Standard Deviation": values["std"],
            "95% CI Lower": values["ci_lower"],
            "95% CI Upper": values["ci_upper"]
        })

    cv_df = pd.DataFrame(rows)

    st.dataframe(
        cv_df.style.format({
            "Mean": "{:.4f}",
            "Standard Deviation": "{:.4f}",
            "95% CI Lower": "{:.4f}",
            "95% CI Upper": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "WDBC XGBoost Classification | "
    "Hyperparameter Tuned | "
    "5-Fold Cross Validation | "
    "SHAP Explainability"
)

st.warning(
    "⚠️ This application is for research and educational "
    "purposes only and is not a substitute for professional "
    "medical diagnosis."
)
