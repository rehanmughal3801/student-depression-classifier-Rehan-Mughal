import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "depression_classifier_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

FEATURES = [
    "Gender",
    "Age",
    "Family_background",
    "Relationship_status",
    "Marital_status",
    "Living_style",
    "Current_semester",
    "CGPA",
    "friend_circle",
    "Health_condition",
    "Cost_without_study",
    "Home_division",
]

CATEGORICAL_CODE_FIELDS = [
    "Gender", "Family_background", "Relationship_status", "Marital_status",
    "Living_style", "Current_semester", "friend_circle", "Health_condition",
    "Cost_without_study", "Home_division",
]

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Keep depression_classifier_model.pkl "
            "and scaler.pkl in the same folder as app.py."
        )
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with SCALER_PATH.open("rb") as f:
        scaler = pickle.load(f)
    return model, scaler

st.set_page_config(page_title="Student Depression Classifier", page_icon="🎓", layout="centered")
st.title("🎓 Student Depression Classifier")
st.write(
    "Enter the same feature values and category codes that were used when "
    "the model was trained. The model returns a prediction from its two classes."
)

st.warning(
    "Important: the training script did not save the LabelEncoder objects or "
    "their category mappings. Therefore, categorical fields below require the "
    "original encoded numeric codes—not category names. The meanings of target "
    "classes 0 and 1 were also not documented, so this app displays the raw class."
)

with st.form("prediction_form"):
    st.subheader("Student information")
    col1, col2 = st.columns(2)

    values = {}
    with col1:
        values["Gender"] = st.number_input("Gender (encoded code)", value=0, step=1)
        values["Age"] = st.number_input("Age", min_value=0.0, value=20.0, step=1.0)
        values["Family_background"] = st.number_input("Family background (encoded code)", value=0, step=1)
        values["Relationship_status"] = st.number_input("Relationship status (encoded code)", value=0, step=1)
        values["Marital_status"] = st.number_input("Marital status (encoded code)", value=0, step=1)
        values["Living_style"] = st.number_input("Living style (encoded code)", value=0, step=1)

    with col2:
        values["Current_semester"] = st.number_input("Current semester (encoded code)", value=1, step=1)
        values["CGPA"] = st.number_input("CGPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
        values["friend_circle"] = st.number_input("Friend circle (encoded code)", value=0, step=1)
        values["Health_condition"] = st.number_input("Health condition (encoded code)", value=0, step=1)
        values["Cost_without_study"] = st.number_input("Cost without study (encoded code)", value=0, step=1)
        values["Home_division"] = st.number_input("Home division (encoded code)", value=0, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        model, scaler = load_artifacts()
        input_df = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        st.subheader("Prediction")
        st.metric("Predicted class", str(prediction))

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            probability_df = pd.DataFrame({
                "Model class": [str(c) for c in model.classes_],
                "Model probability": probabilities,
            })
            st.dataframe(probability_df, hide_index=True, use_container_width=True)

        st.caption(
            "This is an automated model output, not a clinical diagnosis. "
            "It should not be used as a substitute for assessment by a qualified professional."
        )
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
