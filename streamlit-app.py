import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

#Load the models and scaler
lr_model = joblib.load('models/dropout_model.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# SHAP setup
explainer = shap.TreeExplainer(rf_model)
st.set_page_config(page_title="Student Dropout Predictor", layout="centered")
st.title("Student Dropout Predictor")
st.write("This app predicts whether a student will drop out based on various factors.")

#  Sidebar: Select model 
model_choice = st.sidebar.radio("Choose model:", ("Logistic Regression", "Random Forest"))

#  Sidebar: Student Inputs 
st.sidebar.header("Student Info")
active_days = st.sidebar.slider("Active Days", 0, 250, 50)
click_events = st.sidebar.slider("Click Events", 0, 10000, 500)
total_clicks = st.sidebar.slider("Total Clicks", 0, 10000, 500)
studied_credits = st.sidebar.number_input("Studied Credits", 0, 600)
num_prev_attempts = st.sidebar.slider("Previous Attempts", 0, 5, 0)

# Encoded categorical features 
gender = st.sidebar.selectbox("Gender", ["M", "F"])
age_band = st.sidebar.selectbox("Age Band", ["0-35", "35-55", "55<"])
highest_edu = st.sidebar.selectbox("Education", ["Lower Than A Level", "A Level", "HE Qualification", "Post Graduate Qualification"]) 
disability = st.sidebar.selectbox("Disability", ["Y", "N"])

#  The inputs
input_dict = {
    "active_days": active_days,
    "click_events": click_events,
    "total_clicks": total_clicks,
    "# studied_credits": studied_credits,
    "# num_of_prev_attempts": num_prev_attempts,
    "gender_M": 1 if gender == "M" else 0,
    "age_band_35-55": 1 if age_band == "35-55" else 0,
    "age_band_55<": 1 if age_band == "55<" else 0,
    "highest_education_Lower Than A Level": 1 if highest_edu == "Lower Than A Level" else 0,
    "highest_education_Post Graduate Qualification": 1 if highest_edu == "Post Graduate Qualification" else 0,
    "highest_education_HE Qualification": 1 if highest_edu == "HE Qualification" else 0,
    "disability_Y": 1 if disability == "Y" else 0
}

input_df = pd.DataFrame([input_dict])


# Fill all expected columns with 0 if missing (to match training set structure)
expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else input_df.columns
for col in expected_features:
    if col not in input_df:
        input_df[col] = 0
input_df = input_df[expected_features]  # reorder to match training


# Scaled input
X_scaled = scaler.transform(input_df)


# Predict 
if model_choice == "Logistic Regression":
    pred = lr_model.predict(X_scaled)[0]
    prob = lr_model.predict_proba(X_scaled)[0][1]
    model_used = lr_model
else:
    pred = rf_model.predict(X_scaled)[0]
    prob = rf_model.predict_proba(X_scaled)[0][1]
    model_used = rf_model

st.subheader("Prediction")
st.write(f"**Prediction:** {'Dropout' if pred == 1 else 'Not Dropout'}")
st.write(f"**Probability of Dropout:** {prob:.2%}")


# SHAP explanation 
if model_choice == "Random Forest":
    st.subheader("Feature Impact (SHAP)")
    shap_values = explainer.shap_values(input_df)
    fig, ax = plt.subplots()
    shap_expl = shap.Explanation(values=shap_values[0, :, 1], base_values=explainer.expected_value[1], data=input_df.iloc[0], feature_names=input_df.columns)
    shap.plots.waterfall(shap_expl, max_display=10)
    st.pyplot(fig)
else:
    st.info("SHAP explanations only available for Random Forest in this demo.")



