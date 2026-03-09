import streamlit as st
import pandas as pd
import numpy as np  
import joblib

model = joblib.load('knn_heart_model.pkl')
scaler = joblib.load('heart_scaler.pkl')
expected_columns = joblib.load('heart_columns.pkl')

st.title("Heart Disease Prediction by Shailja❤️")
st.markdown("Provide the following details to predict the likelihood of heart disease.")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_bp = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=80, max_value=200, value=120)
cholestrol = st.number_input("Cholesterol (in mg/dl)", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
max_heart_rate = st.slider("Maximum Heart Rate Achieved", 50, 300, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, value=1.5)
st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

if st.button("Predict"):
    raw_data = {
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'chest_pain': chest_pain,
        'resting_bp': resting_bp,
        'cholestrol': cholestrol,
        'fasting_bs': 1 if fasting_bs == "Yes" else 0,
        'rest_ecg': rest_ecg,
        'max_heart_rate': max_heart_rate,
        'exercise_angina': 1 if exercise_angina == "Yes" else 0,
        'oldpeak': oldpeak,
        'st_slope': st_slope
    }
    input_df = pd.DataFrame([raw_data])
     
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.error("⚠️High Risk of Heart Disease detected.")
    else:
        st.success("✅Low Risk of Heart Disease detected. Maintain a healthy lifestyle!")