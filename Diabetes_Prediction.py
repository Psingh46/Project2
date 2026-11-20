import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("diabetes_prediction_model.json")

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details to predict diabetes status.")

with st.form("patient_form"):

    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Female", "Male", "Unknown"])

    bmi = st.number_input("BMI", 0.0, 100.0, 25.0)
    glucose = st.number_input("Glucose", 0.0, 500.0, 100.0)

    bp = st.number_input("Blood Pressure", 0, 250, 80)
    cholesterol = st.number_input("Cholesterol", 0, 600, 180)
    insulin = st.number_input("Insulin", 0, 1000, 80)

    hba1c = st.number_input("HbA1c", 0.0, 30.0, 5.5)

    smoking = st.selectbox(
        "Smoking Status",
        ["Never", "Former", "Current"]
    )

    activity = st.selectbox(
        "Physical Activity",
        ["Low", "Moderate", "High"]
    )

    family = st.selectbox(
        "Family History",
        ["Yes", "No"]
    )

    submit = st.form_submit_button("Predict Diabetes")

if submit:

    patient = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "BMI": bmi,
        "Glucose": glucose,
        "Blood_Pressure": bp,
        "Cholesterol": cholesterol,
        "Insulin": insulin,
        "HbA1c": hba1c,
        "Smoking_Status": smoking,
        "Physical_Activity": activity,
        "Family_History": family
    }])

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Model Prediction: Diabetes")
    else:
        st.success("Model Prediction: No Diabetes")

    st.write(
        f"Model-estimated probability of diabetes: {probability*100:.2f}%"
    )

    st.caption(
        "Educational project only. This prediction is not a medical diagnosis."
    )
