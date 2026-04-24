import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import plot_importance
#import shap

model = joblib.load("xgb_model.pkl")

st.title("Student Score Predictor")

# -------------------------
# Inputs
# -------------------------
age = st.number_input("Age", 15, 30, 18)

hours = st.number_input("Hours Studied", 0.0, 12.0, 5.0)
sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)

attendance = st.number_input("Attendance", 0.0, 100.0, 75.0)
tutor = st.number_input("Tutoring Sessions", 0.0, 10.0, 1.0)

stress = st.number_input("Stress Level", 0.0, 10.0, 5.0)
anxiety = st.number_input("Anxiety Score", 0.0, 10.0, 5.0)

screen = st.number_input("Screen Time", 0.0, 12.0, 4.0)
gpa = st.number_input("Previous GPA", 0.0, 10.0, 6.0)

part_time = st.selectbox("Part Time Job", ["No", "Yes"])
extra = st.selectbox("Extracurricular", ["No", "Yes"])

# categorical
study_method = st.selectbox("Study Method", ["Offline", "Online"])
diet = st.selectbox("Diet Quality", ["Good", "Poor"])
internet = st.selectbox("Internet Quality", ["Excellent", "Good", "Poor"])

# -------------------------
# Feature Engineering
# -------------------------
data = {
    'Age': age,
    'Study_Sleep': hours * sleep,
    'Attend_Tutor': attendance * tutor,
    'Stress_Anxiety': stress * anxiety,
    'Screen_Time': screen,
    'Previous_GPA': gpa,
    'Part_Time_Job': 1 if part_time == "Yes" else 0,
    'Extracurricular': 1 if extra == "Yes" else 0,

    # default 0
    'Study_Method_Offline': 0,
    'Study_Method_Online': 0,

    'Diet_Quality_Good': 0,
    'Diet_Quality_Poor': 0,

    'Internet_Quality_Excellent': 0,
    'Internet_Quality_Good': 0,
    'Internet_Quality_Poor': 0
}

# -------------------------
# One-hot encoding
# -------------------------
data[f"Study_Method_{study_method}"] = 1
data[f"Diet_Quality_{diet}"] = 1
data[f"Internet_Quality_{internet}"] = 1

df = pd.DataFrame([data])
# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):
    try:
        df_fixed = df 
        pred = model.predict(df_fixed)[0]
        st.success(f"Predicted Score: {pred:.2f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    # explainer = shap.TreeExplainer(model)

# if st.button("Explain Prediction"):
#     shap_values = explainer.shap_values(df)
#     st.subheader("SHAP Explaination")
#     st.write("Feature contribution breakdown")

#     shap_df = pd.DataFrame({
#         "Features":df.columns,
#         "SHAP Value":shap_values[0]
#     }).sort_values(by="SHAP Value",key = abs,ascending = False)

#     st.dataframe(shap_df)
#     fig, ax = plt.subplots(figsize=(8, 6))
#     shap.plots._waterfall.waterfall_legacy(
#         explainer.expected_value,
#         shap_values[0],
#         feature_names=df.columns
#     )
    #st.pyplot(fig)
