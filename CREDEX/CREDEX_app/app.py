import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")

model = joblib.load("model.pkl")

st.title("Credex Loan Prediction App")

st.write("Enter Details")

feature2 = st.number_input(" Applicant_Income")
feature3 = st.number_input("Coapplicant_Income")
feature4 = st.number_input("Age ")
feature5 = st.number_input("Dependents")
feature6 = st.number_input("Credit_Score")
feature7 = st.number_input("Existing_Loans")
feature8 = st.number_input("DTI_Ratio")
feature9 = st.number_input("Savings")
feature10 = st.number_input("Collateral_Value")
feature11 = st.number_input("Loan_Amount  ")
feature12 = st.number_input("Loan_Term ")
feature13 = st.number_input("Education_Level ")
feature14 = st.number_input("Employment_Status_Salaried  ")
feature15 = st.number_input("Employment_Status_Self-employed")
feature16 = st.number_input("Employment_Status_Self-Unemployed")
feature17 = st.number_input("Marital_Status_Single")
feature18 = st.number_input("Loan_Purpose_Car")
feature19 = st.number_input("Loan_Purpose_Education")
feature20 = st.number_input("Loan_Purpose_Home")
feature21 = st.number_input("Loan_Purpose_Personal")
feature22 = st.number_input("Gender_Male")
feature23 = st.number_input("Property_Area_Semiurban")
feature24 = st.number_input("Property_Area_Urban ")
feature25 = st.number_input("Employer_Category_Government")
feature26 = st.number_input("Employer_Category_MNC")
feature27 = st.number_input("Employer_Category_Private")
feature28 = st.number_input("Employer_Category_Unemployed ")




if st.button("Predict"):
    input_data = np.array([[ feature2, feature3, feature4,
feature5, feature6, feature7, feature8, feature9, feature10,
feature11, feature12, feature13, feature14, feature15,
feature16, feature17, feature18, feature19, feature20,
feature21, feature22, feature23, feature24,
feature25, feature26, feature27, feature28]])

    st.write("Raw Input:", input_data)
    input_data = scaler.transform(input_data)
    st.write("Scaled Input:", input_data)
    prob = model.predict_proba(input_data)
    st.write("Prediction Probability:", prob)
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")

