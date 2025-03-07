import streamlit as st
import pickle
import os

# Set up the page
st.set_page_config(page_title="Diabetes Prediction", page_icon="⚕️")

# Hide Streamlit default UI elements
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load the diabetes model (Handle missing file)
model_path = 'Models/diabetes_model.sav'

if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        diabetes_model = pickle.load(model_file)
else:
    st.error("⚠️ Model file not found: 'Models/diabetes_model.sav'. Predictions will not work.")
    diabetes_model = None  # Set to None if the model is missing

# Diabetes Prediction Page
st.title('Diabetes Prediction')
st.write("Enter the following details to predict diabetes:")

# Input fields
Pregnancies = st.number_input('Number of Pregnancies', step=1, min_value=0, help='Enter number of times pregnant')
Glucose = st.number_input('Glucose Level', step=1, min_value=0, help='Enter glucose level')
BloodPressure = st.number_input('Blood Pressure', step=1, min_value=0, help='Enter blood pressure value')
SkinThickness = st.number_input('Skin Thickness', step=1, min_value=0, help='Enter skin thickness value')
Insulin = st.number_input('Insulin Level', step=1, min_value=0, help='Enter insulin level')
BMI = st.number_input('BMI', step=0.1, min_value=0.0, help='Enter Body Mass Index value')
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', step=0.01, min_value=0.0, help='Enter diabetes pedigree function value')
Age = st.number_input('Age', step=1, min_value=1, help='Enter age of the person')

# Predict button
if st.button('Predict Diabetes'):
    if diabetes_model:
        input_values = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        prob = diabetes_model.predict_proba(input_values)  # Get probability

        # Print probabilities to the console for debugging
        print(f"Debug: Diabetic Probability: {prob[0][1] * 100:.2f}%")
        print(f"Debug: Non-Diabetic Probability: {prob[0][0] * 100:.2f}%")

        threshold = 0.6  # Set confidence threshold (default is 0.5)
        if prob[0][1] >= threshold:
            result = f'The person is diabetic (Confidence: {prob[0][1] * 100:.2f}%)'
            st.error(result)  # Show in red
        else:
            result = f'The person is not diabetic (Confidence: {prob[0][0] * 100:.2f}%)'
            st.success(result)  # Show in green

        # Show probabilities in UI for better debugging
        st.write(f"🔹 **Diabetic Probability:** {prob[0][1] * 100:.2f}%")
        st.write(f"🔹 **Non-Diabetic Probability:** {prob[0][0] * 100:.2f}%")

    else:
        st.error("Prediction failed. Model not loaded. Please check the file path.")
