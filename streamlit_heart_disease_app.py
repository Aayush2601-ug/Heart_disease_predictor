import streamlit as st
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load model and scaler
with open('model_randomforest.pkl', 'rb') as f:
    model_randomforest = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    sc = pickle.load(f)

# Sidebar
st.sidebar.title("About the App")
st.sidebar.info(
    """
    This application uses a **Random Forest Classifier** to predict whether a person is at risk of heart disease.
    - Input relevant medical parameters.
    - Get predictions and probabilities instantly!
    """
)
st.sidebar.caption("Developed Collaboratively by **Aayush**, **Aman**, and **Lakshya**")

# Main App
st.image('img.png', use_column_width=True)
st.title("Heart Disease Prediction")
st.caption('Developed Collaboratively by Aayush, Aman, and Lakshya')

# Sections
st.header("What is this model?")
st.write("""
The Random Forest Classifier is a powerful machine learning algorithm that builds multiple decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression). This model was trained using heart disease datasets to predict whether an individual is at risk of heart disease based on various medical parameters.
""")

st.header("Use the Model")

# Input Form
st.subheader("Enter Your Details:")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, step=1)
    chol = st.number_input("Serum Cholesterol in mg/dl", min_value=100, max_value=600, step=1)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=220, step=1)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2])

# Centered Predict Button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("Predict"):
    input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])
    scaled_features = sc.transform(input_features)  # Apply scaling
    prediction = model_randomforest.predict(scaled_features)
    probabilities = model_randomforest.predict_proba(scaled_features)

    # Display Prediction
    if prediction == 0:
        st.success("The model predicts: Person is NOT at risk of heart disease.")
    else:
        st.warning("The model predicts: Person IS at risk of heart disease.")
    
    # Display Probabilities
    st.subheader("Prediction Probabilities:")
    st.write(f"Probability of NOT having heart disease: {probabilities[0][0]*100:.2f}%")
    st.write(f"Probability of HAVING heart disease: {probabilities[0][1]*100:.2f}%")
st.markdown("</div>", unsafe_allow_html=True)
