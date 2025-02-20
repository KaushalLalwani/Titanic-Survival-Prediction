import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("titanic_model.pkl")

# Title
st.title("🚢 Titanic Survival Prediction")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical inputs
sex_encoded = 1 if sex == "Female" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Create DataFrame for prediction
input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_Q, embarked_S]],
                          columns=['pclass'	,'sex'	,'age'	,'sibsp','parch','fare' ,'embarked_Q','embarked_S'])

# Predict survival


# Make prediction
prediction = model.predict(input_data)
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("🎉 The passenger would have SURVIVED!")
    else:
        st.error("💀 The passenger would NOT have survived.")
