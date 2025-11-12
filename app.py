import streamlit as st
import joblib
import numpy as np

# Load saved model, scaler, and label mapping
model = joblib.load('crop_recommendation_model.pkl')
scaler = joblib.load('scaler.pkl')
reverse_crop_dict = joblib.load('reverse_crop_dict.pkl')

# Streamlit app title
st.title("🌾 Crop Recommendation System")
st.write("Enter the soil and weather conditions to get the best crop suggestion!")

# Input fields for user
N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
P = st.number_input("Phosphorous (P)", min_value=0.0, max_value=200.0, value=50.0)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)

# Predict button
if st.button("Predict Crop"):
    try:
        # Prepare input for model
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]

        # Decode prediction
        crop_name = reverse_crop_dict[prediction]
        st.success(f"🌱 Recommended Crop: **{crop_name.capitalize()}**")

    except Exception as e:
        st.error(f"An error occurred: {e}")
