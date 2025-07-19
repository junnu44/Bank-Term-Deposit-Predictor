# ✅ THIS LINE MUST BE FIRST — ABOVE ALL STREAMLIT USAGE
import streamlit as st
st.set_page_config(page_title="Bank Term Deposit Predictor", layout="centered")

import pandas as pd
import pickle

# Load model and encoders
with open('model_decision_tree.pkl', 'rb') as f:
    model, label_encoders, target_encoder, feature_columns = pickle.load(f)

st.title("💼 Bank Marketing Prediction App")
st.markdown("Fill in the form below to predict if a customer will subscribe.")

# User Input Section
user_input = {}
for feature in feature_columns:
    if feature in label_encoders:
        options = list(label_encoders[feature].classes_)
        selected = st.selectbox(f"{feature.capitalize()}:", options)
        encoded = label_encoders[feature].transform([selected])[0]
        user_input[feature] = encoded
    else:
        value = st.number_input(f"{feature.capitalize()}:", min_value=0.0, step=1.0)
        user_input[feature] = value

# Predict Button
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    result = target_encoder.inverse_transform([prediction])[0]

    if result == 'yes':
        st.success("✅ The customer is likely to subscribe.")
    else:
        st.error("❌ The customer is NOT likely to subscribe.")

    st.write("🔍 Prediction:", result.upper())
