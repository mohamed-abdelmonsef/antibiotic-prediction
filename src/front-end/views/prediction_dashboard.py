import streamlit as st
import pandas as pd
import requests
import sys
import os

sys.path.append(os.path.dirname(__file__))

from request_preprocessing import change_types
from default_cultures import culture_437706157, culture_681936151, culture_7806863881, culture_339537616

st.set_page_config(page_title="Antibiotic Predictor (Input Only)", layout="centered")


default_profiles = {
    "Custom": {},
    "Profile 437706157": culture_437706157,
    "Profile 681936151": culture_681936151,
    "Profile 7806863881": culture_7806863881,
    "Profile 339537616": culture_339537616
}

selected_profile_name = st.sidebar.selectbox(
    "Choose a default profile",
    list(default_profiles.keys())
)

profile = default_profiles[selected_profile_name]


st.title("🧪 Enter Clinical Features")

# Grouped input fields
st.header("🩺 Vital Signs")
median_heartrate = st.number_input("Median Heart Rate", value=profile.get("median_heartrate", 80.0))
median_resprate = st.number_input("Median Respiratory Rate", value=profile.get("median_resprate", 18.0))
median_temp = st.number_input("Median Temperature (°C)", value=profile.get("median_temp", 37.0))
median_sysbp = st.number_input("Median Systolic BP", value=profile.get("median_sysbp", 120.0))
median_diasbp = st.number_input("Median Diastolic BP", value=profile.get("median_diasbp", 80.0))

st.header("🧬 Lab Results")
median_wbc = st.number_input("WBC Count", value=profile.get("median_wbc", 7.0))
median_hgb = st.number_input("Hemoglobin", value=profile.get("median_hgb", 13.5))
median_plt = st.number_input("Platelet Count", value=profile.get("median_plt", 250.0))
median_na = st.number_input("Sodium", value=profile.get("median_na", 140.0))
median_hco3 = st.number_input("Bicarbonate (HCO3)", value=profile.get("median_hco3", 22.0))
median_bun = st.number_input("BUN", value=profile.get("median_bun", 15.0))
median_cr = st.number_input("Creatinine", value=profile.get("median_cr", 1.0))

st.header("📊 Demographics Inputs")
culture_description = st.selectbox(
    "Culture Description",
    ['URINE', 'BLOOD', 'RESPIRATORY'],
    index=['URINE', 'BLOOD', 'RESPIRATORY'].index(profile.get("culture_description", 'URINE'))
)
age = st.number_input("AGE", value=profile.get("age", 55))

gender_value = profile.get("gender", "").lower()  # Get from profile if exists
if gender_value == "f":
    gender_value = "female"
elif gender_value == "m":
    gender_value = "male"
elif gender_value not in ['male', 'female']:
    gender_value = 'male'  # fallback default

gender = st.selectbox("Gender", ['male', 'female'], index=['male', 'female'].index(gender_value))

# collect inputs into a dictionary
inputs = {
    "median_heartrate": median_heartrate,
    "median_resprate": median_resprate,
    "median_temp": median_temp,
    "median_sysbp": median_sysbp,
    "median_diasbp": median_diasbp,
    "median_wbc": median_wbc,
    "median_hgb": median_hgb,
    "median_plt": median_plt,
    "median_na": median_na,
    "median_hco3": median_hco3,
    "median_bun": median_bun,
    "median_cr": median_cr,
    "culture_description": culture_description,
    "age": age,
    "gender":gender
}



# Change types of inputs
change_types(inputs)


# Submit button (currently no model)
if st.button("Submit"):
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=inputs)
        
        if response.status_code == 200:
            result = response.json()
            # Success message
            st.success("Prediction received successfully!")

            # Show header
            st.subheader("Top 5 Antibiotic Recommendations")
            
            # Convert to DataFrame for clean table display
            df = pd.DataFrame(result["top_5_recommendations"])

            # Format probabilities to 2 decimal places (optional)
            df["probability"] = df["probability"].apply(lambda x: round(x * 100, 2))

            # Rename columns for better readability
            df.columns = ["Antibiotic", "Probability (%)"]

            # Show table
            st.table(df)
            
        else:
            st.error(f"Request failed with status code {response.status_code}")
            st.text(response.text)  # Show error details
    except requests.exceptions.RequestException as e:
        st.error("Error connecting to the prediction API.")
        st.text(str(e))

    
    print(inputs['gender'],inputs['age'])

