import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the trained model


# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_regressor_singapoor.pkl")  # Replace with your model's file name
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Streamlit app
st.title("Selling Price Prediction")

st.sidebar.header("User Input Features")

# User input for each column
year = st.sidebar.slider("Year", min_value=2000, max_value=2025, value=2020, step=1)

flat_type = st.sidebar.selectbox("Flat Type", options=['1 ROOM','2 ROOM', '3 ROOM', '4 ROOM','5 ROOM','EXECUTIVE','MULTI GENERATION']) 

floor_area_sqm_year_interaction = st.sidebar.number_input(
    "Floor Area (sqm) x Year Interaction",
    min_value=0.0,
    value=290.32
)

lease_commence_date = st.sidebar.slider("Lease Commence Date", min_value=1970, max_value=2025, value=2000, step=1)

age_group = st.sidebar.selectbox("Age Group", options=['new', 'mid_aged', 'old'], index=1)

price_per_sqm = st.sidebar.number_input("Price Per Sqm", min_value=0.0, value=1000.0)

floor_start = st.sidebar.text_input("Floor Start", value="01")

floor_end = st.sidebar.text_input("Floor End", value="10")

# Prepare input data as a DataFrame
input_data = pd.DataFrame({
    'year': [year],
    'flat_type': [flat_type],
    'floor_area_sqm_year_interaction': [floor_area_sqm_year_interaction],
    'lease_commence_date': [lease_commence_date],
    'age_group': [age_group],
    'prcie_per_sqm': [price_per_sqm],
    'floor_start': [floor_start],
    'floor_end': [floor_end]
})

st.write("### User Inputs")
st.write(input_data)

# Map categorical features if necessary
# Example mapping for 'flat_type' or 'age_group' (Ensure consistency with model training)
flat_type_mapping = {'1 ROOM':0,'2 ROOM':1, '3 ROOM':3, '4 ROOM':4,'5 ROOM':5,'EXECUTIVE':6,'MULTI GENERATION':7}
age_group_mapping = {'new': 0, 'mid_aged': 1, 'old': 2}

input_data['flat_type'] = input_data['flat_type'].map(flat_type_mapping)
input_data['age_group'] = input_data['age_group'].map(age_group_mapping)

# Load the model
model = load_model()

# Predict selling price
if st.button("Predict Selling Price"):
    if model is not None:
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Selling Price: ₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Model is not loaded. Please check the model file.")