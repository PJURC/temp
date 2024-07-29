import streamlit as st
import joblib
import pandas as pd
from helper_functions import add_new_features



def predict_stroke(df):
    """
    Predicts the likelihood of a stroke based on the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the patient's features.

    Returns:
        tuple: A tuple containing the predicted stroke label (0 for no stroke, 1 for stroke) and the corresponding probabilities.
    """
    
    # Make predictions
    prediction = model.predict(df)
    probabilities = model.predict_proba(df)

    return prediction[0], probabilities[0]


# Render website, get inputs
st.title('Stroke Prediction')
st.write('Enter the patient details to predict the likelihood of stroke.')
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 0, 120, 50)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.slider('Average Glucose Level', 0, 300, 100)
bmi = st.slider('BMI', 0.0, 70.0, 25.0)
smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])


# Define features
features = {
    'gender': gender,
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'ever_married': ever_married,
    'work_type': work_type,
    'residence_type': residence_type,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': smoking_status
}

# Convert input features to a DataFrame
input_df = pd.DataFrame([features])

# Transform features
input_df['hypertension'] = input_df['hypertension'].map({'No': 0, 'Yes': 1})
input_df['heart_disease'] = input_df['heart_disease'].map({'No': 0, 'Yes': 1})

# Add new features
input_df = add_new_features(input_df)

# Load the model
model = joblib.load('stroke_prediction_model.joblib')

if st.button('Predict'):
    result, probs = predict_stroke(input_df)
    # Calculate percentages
    no_stroke_prob = probs[0] * 100
    stroke_prob = probs[1] * 100

    st.write(f'Prediction: {"Stroke" if result == 1 else "No Stroke"}')
    st.write(f'Probability of No Stroke: {no_stroke_prob:.2f}%')
    st.write(f'Probability of Stroke: {stroke_prob:.2f}%')

    # Add a warning for high stroke probability
    if stroke_prob > 50:
        st.warning('Warning: Risk of stroke! Please consult a healthcare professional immediately.')

    st.bar_chart({
        'No Stroke': no_stroke_prob,
        'Stroke': stroke_prob
    })