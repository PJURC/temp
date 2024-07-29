import streamlit as st
import joblib
import pandas as pd

def bmi_category(bmi):
    """
    Determines the BMI category based on the BMI value provided.

    Parameters:
    bmi (float): The BMI value to categorize.

    Returns:
    str: The category of the provided BMI value ('Underweight', 'Normal', 'Overweight', 'Obese').
    """
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


def age_group(age):
    """
    Determines the age group category based on the provided age.

    Parameters:
    age (int): The age to categorize.

    Returns:
    str: The age group category ('Under 18', '18-29', '30-44', '45-59', '60 and above').
    """
    if age < 18:
        return 'Under 18'
    elif 18 <= age < 30:
        return '18-29'
    elif 30 <= age < 45:
        return '30-44'
    elif 45 <= age < 60:
        return '45-59'
    else:
        return '60 and above'


def glucose_category(glucose):
    """
    Determines the glucose category based on the provided glucose value.

    Parameters:
    glucose (float): The blood glucose value to categorize.

    Returns:
    str: The category of the provided glucose value ('Normal', 'Prediabetes', 'Diabetes').
    """
    if glucose < 100:
        return 'Normal'
    elif 100 <= glucose < 126:
        return 'Prediabetes'
    else:
        return 'Diabetes'


def lifestyle_score(row):
    """
    Calculates a lifestyle score based on the provided row information containing 'smoking_status', 'bmi', and 'avg_glucose_level'.

    Parameters:
    row (dict): A dictionary containing keys for 'smoking_status', 'bmi', and 'avg_glucose_level'.

    Returns:
    int: The calculated lifestyle score based on the provided row information.
    """
    score = 0
    if row['smoking_status'] == 'never smoked':
        score += 2
    elif row['smoking_status'] == 'formerly smoked':
        score += 1
    if 18.5 <= row['bmi'] < 25:  # Normal BMI range
        score += 2
    if 70 <= row['avg_glucose_level'] <= 100:  # Normal fasting glucose range
        score += 2
    return score


def work_stress_proxy(row):
    """
    Calculates the stress level based on the attributes of the input row.

    Parameters:
    row (dict): A dictionary containing information about 'work_type', 'hypertension', 'heart_disease', 'bmi', and 'avg_glucose_level'.

    Returns:
    int: The calculated stress level based on the input row attributes.
    """
    stress = 0
    if row['work_type'] in ['Private', 'Self-employed']:
        stress += 1
    if row['hypertension'] == 1 or row['heart_disease'] == 1:
        stress += 1
    if row['bmi'] > 25:  # Overweight or obese
        stress += 1
    if row['avg_glucose_level'] > 100:  # Above normal
        stress += 1
    return stress


def add_new_features(X):
    """
    Adds new features to the input DataFrame by applying various transformations and calculations.

    Parameters:
    X (pandas.DataFrame): The input DataFrame containing the features to be transformed.

    Returns:
    pandas.DataFrame: The input DataFrame with additional features added.

    The function performs the following transformations and calculations:
    1. Grouping:
        - Adds a new column 'bmi_category' by applying the `bmi_category` function to the 'bmi' column.
        - Adds a new column 'age_group' by applying the `age_group` function to the 'age' column.
        - Adds a new column 'glucose_category' by applying the `glucose_category` function to the 'avg_glucose_level' column.
    2. Score calculation:
        - Adds a new column 'lifestyle_score' by applying the `lifestyle_score` function to each row of `X` along the rows axis.
        - Adds a new column 'work_stress_proxy' by applying the `work_stress_proxy` function to each row of `X` along the rows axis.
    3. Categorical to numerical / binary:
        - Creates a dictionary `smoking_risk` mapping the values of the 'smoking_status' column to numerical values.
        - Adds a new column 'smoking_risk' by mapping the values of the 'smoking_status' column using the `smoking_risk` dictionary.
    4. Numerical interactions:
        - Adds a new column 'age_bmi_interaction' by multiplying the 'age' and 'bmi' columns.
        - Adds a new column 'glucose_bmi_interaction' by multiplying the 'avg_glucose_level' and 'bmi' columns.
    5. Log transformation:
        - Adds a new column 'bmi_log' by taking the natural logarithm of the 'bmi' column.
        - Adds a new column 'avg_glucose_level_log' by taking the natural logarithm of the 'avg_glucose_level' column.
    """

    # Grouping
    X['bmi_category'] = X['bmi'].apply(bmi_category)
    X['age_group'] = X['age'].apply(age_group)
    X['glucose_category'] = X['avg_glucose_level'].apply(glucose_category)

    # Score calculation
    X['lifestyle_score'] = X.apply(lifestyle_score, axis=1)
    X['work_stress_proxy'] = X.apply(work_stress_proxy, axis=1)

    # Categorical to numerical / binary
    smoking_risk = {
        'never smoked': 0,
        'formerly smoked': 1,
        'smokes': 2,
        'Unknown': 0}
    X['smoking_risk'] = X['smoking_status'].map(smoking_risk)

    # Numerical interactions
    X['age_bmi_interaction'] = X['age'] * X['bmi']
    X['glucose_bmi_interaction'] = X['avg_glucose_level'] * X['bmi']

    # Log transformation
    X['bmi_log'] = np.log(X['bmi'])
    X['avg_glucose_level_log'] = np.log(X['avg_glucose_level'])

    return X


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
