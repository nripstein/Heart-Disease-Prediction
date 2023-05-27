import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# streamlit run heart_disease_prediction_app.py

# Create the Streamlit app
st.title("Heart Disease Prediction")

# Create input fields for each feature
age = st.number_input("Age", min_value=18, max_value=150, step=1)
resting_BP = st.number_input("Resting Systolic Blood Pressure (mm Hg)", step=1)
cholesterol = st.number_input("Serum Cholesterol (mm/dl)", step=1)
MaxHR = st.number_input("Maximum Heart Rate Achieved in Exhaustion Test", step=1)
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest (mm)")
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Does the Patient Experience Chest Pain?", ["No Chest Pain", "Typical Angina Pain", "Atypical Angina Pain", "Non-Anginal Pain"])
fasting_bs = st.selectbox("Blood Sugar After Fast (mg/dl)", ["Over 120", "120 or Under"])
resting_ECG = st.selectbox("Resting Electrocardiogram Results", ["Normal", "ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)", "Showing probable or definite left ventricular hypertrophy by Estes' criteria"])
ExerciseAngina = st.selectbox("Does the Patient Experience Exercise-Induced Angina?", ["No", "Yes"])
ST_Slope = st.selectbox("ST Segment Slope during Exercise", ["Sloping Upwards", "Flat", "Sloping Downwards"])
# Pick model
model_options = ["Logistic Regression (default)",
                 "Gaussian Naive Bayes",
                 "Random Forest Classifier",
                 "Neural Network"]
selected_model = st.selectbox("Classification Model", model_options) #  (THIS SELECTOR DOES NOTHING CURRENTLY)


def convert_categorical_variables(sex_, chest_pain_, fasting_bs_, resting_ECG_, ExerciseAngina_, ST_Slope_):
    sex_conversion = {
        "Male": "M",
        "Female": "F"
    }
    chest_pain_conversion = {
        "Typical Angina Pain": "TA",
        "Atypical Angina Pain": "ATA",
        "Non-Anginal Pain": "NAP",
        "No Chest Pain": "ASY"
    }
    fasting_bs_conversion = {
        "Over 120": 1,
        "120 or Under": 0
    }
    resting_ECG_conversion = {
        "ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)": "ST",
        "Normal": "Normal",
        "Showing probable or definite left ventricular hypertrophy by Estes' criteria": "LVH"
    }
    ExerciseAngina_conversion = {
        "Yes": "Y",
        "No": "N"
    }
    ST_Slope_conversion = {
        "Sloping Upwards": "Up",
        "Flat": "Flat",
        "Sloping Downwards": "Down"
    }
    return sex_conversion[sex_], chest_pain_conversion[chest_pain_], fasting_bs_conversion[fasting_bs_], resting_ECG_conversion[resting_ECG_], ExerciseAngina_conversion[ExerciseAngina_], ST_Slope_conversion[ST_Slope_]

sex, chest_pain, fasting_bs, resting_ECG, ExerciseAngina, ST_Slope = convert_categorical_variables(sex, chest_pain, fasting_bs, resting_ECG, ExerciseAngina, ST_Slope)

new_row = pd.Series({
    "Age": age,
    "Sex": sex,
    "ChestPainType": chest_pain,
    "RestingBP": resting_BP,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "RestingECG": resting_ECG,
    "MaxHR": MaxHR,
    "ExerciseAngina": ExerciseAngina,
    "Oldpeak": oldpeak,
    "ST_Slope": ST_Slope
})

# Load in whole dataset
df = pd.read_csv("heart_failure_data.csv")
df = df.drop(["HeartDisease"], axis=1)

df.loc[len(df)] = new_row  # add new row that user specified

# Augment features the same way I did in training
numerical_features = df.select_dtypes(include=[np.number])
numerical_features = numerical_features.drop(["FastingBS"], axis=1)
continuous_feature_names = numerical_features.columns.tolist()

categorical_features = df.select_dtypes(include=[object])
categorical_feature_names = categorical_features.columns.to_list() + ["FastingBS"]


df2 = df.copy(deep=True)  # make a copy of the original data which we will modify

# Initialize the scalers
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()  # not clear this was required for 'Age', 'RestingBP', or, 'MaxHR' because those were already looking pretty close to Gaussian. Further normalization here is unlikely to hurt, however. A further investigation into normality with QQ-plots and the shapiro wilk test could be a future direction and dictate whether those features get StandardScaler applied to them

# Apply both scalers to each continuous variable
for feature in continuous_feature_names:
    min_max_scaled_data = min_max_scaler.fit_transform(df2[[feature]])  # Perform MinMax scaling
    # Perform Standard scaling on the MinMax scaled data
    min_max_standard_scaled_data = standard_scaler.fit_transform(min_max_scaled_data)
    # Update the original DataFrame with the scaled data
    df2[feature] = min_max_standard_scaled_data.flatten()

# one hot encoding of categorical variables
df2 = pd.get_dummies(df2, columns=categorical_feature_names, dtype=int)

# extract final row to predict
to_predict = df2.tail(1)  # get last row, keep as dataframe structure

# Load the trained model
logistic_regressor1 = joblib.load("saved models/logistic_regressor1.pkl")


# Define a function to handle the prediction
def predict():
    # Perform any necessary preprocessing steps on the input data

    # Make the prediction
    prediction = logistic_regressor1.predict(to_predict)

    # Display the prediction result
    if prediction[0] == 1:
        st.success("It is predicted that the patient has heart disease.")
    else:
        st.success("It is predicted that the patient does not have heart disease.")


# Create a button to trigger the prediction
predict_button = st.button("Predict")
if predict_button:
    predict()
