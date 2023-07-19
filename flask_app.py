from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import os


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the form data
    age = int(request.form['age'])
    resting_BP = int(request.form['resting_BP'])
    cholesterol = int(request.form['cholesterol'])
    MaxHR = int(request.form['MaxHR'])
    oldpeak = float(request.form['oldpeak'])
    sex = request.form['sex']
    chest_pain = request.form['chest_pain']
    fasting_bs = request.form['fasting_bs']
    resting_ECG = request.form['resting_ECG']
    ExerciseAngina = request.form['ExerciseAngina']
    ST_Slope = request.form['ST_Slope']
    selected_model = request.form['selected_model']

    # Conversion function
    def convert_categorical_variables(sex_, chest_pain_, fasting_bs_, resting_ECG_, ExerciseAngina_, ST_Slope_):
        # Conversion dictionaries
        sex_conversion = {"Male": "M", "Female": "F"}
        chest_pain_conversion = {"Typical Angina Pain": "TA", "Atypical Angina Pain": "ATA", "Non-Anginal Pain": "NAP", "No Chest Pain": "ASY"}
        fasting_bs_conversion = {"Over 120": 1, "120 or Under": 0}
        resting_ECG_conversion = {"ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)": "ST", "Normal": "Normal", "Showing probable or definite left ventricular hypertrophy by Estes' criteria": "LVH"}
        ExerciseAngina_conversion = {"Yes": "Y", "No": "N"}
        ST_Slope_conversion = {"Sloping Upwards": "Up", "Flat": "Flat", "Sloping Downwards": "Down"}

        # Conversion
        return (
            sex_conversion[sex_],
            chest_pain_conversion[chest_pain_],
            fasting_bs_conversion[fasting_bs_],
            resting_ECG_conversion[resting_ECG_],
            ExerciseAngina_conversion[ExerciseAngina_],
            ST_Slope_conversion[ST_Slope_]
        )

    # Convert categorical variables
    sex, chest_pain, fasting_bs, resting_ECG, ExerciseAngina, ST_Slope = convert_categorical_variables(
        sex, chest_pain, fasting_bs, resting_ECG, ExerciseAngina, ST_Slope
    )

    def preprocess_new_data():
        # Load the whole dataset
        df = pd.read_csv(os.path.join("data", "heart_disease_data.csv"))  # in data directory
        df = df.drop(["HeartDisease"], axis=1)

        # Create a new row with the user's data
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

        df.loc[len(df)] = new_row  # Add the new row to the full dataset

        # Augment features the same way as in training
        numerical_features = df.select_dtypes(include=[np.number])
        numerical_features = numerical_features.drop(["FastingBS"], axis=1)
        continuous_feature_names = numerical_features.columns.tolist()

        categorical_features = df.select_dtypes(include=[object])
        categorical_feature_names = categorical_features.columns.to_list() + ["FastingBS"]

        preprocessed_df = df.copy(deep=True)  # Make a copy of the original data to modify

        # Initialize the scalers
        min_max_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()

        # Apply both scalers to each continuous variable
        for feature in continuous_feature_names:
            min_max_scaled_data = min_max_scaler.fit_transform(preprocessed_df[[feature]])  # Perform MinMax scaling
            min_max_standard_scaled_data = standard_scaler.fit_transform(min_max_scaled_data)  # Perform Standard scaling on the MinMax scaled data
            preprocessed_df[feature] = min_max_standard_scaled_data.flatten()  # Update the original DataFrame with the scaled data

        # One-hot encoding of categorical variables
        preprocessed_df = pd.get_dummies(preprocessed_df, columns=categorical_feature_names, dtype=int)

        # Return the final row to predict
        return preprocessed_df.tail(1)  # Get the last row and keep it as a DataFrame structure

    to_predict = preprocess_new_data()

    # Load the trained models
    random_forest_classifier = joblib.load("saved models/random_forest_classifier.pkl")
    dl_classifier = tf.keras.models.load_model(os.path.join(os.getcwd(), "saved models/deep_learning_classifier"))

    # Prediction function
    def predict():
        # Make the prediction
        if selected_model == "Random Forest Classifier (Highest Specificity)":
            prediction = random_forest_classifier.predict(to_predict)
            probability_positive = random_forest_classifier.predict_proba(to_predict)[0][1]
        else:
            tf_predictions = dl_classifier.predict(to_predict)
            prediction = np.round(tf_predictions).astype(int)[0]
            probability_positive = tf_predictions[0][0]

        if prediction[0] == 1:
            return f"It is predicted that the patient has heart disease.<br>Chance of being heart disease positive: {100*probability_positive:.2f}%"
        else:
            return f"It is predicted that the patient does not have heart disease.<br>Chance of being heart disease positive: {100*probability_positive:.2f}%"

    # Run the prediction function and pass the result to the template
    prediction_result = predict()

    return str(prediction_result)




if __name__ == '__main__':
    app.run(debug=True)
