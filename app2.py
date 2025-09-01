from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer  # Add this import
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error
from prophet import Prophet

# Define base directory and models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'analysis', 'models')

print(f"Looking for models in: {MODELS_DIR}")

try:
    # Load models with absolute paths
    risk_model = joblib.load(os.path.join(MODELS_DIR, "training_model.joblib"))
    attack_count_model = joblib.load(os.path.join(MODELS_DIR, "regression_model.joblib"))
    forecast_model = joblib.load(os.path.join(MODELS_DIR, "timeseries_model.joblib"))
    hotspot_model = joblib.load(os.path.join(MODELS_DIR, "hotspot_model.joblib"))
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print("Please ensure all model files are present in the models directory")
    raise

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Get form data
            model_type = request.form["model_type"]
            year = int(request.form["year"])
            month = int(request.form["month"])
            country = request.form["country"]
            attacks = int(request.form.get("attacks", 0))

            # Create input DataFrame
            input_data = pd.DataFrame({
                'Year': [year],
                'Month': [month],
                'Country': [country],
                'Previous_Attacks': [attacks]
            })

            # Make prediction based on model type
            if model_type == "training_model":
                result = risk_model.predict(input_data)[0]
                prediction = "High Risk" if result == 1 else "Low Risk"
            
            elif model_type == "regression_count":
                result = int(attack_count_model.predict(input_data)[0])
                prediction = f"Predicted number of attacks: {result}"
            
            elif model_type == "timeseries_model":
                # Create future dates for forecasting
                future = forecast_model.make_future_dataframe(periods=12, freq='M')
                forecast = forecast_model.predict(future)
                prediction = forecast[['ds', 'yhat']].tail(12).to_html(
                    classes='table table-striped',
                    float_format=lambda x: '%.2f' % x
                )
            
            elif model_type == "hotspot_model":
                result = hotspot_model.predict(input_data)[0]
                prediction = "Hotspot Area" if result == 1 else "Non-Hotspot Area"

        except ValueError as ve:
            error = f"Invalid input: {str(ve)}"
        except Exception as e:
            error = f"An error occurred: {str(e)}"
            print(f"Error details: {str(e)}")

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == "__main__":
    # Add debug=True during development, remove in production
    app.run(debug=True, port=5000)