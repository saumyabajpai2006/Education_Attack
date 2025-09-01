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

app = Flask(__name__)

df=pd.read_excel('2020-2025-education-in-danger-incident-data.xlsx')

#loading the models
risk_model=joblib.load('analysis/models/training_model.pkl')
regression_model=joblib.load('analysis/models/regression_model.pkl')
forecast_model=joblib.load('analysis/models/timeseries_model.pkl')
hotspot_model=joblib.load('analysis/models/hotspot_model.pkl')

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/login")
def login():
    return render_template("login.html")

if __name__ == "__main__":
    app.run(debug=True)