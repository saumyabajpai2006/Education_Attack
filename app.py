from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bcrypt import bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Load the dataset
df = pd.read_excel('analysis2/models/2020-2025-education-in-danger-incident-data.xlsx')

# Load the trained models
try:
    xgboost_model = joblib.load('analysis2/models/models/regression_model.joblib')
    timeseries_model = joblib.load('analysis2/models/models/timeseries_model.joblib')
    hotspot_model = joblib.load('analysis2/models/models/hotspot_model.joblib')
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # Create dummy models if loading fails
    xgboost_model = None
    timeseries_model = None
    hotspot_model = None

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email_or_username = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email_or_username).first() or User.query.filter_by(username=email_or_username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        existing_user = User.query.filter((User.username==username)|(User.email==email)).first()
        if existing_user:
            flash('Username or email already exists')
        else:
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route("/xgboost-prediction")
def xgboost_prediction():
    return render_template("xgboost_prediction.html")

@app.route("/timeseries-forecast")
def timeseries_forecast():
    return render_template("timeseries_forecast.html")

@app.route("/hotspot-detection")
def hotspot_detection():
    return render_template("hotspot_detection.html")

@app.route("/predict-attack-count", methods=['POST'])
def predict_attack_count():
    try:
        if xgboost_model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Get form data
        data = request.form
        
        # Create feature vector (simplified for demo)
        # In real scenario, you'd need to preprocess exactly like training
        features = np.zeros(50)  # Adjust size based on your model
        
        # Make prediction
        prediction = xgboost_model.predict([features])[0]
        
        return jsonify({
            "predicted_attacks": float(prediction),
            "message": "Prediction successful"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/forecast-future", methods=['POST'])
def forecast_future():
    try:
        if timeseries_model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Get number of years to forecast
        years = int(request.form.get('years', 5))
        
        # Create future dates
        future_dates = pd.date_range(
            start=df['Date'].max() + timedelta(days=1),
            periods=years*365,
            freq='D'
        )
        
        # Make forecast (simplified)
        # In real scenario, use Prophet model properly
        forecast_data = {
            "dates": [d.strftime('%Y-%m-%d') for d in future_dates],
            "predictions": np.random.randint(100, 200, len(future_dates)).tolist()
        }
        
        return jsonify(forecast_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect-hotspots", methods=['POST'])
def detect_hotspots():
    try:
        if hotspot_model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Get form data
        country = request.form.get('country', '')
        admin1 = request.form.get('admin1', '')
        
        # Simple hotspot detection logic
        if country and admin1:
            # Filter data for the specified region
            region_data = df[(df['Country'] == country) & (df['Admin 1'] == admin1)]
            
            if len(region_data) > 0:
                total_attacks = region_data[['Attacks on Schools', 'Attacks on Universities']].sum().sum()
                
                # Simple risk assessment
                if total_attacks > 10:
                    risk_level = "High Risk"
                elif total_attacks > 5:
                    risk_level = "Medium Risk"
                else:
                    risk_level = "Low Risk"
                
                return jsonify({
                    "region": f"{country}, {admin1}",
                    "total_attacks": int(total_attacks),
                    "risk_level": risk_level,
                    "message": "Hotspot analysis completed"
                })
            else:
                return jsonify({"error": "No data found for this region"}), 404
        
        return jsonify({"error": "Please provide country and admin region"}), 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists("users.db"):
        with app.app_context():
            db.create_all()
    app.run(debug=True)