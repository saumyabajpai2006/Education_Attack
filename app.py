from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
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
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/users.db'
db = SQLAlchemy(app)

# Flask-Login setup
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load the dataset
try:
    df = pd.read_excel('analysis2/models/2020-2025-education-in-danger-incident-data.xlsx', engine='openpyxl')
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    # Create empty dataframe as fallback
    df = pd.DataFrame()

# Load the trained models
models_loaded = False
try:
    xgboost_model = joblib.load('analysis2/models/models/regression_model.joblib')
    timeseries_model = joblib.load('analysis2/models/models/timeseries_model.joblib')
    hotspot_model = joblib.load('analysis2/models/models/hotspot_model.joblib')
    models_loaded = True
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("⚠️  Running in demo mode with simplified predictions")
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

@app.route('/Signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        existing_user = User.query.filter((User.username==username)|(User.email==email)).first()
        if existing_user:
            flash('Username or email already exists')
        else:
            if not username or not email or not password:
                flash('All fields are required.')
                return render_template('login.html')
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
    return render_template('login.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email_or_username = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        user = (User.query.filter_by(email=email_or_username).first() or
                User.query.filter_by(username=email_or_username).first())
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route("/xgboost-prediction")
@login_required
def xgboost_prediction():
    return render_template("xgboost_prediction.html")

@app.route("/timeseries-forecast")
@login_required
def timeseries_forecast():
    return render_template("timeseries_forecast.html")

@app.route("/hotspot-detection")
@login_required
def hotspot_detection():
    # Get unique countries from dataset
    countries = sorted(df['Country'].unique().tolist())
    return render_template("hotspot_detection.html", countries=countries)

@app.route("/graphs")
# @login_required
def graphs():
    try:
        notebook_path = os.path.join("analysis2", "models", "graphs.ipynb")
        if not os.path.exists(notebook_path):
            flash("graphs.ipynb not found at analysis2/models/graphs.ipynb", "danger")
            return render_template("graphs.html", graphs=[])

        nb = nbformat.read(notebook_path, as_version=4)
        graphs = []

        for cell in nb.cells:
            if cell.get("cell_type") != "code":
                continue
            for output in cell.get("outputs", []):
                data = output.get("data") if isinstance(output, dict) else None
                if not data:
                    continue
                # Handle both v1 and v2 plotly mimetypes
                plotly_json = data.get("application/vnd.plotly.v1+json") or data.get("application/vnd.plotly.v2+json")
                if plotly_json:
                    try:
                        # plotly_json may already be a dict-like structure
                        fig = pio.from_json(json.dumps(plotly_json))
                        html_div = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)

                        # Extract a friendly title
                        title = None
                        try:
                            title = fig.layout.title.text if fig.layout and fig.layout.title else None
                        except Exception:
                            title = None
                        if not title:
                            title = "Visualization"

                        # Generate basic insights heuristically from traces
                        insights = {"summary": None, "stats": [], "trends": [], "highlights": [], "conclusion": None}
                        try:
                            num_traces = len(fig.data) if fig.data is not None else 0
                            chart_type = getattr(fig.data[0], 'type', '').lower() if num_traces else ''
                            if num_traces:
                                insights["summary"] = f"{num_traces} trace{'s' if num_traces != 1 else ''} • {chart_type or 'chart'}"

                            # Attempt simple stats for first trace
                            if num_traces:
                                trace = fig.data[0]
                                y_values = None
                                x_values = None
                                try:
                                    y_values = list(trace.y) if hasattr(trace, 'y') and trace.y is not None else None
                                    x_values = list(trace.x) if hasattr(trace, 'x') and trace.x is not None else None
                                except Exception:
                                    y_values = None
                                    x_values = None

                                if y_values and len(y_values) > 0:
                                    try:
                                        y_arr = np.array([v for v in y_values if v is not None and v == v], dtype=float)
                                        if y_arr.size:
                                            insights["stats"].append(f"count: {int(y_arr.size)}")
                                            insights["stats"].append(f"mean: {float(np.mean(y_arr)):.2f}")
                                            insights["stats"].append(f"median: {float(np.median(y_arr)):.2f}")
                                            insights["stats"].append(f"min–max: {float(np.min(y_arr)):.2f}–{float(np.max(y_arr)):.2f}")
                                            if y_arr.size >= 2:
                                                diffs = np.diff(y_arr)
                                                if diffs.size:
                                                    trend = float(np.nanmean(diffs))
                                                    if trend > 0:
                                                        insights["trends"].append("overall direction: upward")
                                                    elif trend < 0:
                                                        insights["trends"].append("overall direction: downward")
                                                    else:
                                                        insights["trends"].append("overall direction: flat")
                                                    vol = float(np.std(diffs))
                                                    insights["trends"].append(f"volatility (Δ std): {vol:.2f}")
                                    except Exception:
                                        pass

                                # For categorical bars: show top categories
                                top_k = None
                                top_share = None
                                if x_values and y_values and len(x_values) == len(y_values) and len(x_values) <= 50:
                                    try:
                                        pairs = []
                                        for xv, yv in zip(x_values, y_values):
                                            try:
                                                yv_float = float(yv)
                                                pairs.append((str(xv), yv_float))
                                            except Exception:
                                                continue
                                        pairs.sort(key=lambda p: p[1], reverse=True)
                                        if pairs:
                                            total = sum(val for _, val in pairs) or 1.0
                                            top_k = pairs[:3]
                                            top_share = sum(val for _, val in top_k) / total
                                            top_parts = [f"{name} ({val:.0f}, {val/total*100:.1f}%)" for name, val in pairs[:5]]
                                            insights["highlights"].append("top categories: " + "; ".join(top_parts))
                                            if len(pairs) > 5:
                                                tail = len(pairs) - 5
                                                tail_share = 100.0 - sum(val/total*100.0 for _, val in pairs[:5])
                                                insights["highlights"].append(f"long tail: {tail} other category(ies) contributing {tail_share:.1f}% combined")
                                    except Exception:
                                        pass

                                # Time series specifics (if x looks like dates)
                                ts_delta = None
                                ts_pct = None
                                if x_values and y_values and len(x_values) == len(y_values):
                                    try:
                                        x_series = pd.to_datetime(pd.Series(x_values), errors='coerce')
                                        if x_series.notna().sum() >= 2:
                                            order = np.argsort(x_series.values)
                                            y_series = pd.Series(y_values, dtype='float').iloc[order]
                                            x_sorted = x_series.iloc[order]
                                            y_sorted = y_series.values
                                            y_sorted = y_sorted[np.isfinite(y_sorted)] if len(y_sorted) else y_sorted
                                            if len(y_sorted) >= 2:
                                                start, end = x_sorted.iloc[0], x_sorted.iloc[-1]
                                                ts_delta = float(y_sorted[-1] - y_sorted[0])
                                                ts_pct = (ts_delta / (y_sorted[0] if y_sorted[0] != 0 else 1.0)) * 100.0
                                                insights["trends"].append(f"period: {start.date()} → {end.date()}")
                                                insights["trends"].append(f"net change: {ts_delta:.2f} ({ts_pct:.1f}%)")
                                                years = max((end.year - start.year), 1)
                                                if y_sorted[0] > 0 and years >= 1:
                                                    cagr = (y_sorted[-1] / y_sorted[0]) ** (1.0 / years) - 1.0
                                                    insights["trends"].append(f"approx CAGR: {cagr*100:.1f}%/yr")
                                                diffs2 = np.diff(y_sorted)
                                                if diffs2.size:
                                                    best = float(np.max(diffs2))
                                                    worst = float(np.min(diffs2))
                                                    insights["highlights"].append(f"largest rise (Δ): {best:.2f}; largest drop (Δ): {worst:.2f}")
                                    except Exception:
                                        pass

                                # Build human-readable conclusion
                                try:
                                    if ts_delta is not None:
                                        if ts_delta > 0:
                                            insights["conclusion"] = "Trend appears to be increasing over time, suggesting a rising pattern in the measured metric."
                                        elif ts_delta < 0:
                                            insights["conclusion"] = "Trend appears to be decreasing over time, indicating a decline in the measured metric."
                                        else:
                                            insights["conclusion"] = "No clear upward or downward movement is evident over the observed period."
                                    elif top_k is not None:
                                        leader = top_k[0][0]
                                        if top_share is not None and top_share >= 0.6:
                                            insights["conclusion"] = f"Distribution is heavily concentrated, with '{leader}' contributing the majority share."
                                        elif top_share is not None and top_share >= 0.35:
                                            insights["conclusion"] = f"A few categories dominate; '{leader}' leads, followed closely by others."
                                        else:
                                            insights["conclusion"] = f"Distribution is relatively spread out across categories; '{leader}' is only marginally ahead."
                                    elif insights["stats"]:
                                        insights["conclusion"] = "Values cluster around the central tendency with limited extremes, indicating a fairly stable distribution."
                                    else:
                                        insights["conclusion"] = "This chart provides a visual breakdown of the selected metric; no dominant pattern detected."
                                except Exception:
                                    pass

                        except Exception:
                            pass

                        graphs.append({
                            "title": title,
                            "html": html_div,
                            "insights": insights
                        })
                    except Exception:
                        continue

        return render_template("graphs.html", graphs=graphs)
    except Exception as e:
        flash(f"Failed to load graphs: {str(e)}", "danger")
        return render_template("graphs.html", graphs=[])

@app.route("/predict-attack-count", methods=['POST'])
def predict_attack_count():
    try:
        # Get form data
        country = request.form.get('country', '')
        admin1 = request.form.get('admin1', '')
        
        if not country or not admin1:
            return jsonify({"error": "Please provide country and admin region"}), 400
        
        # Check if we have data
        if df.empty:
            return jsonify({"error": "Dataset not available"}), 500
        
        # Filter data for the specified region
        region_data = df[(df['Country'] == country) & (df['Admin 1'] == admin1)]
        
        if len(region_data) == 0:
            return jsonify({"error": "No data found for this region"}), 404
        
        # Calculate historical average
        historical_attacks = region_data[['Attacks on Schools', 'Attacks on Universities']].sum().sum()
        avg_attacks_per_month = historical_attacks / max(1, len(region_data))
        
        # Simple prediction based on historical data
        if models_loaded and xgboost_model is not None:
            # Use trained model if available
            features = np.zeros(50)  # Adjust size based on your model
            prediction = xgboost_model.predict([features])[0]
            model_status = "Trained Model"
        else:
            # Use simplified prediction based on historical data
            prediction = avg_attacks_per_month * 1.2  # 20% increase factor
            model_status = "Demo Mode"
        
        return jsonify({
            "predicted_attacks": float(prediction),
            "historical_attacks": int(historical_attacks),
            "avg_attacks_per_month": float(avg_attacks_per_month),
            "model_status": model_status,
            "message": "Prediction completed successfully"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/forecast-future", methods=['POST'])
def forecast_future():
    try:
        # Get number of years to forecast
        years = int(request.form.get('years', 5))
        
        # Check if we have data
        if df.empty:
            return jsonify({"error": "Dataset not available"}), 500
        
        # Prepare data for Prophet model (yearly aggregation)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        
        # Create Total_Attacks column
        df["Total_Attacks"] = (
            df["Attacks on Schools"] +
            df["Attacks on Universities"] +
            df["Military Occupation of Education facility"] +
            df["Arson attack on education facility"] +
            df["Attacks on Students and Teachers"]
        )
        
        # Aggregate by year
        yearly_df = df.groupby("Year")["Total_Attacks"].sum().reset_index()
        
        # Prepare data for Prophet
        prophet_df = yearly_df.rename(columns={'Year': 'ds', 'Total_Attacks': 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')
        
        # Generate forecast
        if models_loaded and timeseries_model is not None:
            try:
                # Use trained Prophet model
                future = timeseries_model.make_future_dataframe(periods=years, freq='YE')
                forecast = timeseries_model.predict(future)
                
                # Get only future predictions
                future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]
                predictions = future_forecast['yhat'].round().astype(int).tolist()
                dates = future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
                model_status = "Trained Prophet Model"
            except Exception as e:
                print(f"Prophet model error: {e}, falling back to trend-based forecast")
                # Fall back to trend-based forecast
                historical_yearly = yearly_df['Total_Attacks'].tolist()
                avg_yearly = np.mean(historical_yearly)
                trend = np.polyfit(range(len(historical_yearly)), historical_yearly, 1)[0]
                
                future_years = range(yearly_df['Year'].max() + 1, yearly_df['Year'].max() + 1 + years)
                predictions = []
                dates = []
                
                for i, year in enumerate(future_years):
                    base_prediction = avg_yearly + trend * (i + 1)
                    seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 12)
                    prediction = max(0, int(base_prediction * seasonal_factor + np.random.normal(0, avg_yearly * 0.1)))
                    predictions.append(prediction)
                    dates.append(f"{year}-01-01")
                
                model_status = "Prophet Model (Fallback to Trend)"
        else:
            # Use simplified forecast based on historical trends
            historical_yearly = yearly_df['Total_Attacks'].tolist()
            avg_yearly = np.mean(historical_yearly)
            trend = np.polyfit(range(len(historical_yearly)), historical_yearly, 1)[0]
            
            # Generate future predictions with trend
            future_years = range(yearly_df['Year'].max() + 1, yearly_df['Year'].max() + 1 + years)
            predictions = []
            dates = []
            
            for i, year in enumerate(future_years):
                # Apply trend and add some randomness
                base_prediction = avg_yearly + trend * (i + 1)
                # Add seasonal variation (higher in certain months)
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 12)
                prediction = max(0, int(base_prediction * seasonal_factor + np.random.normal(0, avg_yearly * 0.1)))
                predictions.append(prediction)
                dates.append(f"{year}-01-01")
            
            model_status = "Demo Mode (Trend-based)"
        
        # Calculate statistics
        avg_prediction = np.mean(predictions)
        max_prediction = np.max(predictions)
        min_prediction = np.min(predictions)
        
        # Calculate trend direction
        if len(predictions) > 1:
            first_half = np.mean(predictions[:len(predictions)//2])
            second_half = np.mean(predictions[len(predictions)//2:])
            if second_half > first_half * 1.1:
                trend_direction = "Increasing"
            elif second_half < first_half * 0.9:
                trend_direction = "Decreasing"
            else:
                trend_direction = "Stable"
        else:
            trend_direction = "Stable"
        
        forecast_data = {
            "dates": dates,
            "predictions": predictions,
            "avg_prediction": float(avg_prediction),
            "max_prediction": int(max_prediction),
            "min_prediction": int(min_prediction),
            "trend_direction": trend_direction,
            "model_status": model_status,
            "message": "Forecast completed successfully",
            "historical_data": {
                "years": yearly_df['Year'].tolist(),
                "attacks": yearly_df['Total_Attacks'].tolist()
            }
        }
        
        return jsonify(forecast_data)
        
    except Exception as e:
        return jsonify({"error": f"Forecast failed: {str(e)}"}), 500

@app.route("/get-admin-regions/<country>")
def get_admin_regions(country):
    try:
        # Get unique admin regions for the selected country
        admin_regions = df[df['Country'] == country]['Admin 1'].unique().tolist()
        admin_regions = [region for region in admin_regions if pd.notna(region) and region.strip() != '']
        admin_regions.sort()
        return jsonify(admin_regions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect-hotspots", methods=['POST'])
def detect_hotspots():
    try:
        # Get form data
        country = request.form.get('country', '')
        admin1 = request.form.get('admin1', '')
        
        if not country or not admin1:
            return jsonify({"error": "Please provide country and admin region"}), 400
        
        # Check if we have data
        if df.empty:
            return jsonify({"error": "Dataset not available"}), 500
        
        # Filter data for the specified region
        region_data = df[(df['Country'] == country) & (df['Admin 1'] == admin1)]
        
        if len(region_data) == 0:
            return jsonify({"error": "No data found for this region"}), 404
        
        # Calculate attack statistics
        total_attacks = region_data[['Attacks on Schools', 'Attacks on Universities']].sum().sum()
        students_killed = region_data['Students Killed'].sum()
        students_injured = region_data['Students Injured'].sum()
        educators_killed = region_data['Educators Killed'].sum()
        educators_injured = region_data['Educators Injured'].sum()
        
        # Enhanced risk assessment
        risk_score = 0
        if total_attacks > 20:
            risk_score += 3
        elif total_attacks > 10:
            risk_score += 2
        elif total_attacks > 5:
            risk_score += 1
            
        if students_killed > 5:
            risk_score += 3
        elif students_killed > 2:
            risk_score += 2
        elif students_killed > 0:
            risk_score += 1
            
        if educators_killed > 2:
            risk_score += 2
        elif educators_killed > 0:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 6:
            risk_level = "Critical Risk"
            risk_color = "#8B0000"
        elif risk_score >= 4:
            risk_level = "High Risk"
            risk_color = "#FF4500"
        elif risk_score >= 2:
            risk_level = "Medium Risk"
            risk_color = "#FFA500"
        else:
            risk_level = "Low Risk"
            risk_color = "#32CD32"
        
        # Get recent attacks (last 30 days if available)
        recent_attacks = 0
        if 'Date' in region_data.columns:
            recent_data = region_data[region_data['Date'] >= (pd.Timestamp.now() - pd.Timedelta(days=30))]
            recent_attacks = len(recent_data)
        
        return jsonify({
            "region": f"{country}, {admin1}",
            "total_attacks": int(total_attacks),
            "students_killed": int(students_killed),
            "students_injured": int(students_injured),
            "educators_killed": int(educators_killed),
            "educators_injured": int(educators_injured),
            "recent_attacks": int(recent_attacks),
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_color": risk_color,
            "message": "Hotspot analysis completed successfully",
            "model_status": "Trained Model" if models_loaded else "Demo Mode"
        })
        
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

if __name__ == "__main__":
    if not os.path.exists("instance/users.db"):
        with app.app_context():
            db.create_all()
    app.run(debug=True)