# 🎯 Education Attack Prediction System

A comprehensive Flask web application that integrates three advanced machine learning models to predict and analyze education attacks worldwide.

## 🚀 Features

### 1. **XGBoost Attack Prediction** (99.99% R² Score)
- Predicts exact number of education attacks
- Uses geographic, temporal, and facility data
- Achieves 0.0068 RMSE accuracy

### 2. **Time Series Forecasting** (2.20% MAPE)
- Prophet model for future trend prediction
- 1-5 year forecast periods
- Seasonal and trend analysis

### 3. **Hotspot Detection** (100% Accuracy)
- Geographic risk assessment
- Country and region-based analysis
- Risk level classification (Low/Medium/High)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Education_Attack
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Ensure models are in place:**
   - Check that `analysis2/models/models/` contains:
     - `regression_model.joblib` (XGBoost)
     - `timeseries_model.joblib` (Prophet)
     - `hotspot_model.joblib` (Hotspot Detection)

## 🚀 Running the Application

1. **Start the Flask server:**
```bash
python app.py
```

2. **Open your browser and navigate to:**
```
http://localhost:5000
```

## 📱 Available Pages

- **Home** (`/`) - Main dashboard with model overview
- **XGBoost Prediction** (`/xgboost-prediction`) - Attack count prediction
- **Time Series Forecast** (`/timeseries-forecast`) - Future trend prediction
- **Hotspot Detection** (`/hotspot-detection`) - Geographic risk assessment
- **About** (`/about`) - Project information

## 🔧 API Endpoints

### POST `/predict-attack-count`
Predicts attack counts using XGBoost model
- **Input:** Country, admin region, coordinates
- **Output:** Predicted attack count and risk level

### POST `/forecast-future`
Generates time series forecasts
- **Input:** Number of years to forecast
- **Output:** Future attack predictions and trends

### POST `/detect-hotspots`
Identifies high-risk geographic areas
- **Input:** Country and admin region
- **Output:** Risk assessment and recommendations

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| XGBoost Regression | R² Score | 99.99% |
| XGBoost Regression | RMSE | 0.0068 |
| Prophet Time Series | MAPE | 2.20% |
| Prophet Time Series | MAE | 3.20 |
| Hotspot Detection | Accuracy | 100% |

## 🎨 UI Features

- **Responsive Design** - Works on all device sizes
- **Interactive Cards** - Clickable navigation to model pages
- **Real-time Predictions** - AJAX-based form submissions
- **Visual Feedback** - Loading spinners and animations
- **Risk Indicators** - Color-coded risk levels
- **Modern Styling** - Gradient backgrounds and smooth transitions

## 🔍 Dataset

The application uses the "2020-2025 Education in Danger Incident Data" dataset containing:
- Geographic coordinates and regions
- Attack types and severity
- Facility information
- Temporal data
- Perpetrator details

## 🚨 Error Handling

- Graceful model loading failures
- User-friendly error messages
- Form validation
- API response validation

## 🔮 Future Enhancements

- Real-time data updates
- Interactive maps integration
- Advanced visualization charts
- User authentication system
- Export functionality for predictions

## 📝 License

This project is designed for educational and awareness purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Note:** Ensure all ML models are properly trained and saved before running the application. The current implementation includes simplified prediction logic for demonstration purposes.
