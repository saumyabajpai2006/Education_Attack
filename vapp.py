from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # For Flask (no GUI backend)
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
risk_model = joblib.load(os.path.join("models", "training_model.joblib"))
attack_count_model = joblib.load(os.path.join("models", "attack_count_model.joblib"))
forecast_model = joblib.load(os.path.join("models", "forecast_model.joblib"))
hotspot_model = joblib.load(os.path.join("models", "hotspot_model.joblib"))

# Load dataset (for visualization)
file_path = "e0c5e4d9-aea6-462f-92f1-f42669af5fc9.xlsx"
data = pd.read_excel(file_path, sheet_name="Sheet1")
data = data.fillna(0)
data["Total_Attacks"] = (
    data["Attacks on Schools"] +
    data["Attacks on Universities"] +
    data["Military Occupation of Education facility"] +
    data["Arson attack on education facility"] +
    data["Attacks on Students and Teachers"]
)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        model_type = request.form["model_type"]
        year = int(request.form.get("year", 2024))
        attacks = int(request.form.get("attacks", 10))

        if model_type == "risk":
            input_data = np.array([[year, attacks]])
            prediction = risk_model.predict(input_data)[0]

        elif model_type == "attack_count":
            input_data = np.array([[year, attacks]])
            prediction = attack_count_model.predict(input_data)[0]

        elif model_type == "forecast":
            future = forecast_model.make_future_dataframe(periods=5, freq="Y")
            forecast = forecast_model.predict(future)
            prediction = forecast[["ds", "yhat"]].tail(5).to_html()

        elif model_type == "hotspot":
            input_data = np.array([[year, attacks]])
            prediction = hotspot_model.predict(input_data)[0]

    return render_template("index.html", prediction=prediction)


# 🔹 Visualization Page
@app.route("/visualizations")
def visualizations():
    # 1. Yearly Trend
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="Year", y="Total_Attacks", data=data, marker="o")
    plt.title("Yearly Total Attacks Trend")
    plt.xlabel("Year")
    plt.ylabel("Total Attacks")
    plt.tight_layout()
    plt.savefig("static/yearly_trend.png")
    plt.close()

    # 2. Attack Type Distribution
    plt.figure(figsize=(8, 5))
    attack_types = [
        "Attacks on Schools",
        "Attacks on Universities",
        "Military Occupation of Education facility",
        "Arson attack on education facility",
        "Attacks on Students and Teachers"
    ]
    data[attack_types].sum().plot(kind="bar", color="tomato")
    plt.title("Distribution of Attack Types (2020–2025)")
    plt.ylabel("Number of Attacks")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/attack_types.png")
    plt.close()

    # 3. Heatmap Correlation
    plt.figure(figsize=(8, 6))
    corr = data[attack_types + ["Total_Attacks"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Between Attack Features")
    plt.tight_layout()
    plt.savefig("static/heatmap.png")
    plt.close()

    return render_template("visualization.html")

if __name__ == "__main__":
    app.run(debug=True)
