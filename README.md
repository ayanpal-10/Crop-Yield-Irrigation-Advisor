# 🌾 Crop Yield & Irrigation Advisor

*AI-powered advisory for smarter, water-efficient farming*

## 1. Problem Statement

Farmers, especially small and marginal ones, often lack real-time access to
weather, soil, and yield insights. This leads to inefficient irrigation
(over- or under-watering), unpredictable crop yields, and limited access to
expert agricultural advisory. This project is a first-round academic
prototype of a system that uses crop, soil, and weather data to estimate
crop yield and recommend irrigation, presented in a simple, farmer-friendly
way.

## 2. Objectives

1. Predict expected crop yield using a Machine Learning model.
2. Recommend irrigation requirements using simple, explainable rules.
3. Provide a basic farm risk assessment.
4. Visualize key agricultural relationships through charts.
5. Present all of this through a simple, easy-to-use dashboard.

## 3. Features

- 🌱 **Yield Prediction** — Random Forest model estimates yield in tons/hectare
- 💧 **Smart Irrigation** — rule-based recommendation (level + duration + reason)
- ⚠️ **Farm Risk Assessment** — flags soil/weather factors outside safe ranges
- 🌱 **Farmer Advisory** — plain-language summary of all the above
- 🔍 **Pattern Insights** — simple relationships computed live from the dataset
- 📊 **Data Visualization** — interactive Plotly charts (crop, rainfall, moisture, temperature vs yield)

## 4. Technologies Used

| Layer | Tool |
|---|---|
| Language | Python |
| UI / Dashboard | Streamlit |
| Data handling | Pandas, NumPy |
| Machine Learning | Scikit-learn (RandomForestRegressor) |
| Visualization | Plotly |
| Storage | CSV file |

No external APIs, databases, or paid services are used — the app works
completely offline after installation.

## 5. Dataset

`crop_data.csv` is a generated **sample** dataset (~900 rows) covering 7
common crops (Wheat, Rice, Maize, Cotton, Sugarcane, Potato, Tomato). Each
row simulates realistic growing conditions (soil moisture, temperature,
rainfall, N-P-K levels, pH, irrigation hours) and a resulting yield value,
generated using simplified science-based rules so the data has genuine,
learnable patterns rather than being random. See `generate_dataset.py` for
exactly how it's built.

## 6. Machine Learning Model

- **Algorithm:** RandomForestRegressor (scikit-learn)
- **Target:** `Yield_Ton_Per_Hectare`
- **Features:** Crop, Soil Type, Soil Moisture, Temperature, Humidity,
  Rainfall, Nitrogen, Phosphorus, Potassium, pH, Area, Irrigation Hours
- **Split:** 80% train / 20% test (`train_test_split`, `random_state=42`)
- **Evaluation:** R² Score and Mean Absolute Error (MAE), shown live in the app

> ⚠️ This is a prototype/estimated prediction based on a sample dataset,
> **not** an accurate real-world agricultural forecast.

## 7. Installation

```bash
pip install -r requirements.txt
```

## 8. How to Run

```bash
streamlit run app.py
```

The app will open automatically in your browser (usually at
`http://localhost:8501`). The dataset is generated automatically on first
run if `crop_data.csv` isn't already present.

## 9. Future Improvements

1. NASA POWER / OpenWeatherMap integration for live weather data
2. SoilGrids integration for live soil property data
3. Real government / research agricultural datasets
4. IoT soil moisture sensors for real-time field readings
5. Satellite NDVI data for crop-health monitoring
6. Multilingual advisory output
7. Voice interface for low-literacy users
8. SMS / WhatsApp notifications for farmers
9. LLM-based conversational assistant (GenAI) — `generate_advisory()` in
   `app.py` is already structured so an LLM API/Ollama call can be dropped
   in later
10. Government / NGO pilot deployment

## 10. Limitations

This is an academic prototype. Predictions and irrigation recommendations
are based on a sample dataset and simplified rules. They should not be
treated as professional agricultural advice.


Link - https://crop-yield-irrigation-advisor-vuyrr53yzppgiv9h4bkpsq.streamlit.app/
