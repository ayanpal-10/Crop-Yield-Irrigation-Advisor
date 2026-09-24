"""
app.py
------
Crop Yield & Irrigation Advisor - Streamlit prototype application.

This is a single-file Streamlit app. It:
1. Loads (or generates) the sample dataset.
2. Trains a simple Machine Learning model to predict crop yield.
3. Lets the user enter farm details in the sidebar.
4. Shows a predicted yield, an irrigation recommendation, a basic risk
   assessment, a plain-language advisory, and some data visualizations.

Run it with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

from generate_dataset import generate_and_save

DATA_PATH = "crop_data.csv"

# Features the ML model is trained on. Crop_Code and Soil_Type_Code are
# numeric versions of the Crop / Soil_Type text columns (see encode step).
FEATURE_COLUMNS = [
    "Crop_Code", "Soil_Type_Code", "Soil_Moisture", "Temperature", "Humidity",
    "Rainfall", "Nitrogen", "Phosphorus", "Potassium", "pH",
    "Area_Hectare", "Irrigation_Hours",
]
TARGET_COLUMN = "Yield_Ton_Per_Hectare"


# =====================================================================
# 1. DATA LOADING
# =====================================================================
@st.cache_data
def load_data():
    """
    Loads crop_data.csv. If it doesn't exist yet (first run), generates
    it automatically so the user doesn't have to run a separate script.
    """
    if not os.path.exists(DATA_PATH):
        generate_and_save(DATA_PATH)
    return pd.read_csv(DATA_PATH)


# =====================================================================
# 2. MODEL TRAINING
# =====================================================================
@st.cache_resource
def train_model(df):
    """
    Trains a RandomForestRegressor to predict Yield_Ton_Per_Hectare.
    Cached with st.cache_resource so this only runs once per session,
    not on every button click / rerun.
    """
    df = df.copy()

    # Machine learning models need numbers, not text, so we convert the
    # Crop and Soil_Type text columns into numeric codes using
    # LabelEncoder. We keep the encoders so we can apply the SAME
    # encoding later to whatever the user selects in the sidebar.
    crop_encoder = LabelEncoder()
    soil_encoder = LabelEncoder()
    df["Crop_Code"] = crop_encoder.fit_transform(df["Crop"])
    df["Soil_Type_Code"] = soil_encoder.fit_transform(df["Soil_Type"])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 80% of the data trains the model, 20% is held back to test it on
    # data it has never seen, which is how we get an honest R2 / MAE.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "crop_encoder": crop_encoder,
        "soil_encoder": soil_encoder,
        "mae": mae,
        "r2": r2,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def predict_yield(trained, crop, soil_type, soil_moisture, temperature, humidity,
                   rainfall, nitrogen, phosphorus, potassium, ph, area, irrigation_hours):
    """Builds one input row from the sidebar values and predicts yield."""
    crop_code = trained["crop_encoder"].transform([crop])[0]
    soil_code = trained["soil_encoder"].transform([soil_type])[0]

    input_row = pd.DataFrame([{
        "Crop_Code": crop_code,
        "Soil_Type_Code": soil_code,
        "Soil_Moisture": soil_moisture,
        "Temperature": temperature,
        "Humidity": humidity,
        "Rainfall": rainfall,
        "Nitrogen": nitrogen,
        "Phosphorus": phosphorus,
        "Potassium": potassium,
        "pH": ph,
        "Area_Hectare": area,
        "Irrigation_Hours": irrigation_hours,
    }])[FEATURE_COLUMNS]

    prediction = trained["model"].predict(input_row)[0]
    return round(float(prediction), 2)


# =====================================================================
# 3. IRRIGATION RECOMMENDATION (simple rule-based logic)
# =====================================================================
def irrigation_recommendation(soil_moisture, rainfall, temperature):
    """
    Simple, explainable rules (not a trained model) for irrigation advice.
    This is intentionally easy to read out loud in a viva.
    """
    # Step 1: base recommendation from soil moisture
    if soil_moisture < 35:
        level, duration = "High", "3-4 hours"
    elif soil_moisture < 60:
        level, duration = "Medium", "1-2 hours"
    else:
        level, duration = "Low", "0-1 hour"

    reason_parts = [f"soil moisture is {soil_moisture:.0f}%"]

    # Step 2: adjust based on recent rainfall
    if rainfall > 900 and level != "Low":
        # Plenty of rain recently -> step the recommendation down one level
        level = "Medium" if level == "High" else "Low"
        duration = "1-2 hours" if level == "Medium" else "0-1 hour"
        reason_parts.append("recent rainfall is high, so irrigation need is reduced")
    elif rainfall < 250 and soil_moisture < 45:
        # Very little rain AND dry soil -> bump the recommendation up
        level = "High"
        duration = "3-4 hours"
        reason_parts.append("rainfall is very low, which increases the need for irrigation")
    else:
        reason_parts.append(f"rainfall is {rainfall:.0f}mm")

    # Step 3: mention temperature for context (doesn't change the level,
    # just explains it — high temperature increases water loss)
    if temperature > 32:
        reason_parts.append("high temperature increases water loss from soil")

    reason = "; ".join(reason_parts).capitalize() + "."
    return level, duration, reason


# =====================================================================
# 4. FARM RISK ASSESSMENT
# =====================================================================
def risk_assessment(soil_moisture, temperature, rainfall, nitrogen, potassium, ph):
    """Checks a handful of simple thresholds and returns an overall risk level."""
    checks = []
    risk_points = 0

    if soil_moisture < 30:
        checks.append(("warn", "Soil moisture is low"))
        risk_points += 1
    else:
        checks.append(("ok", "Soil moisture is acceptable"))

    if temperature > 38:
        checks.append(("warn", "Temperature is excessively high"))
        risk_points += 1
    else:
        checks.append(("ok", "Temperature is within acceptable range"))

    if rainfall > 1800:
        checks.append(("warn", "Rainfall is very heavy (possible waterlogging)"))
        risk_points += 1
    elif rainfall < 200:
        checks.append(("warn", "Rainfall is very low"))
        risk_points += 1
    else:
        checks.append(("ok", "Rainfall is in a reasonable range"))

    if nitrogen < 60:
        checks.append(("warn", "Nitrogen level is low"))
        risk_points += 1
    else:
        checks.append(("ok", "Nitrogen level is sufficient"))

    if potassium < 30:
        checks.append(("warn", "Potassium level is low"))
        risk_points += 1
    else:
        checks.append(("ok", "Potassium level is sufficient"))

    if ph < 5.5 or ph > 7.8:
        checks.append(("warn", "Soil pH is outside the ideal range"))
        risk_points += 1
    else:
        checks.append(("ok", "Soil pH is suitable"))

    if risk_points >= 3:
        overall = "High"
    elif risk_points >= 1:
        overall = "Medium"
    else:
        overall = "Low"

    return overall, checks


# =====================================================================
# 5. FARMER ADVISORY (plain-language text generation)
# =====================================================================
def generate_advisory(crop, predicted_yield, irrigation_level, irrigation_duration,
                       risk_level, soil_moisture, rainfall):
    """
    Converts the technical results into a short, simple, farmer-friendly
    message. This is a template-based (rule-based) function for the
    prototype — no external AI service is called.

    # Future version: connect this function to an LLM API/Ollama
    # for multilingual and conversational farmer assistance.
    """
    moisture_desc = "low" if soil_moisture < 40 else ("moderate" if soil_moisture < 65 else "high")
    rainfall_desc = "limited" if rainfall < 400 else ("moderate" if rainfall < 1000 else "plentiful")

    message = (
        f"For your {crop} field, the prototype estimates a yield of about "
        f"{predicted_yield} tonnes per hectare. Soil moisture is currently {moisture_desc} "
        f"and rainfall has been {rainfall_desc}. "
        f"The recommended irrigation level is {irrigation_level.upper()} "
        f"(approximately {irrigation_duration}). "
    )

    if risk_level == "High":
        message += "Overall field risk is HIGH — please review the risk factors below before taking action. "
    elif risk_level == "Medium":
        message += "Overall field risk is MEDIUM — keep an eye on the flagged factors below. "
    else:
        message += "Overall field risk is LOW, conditions look reasonably favorable. "

    message += "Please check the field in person before irrigating, and avoid overwatering."
    return message


# =====================================================================
# 6. PATTERN INSIGHTS (computed from the dataset, not hard-coded)
# =====================================================================
def generate_pattern_insights(df):
    """
    Looks at simple correlations and group averages in the dataset and
    turns them into plain-language sentences. Values are calculated live
    from crop_data.csv, not hard-coded, so they reflect the actual data.
    """
    insights = []

    corr_moisture = df["Soil_Moisture"].corr(df[TARGET_COLUMN])
    corr_rainfall = df["Rainfall"].corr(df[TARGET_COLUMN])
    corr_temp = df["Temperature"].corr(df[TARGET_COLUMN])

    if corr_moisture > 0.15:
        insights.append(
            f"Higher soil moisture tends to go with higher yield in this dataset "
            f"(correlation: {corr_moisture:.2f})."
        )
    if corr_rainfall > 0.15:
        insights.append(
            f"Adequate rainfall is associated with higher yield in this dataset "
            f"(correlation: {corr_rainfall:.2f})."
        )

    # Compare average irrigation hours for the driest third of records
    # versus the rest, to produce the "low moisture -> more irrigation" insight.
    low_cutoff = df["Soil_Moisture"].quantile(0.33)
    dry_group = df[df["Soil_Moisture"] <= low_cutoff]
    wet_group = df[df["Soil_Moisture"] > low_cutoff]
    if dry_group["Irrigation_Hours"].mean() > wet_group["Irrigation_Hours"].mean():
        insights.append(
            "Records with low soil moisture tend to use more irrigation hours than "
            "records with higher soil moisture, as expected."
        )

    # Compare average yield for the hottest third vs the rest.
    hot_cutoff = df["Temperature"].quantile(0.67)
    hot_group = df[df["Temperature"] >= hot_cutoff]
    rest_group = df[df["Temperature"] < hot_cutoff]
    if hot_group[TARGET_COLUMN].mean() < rest_group[TARGET_COLUMN].mean():
        insights.append(
            "Records with very high temperature show somewhat lower average yield "
            "compared to the rest of the sample dataset."
        )

    if not insights:
        insights.append("No strong patterns were detected in the current sample dataset.")

    return insights


# =====================================================================
# 7. INPUT VALIDATION
# =====================================================================
def validate_inputs(area, rainfall, ph, soil_moisture, humidity, temperature):
    errors = []
    if area <= 0:
        errors.append("Area must be greater than 0 hectares.")
    if rainfall < 0:
        errors.append("Rainfall cannot be negative.")
    if not (0 <= ph <= 14):
        errors.append("Soil pH must be between 0 and 14.")
    if not (0 <= soil_moisture <= 100):
        errors.append("Soil moisture must be between 0 and 100%.")
    if not (0 <= humidity <= 100):
        errors.append("Humidity must be between 0 and 100%.")
    if not (-10 <= temperature <= 55):
        errors.append("Temperature must be in a realistic range (-10 to 55°C).")
    return errors


# =====================================================================
# 8. STREAMLIT PAGE CONFIG + LIGHT STYLING
# =====================================================================
st.set_page_config(page_title="Crop Yield & Irrigation Advisor", page_icon="🌾", layout="wide")

st.markdown("""
<style>
.feature-card {
    background-color: #FFFFFF;
    border: 1px solid #DCE7D4;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    height: 100%;
}
.feature-card h4 { margin-bottom: 6px; }
.disclaimer-box {
    background-color: #FFF6E5;
    border: 1px solid #F0D68A;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 9. HOME / HEADER
# =====================================================================
st.title("🌾 Crop Yield & Irrigation Advisor")
st.markdown("*AI-powered advisory for smarter, water-efficient farming*")
st.write(
    "This prototype combines crop, soil and weather information to estimate crop "
    "yield and provide irrigation recommendations."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="feature-card"><h4>🌱 Yield Prediction</h4>
    <p>Estimate expected crop yield using machine learning.</p></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="feature-card"><h4>💧 Smart Irrigation</h4>
    <p>Recommend irrigation based on soil and weather conditions.</p></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="feature-card"><h4>📊 Data Insights</h4>
    <p>Visualize relationships between agricultural factors and yield.</p></div>""", unsafe_allow_html=True)

st.divider()

# =====================================================================
# 10. LOAD DATA + TRAIN MODEL
# =====================================================================
df = load_data()
trained = train_model(df)

st.success(
    f"✅ Model trained successfully — R² Score: {trained['r2']:.2f} | "
    f"MAE: {trained['mae']:.2f} tons/hectare"
)

# =====================================================================
# 11. SIDEBAR INPUTS
# =====================================================================
st.sidebar.header("🌱 Farm Information")

crop = st.sidebar.selectbox("Select Crop", sorted(df["Crop"].unique()))
soil_type = st.sidebar.selectbox("Select Soil Type", sorted(df["Soil_Type"].unique()))
area = st.sidebar.number_input("Area (hectares)", min_value=0.1, value=1.0, step=0.1)
soil_moisture = st.sidebar.slider("Soil Moisture (%)", 0, 100, 45)
temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, value=600.0, step=10.0)
nitrogen = st.sidebar.slider("Nitrogen (kg/ha)", 0, 200, 100)
phosphorus = st.sidebar.slider("Phosphorus (kg/ha)", 0, 150, 60)
potassium = st.sidebar.slider("Potassium (kg/ha)", 0, 150, 50)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5, 0.1)
irrigation_hours = st.sidebar.number_input("Current Irrigation Hours", 0.0, 10.0, 2.0, 0.5)

generate_clicked = st.sidebar.button("🔍 Generate Advisory", type="primary")

# =====================================================================
# 12. RUN PREDICTION ON BUTTON CLICK (stored in session_state so it
#     stays visible even when the user later changes the chart selector)
# =====================================================================
if generate_clicked:
    errors = validate_inputs(area, rainfall, ph, soil_moisture, humidity, temperature)
    if errors:
        for e in errors:
            st.sidebar.error(e)
    else:
        predicted_yield = predict_yield(
            trained, crop, soil_type, soil_moisture, temperature, humidity,
            rainfall, nitrogen, phosphorus, potassium, ph, area, irrigation_hours
        )
        level, duration, reason = irrigation_recommendation(soil_moisture, rainfall, temperature)
        risk_level, risk_checks = risk_assessment(soil_moisture, temperature, rainfall, nitrogen, potassium, ph)
        advisory_text = generate_advisory(
            crop, predicted_yield, level, duration, risk_level, soil_moisture, rainfall
        )

        st.session_state["result"] = {
            "crop": crop, "predicted_yield": predicted_yield,
            "irrigation_level": level, "irrigation_duration": duration, "irrigation_reason": reason,
            "risk_level": risk_level, "risk_checks": risk_checks,
            "advisory_text": advisory_text, "soil_moisture": soil_moisture,
        }

# =====================================================================
# 13. DISPLAY RESULTS (if available)
# =====================================================================
if "result" in st.session_state:
    r = st.session_state["result"]

    st.subheader("Advisory Summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected Yield", f"{r['predicted_yield']} Ton/Ha")
    k2.metric("Irrigation", r["irrigation_level"])
    k3.metric("Risk", r["risk_level"])
    k4.metric("Soil Moisture", f"{r['soil_moisture']}%")

    st.divider()

    # ---- Yield prediction section ----
    st.subheader("📈 Yield Prediction")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Predicted Yield", f"{r['predicted_yield']} Ton/Ha")
        st.write(f"Model R² Score: **{trained['r2']:.2f}**")
        st.write(f"Model MAE: **{trained['mae']:.2f}** tons/hectare")
        st.caption("This is a prototype/estimated prediction, not an accurate real-world forecast.")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trained["y_test"], y=trained["y_pred"], mode="markers",
            marker=dict(color="#2C5F2D", opacity=0.6), name="Test predictions"
        ))
        min_v = float(min(trained["y_test"].min(), trained["y_pred"].min()))
        max_v = float(max(trained["y_test"].max(), trained["y_pred"].max()))
        fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                                  line=dict(color="#E8B84B", dash="dash"), name="Perfect prediction"))
        fig.update_layout(title="Actual vs Predicted Yield (test data)",
                           xaxis_title="Actual Yield (Ton/Ha)", yaxis_title="Predicted Yield (Ton/Ha)",
                           height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width='stretch')

    st.divider()

    # ---- Irrigation section ----
    st.subheader("💧 Irrigation Recommendation")
    color_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    st.markdown(f"**Recommended irrigation:** {color_map.get(r['irrigation_level'], '')} {r['irrigation_level'].upper()}")
    st.markdown(f"**Recommended duration:** {r['irrigation_duration']}")
    st.markdown(f"**Reason:** {r['irrigation_reason']}")
    st.markdown(
        '<div class="disclaimer-box">Prototype recommendation — actual irrigation should consider '
        'local agricultural guidance, crop stage and field conditions.</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ---- Risk section ----
    st.subheader("⚠️ Farm Risk Assessment")
    st.markdown(f"**Overall Risk: {r['risk_level'].upper()}**")
    st.write("Risk factors detected:")
    for status, msg in r["risk_checks"]:
        icon = "✓" if status == "ok" else "⚠"
        st.write(f"{icon} {msg}")

    st.divider()

    # ---- Farmer advisory section ----
    st.subheader("🌱 Farmer Advisory")
    st.info(r["advisory_text"])

    st.divider()

else:
    st.info("Fill in your farm details in the sidebar and click **Generate Advisory** to see results.")
    st.divider()

# =====================================================================
# 14. PATTERN INSIGHTS (always visible — computed from full dataset)
# =====================================================================
st.subheader("🔍 Pattern Insights")
st.caption("Patterns observed in the prototype dataset.")
for insight in generate_pattern_insights(df):
    st.write(f"- {insight}")

st.divider()

# =====================================================================
# 15. DATA VISUALIZATION
# =====================================================================
st.subheader("📊 Agricultural Insights")
chart_choice = st.selectbox(
    "Choose a chart",
    ["Crop vs Average Yield", "Rainfall vs Yield", "Soil Moisture vs Yield", "Temperature vs Yield"]
)

if chart_choice == "Crop vs Average Yield":
    avg_yield = df.groupby("Crop")[TARGET_COLUMN].mean().reset_index().sort_values(TARGET_COLUMN)
    fig = px.bar(avg_yield, x="Crop", y=TARGET_COLUMN, color="Crop",
                 title="Average Yield by Crop", labels={TARGET_COLUMN: "Avg Yield (Ton/Ha)"})
elif chart_choice == "Rainfall vs Yield":
    fig = px.scatter(df, x="Rainfall", y=TARGET_COLUMN, color="Crop",
                      title="Rainfall vs Yield", opacity=0.6)
elif chart_choice == "Soil Moisture vs Yield":
    fig = px.scatter(df, x="Soil_Moisture", y=TARGET_COLUMN, color="Crop",
                      title="Soil Moisture vs Yield", opacity=0.6)
else:
    fig = px.scatter(df, x="Temperature", y=TARGET_COLUMN, color="Crop",
                      title="Temperature vs Yield", opacity=0.6)

fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, width='stretch')

st.divider()

# =====================================================================
# 16. FUTURE SCOPE
# =====================================================================
st.subheader("🚀 Future Scope")
future_items = [
    "NASA POWER / OpenWeatherMap integration for live weather data",
    "SoilGrids integration for live soil property data",
    "Real government / research agricultural datasets",
    "IoT soil moisture sensors for real-time field readings",
    "Satellite NDVI data for crop-health monitoring",
    "Multilingual advisory output",
    "Voice interface for low-literacy users",
    "SMS / WhatsApp notifications for farmers",
    "LLM-based conversational assistant (GenAI)",
    "Government / NGO pilot deployment",
]
fcol1, fcol2 = st.columns(2)
for i, item in enumerate(future_items):
    (fcol1 if i % 2 == 0 else fcol2).write(f"- {item}")

st.divider()

# =====================================================================
# 17. FOOTER / DISCLAIMER
# =====================================================================
st.caption(
    "This is an academic prototype. Predictions and irrigation recommendations are based on "
    "the sample dataset and simplified rules. They should not be treated as professional "
    "agricultural advice."
)
