import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from warehouse_etl import load_warehouse_dataset
from ml_engine import train_and_evaluate_models, mine_association_rules
from services import fetch_live_weather, calculate_irrigation_recommendation, generate_genai_advisory

st.set_page_config(page_title="Crop Yield & Smart Irrigation Advisor", page_icon="🌾", layout="wide")

@st.cache_data
def get_warehouse_data():
    return load_warehouse_dataset()

@st.cache_resource
def get_trained_models(df):
    return train_and_evaluate_models(df)

@st.cache_data
def get_rules(df):
    return mine_association_rules(df)

df = get_warehouse_data()
model_package = get_trained_models(df)
benchmarks = model_package["benchmarks"]
best_model_name = model_package["best_model_name"]
best_pipeline = model_package["best_pipeline"]

st.sidebar.title("🌾 Farm Parameters")
crops_list = sorted(df['Crop'].unique().tolist())
states_list = sorted(df['State'].unique().tolist())
seasons_list = sorted(df['Season'].unique().tolist())

crop_input = st.sidebar.selectbox("Crop", crops_list, index=crops_list.index("Wheat") if "Wheat" in crops_list else 0)
state_input = st.sidebar.selectbox("State", states_list, index=states_list.index("Punjab") if "Punjab" in states_list else 0)
season_input = st.sidebar.selectbox("Season", seasons_list, index=seasons_list.index("Rabi") if "Rabi" in seasons_list else 0)
area_input = st.sidebar.number_input("Area (Hectares)", min_value=0.1, value=2.0, step=0.5)

avg_rain = float(df[df['State'] == state_input]['Annual_Rainfall'].mean())
avg_fert = float(df[df['Crop'] == crop_input]['Fertilizer'].mean() / max(df[df['Crop'] == crop_input]['Area'].mean(), 1.0)) * area_input
avg_pest = float(df[df['Crop'] == crop_input]['Pesticide'].mean() / max(df[df['Crop'] == crop_input]['Area'].mean(), 1.0)) * area_input

rainfall_input = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, value=round(avg_rain, 1))
fertilizer_input = st.sidebar.number_input("Fertilizer (kg)", min_value=0.0, value=round(avg_fert, 1))
pesticide_input = st.sidebar.number_input("Pesticide (kg)", min_value=0.0, value=round(avg_pest, 1))
soil_moisture_input = st.sidebar.slider("Soil Moisture (%)", min_value=5, max_value=95, value=45)

gemini_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password")
language_choice = st.sidebar.selectbox("Language", ["English", "Hindi", "Punjabi"])

live_weather = fetch_live_weather(state_input)

st.title("🌾 Crop Yield & Smart Irrigation Advisory System")
st.caption(f"Enterprise Analytics on {len(df):,} Agricultural Records • Live Weather via Open-Meteo")

tab_advisory, tab_warehouse, tab_models, tab_mining = st.tabs([
    "🚀 Advisory & Yield Forecast", "🏛️ Data Warehouse", "🤖 ML Benchmarking", "🔍 Pattern Mining"
])

with tab_advisory:
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Live Temp", f"{live_weather['temperature']} °C")
    w2.metric("Live Humidity", f"{live_weather['humidity']} %")
    w3.metric("Precipitation", f"{live_weather['precipitation']} mm")
    w4.metric("Stream Status", live_weather['source'])

    input_payload = pd.DataFrame([{"Crop": crop_input, "State": state_input, "Season": season_input, "Area": area_input, "Annual_Rainfall": rainfall_input, "Fertilizer": fertilizer_input, "Pesticide": pesticide_input}])
    pred_yield = float(best_pipeline.predict(input_payload)[0])
    irr_level, irr_hours, irr_reason = calculate_irrigation_recommendation(crop_input, soil_moisture_input, rainfall_input, live_weather['temperature'])
    advisory = generate_genai_advisory(crop_input, state_input, season_input, pred_yield, irr_level, irr_hours, irr_reason, gemini_key, language_choice)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Yield", f"{pred_yield:.2f} Ton/Ha", delta=f"{pred_yield * area_input:.1f} Total Tons")
    c2.metric("Irrigation Schedule", irr_level, delta=irr_hours)
    c3.metric("Champion Model", best_model_name.split()[0], delta=f"R²: {benchmarks[best_model_name]['r2']:.3f}")

    st.subheader("Farmer Advisory")
    st.info(advisory)
    st.caption(f"Reasoning: {irr_reason}")

with tab_warehouse:
    st.dataframe(df.head(10), width='stretch')
    t_crops = df.groupby('Crop')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)
    st.plotly_chart(px.bar(t_crops, x='Crop', y='Production', title="Top 10 Crops by Production (Tonnes)", color='Crop'), width='stretch')

with tab_models:
    eval_rows = [{"Algorithm": k, "R² Score": round(v["r2"], 4), "MAE (Ton/Ha)": round(v["mae"], 4), "RMSE (Ton/Ha)": round(v["rmse"], 4)} for k, v in benchmarks.items()]
    st.table(pd.DataFrame(eval_rows).sort_values(by="R² Score", ascending=False))

with tab_mining:
    rules_df = get_rules(df)
    if not rules_df.empty:
        st.dataframe(rules_df.head(15), width='stretch')
        st.plotly_chart(px.scatter(rules_df.head(40), x="Support", y="Confidence", size="Lift", color="Lift", hover_data=["Antecedent", "Consequent"], title="Association Rules: Support vs Confidence"), width='stretch')
