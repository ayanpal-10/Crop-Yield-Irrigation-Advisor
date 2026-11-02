"""
services.py
-----------
1. Live Weather & Agro-climatic API Connectors.
2. Explainable Rule-Based Irrigation Logic.
3. Generative AI Farmer Advisory Layer (Safe Gemini import + template fallback).
"""

import requests

# Safe import to prevent ModuleNotFoundError on any host
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

STATE_COORDINATES = {
    "Punjab": (31.1471, 75.3412), "Haryana": (29.0588, 76.0856),
    "Uttar Pradesh": (26.8467, 80.9462), "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139), "West Bengal": (22.9868, 87.8550),
    "Gujarat": (22.2587, 71.1924), "Rajasthan": (27.0238, 74.2179),
    "Karnataka": (15.3173, 75.7139), "Tamil Nadu": (11.1271, 78.6569)
}

def fetch_live_weather(state_name):
    coords = STATE_COORDINATES.get(state_name, (20.5937, 78.9629))
    lat, lon = coords
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"
    try:
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            curr = res.json().get("current", {})
            return {
                "temperature": curr.get("temperature_2m", 27.5),
                "humidity": curr.get("relative_humidity_2m", 62.0),
                "precipitation": curr.get("precipitation", 0.0),
                "source": "Live Open-Meteo API"
            }
    except Exception:
        pass
    return {"temperature": 28.0, "humidity": 65.0, "precipitation": 0.0, "source": "Regional Baseline"}

def calculate_irrigation_recommendation(crop, soil_moisture, rainfall, temperature):
    if soil_moisture < 35:
        level, hours = "High", "3.5 - 4.5 hrs"
    elif soil_moisture <= 65:
        level, hours = "Medium", "1.5 - 2.5 hrs"
    else:
        level, hours = "Low", "0 - 1.0 hr"

    reasons = [f"current soil moisture is {soil_moisture:.1f}%"]
    if rainfall > 1400:
        level, hours = ("Medium", "2.0 hrs") if level == "High" else ("Low", "0.5 hrs")
        reasons.append("high annual rainfall buffer lowers immediate demand")
    elif rainfall < 500 and soil_moisture < 45:
        level, hours = "Critical High", "4.5 - 5.5 hrs"
        reasons.append("rainfall deficit detected")

    if temperature > 34.0:
        reasons.append(f"high temperature ({temperature}°C) increases evapotranspiration")

    return level, hours, "; ".join(reasons).capitalize() + "."

def generate_genai_advisory(crop, state, season, yield_val, irr_level, irr_hours, irr_reason, api_key=None, language="English"):
    if api_key and HAS_GENAI:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"Expert advice for {crop} farmer in {state} during {season}. Expected Yield: {yield_val:.2f} t/ha. Irrigation: {irr_level} ({irr_hours}) because {irr_reason}. Language: {language}. Write a 3-sentence encouraging, practical advisory."
            res = model.generate_content(prompt)
            if res and res.text:
                return res.text.strip()
        except Exception:
            pass

    advisories = {
        "English": f"For your {crop} crop in {state} ({season}), projected yield is {yield_val:.2f} tonnes/hectare. Schedule a {irr_level} irrigation cycle ({irr_hours}) as {irr_reason.lower()} Monitor drainage before applying remaining fertilizer.",
        "Hindi": f"{state} में {season} मौसम के दौरान आपकी {crop} की अनुमानित उपज {yield_val:.2f} टन/हेक्टेयर है। {irr_hours} की {irr_level} सिंचाई करें क्योंकि {irr_reason.lower()}",
        "Punjabi": f"{state} ਵਿੱਚ {season} ਸੀਜ਼ਨ ਦੌਰਾਨ {crop} ਦੀ ਸੰਭਾਵੀ ਪੈਦਾਵਾਰ {yield_val:.2f} ਟਨ/ਹੈਕਟੇਅਰ ਹੈ। {irr_hours} ਲਈ {irr_level} ਸਿੰਚਾਈ ਕਰੋ ਕਿਉਂਕਿ {irr_reason.lower()}"
    }
    return advisories.get(language, advisories["English"])
