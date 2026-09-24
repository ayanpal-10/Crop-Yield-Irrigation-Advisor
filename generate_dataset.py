"""
generate_dataset.py
--------------------
This script creates a SAMPLE agricultural dataset for the Crop Yield &
Irrigation Advisor prototype.

Why do we need this?
Real government/weather agricultural datasets are hard to line up for a
first-round prototype, so instead we generate a REALISTIC sample dataset
using simple science-based rules (each crop has an "ideal" soil moisture,
temperature, rainfall, etc.). Values close to the ideal produce higher
yield, values far from the ideal produce lower yield, plus a bit of random
noise so it behaves like real-world data. This way, the Machine Learning
model in app.py has genuine patterns to learn from.

Run this file directly to create crop_data.csv:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd

# Using a fixed seed so the dataset is the same every time we generate it.
# This makes results reproducible when demonstrating the project.
np.random.seed(42)

# ---------------------------------------------------------------------
# 1. Crop profiles: the "ideal" growing conditions for each crop.
#    These numbers are simplified approximations for teaching purposes,
#    not exact agronomy figures.
# ---------------------------------------------------------------------
CROP_PROFILES = {
    "Wheat":     {"ideal_moisture": 45, "moisture_tol": 15, "ideal_temp": 22, "temp_tol": 6,
                  "ideal_rainfall": 650,  "rainfall_tol": 250, "base_yield": 3.5,
                  "ideal_N": 120, "ideal_P": 60, "ideal_K": 40, "ideal_ph": 6.5, "ph_tol": 1.0,
                  "ideal_irrigation": 2.0},
    "Rice":      {"ideal_moisture": 70, "moisture_tol": 15, "ideal_temp": 27, "temp_tol": 5,
                  "ideal_rainfall": 1200, "rainfall_tol": 300, "base_yield": 4.5,
                  "ideal_N": 100, "ideal_P": 50, "ideal_K": 50, "ideal_ph": 6.0, "ph_tol": 1.0,
                  "ideal_irrigation": 3.5},
    "Maize":     {"ideal_moisture": 50, "moisture_tol": 15, "ideal_temp": 24, "temp_tol": 6,
                  "ideal_rainfall": 700,  "rainfall_tol": 250, "base_yield": 3.0,
                  "ideal_N": 140, "ideal_P": 60, "ideal_K": 40, "ideal_ph": 6.2, "ph_tol": 1.0,
                  "ideal_irrigation": 2.2},
    "Cotton":    {"ideal_moisture": 40, "moisture_tol": 15, "ideal_temp": 28, "temp_tol": 6,
                  "ideal_rainfall": 600,  "rainfall_tol": 250, "base_yield": 2.0,
                  "ideal_N": 100, "ideal_P": 50, "ideal_K": 50, "ideal_ph": 7.0, "ph_tol": 1.2,
                  "ideal_irrigation": 2.5},
    "Sugarcane": {"ideal_moisture": 65, "moisture_tol": 15, "ideal_temp": 26, "temp_tol": 5,
                  "ideal_rainfall": 1500, "rainfall_tol": 400, "base_yield": 70.0,
                  "ideal_N": 150, "ideal_P": 70, "ideal_K": 60, "ideal_ph": 6.5, "ph_tol": 1.0,
                  "ideal_irrigation": 4.0},
    "Potato":    {"ideal_moisture": 55, "moisture_tol": 12, "ideal_temp": 18, "temp_tol": 5,
                  "ideal_rainfall": 500,  "rainfall_tol": 200, "base_yield": 22.0,
                  "ideal_N": 120, "ideal_P": 80, "ideal_K": 100, "ideal_ph": 5.5, "ph_tol": 1.0,
                  "ideal_irrigation": 2.0},
    "Tomato":    {"ideal_moisture": 55, "moisture_tol": 12, "ideal_temp": 23, "temp_tol": 5,
                  "ideal_rainfall": 450,  "rainfall_tol": 180, "base_yield": 30.0,
                  "ideal_N": 110, "ideal_P": 70, "ideal_K": 90, "ideal_ph": 6.3, "ph_tol": 1.0,
                  "ideal_irrigation": 2.3},
}

SOIL_TYPES = ["Loamy", "Sandy", "Clay", "Black", "Red", "Alluvial"]

N_ROWS = 900  # total rows to generate (spec asked for ~500-1000)


def gaussian_factor(value, ideal, tolerance):
    """
    Returns a multiplier between 0 and 1 showing how close `value` is to
    the `ideal` value. 1.0 = perfect match, drops off the further away
    `value` is, based on a bell-curve (Gaussian) shape.
    This is the core trick used to make yield depend realistically on
    each condition (moisture, temperature, etc).
    """
    return np.exp(-((value - ideal) ** 2) / (2 * tolerance ** 2))


def generate_row(crop_name):
    """Generates one realistic data row for a given crop."""
    profile = CROP_PROFILES[crop_name]

    # Sample each condition around the crop's ideal value with some spread,
    # so we get a realistic range of "good" and "bad" growing conditions.
    soil_moisture = np.clip(np.random.normal(profile["ideal_moisture"], 18), 5, 95)
    temperature = np.clip(np.random.normal(profile["ideal_temp"], 8), 5, 45)
    humidity = np.clip(np.random.normal(60, 15), 10, 95)
    rainfall = np.clip(np.random.normal(profile["ideal_rainfall"], profile["rainfall_tol"] * 1.3), 20, 2500)
    nitrogen = np.clip(np.random.normal(profile["ideal_N"], 30), 10, 200)
    phosphorus = np.clip(np.random.normal(profile["ideal_P"], 20), 5, 150)
    potassium = np.clip(np.random.normal(profile["ideal_K"], 25), 5, 150)
    ph = np.clip(np.random.normal(profile["ideal_ph"], 0.8), 4.0, 9.0)
    area = np.round(np.random.uniform(0.5, 10.0), 2)
    irrigation_hours = np.clip(np.random.normal(profile["ideal_irrigation"], 1.0), 0, 8)

    soil_type = np.random.choice(SOIL_TYPES)

    # --- compute how "favorable" each condition is (0 to 1 scale) ---
    moisture_factor = gaussian_factor(soil_moisture, profile["ideal_moisture"], profile["moisture_tol"])
    temp_factor = gaussian_factor(temperature, profile["ideal_temp"], profile["temp_tol"])
    rainfall_factor = gaussian_factor(rainfall, profile["ideal_rainfall"], profile["rainfall_tol"])
    n_factor = gaussian_factor(nitrogen, profile["ideal_N"], 45)
    p_factor = gaussian_factor(phosphorus, profile["ideal_P"], 35)
    k_factor = gaussian_factor(potassium, profile["ideal_K"], 35)
    ph_factor = gaussian_factor(ph, profile["ideal_ph"], profile["ph_tol"])

    # Irrigation has a mild positive effect when close to the ideal amount.
    irrigation_factor = 0.85 + 0.15 * gaussian_factor(
        irrigation_hours, profile["ideal_irrigation"], 1.5
    )

    # Combine all factors. Averaging (rather than multiplying everything)
    # keeps the yield from collapsing to near-zero when just one factor
    # is slightly off, which better matches real crop behaviour.
    combined_factor = np.mean([
        moisture_factor, temp_factor, rainfall_factor,
        n_factor, p_factor, k_factor, ph_factor, irrigation_factor
    ])

    # Add a little random noise so the data isn't a perfectly clean formula.
    noise = np.random.normal(0, 0.05)
    yield_value = profile["base_yield"] * (0.4 + 0.6 * combined_factor) * (1 + noise)
    yield_value = max(round(yield_value, 2), 0.1)

    return {
        "Crop": crop_name,
        "Soil_Type": soil_type,
        "Soil_Moisture": round(soil_moisture, 1),
        "Temperature": round(temperature, 1),
        "Humidity": round(humidity, 1),
        "Rainfall": round(rainfall, 1),
        "Nitrogen": round(nitrogen, 1),
        "Phosphorus": round(phosphorus, 1),
        "Potassium": round(potassium, 1),
        "pH": round(ph, 2),
        "Area_Hectare": area,
        "Irrigation_Hours": round(irrigation_hours, 1),
        "Yield_Ton_Per_Hectare": yield_value,
    }


def generate_and_save(path="crop_data.csv", n_rows=N_ROWS):
    """Generates the full dataset and saves it as a CSV file."""
    crops = list(CROP_PROFILES.keys())
    rows = [generate_row(np.random.choice(crops)) for _ in range(n_rows)]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} rows and saved to '{path}'")
    return df


if __name__ == "__main__":
    generate_and_save()
