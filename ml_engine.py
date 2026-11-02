import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

FEATURE_COLS = ['Crop', 'State', 'Season', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
TARGET_COL = 'Yield'

def train_and_evaluate_models(df):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Crop', 'State', 'Season'])
        ],
        remainder='passthrough'
    )

    models = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0)
    }

    results = {}
    best_name, best_r2, best_pipeline = None, -float("inf"), None

    for name, model in models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('regressor', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        results[name] = {"pipeline": pipe, "r2": r2, "mae": mae, "rmse": rmse, "y_test": y_test, "y_pred": y_pred}

        if r2 > best_r2:
            best_r2, best_name, best_pipeline = r2, name, pipe

    return {"benchmarks": results, "best_model_name": best_name, "best_pipeline": best_pipeline}

def mine_association_rules(df, min_support=0.04, min_confidence=0.5):
    df_disc = pd.DataFrame()
    df_disc['State'] = 'State=' + df['State'].str.strip()
    df_disc['Crop'] = 'Crop=' + df['Crop'].str.strip()
    df_disc['Season'] = 'Season=' + df['Season'].str.strip()

    df_disc['Rainfall'] = pd.cut(
        df['Annual_Rainfall'],
        bins=[-np.inf, 1000, 2000, np.inf],
        labels=['Rainfall=Low(<1000mm)', 'Rainfall=Mod(1000-2000mm)', 'Rainfall=High(>2000mm)']
    ).astype(str)

    f_med = df['Fertilizer'].median()
    df_disc['Fertilizer'] = np.where(df['Fertilizer'] > f_med, 'Fertilizer=High', 'Fertilizer=Low')

    y_q33, y_q66 = df['Yield'].quantile(0.33), df['Yield'].quantile(0.66)
    df_disc['Yield_Tier'] = pd.cut(
        df['Yield'],
        bins=[-np.inf, y_q33, y_q66, np.inf],
        labels=['Yield=Tier-Low', 'Yield=Tier-Med', 'Yield=Tier-High']
    ).astype(str)

    records = df_disc.values.tolist()
    total_records = len(records)
    item_counts, pair_counts = defaultdict(int), defaultdict(int)

    for record in records:
        for item in record:
            item_counts[item] += 1
        for i in range(len(record)):
            for j in range(i + 1, len(record)):
                pair = tuple(sorted((record[i], record[j])))
                pair_counts[pair] += 1

    rules = []
    for (item_a, item_b), count in pair_counts.items():
        support = count / total_records
        if support >= min_support:
            conf_a = count / item_counts[item_a]
            lift_a = conf_a / (item_counts[item_b] / total_records)
            if conf_a >= min_confidence:
                rules.append({"Antecedent": item_a, "Consequent": item_b, "Support": support, "Confidence": conf_a, "Lift": lift_a})

            conf_b = count / item_counts[item_b]
            lift_b = conf_b / (item_counts[item_a] / total_records)
            if conf_b >= min_confidence:
                rules.append({"Antecedent": item_b, "Consequent": item_a, "Support": support, "Confidence": conf_b, "Lift": lift_b})

    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(by="Lift", ascending=False).drop_duplicates(subset=['Antecedent', 'Consequent'])
    return rules_df
