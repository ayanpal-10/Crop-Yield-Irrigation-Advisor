import os
import sqlite3
import pandas as pd

DB_PATH = "agri_warehouse.db"
CSV_PATH = "crop_yield.csv"

def run_etl(csv_path=CSV_PATH, db_path=DB_PATH):
    print(f"[*] Starting ETL pipeline from '{csv_path}'...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file {csv_path} not found.")

    df = pd.read_csv(csv_path)

    # 1. Standardize text & handle types
    df['Crop'] = df['Crop'].astype(str).str.strip().str.title()
    df['State'] = df['State'].astype(str).str.strip().str.title()
    df['Season'] = df['Season'].astype(str).str.strip().str.title()
    df['Crop_Year'] = df['Crop_Year'].astype(int)

    df['Area'] = df['Area'].apply(lambda x: max(float(x), 0.1))
    df['Annual_Rainfall'] = df['Annual_Rainfall'].apply(lambda x: max(float(x), 0.0))
    df['Fertilizer'] = df['Fertilizer'].apply(lambda x: max(float(x), 0.0))
    df['Pesticide'] = df['Pesticide'].apply(lambda x: max(float(x), 0.0))
    df['Yield'] = df['Yield'].apply(lambda x: max(float(x), 0.001))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 2. Dimensions
    dim_crop = pd.DataFrame({'crop_id': range(1, df['Crop'].nunique() + 1), 'crop_name': sorted(df['Crop'].unique())})
    dim_crop.to_sql('dim_crop', conn, if_exists='replace', index=False)

    dim_state = pd.DataFrame({'state_id': range(1, df['State'].nunique() + 1), 'state_name': sorted(df['State'].unique())})
    dim_state.to_sql('dim_state', conn, if_exists='replace', index=False)

    dim_season = pd.DataFrame({'season_id': range(1, df['Season'].nunique() + 1), 'season_name': sorted(df['Season'].unique())})
    dim_season.to_sql('dim_season', conn, if_exists='replace', index=False)

    dim_year = pd.DataFrame({'year_id': range(1, df['Crop_Year'].nunique() + 1), 'year': sorted(df['Crop_Year'].unique())})
    dim_year.to_sql('dim_year', conn, if_exists='replace', index=False)

    # 3. Fact Table
    fact_df = df.merge(dim_crop, left_on='Crop', right_on='crop_name')
    fact_df = fact_df.merge(dim_state, left_on='State', right_on='state_name')
    fact_df = fact_df.merge(dim_season, left_on='Season', right_on='season_name')
    fact_df = fact_df.merge(dim_year, left_on='Crop_Year', right_on='year')

    fact_table = fact_df[[
        'crop_id', 'state_id', 'season_id', 'year_id',
        'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield'
    ]].copy()
    fact_table.reset_index(drop=True, inplace=True)
    fact_table['fact_id'] = range(1, len(fact_table) + 1)
    fact_table.to_sql('fact_crop_production', conn, if_exists='replace', index=False)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_crop ON fact_crop_production(crop_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_state ON fact_crop_production(state_id);")
    conn.commit()
    conn.close()
    print("[✓] Star Schema warehouse built successfully.")

def load_warehouse_dataset(db_path=DB_PATH):
    if not os.path.exists(db_path):
        run_etl(CSV_PATH, db_path)

    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        f.fact_id, c.crop_name AS Crop, s.state_name AS State,
        se.season_name AS Season, y.year AS Crop_Year,
        f.Area, f.Production, f.Annual_Rainfall, f.Fertilizer, f.Pesticide, f.Yield
    FROM fact_crop_production f
    JOIN dim_crop c ON f.crop_id = c.crop_id
    JOIN dim_state s ON f.state_id = s.state_id
    JOIN dim_season se ON f.season_id = se.season_id
    JOIN dim_year y ON f.year_id = y.year_id;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
