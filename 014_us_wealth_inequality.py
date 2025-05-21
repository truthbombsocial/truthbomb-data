import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
print("Current Working Directory:", os.getcwd())

def load_census_csv(path: str, lookahead: int = 15) -> pd.DataFrame:
    for skip in range(lookahead):
        df = pd.read_csv(path, skiprows=skip)
        cols = [c.strip().lower() for c in df.columns]
        if 'us_total_households' in cols and 'year' in cols:
            df.columns = cols # Normalize column names
            print(f"Census header found after skipping {skip} rows.")
            return df
    raise ValueError("Could not locate header row with 'us_total_households' + 'year' in the CSV file.")

census_data = load_census_csv('data/014_census_household_data.csv')
fed_reserve_data = pd.read_csv('data/014_federal_reserve_data.csv')
print("\nCensus columns actually read: ", census_data.columns.tolist()[:10])
print("First 2 rows\n", census_data.head(2))

# Transform (Cleaning & Normalization)
census_data.columns = [col.strip().lower() for col in census_data.columns]
census_data['us_total_households'] = census_data['us_total_households'].str.replace(',', '').astype(int) * 1000
census_data['year'] = census_data['year'].astype(int)

fed_reserve_data['year'] = fed_reserve_data['date'].str[:4].astype(int)
fed_reserve_data = fed_reserve_data[(fed_reserve_data['year'] >= 1990) & (fed_reserve_data['year'] <= 2023)]
fed_reserve_data['assets'] *= 1_000_000  # trillions -> millions

# Transform (AVERAGE quarterly values per year)
annual_avg = fed_reserve_data.groupby(['year', 'category'])[['assets']].mean().reset_index()
annual_wealth_data = annual_avg.pivot(index='year', columns='category', values='assets').reset_index()
annual_wealth_data = annual_wealth_data.rename(columns={
    'TopPt1': 'top_pt1_assets',
    'RemainingTop1': 'remaining_top_1_assets',
    'Next9': 'next9_assets',
    'Next40': 'next40_assets',
    'Bottom50': 'bottom50_assets'
})

# Transform (Merge and Calculate Per-Household Figures, in Millions)
merged_data = pd.merge(annual_wealth_data, census_data, on='year', how='left')
merged_data['top_pt1_per_household'] = merged_data['top_pt1_assets'] / (merged_data['us_total_households'] * 0.001)
merged_data['remaining_top_1_per_household'] = merged_data['remaining_top_1_assets'] / (
            merged_data['us_total_households'] * 0.009)
merged_data['next9_per_household'] = merged_data['next9_assets'] / (merged_data['us_total_households'] * 0.09)
merged_data['next40_per_household'] = merged_data['next40_assets'] / (merged_data['us_total_households'] * 0.40)
merged_data['bottom50_per_household'] = merged_data['bottom50_assets'] / (merged_data['us_total_households'] * 0.50)

columns_to_round = [columns_out for columns_out in merged_data.columns if
                    columns_out not in ['year', 'us_total_households']]
merged_data[columns_to_round] = merged_data[columns_to_round].round(2)

# Validation Check
def validate_dataset(df):
    expected_columns = [
        'year',
        'top_pt1_per_household',
        'remaining_top_1_per_household',
        'next9_per_household',
        'next40_per_household',
        'bottom50_per_household']
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"
        assert df[col].notnull().all(), f"Null values in column: {col}"
        assert (df[col] >= 0).all(), f"Negative values in column: {col}"
    print("Dataset validation passed!")

validate_dataset(merged_data)

# -- Plot --
plt.figure(figsize=(12, 6))
plt.plot(merged_data['year'], merged_data['top_pt1_per_household'], label='Top 0.1% per Household', marker='o')
plt.plot(merged_data['year'], merged_data['remaining_top_1_per_household'], label='Remaining Top 0.9% per Household',marker='o')
plt.plot(merged_data['year'], merged_data['next9_per_household'], label='Next 9% per Household', marker='o')
plt.plot(merged_data['year'], merged_data['next40_per_household'], label='Next 40% per Household', marker='o')
plt.plot(merged_data['year'], merged_data['bottom50_per_household'], label='Bottom 50% per Household', marker='o')

plt.xlabel('Year', fontsize=14)
plt.ylabel('Assets per Household (in Millions of Dollars)', fontsize=10)
plt.suptitle("99.9% of Americans Aren't Losing to the Poor, But to the Richest .1%", fontsize=16)
plt.title("Household Wealth by Year (1990 - 2023)", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

# Save CSV
print("Saving merged data to CSV...")

columns_out = [
    'year',
    'top_pt1_per_household',
    'remaining_top_1_per_household',
    'next9_per_household',
    'next40_per_household',
    'bottom50_per_household'
]

out_dir = Path("result_set")
out_dir.mkdir(parents=True, exist_ok=True)

out_file = out_dir / "014_result_set.csv"

merged_data[columns_out].to_csv(out_file, index=False)

print("Result-set written to:", out_file.resolve())
print("Merged Data (first 5 rows):")
print(merged_data[columns_out].head())

