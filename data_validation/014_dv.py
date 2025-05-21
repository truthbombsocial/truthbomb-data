# detailed code review for data validation team
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Print working directory to confirm file paths
print("Current Working Directory:", os.getcwd())

# ------------------------------------------------------------
# 1. Load Census Data
#    - The raw census CSV has extraneous header lines.
#    - We search for the first row where both 'year' and 'us_total_households' appear.
# ------------------------------------------------------------

def load_census_csv(path: str, lookahead: int = 15) -> pd.DataFrame:
    for skip in range(lookahead):
        df = pd.read_csv(path, skiprows=skip)
        cols = [c.strip().lower() for c in df.columns]
        if 'us_total_households' in cols and 'year' in cols:
            df.columns = cols # Normalize column names
            print(f"Census header found after skipping {skip} rows.")
            return df
    raise ValueError("Could not locate header row with 'us_total_households' + 'year' in the CSV file.")


# Load the census data and print first two rows
census_data = load_census_csv('data/014_census_household_data.csv')
fed_reserve_data = pd.read_csv('data/014_federal_reserve_data.csv')
print("\nCensus columns actually read: ", census_data.columns.tolist()[:10])
print("First 2 rows\n", census_data.head(2))

# Normalize column names for consistency:
# - strip any leading/trailing spaces
# - convert to lowercase so downstream code can reference columns reliably
census_data.columns = [col.strip().lower() for col in census_data.columns]

# Clean up the household counts and convert to absolute numbers:
# 1) Remove thousand‐separating commas (e.g. "128,451" → "128451")
# 2) Cast the resulting strings to integers
# 3) Multiply by 1,000 because the source CSV reports households in thousands
census_data['us_total_households'] = census_data['us_total_households'].str.replace(',', '').astype(int) * 1000

# Ensure the 'year' column is treated as numeric rather than text,
# so we can filter, sort, and join on it without surprises.
census_data['year'] = census_data['year'].astype(int)

# Derive a numeric 'year' column from the quarterly 'date' string:
# - The 'date' format is "YYYY:Qn" (e.g. "1990:Q1"), so taking the first 4 characters gives the year.
# - Converting to int lets us filter and group by year reliably.
fed_reserve_data['year'] = fed_reserve_data['date'].str[:4].astype(int)

# Keep only the timeframe of interest (1990 through 2023):
# - This removes any quarters outside our analysis window.
fed_reserve_data = fed_reserve_data[(fed_reserve_data['year'] >= 1990) & (fed_reserve_data['year'] <= 2023)]

# Convert asset figures from trillions of dollars to millions of dollars:
# - The raw 'assets' values are in trillions, so multiplying by 1,000,000 yields values in millions.
# - This makes downstream per-household numbers more interpretable (in millions).
fed_reserve_data['assets'] *= 1_000_000  # trillions -> millions

# Aggregate quarterly asset data into annual totals by category:
# 1) Group by year and wealth category, take the mean of the four quarters.
# 2) Pivot the result so each category becomes its own column (one row per year).
# 3) Rename the raw category labels to clear, descriptive column names.
annual_avg = fed_reserve_data.groupby(['year', 'category'])[['assets']].mean().reset_index()
annual_wealth_data = annual_avg.pivot(index='year', columns='category', values='assets').reset_index()
annual_wealth_data = annual_wealth_data.rename(columns={
    'TopPt1': 'top_pt1_assets',
    'RemainingTop1': 'remaining_top_1_assets',
    'Next9': 'next9_assets',
    'Next40': 'next40_assets',
    'Bottom50': 'bottom50_assets'
})
#  Merge annual wealth totals with total U.S. households by year:
#    - Left join ensures we keep every year’s wealth data even if census is missing (shouldn’t happen).
merged_data = pd.merge(annual_wealth_data, census_data, on='year', how='left')

#  Compute assets per household for each wealth bracket:
#    - Top 0.1%   → divide top_pt1_assets by 0.001 of total households
#    - Next 0.9%  → remaining_top_1_assets ÷ 0.009 of total households
#    - Next 9%    → next9_assets           ÷ 0.09 of total households
#    - Next 40%   → next40_assets          ÷ 0.40 of total households
#    - Bottom 50% → bottom50_assets        ÷ 0.50 of total households
merged_data = pd.merge(annual_wealth_data, census_data, on='year', how='left')
merged_data['top_pt1_per_household'] = merged_data['top_pt1_assets'] / (merged_data['us_total_households'] * 0.001)
merged_data['remaining_top_1_per_household'] = merged_data['remaining_top_1_assets'] / (
            merged_data['us_total_households'] * 0.009)
merged_data['next9_per_household'] = merged_data['next9_assets'] / (merged_data['us_total_households'] * 0.09)
merged_data['next40_per_household'] = merged_data['next40_assets'] / (merged_data['us_total_households'] * 0.40)
merged_data['bottom50_per_household'] = merged_data['bottom50_assets'] / (merged_data['us_total_households'] * 0.50)

#  Round all per-household metrics to one decimal place:
#    - Excludes 'year' and raw household count columns
columns_to_round = [columns_out for columns_out in merged_data.columns if
                    columns_out not in ['year', 'us_total_households']]
merged_data[columns_to_round] = merged_data[columns_to_round].round(1)

# Basic data integrity checks—fail early if something is off:
# 1️⃣ Verify all expected columns are present.
# 2️⃣ Ensure none of those columns contain missing (null) values.
# 3️⃣ Confirm every value is zero or positive (no negatives in asset figures).
# If everything passes, print a success message.
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

# --- Plot ---
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

# Save CSV (creates the result_set and saves it to the directory)
# then prints the data frame to the console
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