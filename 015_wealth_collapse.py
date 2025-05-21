# 015_wealth_collapse.py
"""
Wealth Collapse: % change in average assets per household
from 1990 to 2023, by income tier.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1) Load the per-household data produced by 014
infile = Path("result_set") / "014_result_set.csv"
df = pd.read_csv(infile)
print("Loaded 014 result_set data (first 5 rows):")
print(df.head(), "\n")

# 2) Pull out each year as a Series (so we get a single row of 5 tiers each)
by_year = df.set_index("year")
start_s = by_year.loc[1990]
end_s   = by_year.loc[2023]

# 3) Compute % change = (end - start) / start * 100, round to 1 decimal
pct_change = ((end_s - start_s) / start_s * 100).round(1)
# Turn that Series into a 1-column DataFrame, name the column & index
pct_change = pct_change.to_frame(name="pct_change")
pct_change.index.name = "tier"

# 4) Save the new result set
out_dir = Path("result_set")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "015_result_set.csv"
pct_change.to_csv(out_file, index=True)
print(f"015 result set written to: {out_file.resolve()}\n")

# Print the %-change table for console inspection
print("015 %-change table:")
print(pct_change, "\n")

# 5) Plot the collapse as a bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(
    pct_change.index,
    pct_change["pct_change"],
    color=["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
)
plt.ylabel("% Change in Average Assets per Household\n(1990 → 2023)", fontsize=10)
plt.title("Wealth Collapse: How Much the Bottom 99.9% Has Lost Since 1990", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
