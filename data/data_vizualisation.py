import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data/clean/t_out.csv", header=0)
df["T_out"] = df["T_out"].astype(float)
df["Date"] = pd.to_datetime(df["Date"])

# Extract the date (without time) for grouping
df["Day"] = df["Date"].dt.date

# Calculate the percentage of readings above 21°C for each day
daily_stats = (
    df.groupby("Day")
    .apply(lambda x: (x["T_out"] > 21).mean() * 100)
    .reset_index(name="Percent_Above_21")
)

# Filter days with more than 25% of readings above 21°C
hot_days = daily_stats[daily_stats["Percent_Above_21"] > 25]

# Plot the results
plt.figure(figsize=(12, 6))
plt.bar(daily_stats["Day"], daily_stats["Percent_Above_21"], color="skyblue", label="% Above 21°C")
plt.axhline(y=25, color="r", linestyle="--", label="25% Threshold")
plt.scatter(hot_days["Day"], hot_days["Percent_Above_21"], color="red", zorder=5, label="Hot Days (>25%)")
plt.xlabel("Day")
plt.ylabel("Percentage of Readings Above 21°C")
plt.title("Daily Percentage of Temperature Readings Above 21°C")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

# Print the hot days
print("Days with more than 25% of readings above 21°C:")
print(hot_days)
