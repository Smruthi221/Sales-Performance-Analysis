import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("SuperstoreOrders.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display number of rows and columns
print("\nShape of dataset:")
print(df.shape)

# Display information about columns and data types
print("\nDataset information:")
df.info()

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

df["order_date"] = pd.to_datetime(df["order_date"], format="mixed")
df["ship_date"] = pd.to_datetime(df["ship_date"], format="mixed")

print("\nDate columns converted successfully.")
print(df[["order_date", "ship_date"]].dtypes)

df.to_csv("cleaned_superstore.csv", index=False)

print("\nCleaned dataset saved successfully!")

print("\nStatistical Summary:")
print(df.describe())

print("\nCorrelation Matrix:")
print(df.select_dtypes(include="number").corr())



numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, fmt=".2f")

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")

plt.figure(figsize=(8, 5))
sns.boxplot(data=df[["sales", "profit", "discount"]])

plt.title("Outlier Analysis")
plt.ylabel("Values")
plt.tight_layout()
plt.savefig("outlier_analysis.png")

category_sales = df.groupby("category")["sales"].sum().sort_values(ascending=False)

print("\nSales by Category:")
print(category_sales)

category_profit = df.groupby("category")["profit"].sum().sort_values(ascending=False)

print("\nProfit by Category:")
print(category_profit)

region_sales = df.groupby("region")["sales"].sum().sort_values(ascending=False)

print("\nSales by Region:")
print(region_sales)

df["month"] = df["order_date"].dt.to_period("M")

monthly_sales = df.groupby("month")["sales"].sum()

print("\nMonthly Sales Trend:")
print(monthly_sales)

plt.figure(figsize=(8, 5))

plt.bar(category_sales.index, category_sales.values)

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig("sales_by_category.png")

plt.figure(figsize=(8, 5))

plt.bar(region_sales.index, region_sales.values)

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig("sales_by_region.png")

plt.figure(figsize=(10, 5))

plt.plot(monthly_sales.index.astype(str), monthly_sales.values, marker="o")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_sales_trend.png")

plt.figure(figsize=(8, 5))

plt.bar(category_profit.index, category_profit.values)

plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Total Profit")

plt.tight_layout()
plt.savefig("profit_by_category.png")

# Convert Sales and Profit to numeric
df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
df["profit"] = pd.to_numeric(df["profit"], errors="coerce")

# Remove missing values
scatter_data = df.dropna(subset=["sales", "profit"])

# Sales vs Profit chart
plt.figure(figsize=(8, 5))

plt.scatter(
    scatter_data["sales"],
    scatter_data["profit"],
    alpha=0.5
)

plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")

plt.tight_layout()
plt.savefig("sales_vs_profit.png")
