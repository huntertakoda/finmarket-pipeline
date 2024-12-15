import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf

# loading

file_path = "full_financial_data.csv"
df = pd.read_csv(file_path, parse_dates=["Timestamp"])

# sort data by timestamp

df = df.sort_values(by="Timestamp").reset_index(drop=True)

# ftr1 . moving averages

df["Close_7D_MA"] = df["Close"].rolling(window=7).mean()
df["Close_30D_MA"] = df["Close"].rolling(window=30).mean()

# ftr2 . volatility (high - low)

df["Volatility"] = df["High"] - df["Low"]

# ftr3 . volume z-scores

df["Volume_ZScore"] = (df["Volume"] - df["Volume"].mean()) / df["Volume"].std()

# ftr4 . resample data by day

df_daily = df.resample("D", on="Timestamp").agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum"
}).dropna()

# basic visualizations

plt.figure(figsize=(12, 6))
sns.lineplot(x='Timestamp', y='Close', data=df)
plt.title("Close Price Over Time")
plt.xticks(rotation=45)
plt.show()

# plot close price with moving averages

plt.figure(figsize=(12, 6))
plt.plot(df["Timestamp"], df["Close"], label="Close Price", linewidth=1)
plt.plot(df["Timestamp"], df["Close_7D_MA"], label="7-Day MA", linestyle="--", color="orange")
plt.plot(df["Timestamp"], df["Close_30D_MA"], label="30-Day MA", linestyle="--", color="red")
plt.title("Close Price with Moving Averages")
plt.xlabel("Timestamp")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot volatility over time

plt.figure(figsize=(12, 6))
sns.lineplot(x="Timestamp", y="Volatility", data=df)
plt.title("Volatility Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Volatility")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot volume z-scores to identify outliers

plt.figure(figsize=(12, 6))
sns.histplot(df["Volume_ZScore"], bins=50, kde=True)
plt.title("Volume Z-Scores Distribution")
plt.xlabel("Z-Score")
plt.tight_layout()
plt.show()

# plot resampled daily close prices

plt.figure(figsize=(12, 6))
plt.plot(df_daily.index, df_daily["Close"], label="Daily Close Price")
plt.title("Daily Close Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# resample for daily OHLC data

df_candlestick = df.set_index("Timestamp").resample("D").agg({
    "Open": "first", 
    "High": "max", 
    "Low": "min", 
    "Close": "last", 
    "Volume": "sum"
}).dropna()

# plot candlestick chart with volume

mpf.plot(df_candlestick, type='candle', style='charles', 
         title="Daily Candlestick Chart with Volume", 
         ylabel="Price", volume=True, mav=(7, 30))

# rolling correlation between close price and volume

df["Rolling_Corr_Close_Volume"] = df["Close"].rolling(window=30).corr(df["Volume"])

# plot rolling correlation

plt.figure(figsize=(12, 6))
sns.lineplot(x="Timestamp", y="Rolling_Corr_Close_Volume", data=df)
plt.title("30-Day Rolling Correlation: Close Price vs Volume")
plt.xlabel("Timestamp")
plt.ylabel("Rolling Correlation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot volatility vs volume scatter plot

plt.figure(figsize=(12, 6))
sns.scatterplot(x="Volatility", y="Volume", data=df, alpha=0.7)
sns.regplot(x="Volatility", y="Volume", data=df, scatter=False, color="red", ci=None)
plt.title("Volatility vs Volume")
plt.xlabel("Volatility")
plt.ylabel("Volume")
plt.tight_layout()
plt.show()

# pairplot for key numerical columns

sns.pairplot(df[["Open", "High", "Low", "Close", "Volume", "Volatility"]], diag_kind="kde")
plt.suptitle("Pairplot for Key Numerical Features", y=1.02)
plt.tight_layout()
plt.show()

# cumulative volume calculation

df["Cumulative_Volume"] = df["Volume"].cumsum()

# plot cumulative volume

plt.figure(figsize=(12, 6))
sns.lineplot(x="Timestamp", y="Cumulative_Volume", data=df, color="green")
plt.title("Cumulative Trading Volume Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Cumulative Volume")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# saving

df.to_csv("enhanced_financial_data.csv", index=False)
df_daily.to_csv("daily_resampled_data.csv", index=True)

# previewing

print("Enhanced DataFrame:")
print(df.head())

print("\nDaily Resampled DataFrame:")
print(df_daily.head())
