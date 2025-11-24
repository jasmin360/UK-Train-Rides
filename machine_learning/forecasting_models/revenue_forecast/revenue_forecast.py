import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


df = pd.read_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/revenue_forecast/revenue_forecast.csv')

df["Date of Journey"] = pd.to_datetime(df["Date of Journey"])
df = df.sort_values("Date of Journey")
df = df.set_index("Date of Journey")   


y = df["Price"]      


df["Year"] = df.index.year
df["Month"] = df.index.month
df["Day"] = df.index.day
df["Weekday"] = df.index.weekday

X = df[["Year", "Month", "Day", "Weekday"]]

#train-test split (last 30 days for testing)
train_size = len(df) - 30
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]


model = GradientBoostingRegressor()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Absolute Error: {mae:.3f}")



#forecast next 30 days
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                             periods=30, freq='D')

future_df = pd.DataFrame(index=future_dates)

for d in future_dates:
    row = {
        "Year": d.year,
        "Month": d.month,
        "Day": d.day,
        "Weekday": d.weekday()
    }
    
    X_future = pd.DataFrame([row])
    pred_value = model.predict(X_future)[0]

    future_df.loc[d, "Predicted Revenue"] = pred_value


#forecast plot
plt.figure(figsize=(12,5))
lookback = 90
plt.plot(df.index, y, label="Historical Revenue")
plt.plot(future_df.index, future_df["Predicted Revenue"], label="Forecasted Revenue", linestyle='--', color='green')
plt.plot(X_test.index, y_pred, label="Test Predictions", linestyle='--', color='orange')
plt.title("Revenue Forecast")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.grid(True)
plt.show()

print("\nFuture 30 Days Revenue Forecast:")
print(future_df)