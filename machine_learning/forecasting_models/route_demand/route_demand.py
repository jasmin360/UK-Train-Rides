import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os


df = pd.read_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_demand_forecast/route_demand_forecast.csv')
df["Date of Journey"] = pd.to_datetime(df["Date of Journey"])
df = df.sort_values("Date of Journey")
df = df.set_index("Date of Journey")

ROUTE_COL = "Route"
DEMAND_COL = "Demand"
FORECAST_HORIZON = 30
TEST_HOLDOUT_DAYS = 30

# Ensure output folder for plots
os.makedirs("forecast_plots_routes", exist_ok=True)


df_reset = df.reset_index()
df_agg = df_reset.groupby(["Date of Journey", ROUTE_COL], as_index=False)[DEMAND_COL].sum()
df_agg = df_agg.set_index("Date of Journey")

# List of unique routes
routes = df_agg[ROUTE_COL].unique().tolist()
print(f"Found routes: {routes}")


def make_features(series: pd.Series):
    df_feat = pd.DataFrame({"demand": series})
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["day"] = df_feat.index.day
    df_feat["month"] = df_feat.index.month
    df_feat["time_idx"] = np.arange(len(df_feat))
    return df_feat


for route in routes:
    print(f"\n--- Processing route: {route} ---")
    
    series = df_agg[df_agg[ROUTE_COL] == route][DEMAND_COL].copy()
    
    # Ensure all days exist
    idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq="D")
    series = series.reindex(idx, fill_value=0)
    series.index.name = "Date"
    
    # Create features
    feat = make_features(series).drop(columns=["demand"])
    target = series.copy()
    data = feat.join(target.rename("demand")).dropna()
    
    # Train/test split
    train_size = max(0, len(data) - TEST_HOLDOUT_DAYS)
    X_train = data.iloc[:train_size].drop(columns="demand")
    y_train = data.iloc[:train_size]["demand"]
    X_test = data.iloc[train_size:].drop(columns="demand")
    y_test = data.iloc[train_size:]["demand"]
    
    # Train model
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Test predictions
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE on test set: {mae:.2f}")
    
    # Forecast next 30 days
    future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1),
                                 periods=FORECAST_HORIZON, freq="D")
    future_preds = []
    last_known = series.copy()
    
    for d in future_dates:
        row = {
            "dayofweek": d.dayofweek,
            "day": d.day,
            "month": d.month,
            "time_idx": len(series) + len(future_preds)
        }
        X_row = pd.DataFrame([row])
        pred_val = int(model.predict(X_row)[0])
        future_preds.append(pred_val)
        last_known = pd.concat([last_known, pd.Series([pred_val], index=[d])])
    
    # Plot history + test + forecast
    plt.figure(figsize=(12,5))
    lookback = 90
    plt.plot(series.index[-lookback:], series[-lookback:], label="History")
    plt.plot(y_test.index, y_pred, label="Test Predictions", linestyle='--', color='orange')
    plt.plot(future_dates, future_preds, label=f"Forecast (next {FORECAST_HORIZON} days)", linestyle='--', color='green')
    plt.title(f"Route: {route} — MAE: {mae:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    safe_name = str(route).replace(" ", "_").replace("/", "_")
    os.makedirs("forecast_results/forecast_plots_routes", exist_ok=True)
    plt.savefig(f"forecast_results/forecast_plots_routes/forecast_{safe_name}.png")
    plt.close()
    
print("\nForecast plots saved in folder 'forecast_plots_routes/'")
print("\nFuture 30 Days Route Demand Forecasts")
print(future_preds)
print(future_dates)