import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os


df = pd.read_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/class_demand_forecast/class_demand_forecast.csv')
df["Date of Journey"] = pd.to_datetime(df["Date of Journey"])
df = df.sort_values("Date of Journey")
df = df.set_index("Date of Journey")


CLASS_COL = "Ticket Class"
DEMAND_COL = "Demand"
FORECAST_HORIZON = 30       # Next 30 days to forecast
TEST_HOLDOUT_DAYS = 30      # Last N days for test evaluation
MIN_TRAIN_ROWS = 60         # Minimum rows to train model


df_reset = df.reset_index()
df_agg = df_reset.groupby(["Date of Journey", CLASS_COL], as_index=False)[DEMAND_COL].sum()
df_agg = df_agg.set_index("Date of Journey")  # index is date


classes = df_agg[CLASS_COL].unique().tolist()
print(f"Found ticket classes: {classes}")


all_forecasts = []


def make_features(series: pd.Series):
    df_feat = pd.DataFrame({"demand": series})
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["day"] = df_feat.index.day
    df_feat["month"] = df_feat.index.month
    df_feat["time_idx"] = np.arange(len(df_feat))
    return df_feat


for cls in classes:
    print(f"\n--- Processing class: {cls} ---")
    
    # Select series for this class
    series = df_agg[df_agg[CLASS_COL] == cls][DEMAND_COL].copy()
    
    # Ensure all days are present
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
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE on last {TEST_HOLDOUT_DAYS} days: {mae:.2f}")
    
    # Forecast next 30 days autoregressively
    future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1),
                                 periods=FORECAST_HORIZON, freq="D")
    future_preds = []
    last_known = series.copy()
    
    for d in future_dates:
        row = {}
        row["dayofweek"] = d.dayofweek
        row["day"] = d.day
        row["month"] = d.month
        row["time_idx"] = len(series) + len(future_preds)
        
        X_row = pd.DataFrame([row])
        pred_val = float(model.predict(X_row)[0])
        future_preds.append(pred_val)
        
        last_known = pd.concat([last_known, pd.Series([pred_val], index=[d])])

    
    # Save forecast
    df_future = pd.DataFrame({
        "Date": future_dates,
        CLASS_COL: cls,
        "Predicted_Demand": future_preds
    })
    all_forecasts.append(df_future)
    
#plots
    plt.figure(figsize=(12,5))
    lookback = 90
    plt.plot(series.index[-lookback:], series[-lookback:], label="History (last 90 days)")
    plt.plot(X_test.index, y_pred, label="Test Predictions", linestyle='--', color='orange')
    plt.plot(future_dates, future_preds, label=f"Forecast (next {FORECAST_HORIZON} days)", linestyle='--', color='green')
    plt.title(f"Ticket Class: '{cls}' — MAE: {mae:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Predicted Demand")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    safe_name = str(cls)
    os.makedirs("forecast_results/class_demand_forecast", exist_ok=True)
    plt.savefig(f"forecast_results/class_demand_forecast/forecast_{safe_name}.png")