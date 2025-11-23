
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/class_demand_forecast/class_demand_forecast.csv')  # replace with your file


df["Date of Journey"] = pd.to_datetime(df["Date of Journey"])


df["Year"] = df["Date of Journey"].dt.year
df["Month"] = df["Date of Journey"].dt.month
df["Day"] = df["Date of Journey"].dt.day
df["Weekday"] = df["Date of Journey"].dt.weekday


X = df[["Ticket Class", "Year", "Month", "Day", "Weekday"]]
y = df["Demand"]   

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ["Ticket Class"]),
        ('num', 'passthrough', ["Year", "Month", "Day", "Weekday"])
    ]
)


model = Pipeline(steps=[
    ('prep', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining model...")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.3f}")


plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted Class Demand")
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Ticket Class", y="Demand")
plt.title("Demand Distribution by Class")
plt.ylabel("Demand")
plt.show()


plt.figure(figsize=(12, 5))
plt.plot(df["Date of Journey"], df["Demand"])
plt.title("Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.xticks(rotation=45)
plt.show()

# --------------------------------
# Future Forecast Example
# --------------------------------
future_input = pd.DataFrame({
    "Ticket Class": ["first", "standard"],
    "Year": [2026, 2026],
    "Month": [1, 1],
    "Day": [15, 15],
    "Weekday": [4, 4]
})

future_pred = model.predict(future_input)

print("\nFuture Prediction Example:")
for cls, pred in zip(future_input["Ticket Class"], future_pred):
    print(f"Predicted demand for {cls}: {pred:.1f}")
