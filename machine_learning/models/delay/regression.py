import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------
# Load and preprocess data
# -----------------------
route_delay = pd.read_csv(r'../UK-Train-Rides/machine_learning/datasets/processed.csv')
route_delay['Departure Time'] = pd.to_datetime(route_delay['Departure Time'], errors='coerce')
route_delay = route_delay[route_delay['Percentage Delay'] != 'cancelled']
route_delay = route_delay.loc[:, ['Departure Station', 'Arrival Destination','Departure Time','Percentage Delay']]
route_delay['Departure Hour'] = route_delay['Departure Time'].dt.hour
route_delay['DayOfWeek'] = route_delay['Departure Time'].dt.dayofweek
route_delay['IsWeekend'] = route_delay['DayOfWeek'].isin([5, 6]).astype(int)
route_delay['Month'] = route_delay['Departure Time'].dt.month
route_delay['IsPeakHour'] = route_delay['Departure Hour'].between(7, 9) | route_delay['Departure Hour'].between(16, 19)

X_route_delay = route_delay[['Departure Station', 'Arrival Destination','Departure Time','Departure Hour','DayOfWeek','IsWeekend','Month','IsPeakHour']]
y_route_delay = pd.to_numeric(route_delay['Percentage Delay'], errors='coerce')

# -----------------------
# Clean target
# -----------------------
y_route_delay = y_route_delay.replace([np.inf, -np.inf], np.nan)   # Remove infinite values
y_route_delay = y_route_delay.dropna()                             # Drop NaNs

# -----------------------
# Feature processing
# -----------------------
X_route_delay['Departure Day'] = X_route_delay['Departure Time'].dt.dayofweek
X_route_delay = X_route_delay.drop(columns=['Departure Time'])
X_route_delay['Route'] = X_route_delay['Departure Station'] + "_" + X_route_delay['Arrival Destination']

# Average route delay
temp_df = X_route_delay.copy()
temp_df['Percentage Delay'] = y_route_delay
avg_delay = temp_df.groupby('Route')['Percentage Delay'].mean().to_dict()
X_route_delay['AvgRouteDelay'] = X_route_delay['Route'].map(avg_delay)
X_route_delay['AvgRouteDelay'] = X_route_delay['AvgRouteDelay'].fillna(X_route_delay['AvgRouteDelay'].mean())

# -----------------------
# Train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X_route_delay, y_route_delay, test_size=0.3, random_state=42)

categorical_features = ['Departure Station', 'Arrival Destination', 'Route']
numeric_features = ['Departure Hour', 'Departure Day', 'IsPeakHour','AvgRouteDelay',  'DayOfWeek', 'IsWeekend', 'Month']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# -----------------------
# Train model
# -----------------------
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# -----------------------
# Evaluation
# -----------------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = pipeline.score(X_test, y_test)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")
print("Model Accuracy:", accuracy*100)

# -----------------------
# Save model
# -----------------------
joblib.dump(pipeline, '../UK-Train-Rides/machine_learning/models/delay/regression.pkl')
print("Model saved successfully!")

# -----------------------
# Comparison & plots
# -----------------------
comparison_df = pd.DataFrame({'Actual Delay (%)': y_test.values, 'Predicted Delay (%)': y_pred})
print("\n sample (first 20):")
print(comparison_df.head(20).to_string(index=False))

# Plot histogram of actual delays (no log transform)
plt.figure(figsize=(8,4))
sns.histplot(y_route_delay, bins=50, kde=False)
plt.title("Distribution of Percentage Delays")
plt.xlabel("Delay (%)")
plt.ylabel("Frequency")
plt.show()

print(y_route_delay.describe())
