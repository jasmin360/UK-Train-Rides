import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load the route delay dataset
X_train_route_delay = pd.read_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/training/entire/X_train.csv')
y_train_route_delay = pd.read_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/training/entire/y_train.csv')

# Ensure 'Percentage Delay' is numeric
route_delay = pd.concat([X_train_route_delay, y_train_route_delay], axis=1)
route_delay['Percentage Delay'] = pd.to_numeric(route_delay['Percentage Delay'], errors='coerce')
route_delay = route_delay.dropna(subset=['Percentage Delay'])

# Split features and target
X = route_delay[['Departure Station', 'Arrival Destination']]
y = route_delay['Percentage Delay']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: One-hot encode categorical columns
categorical_features = ['Departure Station', 'Arrival Destination']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Model: Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create a pipeline: preprocessing + model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the trained model
joblib.dump(pipeline, '../UK-Train-Rides/machine_learning/route_delay_model.pkl')

print("Model saved successfully!")
