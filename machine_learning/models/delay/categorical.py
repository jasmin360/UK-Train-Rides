import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import  accuracy_score
from sklearn.ensemble import RandomForestClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numberize as num
import joblib


X_route_delay = pd.read_csv( '../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/training/entire/X_train.csv')
y_route_delay = pd.read_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/training/entire/y_train.csv').squeeze("columns")


y_route_delay= y_route_delay.apply(num.numberize_delay_category)

# Ensure numeric
y_route_delay = pd.to_numeric(y_route_delay, errors='coerce')


X_route_delay['Departure Time'] = pd.to_datetime(X_route_delay['Departure Time'], errors='coerce')
X_route_delay['Departure Day'] = X_route_delay['Departure Time'].dt.dayofweek
X_route_delay = X_route_delay.drop(columns=['Departure Time'])

X_route_delay['Route'] = X_route_delay['Departure Station'] + "_" + X_route_delay['Arrival Destination']

# Add AvgRouteDelay
temp_df = X_route_delay.copy()
temp_df['Percentage Delay'] = y_route_delay
avg_delay = temp_df.groupby('Route')['Percentage Delay'].mean().to_dict()
X_route_delay['AvgRouteDelay'] = X_route_delay['Route'].map(avg_delay)
X_route_delay['AvgRouteDelay'] = X_route_delay['AvgRouteDelay'].fillna(X_route_delay['AvgRouteDelay'].mean())



X_train, X_test, y_train, y_test = train_test_split(X_route_delay, y_route_delay, test_size=0.3, random_state=42)

X_train.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/training/train_test_split/X_train.csv', index=False)
y_train.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/training/train_test_split/y_train.csv', index=False)    
X_test.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/training/train_test_split/X_test.csv', index=False)
y_test.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/training/train_test_split/y_test.csv', index=False)

categorical_features = ['Departure Station', 'Arrival Destination', 'Route']
numeric_features = ['Departure Hour', 'Departure Day', 'IsPeakHour', 'AvgRouteDelay', 'DayOfWeek', 'IsWeekend', 'Month']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)



model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")


joblib.dump(pipeline, '../UK-Train-Rides/machine_learning/models/delay/categorical.pkl')
print("model saved")


comparison_df = pd.DataFrame({
    'Actual Category': y_test.values,
    'Predicted Category': y_pred
})
print("\n sample")
print(comparison_df.head(50).to_string(index=False))
