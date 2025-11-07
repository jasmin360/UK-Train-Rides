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
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numberize as num

X_ticket_price = pd.read_csv( '../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_price/training/entire/X_train.csv')
y_ticket_price = pd.read_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_price/training/entire/y_train.csv').squeeze("columns")

X_ticket_price['Ticket Type'] = X_ticket_price['Ticket Type'].apply(num.numberize_ticket_type)
X_ticket_price['Railcard'] = X_ticket_price['Railcard'].apply(num.numberize_railcard)
X_ticket_price['Ticket Class'] = X_ticket_price['Ticket Class'].apply(num.numberize_ticket_class)
X_ticket_price['Lead Time'] = pd.to_timedelta(X_ticket_price['Lead Time']).dt.days

X_train, X_test, y_train, y_test = train_test_split(X_ticket_price, y_ticket_price, test_size=0.3, random_state=42)

categorical_features = ['Departure Station', 'Arrival Destination']
numeric_features = ['Ticket Type', 'Railcard', 'Ticket Class', 'Lead Time']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])


pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = pipeline.score(X_test, y_test)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")
print("Model Accuracy:", accuracy*100)


joblib.dump(pipeline, '../UK-Train-Rides/machine_learning/models/ticket_price/regression.pkl')
print("model saved")


