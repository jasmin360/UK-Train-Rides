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

X_ticket_type = pd.read_csv( '../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_type_choice/training/entire/X_train.csv')
y_ticket_type = pd.read_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_type_choice/training/entire/y_train.csv').squeeze("columns")


y_ticket_type = y_ticket_type.apply(num.numberize_ticket_type)
y_ticket_type = pd.to_numeric(y_ticket_type, errors='coerce')

X_ticket_type['Lead Time'] = pd.to_timedelta(X_ticket_type['Lead Time']).dt.days
print(X_ticket_type.isna().sum())

X_ticket_type['Day of Week'] = X_ticket_type['Day of Week'].apply(num.numberize_day_of_week)

X_train, X_test, y_train, y_test = train_test_split(X_ticket_type, y_ticket_type,test_size=0.3, random_state=42, stratify=y_ticket_type)

X_train.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_type_choice/training/train_test_split/X_train.csv', index=False)
y_train.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_type_choice/training/train_test_split/y_train.csv', index=False)    
X_test.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_type_choice/training/train_test_split/X_test.csv', index=False)
y_test.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_type_choice/training/train_test_split/y_test.csv', index=False)



categorical_features = ['Departure Station', 'Arrival Destination']
numeric_features = ['Lead Time', 'Day of Week']

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

joblib.dump(pipeline, '../UK-Train-Rides/machine_learning/models/ticket_type/categorical.pkl')