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

X_ticket_choice = pd.read_csv( '../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_class_choice/training/entire/X_train.csv')
y_ticket_choice = pd.read_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_class_choice/training/entire/y_train.csv').squeeze("columns")


y_ticket_choice = y_ticket_choice.apply(num.numberize_ticket_class)
y_ticket_choice = pd.to_numeric(y_ticket_choice, errors='coerce')

X_ticket_choice['Lead Time'] = pd.to_timedelta(X_ticket_choice['Lead Time']).dt.days
X_ticket_choice['Railcard'] = X_ticket_choice['Railcard'].apply(num.numberize_railcard)
X_ticket_choice['Ticket Type'] = X_ticket_choice['Ticket Type'].apply(num.numberize_ticket_type)
print(X_ticket_choice.isna().sum())


X_train, X_test, y_train, y_test = train_test_split(X_ticket_choice, y_ticket_choice,test_size=0.3, random_state=42, stratify=y_ticket_choice)

X_train.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_class_choice/train_test_split/X_train.csv', index=False)
y_train.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_class_choice/train_test_split/y_train.csv', index=False)    
X_test.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_class_choice/train_test_split/X_test.csv', index=False)
y_test.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_class_choice/train_test_split/y_test.csv', index=False)



categorical_features = ['Departure Station', 'Arrival Destination']
numeric_features = ['Lead Time', 'Railcard', 'Ticket Type', 'Price']

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

joblib.dump(pipeline, '../UK-Train-Rides/machine_learning/models/ticket_class/categorical.pkl')