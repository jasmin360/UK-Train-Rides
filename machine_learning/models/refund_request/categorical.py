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

X_refund = pd.read_csv( '../UK-Train-Rides/machine_learning/datasets/model_datasets/refund_request/training/entire/X_train.csv')
y_refund = pd.read_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/refund_request/training/entire/y_train.csv').squeeze("columns")



y_refund = y_refund.apply(num.numberize_refund)
y_refund = pd.to_numeric(y_refund, errors='coerce')



X_refund['Ticket Class'] = X_refund['Ticket Class'].apply(num.numberize_ticket_class)
X_refund['Delay Category'] = X_refund['Delay Category'].apply(num.numberize_delay_category) 

X_train, X_test, y_train, y_test = train_test_split(X_refund, y_refund,test_size=0.3, random_state=42, stratify=y_refund)

X_train.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/refund_request/training/train_test_split/X_train.csv', index=False)
y_train.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/refund_request/training/train_test_split/y_train.csv', index=False)    
X_test.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/refund_request/training/train_test_split/X_test.csv', index=False)
y_test.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/refund_request/training/train_test_split/y_test.csv', index=False)


numeric_features = ['Price', 'Delay Category', 'Ticket Class']

preprocessor = ColumnTransformer(
    transformers=[
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

#joblib.dump(pipeline, '../UK-Train-Rides/machine_learning/models/purchase_channel/categorical.pkl')