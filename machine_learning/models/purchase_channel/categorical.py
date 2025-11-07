import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import  accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

X_purchase_channel = pd.read_csv( '../UK-Train-Rides/machine_learning/datasets/model_datasets/purchase_channel/training/entire/X_train.csv')
y_purchase_channel = pd.read_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/purchase_channel/training/entire/y_train.csv').squeeze("columns")

def numberize_purchase_channel(x):
    if x.lower() == "online":
        return 0
    elif x.lower() == "station":
        return 1

y_purchase_channel = y_purchase_channel.apply(numberize_purchase_channel)
y_purchase_channel = pd.to_numeric(y_purchase_channel, errors='coerce')

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h*3600 + m*60 + s

X_purchase_channel['Time of Purchase'] = X_purchase_channel['Time of Purchase'].apply(time_to_seconds)

X_purchase_channel['Lead Time'] = pd.to_timedelta(X_purchase_channel['Lead Time']).dt.days

X_purchase_channel['Purchase Hour'] = X_purchase_channel['Time of Purchase'] // 3600
X_purchase_channel['Purchase Minute'] = (X_purchase_channel['Time of Purchase'] % 3600) // 60
X_purchase_channel['Is Morning'] = X_purchase_channel['Purchase Hour'].between(6, 11).astype(int)
X_purchase_channel['Is Afternoon'] = X_purchase_channel['Purchase Hour'].between(12, 17).astype(int)
X_purchase_channel['Is Evening'] = X_purchase_channel['Purchase Hour'].between(18, 23).astype(int)

X_purchase_channel['Date of Purchase'] = pd.to_datetime( X_purchase_channel['Date of Purchase'], errors='coerce')
X_purchase_channel['DayOfWeek'] = X_purchase_channel['Date of Purchase'].dt.dayofweek
X_purchase_channel['IsWeekend'] = X_purchase_channel['Date of Purchase'].dt.weekday >= 5
X_purchase_channel['Purchase Month'] = X_purchase_channel['Date of Purchase'].dt.month


X_train, X_test, y_train, y_test = train_test_split(X_purchase_channel, y_purchase_channel,test_size=0.3, random_state=42, stratify=y_purchase_channel)

X_train.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/purchase_channel/training/train_test_split/X_train.csv', index=False)
y_train.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/purchase_channel/training/train_test_split/y_train.csv', index=False)    
X_test.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/purchase_channel/training/train_test_split/X_test.csv', index=False)
y_test.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/purchase_channel/training/train_test_split/y_test.csv', index=False)



categorical_features = ['Ticket Type']
numeric_features = ['Lead Time','Time of Purchase', 'Purchase Hour', 'Purchase Minute', 'Is Morning', 'Is Afternoon', 'Is Evening', 'DayOfWeek', 'IsWeekend', 'Purchase Month']

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

joblib.dump(pipeline, '../UK-Train-Rides/machine_learning/models/purchase_channel/categorical.pkl')