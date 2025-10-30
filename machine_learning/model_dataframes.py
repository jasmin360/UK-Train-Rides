import machine_learning_dataset_processing as mldp
import pandas as pd

#average percentage delay per route
route_delay = mldp.df.copy()
route_delay= route_delay[route_delay['Percentage Delay'] != 'cancelled']
route_delay= route_delay.groupby(['Departure Station', 'Arrival Destination'])['Percentage Delay'].mean().reset_index()

route_delay.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/route_delay.csv', index=False)

X_train_route_delay = route_delay[['Departure Station', 'Arrival Destination']]
y_train_route_delay = route_delay['Percentage Delay']

X_train_route_delay.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/training/entire/X_train.csv', index=False)
y_train_route_delay.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/training/entire/y_train.csv', index=False)