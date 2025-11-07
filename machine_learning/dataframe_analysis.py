import machine_learning_dataset_processing as mldp
import pandas as pd

#average percentage delay per route
route_delay = mldp.df.copy()
route_delay= route_delay[route_delay['Percentage Delay'] != 'cancelled']
route_delay= route_delay.groupby(['Departure Station', 'Arrival Destination'])['Percentage Delay'].mean().reset_index()

route_delay.to_csv('../UK-Train-Rides/machine_learning/datasets/analysed_Data/average_route_delay.csv', index=False)


#el analysis Qs ely liha 3laqa bl revenue wl demand 8aleban gowa el model datasetsss