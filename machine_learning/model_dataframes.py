import machine_learning_dataset_processing as mldp
import pandas as pd

#percentage delay per route
route_delay =pd.read_csv(r'../UK-Train-Rides/machine_learning/datasets/processed.csv')
print(route_delay.columns)
route_delay['Departure Time'] = pd.to_datetime(route_delay['Departure Time'], errors='coerce')
route_delay= route_delay.loc[:,['Departure Station', 'Arrival Destination','Departure Time','Percentage Delay','Delay Category' ]]
route_delay['Departure Hour'] = route_delay['Departure Time'].dt.hour
route_delay['DayOfWeek'] = route_delay['Departure Time'].dt.dayofweek
route_delay['IsWeekend'] = route_delay['DayOfWeek'].isin([5, 6]).astype(int)
route_delay['Month'] = route_delay['Departure Time'].dt.month
route_delay['IsPeakHour'] = route_delay['Departure Hour'].between(7, 9) | route_delay['Departure Hour'].between(16, 19)

route_delay.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/delay.csv', index=False)

X_train_route_delay = route_delay[['Departure Station', 'Arrival Destination','Departure Time','Departure Hour','DayOfWeek','IsWeekend','Month','IsPeakHour']]
y_train_route_delay = route_delay['Delay Category']

X_train_route_delay.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/training/entire/X_train.csv', index=False)
y_train_route_delay.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/training/entire/y_train.csv', index=False)


#ticket price
ticket_price = mldp.df.copy()
ticket_price= ticket_price.loc[:,['Ticket Type','Ticket Class','Departure Station', 'Arrival Destination', 'Railcard','Lead Time','Price' ]]

ticket_price.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_price/ticket_price.csv', index=False)

X_train_ticket_price = ticket_price[['Ticket Type','Ticket Class','Departure Station', 'Arrival Destination', 'Railcard','Lead Time']]
y_train_ticket_price = ticket_price['Price']

X_train_ticket_price.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_price/training/entire/X_train.csv', index=False)
y_train_ticket_price.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_price/training/entire/y_train.csv', index=False)


#Ticket Class Choice
ticket_class_choice = mldp.df.copy()
ticket_class_choice = ticket_class_choice.loc[:,['Departure Station', 'Arrival Destination','Railcard','Ticket Type','Lead Time','Price','Ticket Class' ]]

ticket_class_choice.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_class_choice/ticket_class_choice.csv', index=False)

X_train_ticket_class_choice = ticket_class_choice[['Departure Station', 'Arrival Destination','Railcard','Ticket Type','Lead Time','Price']]
y_train_ticket_class_choice = ticket_class_choice['Ticket Class']

X_train_ticket_class_choice.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_class_choice/training/entire/X_train.csv', index=False)
y_train_ticket_class_choice.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_class_choice/training/entire/y_train.csv', index=False)


#purchase type (channel) prediction
purchase_channel = mldp.df.copy()
purchase_channel = purchase_channel.loc[:, ['Time of Purchase','Lead Time','Ticket Type','Purchase Type', 'Date of Purchase']]

purchase_channel.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/purchase_channel/purchase_channel.csv', index=False)

X_train_purchase_channel = purchase_channel[['Time of Purchase','Lead Time','Ticket Type','Date of Purchase']]
y_train_purchase_channel = purchase_channel['Purchase Type']

X_train_purchase_channel.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/purchase_channel/training/entire/X_train.csv', index=False)
y_train_purchase_channel.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/purchase_channel/training/entire/y_train.csv', index=False)


#ticket type predicition (how early will a person buy the ticket)
ticket_type_choice = mldp.df.copy()
ticket_type_choice = ticket_type_choice.loc[:, ['Lead Time','Departure Station', 'Arrival Destination','Day of Week','Ticket Type']]

ticket_type_choice.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_type_choice/ticket_type_choice.csv', index=False)

X_train_ticket_type_choice = ticket_type_choice[['Lead Time','Departure Station', 'Arrival Destination','Day of Week']] 
y_train_ticket_type_choice = ticket_type_choice['Ticket Type']

X_train_ticket_type_choice.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_type_choice/training/entire/X_train.csv', index=False)
y_train_ticket_type_choice.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/ticket_type_choice/training/entire/y_train.csv', index=False)



#refund request likelihood 
refund_request = mldp.df.copy()
refund_request = refund_request[refund_request['Delay Category'] != 'On Time']
refund_request = refund_request.loc[:, ['Price','Ticket Class','Refund Request','Delay Category']]

refund_request.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/refund_request/refund_request.csv', index=False)

X_train_refund_request = refund_request[['Price','Ticket Class','Delay Category']] 
y_train_refund_request = refund_request['Refund Request']

X_train_refund_request.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/refund_request/training/entire/X_train.csv', index=False)
y_train_refund_request.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/refund_request/training/entire/y_train.csv', index=False)