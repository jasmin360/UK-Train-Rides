import machine_learning_dataset_processing as mldp
import pandas as pd

#percentage delay per route
route_delay = mldp.df.copy()
route_delay= route_delay[route_delay['Percentage Delay'] != 'cancelled']
route_delay= route_delay.loc[:,['Departure Station', 'Arrival Destination','Percentage Delay']]

route_delay.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/route_delay/route_delay.csv', index=False)

X_train_route_delay = route_delay[['Departure Station', 'Arrival Destination']]
y_train_route_delay = route_delay['Percentage Delay']

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
purchase_channel = purchase_channel.loc[:, ['Time of Purchase','Lead Time','Ticket Type','Purchase Type']]

purchase_channel.to_csv('../UK-Train-Rides/machine_learning/datasets/model_datasets/purchase_channel/purchase_channel.csv', index=False)

X_train_purchase_channel = purchase_channel[['Time of Purchase','Lead Time','Ticket Type']]
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