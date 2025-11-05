import numpy as np
import pandas as pd
import processing_function_implementation as pfi
df=pd.read_csv(r'../UK-Train-Rides/cleaned_dataset.csv')

# Convert time columns to datetime
df["Departure Time"] = pd.to_datetime(df["Departure Time"])
df["Actual Arrival Time"] = pd.to_datetime(df["Actual Arrival Time"])
df["Date of Purchase"] = pd.to_datetime(df["Date of Purchase"])
df["Date of Journey"] = pd.to_datetime(df["Date of Journey"])
df["Arrival Time"] = pd.to_datetime(df["Arrival Time"])

print(df.dtypes)

# Fill missing values due to cancellations
df['Actual Arrival Time'] = df['Actual Arrival Time'].fillna('cancelled')



# Calculate trip time and actual trip time
df["Actual Trip Time"]= df.apply(pfi.actual_trip_time, axis=1)
df["Trip Time"]= df.apply(pfi.trip_time, axis=1)

# Calculate percentage delay
df["Delay Time"]=df.apply(pfi.delay_time, axis=1)
df["Percentage Delay"]= df.apply(pfi.delay_percentage, axis=1)

#Calculating lead time
df["Lead Time"] = ( df["Date of Journey"] - df["Date of Purchase"])

# Categorize delays
df['Delay Category'] = df.apply(pfi.categorize_delay, axis=1)
df.to_csv('../UK-Train-Rides/machine_learning/datasets/processed.csv', index=False)

# Extract 'Day of Week' for journey
df['Day of Week'] = df['Date of Journey'].dt.dayofweek
df['Day of Week'] = df['Day of Week'].apply(pfi.number_to_day)



