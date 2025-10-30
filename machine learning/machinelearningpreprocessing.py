import numpy as np
import pandas as pd
df=pd.read_csv(r'../UK-Train-Rides/cleaned_dataset.csv')

# Convert time columns to datetime
df["Departure Time"] = pd.to_datetime(df["Departure Time"])
df["Actual Arrival Time"] = pd.to_datetime(df["Actual Arrival Time"])
df["Date of Purchase"] = pd.to_datetime(df["Date of Purchase"])
df["Date of Journey"] = pd.to_datetime(df["Date of Journey"])
df["Arrival Time"] = pd.to_datetime(df["Arrival Time"])


print(df.dtypes)


# Calculate trip time and actual trip time
df["Actual Trip Time"]= df["Actual Arrival Time"] - df["Departure Time"]
df["Trip Time"]= df["Arrival Time"] - df["Departure Time"]

# Calculate percentage delay
df["Percentage Delay"]= ((df["Actual Trip Time"]-df["Trip Time"]) / df["Trip Time"]) * 100

#Calculating lead time
df["Lead Time"] = ( df["Date of Journey"] - df["Date of Purchase"])

df.to_csv('../UK-Train-Rides/machine learning/cleaned_dataset2.csv', index=False)