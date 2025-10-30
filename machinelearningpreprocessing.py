import numpy as np
import pandas as pd
df=pd.read_csv(r'../UK-Train-Rides/cleaned_dataset.csv')

df["Departure Time"] = pd.to_datetime(df["Departure Time"])
df["Actual Arrival Time"] = pd.to_datetime(df["Actual Arrival Time"])

df["Trip Time"]= df["Actual Arrival Time"] - df["Departure Time"]