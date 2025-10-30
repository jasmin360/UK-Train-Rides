import pandas as pd
import numpy as np

def categorize_delay(row):
    if row['Journey Status'].lower() == 'cancelled':
        return 'Cancelled'
    elif row['Percentage Delay'] <= 0:
        return 'On Time'
    elif row['Percentage Delay'] <= 10:
        return 'Short'
    elif row['Percentage Delay'] <= 30:
        return 'Medium'
    elif row['Percentage Delay'] < 100:
        return 'Long'
    else:
        return 'Extreme'


def actual_trip_time(row):
    if row['Journey Status'].lower() == 'cancelled':
        return 'cancelled'
    return row['Actual Arrival Time'] - row['Departure Time']

def trip_time(row):
    return row['Arrival Time'] - row['Departure Time']

def delay_time(row):
    if row['Journey Status'].lower() == 'cancelled':
        return 'cancelled'
    return row['Actual Trip Time'] - row['Trip Time']

def delay_percentage(row):
    if row['Journey Status'].lower() == 'cancelled':
        return 'cancelled'
    return ((row['Actual Trip Time'] - row['Trip Time']) / row['Trip Time']) * 100