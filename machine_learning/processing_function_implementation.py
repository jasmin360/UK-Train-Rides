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

def number_to_day(day_number):
    if day_number == 0:
        return 'Monday'
    elif day_number == 1:
        return 'Tuesday'
    elif day_number == 2:
        return 'Wednesday'
    elif day_number == 3:
        return 'Thursday'
    elif day_number == 4:
        return 'Friday'
    elif day_number == 5:
        return 'Saturday'
    elif day_number == 6:
        return 'Sunday'
    else:
        return None # Just in case