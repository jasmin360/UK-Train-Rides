def numberize_refund(x):
    if x.lower() == "no":
        return 0
    elif x.lower() == "yes":
        return 1
    
def numberize_delay_category(x):
    if x == "On Time":
        return 0
    elif x == "Short":
        return 1
    elif x == "Medium":
        return 2
    elif x == "Long":
        return 3
    elif x== "Extreme":
        return 4
    elif x == "Cancelled":
        return -1
    
def numberize_ticket_class(x):
    if x.lower() == "standard":
        return 0
    elif x.lower() == "first class":
        return 1
    
def numberize_purchase_channel(x):
    if x.lower() == "online":
        return 0
    elif x.lower() == "station":
        return 1

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h*3600 + m*60 + s

def numberize_ticket_type(x):
    if x.lower() == "advance":
        return 0
    elif x.lower() == "anytime":
        return 1
    elif x.lower() == "off-peak":
        return 2
    
def numberize_railcard(x):
    if x == "no railcard":
        return 0
    elif x == "senior":
        return 1
    elif x == "disabled":
        return 2
    elif x == "adult":
        return 3


def numberize_day_of_week(x):
    days = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    return days.get(x, -1)  