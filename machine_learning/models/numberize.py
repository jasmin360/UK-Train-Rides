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
    if x == "Standard":
        return 0
    elif x == "First Class":
        return 1
    
def numberize_purchase_channel(x):
    if x.lower() == "online":
        return 0
    elif x.lower() == "station":
        return 1

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h*3600 + m*60 + s
