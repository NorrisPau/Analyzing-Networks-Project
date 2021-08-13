# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

feasible = []
import copy
def combinations(target,data):
    for i in range(len(data)):
        #print(i)
        new_target = copy.copy(target)
        new_data = copy.copy(data)
        new_target.append(data[i])
        new_data = data[i + 1:]
        #print(list(new_target), list(new_data))
        weights = table.loc[new_target]["weights"]
        if weights.sum() < 5:
            if new_target not in feasible:
            feasible.append(new_target)
        else:
            new_target = new_target[:-1]
        combinations(new_target, new_data)


target = []
data = range(6)
y_data = [1.0,3.0,2.0,3.0,1.0,1.0]
table = pd.DataFrame({'orders':data, 'weights':y_data})
#print(table)

combinations(target,data)


import numpy as np

time1 = np.random.randint(40, 70, 22)
time_diff = np.random.randint(-5, 5, 22)
time2 = time1 + time_diff
time1

test_df = pd.DataFrame({'orders':feasible, 'time1':time1, 'time2':time2})
test_df['time1PerItem'] = test_df['time1']/test_df['orders'].apply(len)
test_df['time2PerItem'] = test_df['time2']/test_df['orders'].apply(len)

# Assumption: there are only 2 packing station and 2 cobots. This is consistent with both 24 and 360 layouts
# Keep track of time for each packer and cobot. Prep time is 30 seconds (only assigned at the beginning), so each time is
# initialized at 30 seconds.

CobotCount = 2
TimePacker1 = 30
TimePacker2 = 30
TimeCobot1 = 30
TimeCobot2 = 30

BatchAssignCobot1 = []
BatchAssignCobot2 = []

# Question: We need to reduce overall time to complete orders. However that time accumulates from robots picking up orders
# and packers packing the orders. Is it more beneficial to keep the packers busy or the cobots busy?
# Strategy: We will first send the cobots to pick up the quickest batch, thereby getting the packers working in the shortest
# amount of time (the packers stand idle until the first batch is returned). After that, the cobots will pick up batches
# that have the shortest time PER item.

# Initialize table of feasible orders
OrdersTable = test_df

# First batch assignment

# Find batch with shortest time
next_batch = OrdersTable.orders[OrdersTable.time1.idxmin()]
next_batch_time = OrdersTable.time1.min()

# Assign batch to cobot. Add time to cobot and packer (packing time is 60 sec per item)
BatchAssignCobot1.append(next_batch)
TimeCobot1 += next_batch_time + 20
TimePacker1 += next_batch_time + 20 + 60*len(next_batch)

# Remove any rows containing any of these orders
dropping_rows = OrdersTable.orders.apply(lambda x: any(item in next_batch for item in x))
OrdersTable = OrdersTable[dropping_rows==False]

### Do the exact same thing for cobot2
next_batch = OrdersTable.orders[OrdersTable.time2.idxmin()]
next_batch_time = OrdersTable.time2.min()

# Assign batch to cobot. Add time to cobot and packer (packing time is 60 sec per item)
BatchAssignCobot2.append(next_batch)
TimeCobot2 += next_batch_time + 20
TimePacker2 += next_batch_time + 20 + 60*len(next_batch)

# Remove any rows containing any of these orders
dropping_rows = OrdersTable.orders.apply(lambda x: any(item in next_batch for item in x))
OrdersTable = OrdersTable[dropping_rows==False]

cobot_times = pd.Series([TimeCobot1, TimeCobot2])
# The loop will continue to run until there are no feasible orders left. At each iteration the feasible orders will be reduced.

while len(OrdersTable) > 0:
    # Find next available cobot
    next_available_cobot = cobot_times.idxmin()
    


#OrdersList = list(test_df.orders)

#len(test_df.orders[2])

