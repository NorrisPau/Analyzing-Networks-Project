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
data = range(7)
y_data = [1.0,3.0,2.0,3.0,1.0,1.0,3.0]
table = pd.DataFrame({'orders':data, 'weights':y_data})
#print(table)

combinations(target,data)


import numpy as np

batch_penalty = []
for j in feasible:
    batch_penalty.append(10*len(j))

time1 = np.random.randint(40, 60, len(feasible))
time1 += batch_penalty

time_diff = np.random.randint(-5, 5, len(feasible))
time2 = time1 + time_diff


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

# Pair these into lists for easier indexing
TimePacker_List = [TimePacker1, TimePacker2]
TimeCobot_List = [TimeCobot1, TimeCobot2]
BatchAssignCobot_List = [[], []]

# Question: We need to reduce overall time to complete orders. However that time accumulates from robots picking up orders
# and packers packing the orders. Is it more beneficial to keep the packers busy or the cobots busy?
# Strategy: We will first send the cobots to pick up the quickest batch, thereby getting the packers working in the shortest
# amount of time (the packers stand idle until the first batch is returned). After that, the cobots will pick up batches
# that have the shortest time PER item.

# Initialize table of feasible orders
OrdersTable = test_df

##### I could write this in a loop. If there's time
# First batch assignment

# Find batch with shortest time
next_batch = OrdersTable.orders[OrdersTable.time1.idxmin()]
next_batch_time = OrdersTable.time1.min()

# Assign batch to cobot. Add time to cobot and packer (packing time is 60 sec per item)
BatchAssignCobot_List[0].append(next_batch)
TimeCobot_List[0] += next_batch_time + 20
TimePacker_List[0] += next_batch_time + 20 + 60*len(next_batch)

# Remove any rows containing any of these orders
dropping_rows = OrdersTable.orders.apply(lambda x: any(item in next_batch for item in x))
OrdersTable = OrdersTable[dropping_rows==False]

### Do the exact same thing for cobot2
next_batch = OrdersTable.orders[OrdersTable.time2.idxmin()]
next_batch_time = OrdersTable.time2.min()

# Assign batch to cobot. Add time to cobot and packer (packing time is 60 sec per item)
BatchAssignCobot_List[1].append(next_batch)
TimeCobot_List[1] += next_batch_time + 20
TimePacker_List[1] += next_batch_time + 20 + 60*len(next_batch)

# Remove any rows containing any of these orders
dropping_rows = OrdersTable.orders.apply(lambda x: any(item in next_batch for item in x))
OrdersTable = OrdersTable[dropping_rows==False]

# The loop will continue to run until there are no feasible orders left. At each iteration the feasible orders will be reduced.
i = 0
while len(OrdersTable) > 0:
    # Subset orders table into time columns
    i += 1
    print("Iteration", i)
    print("Orders Left", OrdersTable)
    print("Current Batch Assignments", BatchAssignCobot_List)
    print("Cobot Times", TimeCobot_List)
    print("Packer Times", TimePacker_List)

    batch_times = OrdersTable[['time1PerItem', 'time2PerItem']]
    # Find next available cobot
    next_available_cobot = TimeCobot_List.index(min(TimeCobot_List))
    other_cobot = TimeCobot_List.index(max(TimeCobot_List))
    # Find index of next batch based on timePerItem
    next_batch_index = batch_times.iloc[:, next_available_cobot].idxmin()
    next_batch = OrdersTable.orders[next_batch_index]
    next_batch_time = OrdersTable.loc[next_batch_index].iloc[next_available_cobot+1]
    print("next batch", next_batch)
    # Perform a check here! If its faster to send out the other cobot because the packer is still busy, then send the other cobot.
    completion_time1 = max(TimeCobot_List[next_available_cobot] + next_batch_time, TimePacker_List[next_available_cobot])
    completion_time2 = max(TimeCobot_List[other_cobot] + OrdersTable.loc[next_batch_index].iloc[other_cobot+1], TimePacker_List[other_cobot])
    if completion_time1 > completion_time2:
        next_available_cobot = other_cobot

    # Assign batch and calculate what time cobot will arrive at packing station.
    BatchAssignCobot_List[next_available_cobot].append(next_batch)
    TimeCobot_List[next_available_cobot] += next_batch_time

    # Since the unload process cannot start until both the cobot and packer are ready, they are both set to the maximum
    # time between the two of them.
    TimeCobot_List[next_available_cobot] = max(TimeCobot_List[next_available_cobot], TimePacker_List[next_available_cobot])
    TimePacker_List[next_available_cobot] = max(TimeCobot_List[next_available_cobot], TimePacker_List[next_available_cobot])

    # Now that both are ready, we add unpack time to cobot
    TimeCobot_List[next_available_cobot] += 20

    # The packer also adds the unload time, and also packing time for each item
    TimePacker_List[next_available_cobot] += 20 + 60 * len(next_batch)

    # Drop these items from the other feasible batches
    dropping_rows = OrdersTable.orders.apply(lambda x: any(item in next_batch for item in x))
    OrdersTable = OrdersTable[dropping_rows == False]

print(BatchAssignCobot_List)
print(TimeCobot_List)
print(TimePacker_List)


