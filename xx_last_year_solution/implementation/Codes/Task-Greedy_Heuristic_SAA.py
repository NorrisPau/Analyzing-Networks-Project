# The script is divided into three main parts:
# A) importing files and creating instances defined in instance_demo.py (this is the code we were provided with)
# B) functions that contain the algorithms to solve the problems
# C) execution of the given functions

#%% sourcing scripts
import rafs_instance as instance
import instance_demo as demo
import pandas as pd
import copy
import numpy as np
import random as rd
import random as rd
import numpy.random as rn
import sys, os
import pickle
import xml.etree.ElementTree as ET

#%% control knobs

# Specify version of order ("", "_a" or "_b")
orderVersion = ""

# SPecify mean item amount in an order (either 1x6 or 5)
meanItemInOrder = "5"

# Specify the number of orders we receive (either 10 or 20)
orderAmount = 10

# Specify the number of items in the warehouse (either 24 or 360)
itemAmount = 24

# SPecifiy the pod amount
podAmount = 24

# Specify the policy (either "dedicated_1" or "mixed_shevels_1-5")
podPolicy = "dedicated_1"

#%% A) importing files and creating instances defined in instance_demo.py

# not sure what this contains, but it is needed for the Warehouse-class
layoutFile = r'data/layout/1-1-1-2-1.xlayo' 
# loading all the information about the pods
podInfoFile = 'data/sku' + str(itemAmount) + '/pods_infos.txt'   
# loading information about picking locations, packing stations, waypoints,
# pods 
instanceFile = r'data/sku' + str(itemAmount) + '/layout_sku_' + str(itemAmount) + '_2.xml'

# loading information about item storage: contains all SKUs along with their
# attributes
storagePolicyFile = 'data/sku' + str(itemAmount) + '/pods_items_' + str(podPolicy) + '.txt'
#storagePolicies['mixed'] = 'data/sku24/pods_items_mixed_shevels_1-5.txt'

# loading information about the orders: contains list of orders, with number
# of ordered items per SKU-ID
orderFile =r'data/sku' + str(itemAmount) + '/orders_' + str(orderAmount) + '_mean_' + str(meanItemInOrder) + '_sku_' + str(itemAmount) + orderVersion + '.xml'
#orders['20_5']=r'data/sku24/orders_20_mean_5_sku_24.xml'

# trying a different way to get the demonstration running
# function to prepare data
warehouseInstance = instance.Warehouse(layoutFile, instanceFile, podInfoFile, storagePolicyFile, orderFile)

batch_weight = 18
item_id_pod_id_dict = {}

distance_ij = demo.WarehouseDateProcessing(warehouseInstance).CalculateDistance()

#%% Necessary functions

# Functions rely on the script instance_demo.py to be sources, as this script 
# generates the object warehouseInstance that contains all relevant information
# about the project


def F_podInfo(podAmount):
    infoList = []
    for podPosition in range(0, podAmount):
        itemInPod = warehouseInstance.Pods[str(podPosition)].Items
        for item in itemInPod:
            info = {"inPod":podPosition, "Description": item.ID, "itemCount": item.Count}
            infoList.append(info)
    infoList = pd.DataFrame(infoList)
    return(infoList)

def F_itemInfo(itemID):
    """ Return a dictionary with information about the item specified by itemID.
    Function relies on warehouseInstance to be loaded.

    Parameters
    ----------
    itemID : int

    Returns
    -------
    Dictionary contains ID, Type, Weight, Letter, Color and Description.

    """
    item = warehouseInstance.ItemDescriptions[str(itemID)].__dict__
    itemDesc = item["Color"] + "/" + item["Letter"]
    item["Description"] = itemDesc
    return(item)
    
def F_itemInfoList(itemAmount):
    itemInfoList = []
    for itemID in range(0, itemAmount):
        itemInfoList.append(F_itemInfo(itemID))
    itemInfoList = pd.DataFrame(itemInfoList)
    return(itemInfoList)

def F_orderInfo(orderID):
    """Returns a list of dictionaries with information about an order specified
    by orderID. Function relies on warehouseInstance to be loaded.

    Parameters
    ----------
    orderID : int

    Returns
    -------
    List contains one dictionary for each itemID in the specified order, with
    respective weight and count of items for each itemID.
    """
    itemList = []
    for itemID in warehouseInstance.Orders[orderID].Positions.keys():
        weight = F_itemInfo(itemID)["Weight"]
        count = int(warehouseInstance.Orders[orderID].Positions[str(itemID)].Count)
        item = {"itemID": int(itemID), "Weight" : weight, "Count" : count}
        itemList.append(item)
    return(itemList)

def F_itemsInOrder(orderID):
    """For a specified orderID, returns a list of integers. Each integer is an
    item in the order.
    POTENTIALLY A DUPLICATE

    Parameters
    ----------
    orderID : int

    Returns
    -------
    List of integers.

    """
    items = [*warehouseInstance.Orders[orderID].Positions]
    items = [int(i) for i in items]
    return(items)

def F_weightOfOrder(orderID):
    """For a given orderID, calculates the total weight of all items in the
    order.

    Parameters
    ----------
    orderID : int

    Returns
    -------
    Float.

    """
    return(round(sum([dict["Weight"]*dict["Count"] for dict in F_orderInfo(orderID)]), 2))

def F_weightOfOrdersInBatch(orderList):
    """
    For a given list of orders (as orderIDs), calculates the total weight of 
    all items in all orders in the list.

    Parameters
    ----------
    orderList : list of int

    Returns
    -------
    float

    """
    
    #totalWeight = 0
    #for order in orderList:
    #    totalWeight += F_weightOfOrder(order)
    # return(totalWeight)
    return(sum([F_weightOfOrder(orderID) for orderID in orderList]))

def F_createDistMat():
    """For each station in the warehouse, creates one DataFrame in the global
    environment that contains the distance from the station to each pod and 
    all other stations.
    
    Requires
    --------
    distance_ij from the script instance_demo.py must have been loaded.

    Returns
    -------
    None; instead directly writes in the global environment.

    """
    global distToStation0
    global distToStation1
    global distMat

    # What does this do?
    stopAmount = int(np.sqrt(len(distance_ij)))
        # StopAmount is the number of elements in the matrix (number of rows)
    distMat = np.array(list(distance_ij.values())).reshape(stopAmount, stopAmount)
        # reshape distance_ij from a 1d-array to a 2d-matrix
    
    # Convert to pandas DataFrame
    distMat = pd.DataFrame(distMat)
    
    # Add column names
    distMat.columns = ['OutD0', 'OutD1'] + [i for i in range(0,itemAmount)]
    
    # Add an indexing column FromPod
    distMat.insert(0, "fromPod", distMat.columns)
    distMat = distMat.set_index('fromPod')
    
    # Extract distances from pods back to the packing stations. This is the last
    # step in calculating the route.
    distToStation0 = distMat[["OutD0"]]
    distToStation1 = distMat[["OutD1"]]
    distMat = distMat.drop(["OutD0","OutD1"], axis = 1) 

def F_itemsToPodLocation(listOfItems):
    listOfPods = []
    for item in listOfItems:
        listOfPods.append(itemInfoList[itemInfoList.ID == str(item)].inPod.values[0])
    return(listOfPods)

def F_itemsInBatch(batchList): 
    """???

    Parameters
    ----------
    batchList : List of 

    Returns
    -------
    ???

    """
    itemList = []
    for orderID in batchList:
        items = F_itemsInOrder(orderID)
        itemList = list(set(itemList) | set(items))
    return(itemList)

def F_minDist(items, itemInfoList, packingStation):
    """For a given list of items and a packing station, returns the shortest
    possible route of pods and stations, where each item is contained in one 
    of the pods.

    Parameters
    ----------
    items : List of int; itemIDs in a list, usually within one order.
    packingStation : String. Stations names have the form of OutDX, where X is
    an integer representing the station number.

    Returns
    -------
    None.

    """
    # Call for distance matrix
    F_createDistMat()    
    
    # For the list of items, find the corresponding pod locations 
    pods = F_itemsToPodLocation(items)
    
    # Create destructive copy of nodesOfPods, so that each
    # node can be removed later
    currentBatch = pods.copy()
    
    # First node of route is the chosen packing station
    route = [packingStation] 
    
    # Initialize distance
    totalDist = 0

    for i in range(len(items)):
        
        dist = distMat.loc[route[i],currentBatch].to_frame()
        # print("Compare distances:")
        # print(dist)
        nextNode = int(dist.idxmin())
        # print("Next optimal node:")
        # print(nextNode)
        distNode = float(dist.min())
        # print("Weight added")
        # print(distNode)
        route.append(nextNode)   
        # print("Route so far:")
        # print(route)
        totalDist += float(distNode)
        # print("Distance so far:")
        # print(totalDist)
        currentBatch.remove(nextNode)
        # print("Pods left to visit:")
        # print(currentBatch)

    # print("Distance back to packing station:")
    # print(float(distMat.loc[packingStation,route[-1]]))
    totalDist += distMat.loc[packingStation,route[-1]]
    totalDist = round(float(totalDist),2)
    # print("Final dist:")
    # print(totalDist)
    route.append(packingStation)
    return({"route":route, "distance": totalDist})

def F_assignOrderToStation(orderAmount, itemInfoList, type = "full"):
    """ ???
    
    Parameters
    ----------
    orderAmount : int
        Number of orders, has to be specified externally. 
    type : TYPE, optional
        DESCRIPTION. The default is "full".

    Returns
    -------
    None.

    """
    
    station = {"OutD0":[],
               "OutD1":[]}
    
    for orderID in range(orderAmount):
        itemsInOrder = F_itemsInOrder(orderID)
        # print("----Order " + str(count) +"----")
        fromStation0 = F_minDist(itemsInOrder, itemInfoList, "OutD0")
        # print("If start at station 0:")
        # print(fromStation0)
        fromStation1 = F_minDist(itemsInOrder, itemInfoList,  "OutD1")
        # print("If start at station 1:")
        # print(fromStation1)
        if type == "full":
            if fromStation0["distance"] < fromStation1["distance"]:
                station["OutD0"].append({"orderID":orderID, "itemsInOrder" : itemsInOrder, "distance" : fromStation0["distance"]})
                # print("Choose station 0")
            else: 
                station["OutD1"].append({"orderID":orderID, "itemsInOrder" : itemsInOrder, "distance" : fromStation1["distance"]})
                # print("Choose station 1")
        elif type == "lite":
            if fromStation0["distance"] < fromStation1["distance"]:
                station["OutD0"].append(orderID)
            else: 
                station["OutD1"].append(orderID)
    if station["OutD0"] == []:
        station["OutD0"].append(station["OutD1"].pop(0))
    elif station["OutD1"] == []:
        station["OutD1"].append(station["OutD0"].pop(0))
    return(station)

def F_orderToBatch(listOfOrders, itemInfoList, packingStation):
    """ For a given sset of orders, returns a list of all feasible batches that
    can be generated using these orders.

    Parameters
    ----------
    listOfOrders : list
        List of orders, represented as a list of integers.
    packingStation : string
        String that specifies which station the orders belong to.

    Returns
    -------
    A DataFrame with the columns batchID (integer identifying a batch), 
    ordersInBatch (list of integers representing orders that are in the batch),
    routeInBatch (list of integers representing the pods that are visited 
    in the batch), distance (float representing distance travelled in the
    batch) and weight (float, cumulative weight of all items in the batch).
    """
    batch = [] # initialise list of feasible batches as empty list
    batch.append([listOfOrders.pop(0)]) # add the first order to the batch,
        # remove the first order from the list of orders
    
    while (listOfOrders != []):
        # make a copy of the list of batches
        batchCopy = copy.deepcopy(batch) 
        
        # pick the next order from the list orders, remove this order from the 
        # list of orders
        nextOrder = [listOfOrders.pop(0)]
        
        # Add nextOrder to each of the batches in batchCopy to create new batches
        batchCopy = [b + nextOrder for b in batchCopy]
        
        # Check weight capacity constraint of newly created batches
        # batch_weight is created by instance_demo and a global variable
        batchCopy = [b for b in batchCopy if F_weightOfOrdersInBatch(b) <= batch_weight]
        
        # Add nextOrder to batch as a feasible batch
        batch.append(nextOrder)
        
        # Append new feasible batches in batchCopy to batch
        batch = batch + batchCopy
    
    # initialise a new list to store data about the batches
    batchInfo = []    
    batchID = 0
    # for each batch, generate a batchID, extract the orders in the batch, the 
    # route taken to travel the batch, the distance travelled, and the total 
    # weight of all items in the batch
    for b in batch:
        item = F_itemsInBatch(b)
        dist = F_minDist(item, itemInfoList, packingStation)["distance"]
        route = F_minDist(item, itemInfoList, packingStation)["route"]
        del route[0]
        del route[-1]
        batchInfo.append({"batchID": batchID, "ordersInBatch": b, "routeInBatch" : route, "distance": dist, "weight" : F_weightOfOrdersInBatch(b)})
        batchID += 1
    
    batchInfo = pd.DataFrame(batchInfo)
    return(batchInfo)

def F_greedyHeuristic(station, batchFromStation, itemInfoList, packingStation):
    """Greedy heuristic to assign orders to batches, for an already given
    assignment of orders to stations, and for only one station.

    Parameters
    ----------
    batchFromStation : list of dictionaries
        For each station, the list contains one dictionary. Each dictionary
        contains two values: the first is the name of the station, the
        second is a DataFrame of feasible batches as generated by the function
        F_orderToBatch().
        
    packingStation : String
        String that specifies for each station to apply the heuristic.

    Returns
    -------
    DataFrame, containing six columns: batchID (IDs of batches used to cover the
    orders), ordersInBatch (for each batch, which orders it contains),
    routeInBatch (for each batch, which pods it travels to), distance (for each
    batch, the distance necessary to fulfil the batch), weight (weight of all
    items in that batch) and numberOfBatchCovered (number of orders covered by
    the batch)

    """
    # Initialize empty lists
    greedyBatch = []
    batchFromStationCopy = copy.deepcopy(batchFromStation)
    
    # Extract orders from the station: orderToCover is a list of all orders
    # of a station, that is, the set of all orders that need to be covered;
    # station is a dictionary that has one key for each station and as value
    # a list of all orderIDs that are assigned to the station.
    orderToCover = station[packingStation]
    
    # Extract the feasible batch information for the respective station from 
    # batchFromStation object
    batchInfo = [batch['batchInfo'] for batch in batchFromStationCopy if batch['station'] == packingStation]
    # This is somehow necessary because to this point, batchInfo is a list 
    # that contains only one element, which is the dataframe we need
    batchInfo = batchInfo[0]
    
    while orderToCover != []:
        print("-----Next Batch-----")
        # add a new column called numberOfBatchCovered (will be filled later        )
        batchInfo["numberOfBatchCovered"] = 0
        print("Order to cover: " + str(orderToCover))
        
        # For each batch, calculate the number of orders a feasible batch can
        # cover. This number is equal to the orders that are in both a batch
        # and the set of orders that still need to be covered.
        # For each row in the DataFrame that contains the data about feasible
        # batches, count the number of orders that are in both this feasible
        # batch and in the cover
        value = []
        for i in list(batchInfo.index):
            value = value + [len(set(batchInfo.loc[i,"ordersInBatch"]) & set(orderToCover))]
        batchInfo["numberOfBatchCovered"] = value
        
        # # Preliminary criterion 1 - batch has to cover at least 1 order
        # nextBatch = batchInfo.query("numberOfBatchCovered > 0")
        # print("Preliminary criteria - batch has to cover at least 1 order:")
        # print(nextBatch)
        
        # Preliminary criterion 2 - batch is a subset of orderToCover (to avoid
        # a batch being covered more than once)
        nextBatch = copy.deepcopy(batchInfo)
        for i in nextBatch.index:
            canCover = nextBatch.loc[i].ordersInBatch
            if all([compare in orderToCover for compare in canCover]) == False:
                nextBatch = nextBatch.drop(i)
        print("Prelim: Batch in a subset of orderToCover")
        print(nextBatch)    
        
        # Greedy criteria 1 - subset of batches with maximum number of orders covered
        nextBatch = nextBatch.query("numberOfBatchCovered == numberOfBatchCovered.max()")
        print("Max cover criteria:")
        print(nextBatch)
        
        # Greedy criteria 2 - subset of batches with minimum distance travelled
        nextBatch = nextBatch.query("distance == distance.min()")
        print("Min dist criteria (greedy):")
        print(nextBatch)
        
        # From that subset, randomly pick a batch and append it to the final result
        random = rd.randint(0,len(nextBatch)-1)
        nextBatch = dict(nextBatch.iloc[random,:])
        print("Choose random:")
        print(nextBatch)
        greedyBatch.append(nextBatch)
        
        # Delete already covered orders out of orderToCover list
        orderToCover = list(set(orderToCover) - set(nextBatch["ordersInBatch"]))
        
        # Delete the already chosen batch out of the batch list
        batchToDelete = nextBatch["batchID"]
        batchInfo = batchInfo[batchInfo["batchID"] != batchToDelete]
        
    greedyBatch = pd.DataFrame(greedyBatch)
    #greedyBatch.drop(["numberOfBatchCovered"],axis=1)
    return(greedyBatch)

def F_feasibleBatch(station):
    batchFromStation = []
    '''
    the function get all the feasible batches for a station. 
    '''
    
    for i in range(len(station)):
        stationCopy = copy.deepcopy(station)
            # calls the name of the packing station
        packingStation = list(stationCopy.keys())[i]
            # calls the orders that belong to that station
        listOfOrders = list(stationCopy.values())[i] 
            # assign orders from list of orders to batches, procuding all feasible
            # batches for each station
        batchFromStation.append({"station":packingStation, "batchInfo":F_orderToBatch(listOfOrders, itemInfoList, packingStation)})
    del stationCopy
    return(batchFromStation)

def SwapStation(station, n):
    '''
    This function is used in simulate annealing to randomly reassign orders to
    different stations. 

    Parameters
    ----------
    station : the assignment of orders to each station.
    n : number of orders to be swapped.
    -------
    '''
    
    stationCo=copy.deepcopy(station)
    
    print("# of orders to change station is "+str(n))
    
    if len(stationCo['OutD0'])>1 and len(stationCo['OutD0'])>1: 
        #print("Situation 1")
        ordToMoved=rd.sample(range(10),n)
        
    
    elif  len(stationCo['OutD0'])<2: 
        #print("Situation 2")
        ordToMoved=rd.sample(stationCo['OutD1'],n)
        
    else: 
        #print("Situation 3")
        ordToMoved=rd.sample(stationCo['OutD0'],n)
        
    
    print("The "+str(n)+" order(s) to be moved: "+str(ordToMoved))
    for i in range(len(ordToMoved)):
        for key, value in stationCo.items(): 
            if ordToMoved[i] not in value: 
                value.append(ordToMoved[i]) 
            else:
                value.remove(ordToMoved[i])

    print(stationCo)   
    
    return([stationCo, ordToMoved])    

def F_neighbour(station, n): 
    '''
    This is used in simulated annealing algorithm(SAA) to create the neighbor o

    Parameters
    ----------
    station : the assignment of orders to each station.
    n : number of orders to be swapped.
    -------

    '''
    newStation, orderMoved = SwapStation(station, n)
    newFB = F_feasibleBatch(newStation)
    sol0 = F_greedyHeuristic(newStation, newFB, itemInfoList, packingStation = "OutD0")
    sol1 = F_greedyHeuristic(newStation, newFB, itemInfoList, packingStation = "OutD1")
    sol0 = sol0.drop(["numberOfBatchCovered"],axis=1)
    sol1 = sol1.drop(["numberOfBatchCovered"],axis=1)
    print(sol0)
    print(sol1)
    return([sol0, sol1, newStation, orderMoved])

def accept_prob(cost, new_cost, T):
    '''
    This is a small function used in simulated annealing algorithm(SAA) to 
    caluculate the probability to accept a solution.  

    Parameters
    ----------
    cost : original distance
    new_cost : new distance
    T : temperature in SAA.

    Returns
    -------
    p : the probability to accept a solution

    '''
    
    print("oriDis is: "+str(cost))
    #print("newDis is: "+str(new_cost))
    
    if new_cost < cost: return 1
    
    else: 
        p = np.exp(-(new_cost - cost)/T)
        return p

def SAA(oriSol, oriDis, station, T, alpha, tempLimit, n):
    '''
    This is the simulated annealing algorithm, given an original solution, 
    it will keep finding solutions in neighbourhood until the minimum 
    temperature is reached. The distances of every solution found are recorded 
    in DisRec. The optimal results is recorded in optDisRec. 

    Parameters
    ----------
    oriSol : original solution
    oriDis : original distance
    station : the assignment of orders to each station.
    T : temperature
    alpha
    tempLimit : minimum temperature
    n : number of orders to be swapped.
    -------
    '''
    
    print("  ")
    T=T
    alpha = alpha
    epsilon = tempLimit
    print("The original solution: ")
    print(oriSol)
    optSol = []
    optSol = copy.deepcopy(oriSol)
    #optSol = optSol.append(oriSol) 
    print("optSol = oriSol")
    #optDis= sum(optSol['distance'])
    optDis = sum(optSol[0]['distance'])+sum(optSol[1]['distance'])
    print("optDis is the oriDis: "+str(optDis))

    DisRec=[]
    DisRec.append(optDis)
    print("DisRec: "+str(DisRec))
    
    optDisRec=[]
    optDisRec.append(optDis)
    print("optDisRec: "+str(optDisRec))
    
    #if T==1000:    
    round=0
    orderMovedList = [] 
    
    while T > epsilon:  
      round=round+1
      print("   ")
      print(" ************ Round "+str(round)+": T = "+ str(T) +" ***************")
      #oriSol=oriSol.reset_index(drop=True)
      
      print("The original solution: ")
      print(oriSol)
      print("oriDis is: "+str(oriDis))
      print("    ")
      
      print("The optimal solution: ")
      print(optSol)
      print("optDis is: "+str(optDis))
      print("    ")
      
      newSol = F_neighbour(station, n)
      newDis = sum(newSol[0]['distance'])+sum(newSol[1]['distance'])
      print("newDis is: "+str(newDis))
      
      orderMoved = newSol[3]
      orderMovedList.append(orderMoved)
      
      #Accept the neibour if it has smaller distance / if p > random number
      p = accept_prob(oriDis, newDis, T)
      #print("FFF")
      uRan = rn.random()
      
      
      if p > uRan:
          if p==1:
              print("newDis is better / not worse than oriDis --> accept newSol")
          else: 
              print("   ")
              print("p= "+str(p))
              print("uRan= "+str(uRan))
              print("P > random U(0,1) --> accept newSol")
          oriSol = copy.deepcopy(newSol)
          oriDis = sum(oriSol[0]['distance'])+sum(oriSol[1]['distance'])
          station = copy.deepcopy(oriSol[2])
          
      #  reject 
      else: 
          print("p= "+str(p))
          print("uRan= "+str(uRan))
          print("Reject new solution")
      
      if newDis < optDis:
          optSol = copy.deepcopy(newSol)
          optDis = sum(optSol[0]['distance'])+sum(optSol[1]['distance'])
          optDisRec.append(optDis)
      
      DisRec.append(newDis)
      print("   ")
      print("DisRec: "+str(DisRec))  
      print("optDisRec: "+str(optDisRec))
      #print("Optimal distance at round "+str(round)+" is "+str(optDisRec[-1]))
      print("   ")
      print("orderMoverRec: ")
      print(orderMovedList)
      
      print("************* End of round " +str(round)+ ": T = " + str(T) + " ***********************")
      
      T = alpha*T
    
    return([oriSol, optSol, round, DisRec, optDisRec])

def F_convertInputForXMLfromSAA(finalRes):
    result = {"OutD0": finalRes[1][0].to_dict("records"), "OutD1": finalRes[1][1].to_dict("records")}
    
    for station in result.keys():
        for batch in result[station]:
            batch["itemInBatch"] = F_itemsInBatch(batch["ordersInBatch"])
            batch["routeInBatch"] = [station] + batch["routeInBatch"] + [station]
    return(result)

def F_convertInputForXMLfromGreedy(oriSol):
    result = {"OutD0": oriSol[0].to_dict("records"), "OutD1": oriSol[1].to_dict("records")}
    
    for station in result.keys():
        for batch in result[station]:
            batch["itemInBatch"] = F_itemsInBatch(batch["ordersInBatch"])
            batch["routeInBatch"] = [station] + batch["routeInBatch"] + [station]
    return(result)
   
def writeSolutionXML_SAA(result, itemInfoList, filename):
  
    # base element
    root = ET.Element("root")
    # first section "split" contains information about which orders are in which batches, and which batches are assigned to which station (=bot)
    collecting = ET.SubElement(root, "Collecting")
    split = ET.SubElement(collecting, 'Split')
    packingStationNames = result.keys()
    # write each station as a sub-node of split
    for station in packingStationNames:
        Bot_ID = ET.SubElement(split, "Bot")
        Bot_ID.set("ID", station)
        # filter the solution so it only contains batches for the right station
        stationSolution = result[station]
        # write each batch as a sub-node of Bot_ID
        batchID = 1
        for batch in stationSolution:
            Batch_ID = ET.SubElement(Bot_ID, "Batch")
            Batch_ID.set("ID", str(batchID))
            batchID += 1
            # write Orders as the sub-node of Batch_ID
            Orders = ET.SubElement(Batch_ID, "Orders")
            # write each order as sub-node of Batch_ID
            for order in batch["ordersInBatch"]:
                Order = ET.SubElement(Orders, "Order")
                Order.text = str(order)
                
    # first section "bots" contains detailed information about each bot (station)
    bots = ET.SubElement(collecting, "Bots")
    # write each station as a sub-node of bots
    for station in packingStationNames:
        Bot_ID = ET.SubElement(bots, "Bot")
        Bot_ID.set("ID", station)
        stationSolution = result[station]
        # batches are written in sub-node Batches of Bot_ID
        Batches = ET.SubElement(Bot_ID, "Batches")
        # write each batch as a sub-node of Bot_ID
        batchID = 1
        for batch in stationSolution:
            Batch_ID = ET.SubElement(Batches, "Batch")
            Batch_ID.set("BatchNumber", str(batchID))
            Batch_ID.set("Distance", str(batch["distance"]))
            Batch_ID.set("Weight", str(batch["weight"]))
            batchID += 1
            # for each batch, write two sub-nodes: itemsData, edges
            # first write ItemsData
            ItemsData = ET.SubElement(Batch_ID, "ItemsData")
            # ItemsData has a sub-node called Orders
            Orders = ET.SubElement(ItemsData, "Orders")
            # write each order as sub-node of Orders:
            for order in batch["ordersInBatch"]:
                Order = ET.SubElement(Orders, "Order")
                Order.set("ID", str(order))
                
                # write each item in the order as sub-node of Order
                itemList = F_itemsInOrder(int(order))
                for item in itemList:
                    Item = ET.SubElement(Order, "Item")
                    # for each item, conclude information about the itemID and the description
                    Item.set("ID", str(item))
                    Item.set("Type", itemInfoList.loc[item, "Description"])
         
            # write Edges as sub-node of Batch_ID
            Edges = ET.SubElement(Batch_ID, "Edges")       
            # write every edge of the batch
            edgeIndex = list(range(0, len(batch["routeInBatch"]) - 1))
            for edge in edgeIndex:
                Edge = ET.SubElement(Edges, "Edge")
                Edge.set("StartNode", str(batch["routeInBatch"][edge]))
                Edge.set("EndNode", str(batch["routeInBatch"][edge + 1]))
    

    tree = ET.ElementTree(root)
    tree.write(filename)


# Task 1 - Greedy Heurisitc

# Full information on items and their location in the pods
podInfoList = F_podInfo(podAmount)
itemInfoList = F_itemInfoList(itemAmount)
itemInfoList = pd.merge(podInfoList, itemInfoList, how = "left", on = "Description")

# From all orders, assign them to the optimal packing station:
# I use option "lite" here to get two list of orders for two packing station, as
# input for the next function. To see information of the orders, use option "full"
station = F_assignOrderToStation(orderAmount, itemInfoList, "lite")
stationFull = F_assignOrderToStation(orderAmount, itemInfoList, "full")

## Theorical scenario: 
# station = {"OutD0" : [4,5,6,7], "OutD1" : [0,1,2,3,8,9]}   

batchFromStation = F_feasibleBatch(station)

greedyStation0 = F_greedyHeuristic(station, batchFromStation, itemInfoList, packingStation = "OutD0")
greedyStation1 = F_greedyHeuristic(station, batchFromStation, itemInfoList, packingStation = "OutD1")

greedyStation0 = greedyStation0.drop(["numberOfBatchCovered"],axis=1)
greedyStation1 = greedyStation1.drop(["numberOfBatchCovered"],axis=1)

# original solution and its fitting value, the total distance 
oriSol = [greedyStation0, greedyStation1]
oriDis = sum(greedyStation0['distance'])+sum(greedyStation1['distance'])

#%%
# Prepare results as input to write XML
inputXMLGreedy = F_convertInputForXMLfromGreedy(oriSol)

# Write XML
outputName = 'orders_' + str(orderAmount) + '_mean_' + str(meanItemInOrder) + '_sku_' + str(itemAmount) + str(orderVersion) + '_solution_dedicated_greedy.xml'

writeSolutionXML_SAA(inputXMLGreedy, itemInfoList, outputName)
#%%
# Task 2: SAA

# To try first, run the follow for 3 round: 
#finalRes = SAA(oriSol, oriDis, station, T=10, alpha=0.8, tempLimit=5, n=1)

finalRes = SAA(oriSol, oriDis, station, T= 5000, alpha=0.95, tempLimit=0.01, n=1)

print("After round "+str(finalRes[2])+" ,the optimal distance is "+str(finalRes[4][-1]))
print("Optimal Solution is ")
print(finalRes[1][0])
print(finalRes[1][1])

# Prepare results as input to write XML
inputXMLSAA = F_convertInputForXMLfromSAA(finalRes)

# Write XML
outputName = 'orders_' + str(orderAmount) + '_mean_' + str(meanItemInOrder) + '_sku_' + str(itemAmount) + str(orderVersion) + '_solution_dedicated_SAA.xml'
writeSolutionXML_SAA(inputXMLSAA, itemInfoList, outputName)

#%% Task 2: Perturbation

#n = rd.randint(2,orderAmount-1) 
perturRes = SAA(oriSol, oriDis, station, T=10000, alpha=0.8, tempLimit=0.01, n=2)













