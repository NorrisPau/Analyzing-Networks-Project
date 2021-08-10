# The script is divided into three main parts:
# A) importing files and creating instances defined in instance_demo.py (this is the code we were provided with)
# B) functions that contain the algorithms to solve the problems
# C) execution of the given functions

#%% sourcing scripts
import rafs_instance as instance
import instance_demo as demo
import pandas as pd
import copy
# import numpy as np
import random as rd
import os
from os import path
import json
import networkx as nx
import  xml.etree.ElementTree as ET

#%% A) defining the class 
class WarehouseDateProcessing():
    def __init__(self, warehouseInstance, batch_size = None):
        self.Warehouse = warehouseInstance
        self._InitSets(warehouseInstance, batch_size)   

    def preprocessingFilterPods(self, warehouseInstance):
        resize_pods = {}
        print("preprocessingFilterPods")
        item_id_list=[]
        for order in warehouseInstance.Orders:
            for pos in order.Positions.values():
                item = warehouseInstance.ItemDescriptions[pos.ItemDescID].Color.lower() + '/' + warehouseInstance.ItemDescriptions[pos.ItemDescID].Letter
                #item_id = pos.ItemDescID
                if item not in item_id_list:
                    item_id_list.append(item)
                    #print(item_id)
 

        #for item in item_id_list:
        #    print(item)

        # for dedicated
        for pod in warehouseInstance.Pods.values():
            for item in pod.Items:
                #print("item in pod.Items:", item.ID)
                if item.ID in item_id_list:
                    print("item.ID in item_id_list:", item.ID)
                    resize_pods[pod.ID] = pod
        
        print(resize_pods)
        return resize_pods

    # Initialize sets and parameters           
    def _InitSets(self,warehouseInstance, batch_size):
        #V Set of nodes, including shelves V^S and stations (depots)
        # V^D (V=V^S U V^D)
        #Add output and input depots
        
        self.V__D__C = warehouseInstance.OutputStations
        ##self.V__D__F = warehouseInstance.InputStations
        self.V__D__F = {}
        #depot = ('D999', )
        #Old self.V__D = {'D999':depot}
             
        self.V__D = {**self.V__D__C, **self.V__D__F}
        
        #hli
        #self.V__S = warehouseInstance.Pods
        self.V__S = self.preprocessingFilterPods(warehouseInstance)
        
        #Merge dictionaries
        self.V = {**self.V__D, **self.V__S}

    def CalculateDistance(self):

        file_path = r'data/distances/' + os.path.splitext(os.path.basename(self.Warehouse.InstanceFile))[0] + '.json'
        
        if not path.exists(file_path):
            #Create d_ij
            #d_ij = tupledict()
            d_ij = {}
            
            #Loop over all nodes
            for key_i, node_i in self.V.items():
                for key_j, node_j in self.V.items():  
                    
                    source = 'w'+node_i.GetPickWaypoint().ID
                    target = 'w'+node_j.GetPickWaypoint().ID
                    
                    #Calc distance with weighted shortest path
                    d_ij[(key_i,key_j)] = nx.shortest_path_length(self.Graph, source=source, target=target, weight='weight')
            
            #Parse and save
            d_ij_dict = {}
            for key,value in d_ij.items():
                i,j = key  
                if i not in d_ij_dict:
                    d_ij_dict[i]={} 
                d_ij_dict[i][j] = value
            
            with open(file_path, 'w') as fp:
                json.dump(d_ij_dict, fp)
        
        else: 
            #Load and deparse
            with open(file_path, 'r') as fp:
                d_ij_dict = json.load(fp)
            print('d_ij file %s loaded'%(file_path)) 
                
            #d_ij = tupledict()
            d_ij = {}
            for i, values in d_ij_dict.items():
                for j, dist in values.items():
                    d_ij[i,j] = dist                
        return d_ij

#%% 
class Demo():
    def __init__(self, splitOrders = False):
        
        self.batch_weight = 18
        self.item_id_pod_id_dict = {}
        #[0]
        self.warehouseInstance = self.prepareData()
        self.distance_ij = self.initData()




    def initData(self):
        print("[1] changing data format for the algorithm we used here: ")
        warehouse_data_processing = WarehouseDateProcessing(self.warehouseInstance)
        #Distance d_ij between two nodes i,j \in V
        d_ij = warehouse_data_processing.CalculateDistance()
        return d_ij

#%% getting data and putting it into a good format
def prepareData(warehouseInstance):
    # getting data about which items are in which pods, building a dictionary that is easier to read
    podInfoDict = {int(pod):[{"Description":item.ID, "Count":int(item.Count)} for item in warehouseInstance.Pods[pod].Items] for pod in warehouseInstance.Pods}
    
    # getting data about the items
    itemsInfoDict = [{"ID":int(warehouseInstance.ItemDescriptions[SKU].ID), "Description":warehouseInstance.ItemDescriptions[SKU].Color + "/" + warehouseInstance.ItemDescriptions[SKU].Letter, "Weight":warehouseInstance.ItemDescriptions[SKU].Weight} for SKU in warehouseInstance.ItemDescriptions]
    
    # getting data about orders
    orderInfoDict = {order:{"items":[{"itemID":int(position), "quantity":int(warehouseInstance.Orders[order].Positions[position].Count)} for position in warehouseInstance.Orders[order].Positions]} for order in range(0, len(warehouseInstance.Orders))}
    
    # adding information about total order weight
    for key, value in orderInfoDict.items():
        orderWeight = 0
        for item in value["items"]:
            orderWeight += itemsInfoDict[item["itemID"]]["Weight"] * item["quantity"]
        value["total weight"] = orderWeight
    
    # getting data about packing stations
    packingStationNames = list(warehouseInstance.OutputStations.keys())
    
    # replacing item description in podInfoDict with item ID
    # getting an item ID for an item description
    for pod in podInfoDict.keys():
        for podItem in podInfoDict[pod]:
            podItem["ID"] = [item["ID"] for item in itemsInfoDict if item["Description"] == podItem["Description"]][0]
    
    # removing all pods that are empty (needed for 360 SKUs):
    podInfoDict = {podID:SKUs for podID, SKUs in podInfoDict.items() if len(SKUs) >0}
    # converting data to DataFrames, where it helps
    for pod in podInfoDict:
        podInfoDict[pod] = pd.DataFrame(podInfoDict[pod]).set_index("ID")
    for orderID, orderInfo in orderInfoDict.items():
        orderInfo["items"] = pd.DataFrame(orderInfo["items"]).set_index("itemID")
        
    itemsInfoDict = pd.DataFrame(itemsInfoDict).set_index("ID")
    
    return([podInfoDict, itemsInfoDict, orderInfoDict, packingStationNames])

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# to inspect the results of this part, use printouts of the objects
# packingStationNames | for all station names in the warehouse
# itemsInfoDict       | for a DataFrame that contains information about each item
# podInfoDict         | a dictionary whose keys are podIDs, and whose values 
#                       are DataFrames who store itemID, description and count
#                       per item stored in a respective pod
# orderInfoDict       | a dictionary whose keys are orderIDs, and whose values
#                       are dictionaries again, which include the items in that
#                       order (key "items", contains a DataFrame with one row
#                       per item, index is itemID, only column is quantity of
#                       itemID ordered) and the total weight of all items in 
#                       the order (key "total weight", contains a float)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#%%
# greedy heuristic to determine the route with which to fulfil an order, and its fitness value

def F_fitnessValueOrderToStation(order, station, podCopy, pr = False):
    """For a given station and order, returns route the order will take.
    The distance is calculated greedily: for all items in the order, the algorithm looks for the nearest pod that contains any of the items and chooses to visit it. From this pod, it tries to pick as many items as possible.

    Parameters
    ----------
    order : 
        A DataFrame of itemIDs (indexcol) and their respective quantity.
    station : TYPE
        one of packingStationNames.
    podInfo : TYPE
        typically a deep copy of podInfoDict.

    Returns
    -------
    Dictionary with key-value pairs:"distances travelled", "route taken", "route information"

    """
    # get an empty path
    route = [station]
    routeInfo = {}
    # set the distance of the path to 0
    distance = {}
    # iterate over the ordesr and go to the next pod as long as there are items
    # left in the order
    while order.shape[0] != 0:
        if pr: print("-----------------------------------")
        # find the nearest pod that contains at least one item of the SKU
        possiblePodsDistances = []
        # for each SKU, find the nearest pod that contains at least one item of that SKU
        if pr: print("Remaining SKUs in the order are:"), print(list(order.index))
        for SKU in order.index:
            # find all pods that contain at least one item of the SKU and get the minimum
            # distance to any of them
            if pr: print("-- Checking SKU " + str(SKU))
            SKUNearestPod = {}
            for podID, items in podCopy.items():
                if SKU in items.index:
                    if pr: print("---- podID: " + str(podID)), print("---- items in this pod:"), print(items), print("---- distance to this pod: " + str(round(distance_ij[(str(route[-1]), str(podID))], 1)))
                    SKUNearestPod[podID] = round(distance_ij[(str(route[-1]), str(podID))], 1)
            if pr: print("-- Nearest pod for SKU " + str(SKU) + " is " + str(min(SKUNearestPod)))
            if pr: print("-- Distance to the nearest pod for this SKU: " + str(SKUNearestPod[min(SKUNearestPod)]))
            possiblePodsDistances.append({"itemID":SKU,
                                          "pod":min(SKUNearestPod),
                                          "distance":SKUNearestPod[min(SKUNearestPod)]})
            #possiblePodsDistances[SKU] = minDist
        # for all SKUs, find the one whose nearest pod is nearest to the current position
        possiblePodsDistances = pd.DataFrame(possiblePodsDistances)
        if pr: print("Distances to possible pods are:"), print(possiblePodsDistances)
        nextSKU = possiblePodsDistances[possiblePodsDistances.distance == possiblePodsDistances.distance.min()]
        nextSKU = nextSKU.sample()
        # add the distance to the respective pod to the total path distance
        distance[(str(route[-1]), str(int(nextSKU.pod)))] = round(float(nextSKU.distance), 1)
        # remove all items from the order that are in the pod and the order
        if pr: print("Next pod to be visited is: " + str(int(nextSKU.pod)))
        # for each SKU in the order, check whether items of it are in the next pod
        # first iterate over all the items in the order
        picked = []
        if pr: print("Picking items from pod " + str(int(nextSKU.pod)))
        if pr: print(podCopy[int(nextSKU.pod)])
        for index, row in order.iterrows():
            if pr: print("-- Trying to pick SKU " + str(index))
            # next, check if each item is in the pod
            if index in podCopy[int(nextSKU.pod)].index:
                if pr: print("---- SKU " + str(index) + " is in pod " + str(int(nextSKU.pod)))
                # if it is in the pod, determine whether all items from the order can be
                # picked from the pod, or whether there are more items in the pod than
                # there are in the order
                itemsToPick = min(row.quantity, podCopy[int(nextSKU.pod)].loc[index].Count)
                if pr: print("---- "+ str(itemsToPick) + " items of SKU " + str(index) + " can be picked.")
                # Removing items from the order:
                if pr: print("------ Removing " + str(str(itemsToPick)) + " items of SKU " + str(index) + " from the order.")
                order.loc[index, "quantity"] -= itemsToPick
                # Removing the items from the shelves:
                if pr: print("------ Removing " + str(str(itemsToPick)) + " items of SKU " + str(index) + " from pod "+ str(int(nextSKU.pod)))
                podCopy[int(nextSKU.pod)].loc[index, "Count"] -= itemsToPick
                ### print("Removed " +  str(itemsToPick) + " item(s) of SKU " + str(index) + " from the order and from pod " + str(int(nextSKU.pod)) + ".")
                picked.append({"itemID":index, "count":itemsToPick})
        # add the pod to the route
        route.append(int(nextSKU.pod))
        routeInfo[int(nextSKU.pod)] = picked
        # trying to delete all rows from the order and all pods where quantity of a SKU is 0
        order = order.drop(list(order[order.quantity == 0].index))
        podCopy = {podID:podItems.drop(list(podItems[podItems.Count ==0].index)) for podID, podItems in podCopy.items()}
    # add the last path of the route, from the last pod back to the station
    distance[(str(route[-1]), station)] = round(distance_ij[(str(route[-1]), station)], 1)
    route.append(station)
    return({"distances travelled":distance, "route taken":route, "route information":routeInfo})

#%% perform the check of the fitness value for each order and station to decide
# which station the order should go to
def F_orderToStation(orderInfoDict, podInfoDict, pr = False):
    """Function to assign each order to one station.

    Parameters
    ----------
    orderInfoDict : 
        Typically a copy of orderInfoDict.
    podInfoDict : TYPE
        Typically a copy of podInfoDict.

    Returns a list of two objects.
    ------------------------------
        
    orderToStation: dictionary, one key per orderID, value is a string representing the station to which the order is assigned
    
    allOrderInformation:  dictionary, one key per orderID. Value is a dictionary, with four key value pairs: "distances travelled" contains a dictionary, with a tuple as keys that represents the pods / stations visited, and the distance between them. "route taken" is a list of station names / pods, which shows the route an order has taken. "route information" is itslef a dictionary, with one key per pod visited. Values are dictionaries that are a little too complicated.
    
    """
    # simple dict, keys are orderID, value is string representing station to which
    # the order is assigned
    orderToStation = {}
    allOrderInformation = {}
    while len(orderInfoDict) > 0:
        # pick a random key from the orders
        orderID = rd.choice(list(orderInfoDict.keys()))
        if pr: print("order ID selected: " + str(orderID))
        # remove the corresponding key-value pair from the dictionary and save the order
        order = orderInfoDict.pop(orderID)["items"]
        if pr: print("Order:")
        if pr: print(order)
        # this dictionary will contain information about the route of each order
        # for all stations, in order to compare any order for all given stations.
        orderToStationInformation = {}
        # get routing information for the previously selected order for all stations
        for station in packingStationNames:
            # F_fitnessValueOrderToStation returns a dictionary with three keys:
            # "distances travelled" is a dictionary, whose keys are tuples
            # consisting of pods (or stations) visited, and whose values are the 
            # distances travelled for each tuple
            # "route taken" is a list that contains the stations and pods visited,
            # in order of visitation.
            # "route information" is a dictionary, with one key per pod visited.
            # Each value is a list of dictionaries, and each dictionary contains the
            # itemID of an item picked at a respective pod, and the count of that
            # item picked at that pod.
            orderToStationOutput = F_fitnessValueOrderToStation(order, station, podInfoDict, pr = pr)
            if pr: print(orderToStationOutput)
            # storing information about routing of an order for each station
            orderToStationInformation[station] = orderToStationOutput
            # directly calculating the total distance of an order for a given station
            orderToStationInformation[station]["total distance"] = \
                sum(list(orderToStationOutput["distances travelled"].values()))
        # pick one of the stations to assign the order to: select only the total
        # distance to each station
        distanceToStations = \
            {station:orderToStationInformation[station]["total distance"] for \
             station in orderToStationInformation}
        if pr: print("Distance to stations: "), print(distanceToStations)
        # pick the station that minimises the distance travelled, and add it to the
        # list of chosen stations
        chosenStation = min(distanceToStations, key = distanceToStations.get)
        orderToStation[orderID] = chosenStation
        if pr: print("Assigning order " + str(orderID) + " to station " + str(chosenStation))
        # get information about the route of the order for the chosen station
        chosenStationInfo = {orderID:orderToStationInformation[chosenStation]}
        # add the information about the route for the chosen station to the dict
        # that contains this information for all orders
        allOrderInformation[orderID] = orderToStationInformation[chosenStation]
        # update the warehouse: items taken by an order for a given station are
        # then unavailable for all following orders.
        for pod, podRouteInfo in chosenStationInfo[orderID]["route information"].items():
            for podVisited in podRouteInfo:
                podInfoDict[pod].loc[podVisited["itemID"], "Count"] -= podVisited["count"]
    return([orderToStation, allOrderInformation]) 


#%% after having assigned each order to a station, assign each order within 
# each station to a batch
# pick a station

def getNextOrderForBatch(feasibleNextOrders):
    # choose the order in the orderBatchAssignmentInfo  that covers the most items
    maxItems = max([value["countItems"] for key, value in feasibleNextOrders.items()])
    maxItems = [key for key, value in feasibleNextOrders.items() if value["countItems"] == maxItems]
    maxItems = {orderID:feasibleNextOrders[orderID] for orderID in maxItems}
    # if there are multiple orders that cover the max amount of items, choose 
    # the order with the heighest weight
    maxWeight = max([value["weight"] for key, value in maxItems.items()])
    maxWeight = [key for key, value in maxItems.items() if value["weight"] == maxWeight]
    # if there are multiple orders that match both conditions, draw one at random
    chooseOrder = rd.sample(maxWeight, 1)[0]
    # initialise the batch
    return(chooseOrder)

def assignOrderToBatch(station, orderAssignment):
    """

    Parameters
    ----------
    station : String.
        Name of the station on which the assignment of orders to batches is happening.
    orderAssignment : Object.
        Output of the function F_orderToStation. Contains a dictionary that assigns a station to each order, and contains all information about the route of each order.

    Returns
    -------
    None.

    """
    # prepare order information so that it contains information we need
    orderBatchAssignmentInfo = \
        {order:{"route":orderAssignment[1][order]["route taken"], \
                "weight":orderInfoDict[order]["total weight"], \
                    "countItems":sum(orderInfoDict[order]["items"].quantity)} for \
         order in orderAssignment[1].keys() if orderAssignment[0][order] == station}
    batchList = []
    while orderBatchAssignmentInfo!= {}:
        chooseOrder = getNextOrderForBatch(orderBatchAssignmentInfo)
        # initialise the batch
        batchOrders  = [chooseOrder]
        batchWeight = orderBatchAssignmentInfo[chooseOrder]["weight"]
        batchPods = set(orderBatchAssignmentInfo[chooseOrder]["route"][1:-1])
        # remove the order already picked from the list of orders
        orderBatchAssignmentInfo.pop(chooseOrder)
        feasibleNextOrders = copy.deepcopy(orderBatchAssignmentInfo)
        # from the remaining orders, pick as long as possible, with the same criteria
        while feasibleNextOrders != {}:
            # pick the next order
            order = getNextOrderForBatch(feasibleNextOrders)
            # check whether this order can be added to the batch
            if (feasibleNextOrders[order]["weight"] + batchWeight <= batch_weight):
                # when it can be added, update the batch
                batchOrders.append(order)
                batchWeight += orderBatchAssignmentInfo[order]["weight"]
                batchPods = batchPods.union(set(orderBatchAssignmentInfo[order]["route"][1:-2]))
                # if the order has been added to the batch, it has to be deleted from
                # the list of orders available for all batches
                orderBatchAssignmentInfo.pop(order)
            # remove the order from the list of possible next orders
            feasibleNextOrders.pop(order)
        # add the created batch to the batch list
        batchList.append({"batchOrders":batchOrders, "batchWeight":batchWeight, "batchPods":batchPods})
    return(batchList)


#%% inside each station, assign orders to batches
def createSolutionFromOrderAssignment(orderAssignment, packingStationNames):
    """
    For a given assignment of orders to station, returns a solution that contains different batches (list of dictionaries - see "Returns"). Does not alter the object "orderAssignment".

    Parameters
    ----------
    orderAssignment : Object
        Output of F_orderToStation().
    packingStationNames : List
        List of Strings, each representing one station in the warehouse.

    Returns
    -------
    List of dictionaries. Key value-pairs in each dictionary are:
    - batchOrders: list of orders (orderIDs) the batch contains
    - batchWeight: float, sum of weights of items of all orders in the batch
    - batchPods: set of integers, each a podID of a pod that is visited by the batch
    - station: station to which the batch is assigned
    - distances: dictionary, whose keys are tuples representing two consecutive elements of the route, and whose values are floats, representing the distance travelled between these two elements of the route.
    """
    solution = []
    for station in packingStationNames:
        orderToBatchAssignment = assignOrderToBatch(station, orderAssignment)
        for batch in orderToBatchAssignment:
            batch["station"] = station
        solution += orderToBatchAssignment
    # for each batch in the solution, calculate the distance and the route
    for batch in solution:
        batchRoute = podRoute(list(batch["batchPods"]), batch["station"])
        batch["batchRoute"] = batchRoute[0]
        batch["distances"] = batchRoute[1]
    return(solution)

#%% get the routes for the found batches
def podRoute(pods, station):
    """Return a route in form of a list, given some pods to visit (in form of a list).

    Parameters
    ----------
    pods : list
        List of integers, each one representing one pod to be visited.

    Returns
    -------
    A list that contains pods in visiting order.

    """
    route = [station]
    distances = {}
    while pods != []:
        # getting distances for all the pods
        nearestPod = {pod:distance_ij[str(route[-1]), str(pod)] for pod in pods}
        minDist = min(nearestPod.values())
        nearestPod = rd.choice([pod for pod, distance in nearestPod.items() \
                      if distance == minDist])
        distances[(route[-1], nearestPod)] = minDist
        route.append(nearestPod)
        pods = [pod for pod in pods if pod != nearestPod]
    distances[(route[-1], station)] = distance_ij[(str(route[-1]), station)]
    route.append(station)
    return([route, distances])

# for a given solution with distances, calculate the total distance
def solutionDistance(solution):
    solutionDistance = 0
    for batch in solution:
        for distance in batch["distances"].values():
            solutionDistance += distance
    return(solutionDistance)

#%% finding a neighbour for a solution
# pick two orders from the the solution, and exchange them; if they cannot be
# exchanged due to weight constraints, try a different pair
# pick two random orders
# all has to be repeated, until all weights in the solution do not exceed the
# weight limit:
def kOrdersExchange(solution, k, pr = False, maxTries = 100):
    """
    Function to exchange k orders in a given solution. Returns the new route, DOES NOT RE-CALCULATE THE ACTUAL ROUTES TAKEN BY BATCHES.

    Parameters
    ----------
    solution : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    pr : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    solution = copy.deepcopy(solution)
    tries = 0
    while True:
        tries += 1
        if pr: print("Iteration "+  str(tries))
        exchangeOrders = rd.sample(list(orderInfoDict.keys()), k)
        # check if they are in the same batch already; if yes, redraw until they are not
        while any([all([item in solution[0]["batchOrders"] for item in exchangeOrders]) for batch in solution]):
            exchangeOrders = rd.sample(list(orderInfoDict.keys()), k)
        # create a dictionary to determine how they are changed
        # we want a true shuffle: every order should switch place
        # shuffle the list of orders
        rd.shuffle(exchangeOrders)
        # now every order moves one to the right
        exchangeOrders = {exchangeOrders[position]:exchangeOrders[position-1] for position, order in enumerate(exchangeOrders)}
        exchangeOrderOutput = copy.deepcopy(exchangeOrders)
        # exchange the orders in the solution, change the weight of each batch:
            # check for each batch:
        if pr: print("  Batches to be exchanged:")
        if pr: print(exchangeOrders)
        for batchPos, batch in enumerate(solution):
            if pr: print("    Batch: " + str(batch["batchOrders"]))
        # check for each order currently in the batch:
            for orderPos, order in enumerate(batch["batchOrders"]):
                # check for each order in the list of orders to be replaced:
                for oldOrder, newOrder in exchangeOrders.items():
                    if order == oldOrder:
                        batch["batchOrders"][orderPos] = newOrder
                        changeWeight = orderInfoDict[oldOrder]["total weight"] - \
                                        orderInfoDict[newOrder]["total weight"]
                        batch["batchWeight"] += changeWeight
                        if pr: 
                            print("      Exchanged order " + str(oldOrder) + " for order " + str(newOrder) + " in batch " + str(batchPos))
                            print("      Changed weight of the batch by " + str(changeWeight))
                            print("      Removed order-pair " + str(oldOrder) + " (old order), " + str(newOrder) + " (new order)")
        # only end the while loop if all batches have a weight smaller or equal
        # to the capacity
        if all([batch["batchWeight"] <= batch_weight for batch in solution]):
            break
        # also: end if the number of tries is too high
        if tries == maxTries:
            False
    return({"solution": solution, "iterations":tries, "exchanges":exchangeOrderOutput})

#%% for a given (new) solution, re-calculate the routes of the batches completely
# from the scratch (i.e. without paying attention to the routes the orders
# took when calculating their assignment to stations)
    
# recycle the functions already written: F_fitnessValueOrderToStation returns
#
def recalculateRouteForSolution(solution, podInfoDict):
    """For a given solution, it calculates a route taken for each batch. Attention: this function only account for the list of orders for each batch in the solution (batch["batchOrders"] for batch in solution) and re-calculates routes based on these lists. It will therefore overwrite the current solution with this information!
    
    Parameters
    ----------
    solution : Object.
        Generated by 
    podInfoDict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
# checking the orders for each batch in the solution
    for batch in solution:
        # creating the DataFrame that stores the information about the items
        items = pd.DataFrame(columns = ["quantity", "itemID"]).set_index("itemID")
        # checking the items for each order in the bach
        for order in batch["batchOrders"]:
            # append the items of each order to the DataFrame with items per batch
            items = items.append(orderInfoDict[order]["items"])
        items = items.groupby(level = 0).sum()
        orderAssignment = F_fitnessValueOrderToStation(items, batch["station"], podInfoDict)
        batch["distances"] = orderAssignment["distances travelled"]
        batch["batchRoute"] = orderAssignment["route taken"]
        batch["batchPods"] = set(orderAssignment["route taken"][1:-1])

#%% writing the solution to .xml
def writeSolutionXML(solution, filename):
    # base element
    root = ET.Element("root")
    # first section "split" contains information about which orders are in which batches, and which batches are assigned to which station (=bot)
    collecting = ET.SubElement(root, "Collecting")
    split = ET.SubElement(collecting, 'Split')
    # write each station as a sub-node of split
    for station in packingStationNames:
        Bot_ID = ET.SubElement(split, "Bot")
        Bot_ID.set("ID", station)
        # filter the solution so it only contains batches for the right station
        stationSolution = [batch for batch in solution if batch["station"] == station]
        # write each batch as a sub-node of Bot_ID
        batchID = 1
        for batch in stationSolution:
            Batch_ID = ET.SubElement(Bot_ID, "Batch")
            Batch_ID.set("ID", str(batchID))
            batchID += 1
            # write Orders as the sub-node of Batch_ID
            Orders = ET.SubElement(Batch_ID, "Orders")
            # write each order as sub-node of Batch_ID
            for order in batch["batchOrders"]:
                Order = ET.SubElement(Orders, "Order")
                Order.text = str(order)
    # first section "bots" contains detailed information about each bot (station)
    bots = ET.SubElement(collecting, "Bots")
    # write each station as a sub-node of bots
    for station in packingStationNames:
        Bot_ID = ET.SubElement(bots, "Bot")
        Bot_ID.set("ID", station)
        # batches are written in sub-node Batches of Bot_ID
        Batches = ET.SubElement(Bot_ID, "Batches")
        # write each batch as a sub-node of Bot_ID
        batchID = 1
        stationSolution = [batch for batch in solution if batch["station"] == station]
        for batch in stationSolution:
            Batch_ID = ET.SubElement(Batches, "Batch")
            Batch_ID.set("BatchNumber", str(batchID))
            Batch_ID.set("Distance", str(round(sum([distance for distance in batch["distances"].values()]), 2)))
            batchID += 1
            # for each batch, write two sub-nodes: itemsData, edges
            # first write ItemsData
            ItemsData = ET.SubElement(Batch_ID, "ItemsData")
            # ItemsData has a sub-node called Orders
            Orders = ET.SubElement(ItemsData, "Orders")
            # write each order as sub-node of Orders:
            for order in batch["batchOrders"]:
                Order = ET.SubElement(Orders, "Order")
                Order.set("ID", str(order))
                # write each item in the order as sub-node of Order
                for item in orderInfoDict[order]["items"].index:
                    Item = ET.SubElement(Order, "Item")
                    # for each item, conclude information about the itemID and the description
                    Item.set("ID", str(item))
                    Item.set("Type", itemsInfoDict.loc[item, "Description"])
            # write Edges as sub-node of Batch_ID
            Edges = ET.SubElement(Batch_ID, "Edges")
            # write every edge of the batch
            for edge in batch["distances"].keys():
                Edge = ET.SubElement(Edges, "Edge")
                Edge.set("EndNode", str(edge[1]))
                Edge.set("StartNode", str(edge[0]))
    
    tree = ET.ElementTree(root)
    tree.write(filename)

#%% importing files for mixed shelf policy and creating the instances needed
skus = 24
storagepolicy = "mixed" # this can't be changed here

layoutFile = f'data/layout/1-1-1-2-1.xlayo'
# loading all the information about the pods
podInfoFile = f'data/sku{skus}/pods_infos.txt'
# loading information about picking locations, packing stations, waypoints,
# pods 
instanceFile = f'data/sku{skus}/layout_sku_{skus}_2.xml'

# loading information about item storage: contains all SKUs along with their
# attributes
storagePolicyFile = f'data/sku{skus}/pods_items_{storagepolicy}_shevels_1-5.txt'
#storagePolicies['mixed'] = 'data/sku24/pods_items_mixed_shevels_1-5.txt'

# loading information about the orders: contains list of orders, with number
# of ordered items per SKU-ID
orderFile = f'orders_20_mean_5_sku_{skus}_b.xml'
orderPath = f'data/sku{skus}/' + orderFile
#orders['20_5']=r'data/sku24/orders_20_mean_5_sku_24.xml'

#%% preparing the data
warehouseInstance = instance.Warehouse(layoutFile, instanceFile, podInfoFile, storagePolicyFile, orderPath)
batch_weight = 18
item_id_pod_id_dict = {}
distance_ij = demo.WarehouseDateProcessing(warehouseInstance).CalculateDistance()
distance_ij = {key:round(value, 1) for key, value in distance_ij.items()}
# getting the data
podInfoDict, itemsInfoDict, orderInfoDict, packingStationNames = prepareData(warehouseInstance)

#%% creating a first solution
ordersCopy = copy.deepcopy(orderInfoDict)
podCopy = copy.deepcopy(podInfoDict)
# assigning orders to stations
# orderAssignment = 
orderAssignment = F_orderToStation(ordersCopy, podCopy, False)
# creating a solution from the assignment of orders to stations
solution = createSolutionFromOrderAssignment(orderAssignment, packingStationNames)
print(round(solutionDistance(solution), 2))
writeSolutionXML(solution, "solution/solution_" + orderFile[0:-4] + "_mixedpolicy.xml")

#%% to alter the solution by exchanging k orders, use this algorithm: 
k = 2
l = 50
distance = solutionDistance(solution)
while k < int(len(orderInfoDict) * 0.8):
    print("------------------------------------------------")
    print("Finding a neighbour in the " + str(k) + "-opt neighbourhood.")
    currentSolution = copy.deepcopy(solution)
    currentDistance = solutionDistance(currentSolution)
    print("Current distance is " + str(round(currentDistance, 2)))
    # instead of finding the best neighbour in some neighbourhood, select the
    # best neighbour from l neighbours in that neighbourhood
    print("  Number of neighbours in " + str(k) + "-opt neighbourhood visited:")
    for i in range(l):
        print(i)
        suggestedSolution = kOrdersExchange(solution, k)["solution"]
        recalculateRouteForSolution(suggestedSolution, podInfoDict)
        suggestedDistance = solutionDistance(suggestedSolution)
        print("  Distance of suggested solution: "+ str(suggestedDistance) + " (current distance: " + str(currentDistance) + ")")
        if suggestedDistance < currentDistance:
            print("Found a better neighbour with distance " + str(suggestedDistance))
            currentSolution = suggestedSolution
            currentDistance = suggestedDistance
    if currentDistance < distance:
        print("Replacing old solution. Old Distance: " + str(distance) + ". New Distance: " + str(currentDistance))
        distance = currentDistance
        solution = currentSolution
    else:
        k += 1
        print("Keeping old solution. Increasing neighbourhood.")
