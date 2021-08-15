#example to
import time
import numpy as np
import xml.etree.cElementTree as ET
import networkx as nx
import networkx.algorithms.approximation as approximation
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import datetime
import pickle
import traceback
import os
import os.path
from os import path
import math
import json
import time
from operator import itemgetter, attrgetter
from xml.dom import minidom
import rafs_instance as instance
import untangle
import numpy as np
import itertools

layoutFile = r'data/layout/1-1-1-2-1.xlayo'
podInfoFile = 'data/sku24/pods_infos.txt'

# info on the warehouse layout, bots. pick locations, waypoints etc.
instances = {}
instances[24,2] = r'data/sku24/layout_sku_24_2.xml'

storagePolicies = {}
storagePolicies['dedicated'] = 'data/sku24/pods_items_dedicated_1.txt'
#storagePolicies['mixed'] = 'data/sku24/pods_items_mixed_shevels_1-5.txt'

orders = {}
orders['10_5']=r'data/sku24/orders_10_mean_5_sku_24.xml'
#orders['20_5']=r'data/sku24/orders_20_mean_5_sku_24.xml'



class WarehouseDateProcessing():
    def __init__(self, warehouseInstance, batch_size = None):
        self.Warehouse = warehouseInstance
        self._InitSets(warehouseInstance, batch_size)

    def preprocessingFilterPods(self, warehouseInstance):

        resize_pods = {}
        #print("preprocessingFilterPods")
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
                    #print("item.ID in item_id_list:", item.ID)
                    resize_pods[pod.ID] = pod

        #print(resize_pods)
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
            #print('d_ij file %s loaded'%(file_path))

            #d_ij = tupledict()
            d_ij = {}
            for i, values in d_ij_dict.items():
                for j, dist in values.items():
                    d_ij[i,j] = dist

        return d_ij


class Solution():
    def __init__(self):
        # TODO: read the solution template file and use their structure for the solution class
        # figure out how to do this from their case or package
        solutionTemplateFile = r'log_example.xml'

        tree = ET.parse(solutionTemplateFile)
        root = tree.getroot()

        with open(solutionTemplateFile, 'r') as f:
            xmlTemplate = f.read()

        self.solution = untangle.parse(xmlTemplate)


class Demo():
    def __init__(self, splitOrders = False):

        self.batch_weight = 18
        #[0]
        self.warehouseInstance = self.prepareData()
        self.distance_ij = self.initData()
        #[2]
        if storagePolicies.get('dedicated'):
            self.is_storage_dedicated = True
        else:
            self.is_storage_dedicated = False

        self.item_id_pod_id_dict = self.getPodforItems()

        #self.solution1 = self.initSolution()
        #self.solution2 = self.initSolution()
        #self.solution3 = self.initSolution()



    def prepareWarehouseInformation(self):
        timer_start = time.time()

        # calculates all feasible batches of the given orders
        self.warehouseInstance.getFeasibleBatches()

        # calculates the items and pod locations for all the batches and writes it to BatchesDF
        self.warehouseInstance.getItemPodsBatchDF()

        # generates a Graph of the Warehouse to be used in traveling salesman probelem.
        self.generateGraph()

        # calculates shortest route and traveling time for each batch with regards to the output staion
        # and passed the result in the batches dataframe.
        for OutputStationID in self.warehouseInstance.OutputStations:
            self.shortestPathTSP(OutputStationID)

        time_tsp = (time.time() - timer_start)
        print(f"[TIME] Calculated {len(self.warehouseInstance.BatchesDF)} Batches x {len(self.warehouseInstance.OutputStations)} in {time_tsp} seconds")
        print(f"[TIME] That is {len(self.warehouseInstance.BatchesDF)*len(self.warehouseInstance.OutputStations)} TSPs with {len(self.warehouseInstance.BatchesDF)*len(self.warehouseInstance.OutputStations)/time_tsp} TSPs per second")




        # alternative way to work with dataframes and functions
        # kw_df['kw_umlaute'] = kw_df.apply(lambda kw_df: reduce_umlaut(kw_df['kw_umlaute']), axis = 1)

        ''' Fehlgeleitete Idee
        ### zuerst orders an die packing stations assignen ###
        # 1 first of all, we assign the order to the packing stations baes on weight.
        self.assignOrdersToPackingStations()

        # 2 then, we calculate a list of feasible batches of these orders inside the packing stations
        for j in range(len(self.warehouseInstance.OutputStations)):
            OutputStationKey = 'OutD' + str(j)
            self.warehouseInstance.OutputStations[OutputStationKey].getFeasibleBatches()

        # 3 pick a batch with greedy heuristic
        for i in range(len(self.warehouseInstance.OutputStations)):
            OutputStationKey = 'OutD' + str(j)
            iter = 0
            while self.warehouseInstance.OutputStations[OutputStationKey].feasibleBatches != []:

                Batch = 0

                # for the first iteration of batch assignment, we just chose any batch with
                if iter == 0:
                    # pick the first batch with just 1 item/order so that the packer starts working quickly.
                    pass
                else:
                    # pick the first batch that has longer collection time than the packing time of the previous batch
                    pass
                    self.warehouseInstance.getItemsInBatch(['1', '2'], 'OutD0')

                # feasibleBatches: remove all entries that contain fulfilledOrders
                iter += 1



        # 4 determine the visiting sequence for each cobot/pod

        # 5 add the picked batch to the solution S
        # remove all batches that contain the fulfilled orders



        # calculate minim distance/time  (take from chans group) (decision metric for which batch to proicess)
        #self.getTravelRoute()

        print(1)
        return self.solution1
        '''


    def shortestPathTSP(self, OutputStation):
        # input: packing station
               # stationstovisit

        BotVelocity = 2  # dynamically draw this value from BotClass
        PickerVelocity = 1.3  # picker speed
        ItemPickingTime = 3  # dynamially draw this item from warehouse class




        AllChosenRoutes = []
        AllTravelTimes = []
        AllBatchesStations = list(self.warehouseInstance.BatchesDF['StationsToVisit'])

        SA_tsp = approximation.traveling_salesman.simulated_annealing_tsp
        tsp = approximation.traveling_salesman.traveling_salesman_problem
        tsp_method = lambda G, wt: SA_tsp(G, "greedy", weight=wt, temp=500)

        i = 0
        for stationsToVisit in AllBatchesStations:
            stationsToVisit.insert(0, OutputStation)

            ######################################################################
            chosenTravelRoute = tsp(self.warehouseInstance.WarehouseGraph, nodes=stationsToVisit, method=tsp_method)

            TravelTime = nx.classes.function.path_weight(self.warehouseInstance.WarehouseGraph, chosenTravelRoute, 'weight') / BotVelocity
            PickingTime = TravelTime + self.warehouseInstance.BatchesDF['ItemCount'][i] * ItemPickingTime

            TotalRouteTime = TravelTime + PickingTime
            ######################################################################
            AllChosenRoutes.append(chosenTravelRoute)
            AllTravelTimes.append(TotalRouteTime)
            i += 1

        # output: shortest path TSP
        col_name_route = 'shortestRoute_'+OutputStation
        col_name_time = 'travelTime_'+OutputStation
        col_name_time_per_order = 'travelTimeperOrder_'+OutputStation
        self.warehouseInstance.BatchesDF[col_name_route] = AllChosenRoutes
        self.warehouseInstance.BatchesDF[col_name_time] = AllTravelTimes
        self.warehouseInstance.BatchesDF[col_name_time_per_order] = self.warehouseInstance.BatchesDF[col_name_time] / self.warehouseInstance.BatchesDF['OrderCount']
        # TODO: work with this self.warehouseInstance.BatchesDF['ItemCount'][i] indexing instead of making columns to list and iterating over them.



    def greedyHeuristicT1(self):
        # Assumption: there are only 2 packing station and 2 cobots. This is consistent with both 24 and 360 layouts
        # Keep track of time for each packer and cobot. Prep time is 30 seconds (only assigned at the beginning), so each time is
        # initialized at 30 seconds.

        # we start with 30 seconds time on each instance, which reflects the initial setup time for the cobots.
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
        OrdersTable = self.warehouseInstance.BatchesDF[['Batch', 'travelTime_OutD0', 'travelTime_OutD1', 'travelTimeperOrder_OutD0', 'travelTimeperOrder_OutD1']]

        ##### I could write this in a loop. If there's time
        # First batch assignment

        # Find batch with shortest time
        next_batch = OrdersTable.Batch[OrdersTable.travelTime_OutD0.idxmin()]
        next_batch_time = OrdersTable.travelTime_OutD0.min()

        # Assign batch to cobot. Add time to cobot and packer (packing time is 60 sec per item)
        BatchAssignCobot_List[0].append(next_batch)
        TimeCobot_List[0] += next_batch_time + 20
        TimePacker_List[0] += next_batch_time + 20 + 60 * len(next_batch)

        # Remove any rows containing any of these orders
        dropping_rows = OrdersTable.Batch.apply(lambda x: any(item in next_batch for item in x))
        OrdersTable = OrdersTable[dropping_rows == False]

        ### Do the exact same thing for cobot2
        next_batch = OrdersTable.Batch[OrdersTable.travelTime_OutD1.idxmin()]
        next_batch_time = OrdersTable.travelTime_OutD1.min()

        # Assign batch to cobot. Add time to cobot and packer (packing time is 60 sec per item)
        BatchAssignCobot_List[1].append(next_batch)
        TimeCobot_List[1] += next_batch_time + 20
        TimePacker_List[1] += next_batch_time + 20 + 60 * len(next_batch)

        # Remove any rows containing any of these orders
        dropping_rows = OrdersTable.Batch.apply(lambda x: any(item in next_batch for item in x))
        OrdersTable = OrdersTable[dropping_rows == False]

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

            batch_times = OrdersTable[['travelTimeperOrder_OutD0', 'travelTimeperOrder_OutD1']]
            # Find next available cobot
            next_available_cobot = TimeCobot_List.index(min(TimeCobot_List))
            other_cobot = TimeCobot_List.index(max(TimeCobot_List))
            # Find index of next batch based on timePerItem
            next_batch_index = batch_times.iloc[:, next_available_cobot].idxmin()
            next_batch = OrdersTable.Batch[next_batch_index]
            next_batch_time = OrdersTable.loc[next_batch_index].iloc[next_available_cobot + 1]
            print("next batch", next_batch)
            # Perform a check here! If its faster to send out the other cobot because the packer is still busy, then send the other cobot.
            completion_time1 = max(TimeCobot_List[next_available_cobot] + next_batch_time,
                                   TimePacker_List[next_available_cobot])
            completion_time2 = max(
                TimeCobot_List[other_cobot] + OrdersTable.loc[next_batch_index].iloc[other_cobot + 1],
                TimePacker_List[other_cobot])
            if completion_time1 > completion_time2:
                next_available_cobot = other_cobot

            # Assign batch and calculate what time cobot will arrive at packing station.
            BatchAssignCobot_List[next_available_cobot].append(next_batch)
            TimeCobot_List[next_available_cobot] += next_batch_time

            # Since the unload process cannot start until both the cobot and packer are ready, they are both set to the maximum
            # time between the two of them.
            TimeCobot_List[next_available_cobot] = max(TimeCobot_List[next_available_cobot],
                                                       TimePacker_List[next_available_cobot])
            TimePacker_List[next_available_cobot] = max(TimeCobot_List[next_available_cobot],
                                                        TimePacker_List[next_available_cobot])

            # Now that both are ready, we add unpack time to cobot
            TimeCobot_List[next_available_cobot] += 20

            # The packer also adds the unload time, and also packing time for each item
            TimePacker_List[next_available_cobot] += 20 + 60 * len(next_batch)

            # Drop these items from the other feasible batches
            dropping_rows = OrdersTable.Batch.apply(lambda x: any(item in next_batch for item in x))
            OrdersTable = OrdersTable[dropping_rows == False]


        print(BatchAssignCobot_List)
        print(TimeCobot_List)
        print(TimePacker_List)

        self.BatchAssignCobot_List = BatchAssignCobot_List
        self.TimeCobot_List = TimeCobot_List
        self.TimePacker_List = TimePacker_List  #TODO: could this be used as a makespan time? max(TimeCobot_List, TimePacker_List)


    # deprecated/unused:
    def getTravelRouteofBatch(self):#, Batch, OutputStation):
        '''
        calculates the route and travel time of each batch, based on the batch and the output station
        :return: travel time and route
        '''
        #INPUTS:
        Batch = [1,2]       ## draw this from function call
        OutputStation = 'OutD0' #'OutD1' ## draw this from function call

        # get list of order of items in batch
        itemsInBatch = []
        for i in Batch:     # the orders in batch
            for j in self.warehouseInstance.Orders[i].Positions:
                itemsInBatch.append(self.warehouseInstance.Orders[i].Positions[str(j)])

        # accumulate items
        ItemsDict = {}
        for j in itemsInBatch:
            ItemsDict[j.ItemDescID] = 0
        for k in itemsInBatch:
            ItemsDict[k.ItemDescID] = int(ItemsDict[k.ItemDescID]) + int(k.Count)

        # get ItemPodID for ItemID
        ItemPodID = {}
        for i in ItemsDict:
            ItemPodID[i] = self.warehouseInstance.ItemDescriptions[i].ItemPodID

        # get the pod ID for each Item
        PodID = {}
        for i in ItemsDict:
            PodID[i] = self.warehouseInstance.ItemPodLocations[ItemPodID[i]]

        # collecting all information in dataframe
        batchItemsDF = pd.DataFrame({'items':ItemsDict.keys(), 'quantity':ItemsDict.values(), 'ItemPodID':ItemPodID.values(), 'PodID': PodID.values()})


        # generating a list of nodes to be visited, with the starting/ending node in the first list position
        # this will be used as an inout to the networkx tsp algorithm
        # generally, it would be possible to make one list for each outputStation and compare the time of traveling sequence
        # that is given my the TSP algo of NX.
        stationsToVisit = [item for elem in list(PodID.values()) for item in elem]
        stationsToVisit.insert(0, OutputStation)

        ################################################################################################
        ################################################################################################
        # Generating a network from the pods/stations and distances
        G = nx.Graph()
        # adding all nodes and edges to the network graph
        all_nodes = list(self.warehouseInstance.OutputStations) + list(self.warehouseInstance.Pods)
        G.add_nodes_from(all_nodes)
        G.add_edges_from(self.distance_ij)     ## weighted edged are taken in the format [[1,2,666],[2,3,5565],[],[],[]]
        # setting the edges weights.
        nx.set_edge_attributes(G, values=self.distance_ij, name='weight')

        ## checking some propeties of the G graph, can be deleted later.
        # TODO: delete
        len(self.distance_ij)
        G.number_of_edges()
        G.number_of_nodes()
        G.nodes['1']

        G.edges['1', 'OutD1']
        G.edges['OutD1', '1']

        G.edges['OutD1', '1']
        G.edges['1', '2']
        G.edges['2', 'OutD1']

        G.edges['2', '2']


        G['4']['5']["weight"]

        nx.draw(G, with_labels = True)
        plt.savefig("network_img1.png")

        nx.is_connected(G) ## has to be true always
        #### TODO: delete until here



        ################################################################################################
        ##################################   TSP from NetworkX   #######################################
        ################################################################################################
        SA_tsp = approximation.traveling_salesman.simulated_annealing_tsp
        tsp = approximation.traveling_salesman.traveling_salesman_problem

        tsp_method = lambda G, wt: SA_tsp(G, "greedy", weight=wt, temp=500)
        chosenTravelRoute = tsp(G, nodes=stationsToVisit, method=tsp_method)
        ################################################################################################
        ################################################################################################

        # TODO: Test TSP algorithm with various sequences of stations to visit
        # TODO: write the chosen route to the feasibleBatches dataframe as an entry in a dedicated column for each output station

        # calculating the travel time of the chosen Route # in
        BotVelocity = 2         # dynamically draw this value from BotClass
        PickerVelocity = 1.3    # picker speed
        ItemPickingTime = 3     # dynamially draw this item from warehouse class
        TravelTime = nx.classes.function.path_weight(G, chosenTravelRoute, 'weight') / BotVelocity
        PickingTime = TravelTime + batchItemsDF['quantity'].sum() * ItemPickingTime

        TotalRouteTime = TravelTime + PickingTime


        self.TravelRoute = chosenTravelRoute
        self.TravelRouteTime = TotalRouteTime
    def getPackingTimeofBatch(self):#, #ofOrdersinBatch):
        # calculates time needed for packing a batch of orers at the OutputStation
        unloadCobot = 20        # per batch

        packingOrder = 60       # per order, 60seconds to pack

        prepCobot = 30          # per batch (only once at the beginning, add 30seconds to total time)

        self.PackingTime = 1
    def assignOrdersToPackingStations(self):
        '''
        This heuristic assigns orders to Packing Stations while trying to balance total weight between the stations.
        Improvements: Find a way to assign Order not only on weight but through a combination of weight and number of orders,
        so that several very heavy orders cannot skew the distribution, which would risk having one station idle for too long.
        '''
        for i in range(len(self.warehouseInstance.openOrders)):
            # assign the first order to any packing station (here we just take the first one)
            if i == 0:
                self.warehouseInstance.OutputStations['OutD0'].Queues.append(self.warehouseInstance.openOrders[i])
                self.warehouseInstance.OutputStations['OutD0'].UpdateWeight()
            #otherwise, assign the outut station with the smalles totalWeight of so far assigned Orders.
            else:
                # find station with lowest weight
                LowestWeightOutputStationWeight = 9999999
                LowestWeightOutputStationKey = ''
                for j in range(len(self.warehouseInstance.OutputStations)):
                    OutputStationDictKey = 'OutD' + str(j)
                    if self.warehouseInstance.OutputStations[OutputStationDictKey].totalWeight <= LowestWeightOutputStationWeight:
                        LowestWeightOutputStationKey = OutputStationDictKey
                        LowestWeightOutputStationWeight = self.warehouseInstance.OutputStations[OutputStationDictKey].totalWeight
                # assign order to that station
                self.warehouseInstance.OutputStations[LowestWeightOutputStationKey].Queues.append(self.warehouseInstance.openOrders[i])
                self.warehouseInstance.OutputStations[LowestWeightOutputStationKey].UpdateWeight()





######################### start functions #########################
	# warehouse instance
    def prepareData(self):
        print("[0] preparing all data with the standard format: ")
        #Every instance
        for key, instanceFile in instances.items():
            podAmount = key[0]
            depotAmount = key[1]
            #For different orders
            for key, orderFile in orders.items():
                orderAmount = key
                #For storage policies
                for storagePolicy, storagePolicyFile in storagePolicies.items():
                    warehouseInstance = instance.Warehouse(layoutFile, instanceFile, podInfoFile, storagePolicyFile, orderFile)
        return warehouseInstance

	# distance
    def initData(self):
        print("[1] changing data format for the algorithm we used here: ")
        warehouse_data_processing = WarehouseDateProcessing(self.warehouseInstance)
        #Distance d_ij between two nodes i,j \in V
        d_ij = warehouse_data_processing.CalculateDistance()
        return d_ij

######################### start functions #########################

    # initlaize solution xml struncture as class object.
    def initSolution(self):
        print("[2] initializing solution class")
        solution = Solution()
        return solution

    # generate ItemID - PodLocation dictionary.
    def getPodforItems(self):
        ''':key
        method to collect the pod in which each item is located. can be adapoted for a mixed storage policy where items can be contained in multiple pods.
        '''
        print("[3] Calculating pro Location for Items ")

        item_id_pod_id_dict = {} #initialize dictionary for item pod location
        for i in self.warehouseInstance.ItemDescriptions:
            itemPodID = self.warehouseInstance.ItemDescriptions[i].ItemPodID
            item_id_pod_id_dict[itemPodID] = []

            for j in self.warehouseInstance.Pods:
                for k in range(len(self.warehouseInstance.Pods[j].Items)):
                    if self.warehouseInstance.Pods[j].Items[k].ID == itemPodID:
                        item_id_pod_id_dict[itemPodID].append(j)


        return item_id_pod_id_dict  # a dictionary with item IDs as keys and a list of pod locations as value.

    # generated a Graph as an attribute of the warehouseInstance.
    def generateGraph(self):
        print("[6] Generating Network Graph for Warehouse Instance")

        G = nx.Graph()
        # adding all nodes and edges to the network graph
        all_nodes = list(self.warehouseInstance.OutputStations) + list(self.warehouseInstance.Pods)
        G.add_nodes_from(all_nodes)
        G.add_edges_from(self.distance_ij)  ## weighted edged are taken in the format [[1,2,666],[2,3,5565],[],[],[]]
        # setting the edges weights.
        nx.set_edge_attributes(G, values=self.distance_ij, name='weight')

        nx.draw(G, with_labels=True)
        plt.savefig("network_img.png")

        if nx.is_connected(G) == True: ## has to be true always
            self.warehouseInstance.WarehouseGraph = G
        else:
            print('WarehouseGraph has unconnected nodes!')





if __name__ == "__main__":

    _demo = Demo()


    # preparing warehouse attributes like network graph, batching, orders dataframes etc.
    _demo.prepareWarehouseInformation()

    # applying greedy heuristic to find solution for task 1
    _demo.greedyHeuristicT1()

    #solution1.savetoxml(path)

    #solution2 =
    #solution3 =


