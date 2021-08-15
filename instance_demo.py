#example to use
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



    def t1Greedy(self):

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

        # TODO: check self.warehouseInstance.BatchesDF which contains all the information.
        print(1)



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
        self.warehouseInstance.BatchesDF[col_name_route] = AllChosenRoutes
        self.warehouseInstance.BatchesDF[col_name_time] = AllTravelTimes


        # TODO: work with this self.warehouseInstance.BatchesDF['ItemCount'][i] indexing instead of making columns to list and iterating over them.




    ### deprecated/unused:
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




    #def chooseNextBatch(self):
        ''':key
        this function chooses next batch
        '''

        # 1. choose fastest batch from feasible batches

        # 2. choose batch that takes longer than packing time of previous order
        #    and satisfies other conditons
        #           not longer that 1.5x packing time of previous order
        #

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



    solution1 = _demo.t1Greedy()

    #solution1.savetoxml(path)

    #solution2 =
    #solution3 =
    print(solution1)
    print("todo:")


