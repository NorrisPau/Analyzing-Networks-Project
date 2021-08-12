#example to use
import numpy as np
import xml.etree.cElementTree as ET
import networkx as nx
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

        self.solution1 = self.initSolution()
        self.solution2 = self.initSolution()
        self.solution3 = self.initSolution()



    def t1Greedy(self):
        # top level method for the solution of task 1
        # TODO: implement the algorithm here
            # calculate feasible batches (weight constraint)
        instance.Warehouse.getFeasibleBatches(self.warehouseInstance)

        # calculate minim distance/time  (take from chans group) (decision metric for which batch to proicess)
        self.getTravelRoute()


        return self.solution1

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


    def getTravelRoute(self):#, Batch, OutputStation):
        '''
        calculates the route and travel time of each batch, based on the batch and the output station
        :return: travel time and route
        '''
        #INPUTS:
        Batch = [1,2]
        OutputStation = 'OutD0' #'OutD1'

        #OUTPUTS:
        total_time = 0
        waypoints_list = []

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
            PodID[i] = self.item_id_pod_id_dict[ItemPodID[i]]

        # collecting all information in dataframe
        itemsdf = pd.DataFrame({'items':ItemsDict.keys(), 'quantity':ItemsDict.values(), 'ItemPodID':ItemPodID.values(), 'PodID': PodID.values()})


        print(ItemsDict)
        print(itemsdf)
        # get list of pods that have to be visited for these items


        # choose pod sequence to be visited


        # initialize route with start = outputstation and distance = 0


        # greedy heuristic, that chooses the first waypoint, then the next etc. until all of them have been visited


        self.TravelRoute = 1


    def getTravelTime(self, feasibleBatches, TravelRoute):
        # calculates time based on the sequence of visited pods.


        self.TravelTime = 1



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


    def initSolution(self):
        print("[2] initializing solution class")
        solution = Solution()
        return solution





if __name__ == "__main__":

    _demo = Demo()
    solution1 = _demo.t1Greedy()

    #solution1.savetoxml(path)

    #solution2 =
    #solution3 =
    print(solution1)
    print("todo:")


