# example to use


#%% Importing libraries
#import numpy as np
#import xml.etree.cElementTree as ET
import networkx as nx
#import matplotlib.pyplot as plt
#import pandas as pd
#import itertools
#import datetime
#import pickle
#import traceback
import os
import os.path
from os import path
#import math
import json
#import time
#from operator import itemgetter, attrgetter 
#from xml.dom import minidom
import rafs_instance as instance
    # this does not load all the classes defined in rafs_instance/__init__.py,
    # but it provides the module instance from which all the classes can be
    # loaded via .-notation

#%% defining the class 
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
        #[2]
        if storagePolicies.get('dedicated'):
            self.is_storage_dedicated = True
        else:
            self.is_storage_dedicated = False


	# warehouse instance
    def prepareData(self):
        print("[0] preparing all data with the standard format: ")
        #Every instance
        for key,instanceFile in instances.items():
            #podAmount = key[0]
            #depotAmount = key[1]   
            #For different orders
            for key, orderFile in orders.items():
                # orderAmount = key
                #For storage policies
                for storagePolicy, storagePolicyFile in storagePolicies.items():   
                    warehouseInstance = instance.Warehouse(layoutFile, instanceFile, podInfoFile, storagePolicyFile, orderFile)
        return warehouseInstance

    def initData(self):
        print("[1] changing data format for the algorithm we used here: ")
        warehouse_data_processing = WarehouseDateProcessing(self.warehouseInstance)
        #Distance d_ij between two nodes i,j \in V
        d_ij = warehouse_data_processing.CalculateDistance()
        return d_ij


#%% Getting the files needed

# not sure what this contains, but it is needed for the Warehouse-class
layoutFile = r'data/layout/1-1-1-2-1.xlayo' 
# loading all the information about the pods
podInfoFile = 'data/sku24/pods_infos.txt'   # 
# loading information about picking locations, packing stations, waypoints,
# pods 
instanceFile = r'data/sku24/layout_sku_24_2.xml'

# loading information about item storage: contains all SKUs along with their
# attributes
storagePolicyFile = 'data/sku24/pods_items_mixed_shevels_1-5.txt'
#storagePolicies['mixed'] = 'data/sku24/pods_items_mixed_shevels_1-5.txt'

# loading information about the orders: contains list of orders, with number
# of ordered items per SKU-ID
orderFile =r'data/sku24/orders_10_mean_5_sku_24.xml'
#orders['20_5']=r'data/sku24/orders_20_mean_5_sku_24.xml'

# trying a different way to get the demonstration running
# function to prepare data
warehouseInstance = instance.Warehouse(layoutFile, instanceFile, podInfoFile, storagePolicyFile, orderFile)

def initData(self):
    print("[1] changing data format for the algorithm we used here: ")
    warehouse_data_processing = WarehouseDateProcessing(self.warehouseInstance)
    #Distance d_ij between two nodes i,j \in V
    d_ij = warehouse_data_processing.CalculateDistance()
    return d_ij

#%% 
batch_weight = 18
item_id_pod_id_dict = {}

distance_ij = WarehouseDateProcessing(warehouseInstance).CalculateDistance()

