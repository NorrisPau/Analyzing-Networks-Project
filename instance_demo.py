import xml.etree.cElementTree as ET
import networkx as nx
import networkx.algorithms.approximation as approximation
import matplotlib.pyplot as plt
import os
import os.path
from os import path
import math
import json
import time
import rafs_instance as instance
import numpy as np
import itertools
import copy
import random

# setting the warehouse (small:24, large:360)
warehouse = '24'
if warehouse == '360':
    layoutFile = r'data/layout/1-1-1-2-1.xlayo'
    podInfoFile = 'data/sku360/pods_infos.txt'

    # info on the warehouse layout, bots. pick locations, waypoints etc.
    instances = {}
    instances[360, 2] = r'data/sku360/layout_sku_360_2.xml'

    storagePolicies = {}
    storagePolicies['dedicated'] = 'data/sku360/pods_items_dedicated_1.txt'
    #storagePolicies['mixed'] = 'data/sku360/pods_items_mixed_shevels_1-5.txt'

    orders = {}
    orders['10_5'] = r'data/sku360/orders_10_mean_5_sku_360.xml'
    #orders['20_5'] = r'data/sku360/orders_20_mean_5_sku_360.xml'
else:
    warehouse = '24'
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
        item_id_list=[]
        for order in warehouseInstance.Orders:
            for pos in order.Positions.values():
                item = warehouseInstance.ItemDescriptions[pos.ItemDescID].Color.lower() + '/' + warehouseInstance.ItemDescriptions[pos.ItemDescID].Letter
                if item not in item_id_list:
                    item_id_list.append(item)

        # for dedicated
        for pod in warehouseInstance.Pods.values():
            for item in pod.Items:
                if item.ID in item_id_list:
                    resize_pods[pod.ID] = pod

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

class Demo():
    def __init__(self, splitOrders = False):

        self.batch_weight = 18
        #[0]
        self.warehouseInstance = self.prepareData()
        #self.distance_ij = self.initData()
        self.warehouseInstance.distance_ij = self.initData()
        #[2]
        if storagePolicies.get('dedicated'):
            self.is_storage_dedicated = True
        else:
            self.is_storage_dedicated = False


    # warehouse instance
    def prepareData(self):
        print("[0] preparing all data with the standard format: ")
        # Every instance
        for key, instanceFile in instances.items():
            podAmount = key[0]
            depotAmount = key[1]
            # For different orders
            for key, orderFile in orders.items():
                orderAmount = key
                # For storage policies
                for storagePolicy, storagePolicyFile in storagePolicies.items():
                    warehouseInstance = instance.Warehouse(layoutFile, instanceFile, podInfoFile, storagePolicyFile,
                                                           orderFile)
        return warehouseInstance

    # distance
    def initData(self):
        print("[2] Changing data format for the algorithm we used here: ")
        warehouse_data_processing = WarehouseDateProcessing(self.warehouseInstance)
        # Distance d_ij between two nodes i,j \in V
        d_ij = warehouse_data_processing.CalculateDistance()
        return d_ij


    # preparing the warehouse information
    def prepareWarehouseInformation(self, storagePolicy='dedicated'):
        '''
        preprares information in thre warehouse class that will be used by other methods to determine the solutions
        '''
        timer_start = time.time()

        # calculates all feasible batches of the given orders
        self.warehouseInstance.getFeasibleBatches()

        # calculates the items and pod locations for all the batches and writes it to BatchesDF
        self.warehouseInstance.getItemPodsBatchDF(storagePolicy)
        self.warehouseInstance.getPodZones()

        # generates a Graph of the Warehouse to be used in traveling salesman probelem.
        self.generateGraph()

        # calculates shortest route and traveling time for each batch with regards to the output staion
        # and passed the result in the batches dataframe.
        for OutputStationID in self.warehouseInstance.OutputStations:
            self.shortestPathTSP(OutputStationID)

        time_tsp = (time.time() - timer_start)
        print(f"[TIME] Calculated {len(self.warehouseInstance.BatchesDF)} Batches x {len(self.warehouseInstance.OutputStations)} in {time_tsp} seconds")
        print(f"[TIME] That is {len(self.warehouseInstance.BatchesDF)*len(self.warehouseInstance.OutputStations)} TSPs with {len(self.warehouseInstance.BatchesDF)*len(self.warehouseInstance.OutputStations)/time_tsp} TSPs per second")

    # calcukating shortest paths for batches with TSP
    def shortestPathTSP(self, OutputStation):
        '''
        uses the method traveling salesman problem from networkx to determine shortest path.
        this could be replaced with a simpler heuristic that is less computationally expensive,
         but due to time constraints we are not able to do so anymore.
        :param OutputStation: the output station that this was assigned to.
        :return: shortest path
        '''
        print("[7] Starting Calculation of shortest Routes with TSP")

        BotVelocity = 2

        AllChosenRoutes = []
        AllTravelTimes = []
        AllTravelDistances = []
        AllBatchesStations = list(self.warehouseInstance.BatchesDF['StationsToVisit'])

        # traveling salesman algorithm
        tsp = approximation.traveling_salesman.traveling_salesman_problem

        # there are several tsp methods available, we choose the greedy version, due to faster runtime.
        #SA_tsp = approximation.traveling_salesman.simulated_annealing_tsp
        GR_tsp = approximation.traveling_salesman.greedy_tsp
        #CF_tsp = approximation.traveling_salesman.christofides
        #TA_tsp = approximation.traveling_salesman.threshold_accepting_tsp

        #tsp_method = lambda G, wt: SA_tsp(G, "greedy", weight='weight', temp=500)       # simmulated annealing TSP
        tsp_method = lambda G, wt: GR_tsp(G, weight='weight' )                          # Greedy TSP
        #tsp_method = lambda G, wt: CF_tsp(G, weight='weight' )                          # Christofides TSP
        #tsp_method = lambda G, wt: TA_tsp(G, "greedy", weight='weight' )                # Treshold Accepting TSP


        # this loop iterates over feasible batches and calculates shortest paths for each of them.
        i = 0
        for j in AllBatchesStations:
            print(f"[7_{i}] TSP instance {i} of {len(AllBatchesStations)}")
            stationsToVisit = copy.deepcopy(j)
            stationsToVisit.insert(0, OutputStation)

            ######################################################################
            chosenTravelRoute = tsp(self.warehouseInstance.WarehouseGraph, nodes=stationsToVisit, method=tsp_method)
            TravelDistance = nx.classes.function.path_weight(self.warehouseInstance.WarehouseGraph, chosenTravelRoute, 'weight')
            TotalRouteTime =  TravelDistance / BotVelocity
            ######################################################################

            AllChosenRoutes.append(chosenTravelRoute)
            AllTravelTimes.append(TotalRouteTime)
            AllTravelDistances.append(TravelDistance)
            i += 1

        # output: shortest path TSP. information from this methods gets collected in batchesDF
        col_name_route = 'shortestRoute_' + OutputStation
        col_name_time = 'travelTime_' + OutputStation
        col_name_dist = 'travelDist_' + OutputStation
        col_name_time_per_order = 'travelTimeperOrder_' + OutputStation
        self.warehouseInstance.BatchesDF[col_name_route] = AllChosenRoutes
        self.warehouseInstance.BatchesDF[col_name_time] = AllTravelTimes
        self.warehouseInstance.BatchesDF[col_name_dist] = AllTravelDistances
        self.warehouseInstance.BatchesDF[col_name_time_per_order] = self.warehouseInstance.BatchesDF[col_name_time] / self.warehouseInstance.BatchesDF['OrderCount']

    # using a greedy heuristic to find a solution to task 1
    def greedyHeuristic_T1(self):
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

        ### First batch assignment
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

        # Do the exact same thing for cobot2
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
        while len(OrdersTable) > 0:
            # Subset orders table into time columns
            batch_times = OrdersTable[['travelTimeperOrder_OutD0', 'travelTimeperOrder_OutD1']]
            # Find next available cobot
            next_available_cobot = TimeCobot_List.index(min(TimeCobot_List))
            other_cobot = TimeCobot_List.index(max(TimeCobot_List))
            # Find index of next batch based on timePerItem
            next_batch_index = batch_times.iloc[:, next_available_cobot].idxmin()
            next_batch = OrdersTable.Batch[next_batch_index]
            next_batch_time = OrdersTable.loc[next_batch_index].iloc[next_available_cobot + 1]
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

        self.BatchAssignCobot_List = BatchAssignCobot_List
        self.TimeCobot_List = TimeCobot_List
        self.TimePacker_List = TimePacker_List

    # method to calculate the makespan of a proposed solution
    def calculateMakeSpan(self, batch_assignments):
        '''
        calculates the makespan of a given solution.
        '''
        OrdersTable = self.warehouseInstance.BatchesDF[
            ['Batch', 'shortestRoute_OutD0', 'shortestRoute_OutD1']]
        TimePackers = [30, 30]
        TimeCobots = [30, 30]
        BatchCounter = [0, 0]
        OrdersPerBatch = [[],[]]
        Routes = [[],[]]
        Full_Route = [[], []]

        # Find routes that cobots will be traveling
        for cobot in range(2):
            for batch in batch_assignments[cobot]:
                route = OrdersTable[OrdersTable.Batch.apply(lambda x: x == batch)].iloc[0,cobot+1]
                Routes[cobot].append(route)
                Full_Route[cobot] = Full_Route[cobot] + route[0:-1]
                OrdersPerBatch[cobot].append(len(batch))
        Full_Route[0].append('OutD0')
        Full_Route[1].append('OutD1')

        # Send cobots to first item
        for cobot in range(2):
            distance = self.warehouseInstance.distance_ij[tuple(Routes[cobot][0][0:2])]
            time = distance/2
            TimeCobots[cobot] += time
            Full_Route[cobot] = Full_Route[cobot][1:]

        picking_zone_list = []
        zone_occupied = {}
        for pod in self.warehouseInstance.Pods:
            picking_zone = self.warehouseInstance.Pods[pod].PickingZone
            if picking_zone not in picking_zone_list:
                picking_zone_list.append(picking_zone)
                zone_occupied[picking_zone] = {'occ_time':[99999,99999], 'last_item':99999}


        ##### This loop will run until both cobots have completed their orders
        # Each iteration of the loop starts (and ends) when a cobot has arrived in a NEW PICKING ZONE.
        # Ex: Cobot1 has arrived in zone1 (start of loop). He completes all orders in that zone and moves to his next picking location...
        # in a new zone (loop ends). His ending time is the arrival time in the next zone. Anytime a cobot completes a batch, the packers time is adjusted,
        # and the cobot is sent to his next picking zone. The loop ends when the cobot arrives in his next PICKING ZONE.

        while (Full_Route[0] != []) or (Full_Route[1] != []):
            # choose cobot with lower time, help this cobot until all items in the zone are completed
            current_cobot = TimeCobots.index(min(TimeCobots))
            # if this cobot has not items remaining, switch cobots
            if Full_Route[current_cobot] == []:
                current_cobot = int(bool(current_cobot) == False)

            current_cobot_item = Full_Route[current_cobot][0]
            current_cobot_next_item = Full_Route[current_cobot][1]
            current_cobot_zone = self.warehouseInstance.Pods[current_cobot_item].PickingZone

            ### Has the current cobot arrived in a picking zone where the picker is busy?
            # Grad occupied details for current zone, and check if the zone has been occupied before. If it hasn't, there no need to check for waiting periods.
            zone_occ_last_item = zone_occupied[current_cobot_zone]['last_item']
            if zone_occ_last_item != 99999:
                zone_occ_time = zone_occupied[current_cobot_zone]['occ_time']
                # Calculate time for picker to walk from other cobot's last item to current cobot's first item
                distance = self.warehouseInstance.distance_ij[tuple([str(zone_occ_last_item), current_cobot_item])]
                time = distance / 1.3
                zone_occ_time[1] += time
                # If the cobot arrives while the picker is busy, it must wait.
                if zone_occ_time[0] <= TimeCobots[current_cobot] <= zone_occ_time[1]:
                    TimeCobots[current_cobot] = max(TimeCobots[current_cobot], zone_occ_time[1])

            # This picker is now busy with the current cobot. Overwrite the starting "occupied picker" time
            zone_occupied[current_cobot_zone]['occ_time'][0] = TimeCobots[current_cobot]

            # Find next item zone and continue to assist cobot until next zone changes
            if current_cobot_next_item in ['OutD0','OutD1']:
                next_item_zone = 99999
            else:
                next_item_zone = self.warehouseInstance.Pods[current_cobot_next_item].PickingZone

            while next_item_zone == current_cobot_zone:
                distance = self.warehouseInstance.distance_ij[tuple(Full_Route[current_cobot][0:2])]
                time = 3 + (distance / 1.3)
                TimeCobots[current_cobot] += time
                Full_Route[current_cobot] = Full_Route[current_cobot][1:]
                current_cobot_next_item = Full_Route[current_cobot][1]
                if current_cobot_next_item in ['OutD0', 'OutD1']:
                    next_item_zone = 99999
                else:
                    next_item_zone = self.warehouseInstance.Pods[current_cobot_next_item].PickingZone

            # The picker is finished after picking the last item for this cobot
            zone_occupied[current_cobot_zone]['occ_time'][1] = (TimeCobots[current_cobot] + 3)
            # Update the item that the picker finished at
            zone_occupied[current_cobot_zone]['last_item'] = Full_Route[current_cobot][0]

            # send cobot to next zone
            distance = self.warehouseInstance.distance_ij[tuple(Full_Route[current_cobot][0:2])]
            time = 3 + (distance / 2)
            TimeCobots[current_cobot] += time
            Full_Route[current_cobot] = Full_Route[current_cobot][1:]

            # If cobot ends in a packing station, then we must account for that and send cobot to next item
            if Full_Route[current_cobot][0] in ['OutD0','OutD1']:
                TimeCobots[current_cobot] = max(TimeCobots[current_cobot], TimePackers[current_cobot])
                TimePackers[current_cobot] = max(TimeCobots[current_cobot], TimePackers[current_cobot])
                # Now that both are ready, add time to unpack
                TimeCobots[current_cobot] += 20

                # The packer also adds the unload time, and also packing time for each item
                TimePackers[current_cobot] += 20 + (60 * OrdersPerBatch[current_cobot][BatchCounter[current_cobot]])
                # Keep track of which batch the cobot is fullfilling
                BatchCounter[current_cobot] += 1

                # Send cobot to next item
                if len(Full_Route[current_cobot]) > 1:
                    distance = self.warehouseInstance.distance_ij[tuple(Full_Route[current_cobot][0:2])]
                    time = 3 + (distance / 2)
                    TimeCobots[current_cobot] += time
                Full_Route[current_cobot] = Full_Route[current_cobot][1:]
        return max(TimePackers)

    # simulated annealing algorithm for dedicated storeage policy
    def saNeighborhood_T2(self, initialSolution, T = 100, T_end = 10, alpha = 0.8):
        '''
        uses simmulated annealing technique to improve the first solution given in task 1
        '''
        print("[SA] Start SA")
        s = initialSolution
        s_optimal = copy.deepcopy(s)
        makespan_s = self.calculateMakeSpan(s)
        makespan_s_optimal = copy.deepcopy(makespan_s)

        i = 0
        while T>T_end:
            i += 1
            # randomly select a cobot
            cobot = int(np.random.randint(0, len(initialSolution), 1))
            other_cobot = int(bool(cobot)==False)
            # randomly select a batch to drop
            dropped_batch_index = int(np.random.randint(0, len(s[cobot]), 1))
            dropped_batch = s[cobot][dropped_batch_index]
            s_prime = copy.deepcopy(s)
            s_prime[cobot].remove(dropped_batch)

            # if the batch contains multiple orders, the orders may get split up
            possible_batches = []
            if len(dropped_batch)>1:
                for L in range(1, len(dropped_batch) + 1):
                    for subset in itertools.combinations(dropped_batch, L):
                        possible_batches.append(list(subset))
            else:
                possible_batches.append(dropped_batch)
            which_cobot = other_cobot
            while len(possible_batches)>0:
                # pick a random batch
                adding_batch = possible_batches[int(np.random.randint(0, len(possible_batches), 1))]
                # add that batch to the other cobot
                s_prime[which_cobot].append(adding_batch)
                # remove this batch and any other containing that order
                dropping_batches = []
                for sub in possible_batches:
                    if any(item in adding_batch for item in sub) == True:
                        dropping_batches.append(sub)
                for drop in dropping_batches:
                    possible_batches.remove(drop)
                # flip cobot so next batch gets assigned to other one
                which_cobot = int(bool(which_cobot) == False)
            makespan_s_prime = self.calculateMakeSpan(s_prime)

            if (makespan_s_prime < makespan_s) or math.exp(-(makespan_s_prime-makespan_s)/T) > random.random():
                s = copy.deepcopy(s_prime)
                makespan_s = copy.deepcopy(makespan_s_prime)
            if makespan_s < makespan_s_optimal:
                s_optimal = copy.deepcopy(s)
                makespan_s_optimal = copy.deepcopy(makespan_s)

            T = T*alpha

            #print(str(makespan_s) + ':   ' + str(s))
            #print(str(makespan_s_optimal) + ':   ' + str(s_optimal))

            filename = 'Solutions/Task_2/'  + warehouse + '_' + str(round(makespan_s)) + '_solution_taks2_simmulated_annealing_neighborhood.xml'

            _demo.writeToXML(filename, s)
        return makespan_s_optimal

    # method that contains a pertubation strategy for task 2.3
    def perturbSA(self, iterations):
        '''
        uses a perturbation strategy to improve the solution even further
        '''
        overall_optimal = 10000000
        while iterations > 0:
            # Create initial solution
            OrdersTable = self.warehouseInstance.BatchesDF[['Batch']]
            BatchAssignCobot_List = [[],[]]
            # randomly select a cobot
            current_cobot = int(np.random.randint(0, 2, 1))

            while len(OrdersTable.Batch) > 0:
                # randomly select a batch to add
                add_batch_index = int(np.random.randint(0, len(OrdersTable.Batch), 1))
                add_batch = list(OrdersTable.Batch.iloc[add_batch_index])
                # Assign that batch to the cobot
                BatchAssignCobot_List[current_cobot].append(add_batch)
                # drop batch from orders
                dropping_rows = OrdersTable.Batch.apply(lambda x: any(item in add_batch for item in x))
                OrdersTable = OrdersTable[dropping_rows == False]

                # switch cobot
                current_cobot = int(bool(current_cobot) == False)

            # Feed this starting solution into SA algorithm
            makespan_s_optimal = self.saNeighborhood_T2(BatchAssignCobot_List)
            if makespan_s_optimal < overall_optimal:
                overall_optimal = copy.deepcopy(makespan_s_optimal)
            iterations -= 1

    # Adaptive Large Neighborhood Search in Mixed Storage Policy Warehouse
    def alNeighborhood(self, iterations = 50):
        '''
        adaptive large neighborhood search for mixed storage policy warehouse
        :param iterations: iterations of the alns
        '''
        # different solutions given in the FullBatchAssignment format, so that we can calculate the makespan on them)
        # s         initial solution (from task 3.1)
        # s_prime   candiate solution (after destroy and repair)
        # s_star    best solution (result of comparison between s and s_prime)
        s = self.BatchAssignCobot_List
        s_prime = copy.deepcopy(s)
        s_star = copy.deepcopy(s)

        # 0. destroy operator set
        destroy_operators = {'random_batch_subset': 1, 'random_cobot': 1}

        # 0. repair operator set
        repair_operators = {'most_items_batch_first': 1, 'fewest_items_batch_first': 1, 'random_fill': 1}#, 'shuffle_pods': 1}


        for it in range(iterations):
            # 1a. choose destroy operator (roulette)
            destroy_operator_choice = random.choices(list(destroy_operators.keys()),
                                                     weights=list(destroy_operators.values()))

            # 1b. destroy
            print(f'[ALNS-mixed][{it}] destroying solution ({destroy_operator_choice})')
            if destroy_operator_choice == ['random_cobot']:
                # choose random cobot and destroy all batches assigned to that cobot, append destroyed batches to the desotryed batches list
                random_cobot = int(np.random.randint(0, 2, 1))
                destroyed_batches = s_prime[random_cobot]
                s_prime[random_cobot] = []
            elif destroy_operator_choice == ['random_batch_subset']:
                # destroy each batch with probability of 0.5 and append the destroyed batch to the destroyed batches list
                destroyed_batches = []
                i = 0
                for cobot in s_prime:
                    j = 0
                    for batch in cobot:
                        rand = int(np.random.randint(0, 2, 1))
                        if rand == 0:
                            destroyed_batches.append(batch)
                            s_prime[i][j] = batch * rand
                        j += 1
                    i += 1

            # move all the batches to the beginning of the solution sequence, i.e. all the way to the left.
            s_prime = [[ele for ele in sub if ele != []] for sub in s_prime]

            # 2a. choose repair operator (roulette)
            repair_operator_choice = random.choices(list(repair_operators.keys()),
                                                     weights=list(repair_operators.values()))
            # 2b. repair
            print(f'[ALNS-mixed][{it}] repairing solution ({repair_operator_choice})')
            if repair_operator_choice == ['most_items_batch_first']:
                # get item count for each of the destroyed batches
                destroyed_batches_item_count_list = []
                for batch in destroyed_batches:
                    destroyed_batches_item_count_list.append(self.warehouseInstance.BatchesDF.loc[self.warehouseInstance.BatchesDF['Batch'].astype(str) == str(batch)]['ItemCount'].values[0])

                # sort by item count descending
                destroyed_batches = [x for _, x in sorted(zip(destroyed_batches_item_count_list, destroyed_batches))]
                destroyed_batches.reverse()

                # reassign batches to cobots, starting with the batch with the most items and the cobot with the least batches
                for i in destroyed_batches:
                    # find index of cobot with less items and assign the batch
                    cobot_assign = s_prime.index(min(s_prime, key=len))
                    s_prime[cobot_assign].append(i)

            elif repair_operator_choice == ['fewest_items_batch_first']:
                # get item count for each of the destroyed batches
                destroyed_batches_item_count_list = []
                for batch in destroyed_batches:
                    destroyed_batches_item_count_list.append(self.warehouseInstance.BatchesDF.loc[
                                                                 self.warehouseInstance.BatchesDF['Batch'].astype(
                                                                     str) == str(batch)]['ItemCount'].values[0])

                # sort by item count ascending
                destroyed_batches = [x for _, x in sorted(zip(destroyed_batches_item_count_list, destroyed_batches))]

                # reassign batches to cobots, starting with the batch with the most items and the cobot with the least batches
                for i in destroyed_batches:
                    # find index of cobot with less items and assign the batch
                    cobot_assign = s_prime.index(min(s_prime, key=len))
                    s_prime[cobot_assign].append(i)

            elif repair_operator_choice == ['random_fill']:
                #randomly shuffle elements of destroyed batches
                random.shuffle(destroyed_batches)
                # assign the batches to the cobots.
                for i in destroyed_batches:
                    # find index of cobot with less items and assign the batch
                    cobot_assign = s_prime.index(min(s_prime, key=len))
                    s_prime[cobot_assign].append(i)


            # 3. compare solutions and adapt operator weights
            omega = [5, 1.5, 0.8, 0.5]
            lmbda = 0.8
            makespan_s_prime = self.calculateMakeSpan(s_prime)
            if makespan_s_prime < self.calculateMakeSpan(s_star):
                print(f'[ALNS-mixed][{i}][SUCCESS] found better solution ({makespan_s_prime})')
                # update s_star
                s_star = copy.deepcopy(s_prime)
                # update weights
                destroy_operators[destroy_operator_choice[0]] = (lmbda * destroy_operators[destroy_operator_choice[0]]) + (omega[0] * (1 - lmbda))
                repair_operators[repair_operator_choice[0]] = (lmbda * repair_operators[repair_operator_choice[0]]) + (omega[0] * (1 - lmbda))
            elif makespan_s_prime < self.calculateMakeSpan(s):
                # update weights
                destroy_operators[destroy_operator_choice[0]] = (lmbda * destroy_operators[destroy_operator_choice[0]]) + (omega[1] * (1 - lmbda))
                repair_operators[repair_operator_choice[0]] = (lmbda * repair_operators[repair_operator_choice[0]]) + (omega[1] * (1 - lmbda))
            elif makespan_s_prime < self.calculateMakeSpan(s_star)*1.5: # we accept s_prime only when its at max double the time of s_star, but its not a better solution
                # update weights
                destroy_operators[destroy_operator_choice[0]] = (lmbda * destroy_operators[destroy_operator_choice[0]]) + (omega[2] * (1 - lmbda))
                repair_operators[repair_operator_choice[0]] = (lmbda * repair_operators[repair_operator_choice[0]]) + (omega[2] * (1 - lmbda))
            else: # we dont accept s prime
                # update weights
                destroy_operators[destroy_operator_choice[0]] = (lmbda * destroy_operators[destroy_operator_choice[0]]) + (omega[4] * (1 - lmbda))
                repair_operators[repair_operator_choice[0]] = (lmbda * repair_operators[repair_operator_choice[0]]) + (omega[4] * (1 - lmbda))

            filename = 'Solutions/Task_3/ALNS/' + warehouse + '_' + str(round(makespan_s_prime)) + '_solution_taks3.2_greedy_heuristic_mixed_policy_adaptive_large_neighborhood.xml'
            self.writeToXML(filename, s_prime)

            # 5. iterate again with s_prime

    # generate ItemID - PodLocation dictionary.
    def getPodforItems(self):
        ''':key
        method to collect the pod in which each item is located. can be adapoted for a mixed storage policy where items can be contained in multiple pods.
        '''
        print("[1] Calculating Pod Location for Items ")

        item_id_pod_id_dict = {} #initialize dictionary for item pod location
        for i in self.warehouseInstance.ItemDescriptions:
            itemPodID = self.warehouseInstance.ItemDescriptions[i].ItemPodID
            item_id_pod_id_dict[itemPodID] = []

            for j in self.warehouseInstance.Pods:
                for k in range(len(self.warehouseInstance.Pods[j].Items)):
                    if self.warehouseInstance.Pods[j].Items[k].ID == itemPodID:
                        item_id_pod_id_dict[itemPodID].append(j)


        return item_id_pod_id_dict  # a dictionary with item IDs as keys and a list of pod locations as value.

    # generated a Graph as an attribute of the warehouseInstance. Used to determine distances and trveling salesman problem
    def generateGraph(self):
        '''
        generating graph for networkx package
        '''
        print("[6] Generating Network Graph for Warehouse Instance")

        G = nx.Graph()
        # adding all nodes and edges to the network graph
        all_nodes = list(self.warehouseInstance.OutputStations) + list(self.warehouseInstance.Pods)
        G.add_nodes_from(all_nodes)
        G.add_edges_from(self.warehouseInstance.distance_ij)  ## weighted edged are taken in the format [[1,2,666],[2,3,5565],[],[],[]]
        # setting the edges weights.
        nx.set_edge_attributes(G, values=self.warehouseInstance.distance_ij, name='weight')

        nx.draw(G, with_labels=True)
        plt.savefig("network_img.png")

        if nx.is_connected(G) == True: ## has to be true always
            self.warehouseInstance.WarehouseGraph = G
        else:
            print('WarehouseGraph has unconnected nodes!')

        print(f"[6_b] Network has {G.number_of_nodes()} nodes")
        print(f"[6_c] Network has {G.number_of_edges()} edges")

    # writes the solution to a xml file in the desired structure
    def writeToXML(self, filename, BatchAssignCobot_List):
        '''
        writing a given solution to xml file in the required structure
        '''
        print("[XX] Exporting Solution to XML file")
        # base element
        root = ET.Element("root")
        # first section "split" contains information about which orders are in which batches, and which batches are assigned to which station (=bot)
        collecting = ET.SubElement(root, "Collecting")
        split = ET.SubElement(collecting, "Split")

        #adding first main section of xml file (Split)
        for bot in self.warehouseInstance.Bots:
            bot_id = ET.SubElement(split, "Bot")
            bot_id.set("ID", str(bot))   # adding cobot ID to the XML structure

            for batch in BatchAssignCobot_List[int(bot)]:
                batch_id = ET.SubElement(bot_id, "Batch")
                batch_id.set("ID", str(BatchAssignCobot_List[int(bot)].index(batch)+1))  # adding batch ID to the XML structure
                orders = ET.SubElement(batch_id, "Orders")

                for order in BatchAssignCobot_List[int(bot)][BatchAssignCobot_List[int(bot)].index(batch)]:
                    order_elem = ET.SubElement(orders, "Order")         #adding orders to the batches
                    order_elem.text = 'OC' + str(order)


        #adding second main section of xml file (Bots)
        bots = ET.SubElement(collecting, "Bots")

        for bot in self.warehouseInstance.Bots:
            bot_id = ET.SubElement(bots, "Bot")
            bot_id.set("ID", str(bot))
            bot_id.set("collect", "1")
            bot_id.set("refill", "-1")
            batch_id = ET.SubElement(bot_id, "Batches")

            for batch in BatchAssignCobot_List[int(bot)]:

                df_col_travelDist = 'travelDist_OutD' + bot
                batches = ET.SubElement(batch_id, "Batch")
                batches.set("BatchNumber", str(BatchAssignCobot_List[int(bot)].index(batch)+1))
                batches.set("Distance", str(self.warehouseInstance.BatchesDF.loc[self.warehouseInstance.BatchesDF['Batch'].astype(str) == str(batch)][df_col_travelDist].values[0]))

                items_data = ET.SubElement(batches, "ItemsData")

                orders = ET.SubElement(items_data, "Orders")
                # adding all orders to the xml file
                for order in self.warehouseInstance.Orders:
                    order_elem = ET.SubElement(orders, "Order")  # adding orders to the batches
                    order_elem.set("ID", 'OC' + str(order.OrderID))
                    if order.OrderID in BatchAssignCobot_List[int(bot)][BatchAssignCobot_List[int(bot)].index(batch)]:
                        #loop over all items in order
                        for item in self.warehouseInstance.Orders[order.OrderID].Positions:
                            for position in range(len(self.warehouseInstance.Orders[order.OrderID].Positions[item].Count)):
                                item_elem = ET.SubElement(order_elem, "Item")  # adding orders to the batches
                                item_elem.set("ID", 'C' + str(item) + '_' + str(position))
                                item_elem.set("Pod", self.warehouseInstance.ItemPodLocations[self.warehouseInstance.ItemDescriptions[item].ItemPodID][0])
                                item_elem.set("Type", self.warehouseInstance.ItemDescriptions[item].ItemPodID) ##


                edges = ET.SubElement(batches, "Edges")
                df_col_path = 'shortestRoute_OutD' + bot
                edges_list = self.warehouseInstance.BatchesDF.loc[self.warehouseInstance.BatchesDF['Batch'].astype(str) == str(batch)][df_col_path].values[0]
                for i in range(len(edges_list)-1):
                    edge_elem = ET.SubElement(edges, "Edge")  # adding orders to the batches
                    edge_elem.set("StartNode", edges_list[i])
                    edge_elem.set("EndNode", edges_list[i+1])



                waypoints = ET.SubElement(batches, "Waypoints")



        # write structure to xml file
        tree = ET.ElementTree(root)
        instance.indentXMLTree(root)
        tree.write(filename, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":

    # preparing warehouse attributes like network graph, batching, orders dataframes etc.
    _demo = Demo()
    _demo.prepareWarehouseInformation()
    _demo.calculateMakeSpan([[[1],[4]], [[2,3,7],[9]]])

    # Task 1.1 Greedy Heuristic
    _demo.greedyHeuristic_T1()
    makespan = _demo.calculateMakeSpan(_demo.BatchAssignCobot_List)
    filename = 'Solutions/Task_1/' + warehouse +'_' + str(round(makespan)) + '_solution_taks1_greedy_heuristic.xml'
    _demo.writeToXML(filename, _demo.BatchAssignCobot_List)

    # Task 2.1 Simmunaletd Annealing
    _demo.saNeighborhood_T2(_demo.BatchAssignCobot_List, T=100, alpha=0.8)
    _demo.perturbSA(3)



    # Task 3.1
    ## develop eveything for a mixed storage policy
    storagePolicies = {}
    if warehouse == '360':
        storagePolicies['mixed'] = 'data/sku360/pods_items_mixed_shevels_1-5.txt'
    else:
        storagePolicies['mixed'] = 'data/sku24/pods_items_mixed_shevels_1-5.txt'

    _demo_mixed = Demo()
    _demo_mixed.prepareWarehouseInformation('mixed')

    # Task 3.1 Greedy Heuristic for Mixed Policy
    _demo_mixed.greedyHeuristic_T1()                # the same method is called here, the mixed storage policy is simply implemented in the choosePodLocation() method of the warehouse class that gets called automatically when we use mixed storage policy.
    makespan_mixed = _demo_mixed.calculateMakeSpan(_demo.BatchAssignCobot_List)
    filename = 'Solutions/Task_3/' + warehouse + '_' + str(round(makespan_mixed)) + '_solution_taks3_greedy_heuristic_mixed_policy.xml'
    _demo_mixed.writeToXML(filename, _demo_mixed.BatchAssignCobot_List)

    # Task 3.2 Adaptive Large Neighborhood search for mixed policy warehouse
    _demo_mixed.alNeighborhood(100)

    print('[END] Finished Calculating All Tasks')
    # Her optimal solution
    # [[1],[4,9],[0,6]], [[2,3,7],[8],[5]]
