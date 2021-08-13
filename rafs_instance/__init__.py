import xml.etree.cElementTree as ET
import numpy as np
import copy
import pandas as pd

class PodLayout:
    def __init__(self, tierCount= None, tierHeight= None, humanCount= None, humanRad= None, humanMaxAcc= None, humanMaxDec= None,
                 humanMaxVel= None, humanTurnSpeed= None, boxCap= None, botCount= None, botRad= None, maxAcc= None, maxDec= None,
                 maxVel= None, TurnSpeed= None, CollPenTime= None, PodTransTime= None, PodAmount= None, PodRad= None, PodCap= None,
                 StatRad= None, ItemTransTime= None, ItemPickTime= None, ItemBunTransTime= None, IStatCapacity= None, OStatCapacity= None,
                 ElevatorTransp= None, AisleLayoutType= None, AislesTwoDirect= None, SingleLane= None, NameLayout= None, NrHoriAisles= None,
                 NrVertAisles= None, HoriLengthBlock= None, WidthHall= None, WidthBuffer= None, DistEntryExitStat= None,
                 CounterClockRingwayDirect= None, NPickStatWest= None, NPickStatEast= None, NPickStatSouth= None, NPickStatNorth= None,
                 NrReplenStatWest= None, NrReplenStatEast= None, NrReplenStatSouth= None, NrReplenStatNorth= None, NElevWest= None,
                 NElevEast= None, NElevSouth= None, NElevNorth= None):
    

        self.TierCount = tierCount
        self.TierHeight = tierHeight
        self.HumanCount = humanCount
        self.HumanRadius = humanRad
        self.HumanMaxAcceleration = humanMaxAcc
        self.HumanMaxDeceleration = humanMaxDec
        self.HumanMaxVelocity = humanMaxVel
        self.HumanTurnspeed = humanTurnSpeed
        self.BoxCapacity = boxCap
        self.BotCount = botCount
        self.BotRadius = botRad
        self.MaxAcceleration = maxAcc
        self.MaxDeceleration = maxDec
        self.MaxVelocity = maxVel
        self.TurnSpeed = TurnSpeed
        self.CollisionPenaltyTime = CollPenTime
        self.PodTransferTime = PodTransTime
        self.PodAmount = PodAmount
        self.PodRadius = PodRad
        self.PodCapacity = PodCap
        self.StationRadius = StatRad
        self.ItemTransferTime = ItemTransTime
        self.ItemPickTime = ItemPickTime
        self.ItemBundleTransferTime = ItemBunTransTime
        self.IStationCapacity = IStatCapacity
        self.OStationCapacity = OStatCapacity
        self.ElevatorTransportationTimePerTier = ElevatorTransp
        self.AisleLayoutType = AisleLayoutType
        self.AislesTwoDirectional = AislesTwoDirect
        self.SingleLane = SingleLane
        self.NameLayout = NameLayout
        self.NrHorizontalAisles = NrHoriAisles
        self.NrVerticalAisles = NrVertAisles
        self.HorizontalLengthBlock = HoriLengthBlock
        self.WidthHall = WidthHall
        self.WidthBuffer = WidthBuffer
        self.DistanceEntryExitStation = DistEntryExitStat
        self.CounterClockwiseRingwayDirection = CounterClockRingwayDirect
        self.NPickStationWest = NPickStatWest
        self.NPickStationEast = NPickStatEast
        self.NPickStationSouth = NPickStatSouth
        self.NPickStationNorth = NPickStatNorth
        self.NReplenishmentStationWest = NrReplenStatWest
        self.NReplenishmentStationEast = NrReplenStatEast
        self.NReplenishmentStationSouth = NrReplenStatSouth
        self.NReplenishmentStationNorth = NrReplenStatNorth
        self.NElevatorsWest = NElevWest
        self.NElevatorsEast = NElevEast
        self.NElevatorsSouth = NElevSouth
        self.NElevatorsNorth = NElevNorth

    def SetAttribute(self, attrName, attrVal):
        
        attributeNames = ['TierCount', 'TierHeight', 'HumanCount', 'HumanRadius', 'HumanMaxAcceleration', 'HumanMaxDeceleration',
                      'HumanMaxVelocity', 'HumanTurnspeed', 'BoxCapacity', 'BotCount', 'BotRadius', 'MaxAcceleration', 'MaxDeceleration',
                      'MaxVelocity', 'TurnSpeed', 'CollisionPenaltyTime', 'PodTransferTime', 'PodAmount', 'PodRadius', 'PodCapacity',
                      'StationRadius', 'ItemTransferTime', 'ItemPickTime','ItemBundleTransferTime', 'IStationCapacity', 'OStationCapacity',
                      'ElevatorTransportationTimePerTier', 'AisleLayoutType', 'AislesTwoDirectional', 'SingleLane', 'NameLayout',
                      'NrHorizontalAisles', 'NrVerticalAisles', 'HorizontalLengthBlock', 'WidthHall', 'WidthBuffer', 'DistanceEntryExitStation',
                      'CounterClockwiseRingwayDirection', 'NPickStationWest', 'NPickStationEast', 'NPickStationSouth', 'NPickStationNorth',
                      'NReplenishmentStationWest', 'NReplenishmentStationEast', 'NReplenishmentStationSouth', 'NReplenishmentStationNorth',
                      'NElevatorsWest', 'NElevatorsEast', 'NElevatorsSouth', 'NElevatorsNorth']
        #Set attribute only if in list
        if attrName in attributeNames:
            setattr(self, attrName, attrVal)    

class Pod:
    def __init__(self, podID = None, posX= 0, posY = 0, usedCapacity = None, maxCapacity = None, 
                 reservedCapacity = None, tag = None, readyForRefill = None, items = None, rad = None, orientation = None, 
                 tier = None):
        self.ID = podID
        self.Position = np.array((float(posX),float(posY)))
        self.UsedCapacity = usedCapacity
        self.MaxCapacity = maxCapacity
        self.ReservedCapacity = reservedCapacity
        self.Tag = tag        
        self.ReadyForRefill = readyForRefill
        if items is None:
            self.Items = []
        else:
            self.Items = items
        self.Radius = rad
        self.Orientation = orientation
        self.Tier = tier

class Bot:
    def __init__(self, ID, podTransTime, maxAcc, maxDec, maxVel, turnSpeed,
             collPenTime, posX, posY, rad, orientation, tier, cap, outputStation):
        self.ID = ID
        self.PodTransferTime = podTransTime
        self.MaxAcceleration = maxAcc
        self.MaxDeceleration = maxDec
        self.MaxVelocity = maxVel
        self.TurnSpeed = turnSpeed
        self.CollisionPenaltyTime = collPenTime
        self.Position = np.array((float(posX),float(posY)))
        self.Radius = rad
        self.Orientation = orientation
        self.Tier = tier
        self.Capacity = int(cap)
        self.OutputStation = 'OutD' + str(outputStation)

class Station:
    def __init__(self, ID, posX, posY, rad = None, tier = None):
        self.ID= ID
        self.Position = np.array((float(posX),float(posY)))
        self.Radius = rad
        self.Tier = tier
        
class ChargingStation(Station):
    def __init__(self, ID, posX, posY, tier):
        super().__init__(ID, posX, posY, tier = tier)

class PickLocation(Station):
    def __init__(self, ID, pod = None, posX = 0, posY = 0, rad = 0, orientation = None, tier = None):
        super().__init__(ID, posX, posY, rad, tier)
        self.Pod = pod
        self.Orientation = orientation

class InputStation(Station):
    def __init__(self, ID, posX, posY, rad, tier, cap, itemBundTransTime, actOrdID, queues = None):
        super().__init__(ID, posX, posY, rad, tier)
        self.Capacity = cap
        self.ItemBundleTransferTime = itemBundTransTime
        self.ActivationOrderID = actOrdID
        if queues is None:
            self.Queues = []
        else:
            self.Queues = queues
        
    def AddQueue(self,queue):
        self.Queues.append(queue)
    
class Queue:
    def __init__(self,text):
        self.Info = text

class OutputStation(Station):
    def __init__(self, ID, posX, posY, rad, tier, cap, itemTransTime, itemPickTime, actOrdID, queues = None):
        super().__init__(ID, posX, posY, rad, tier)
        self.Capacity = cap
        self.ItemTransferTime = itemTransTime
        self.ItemPickTime = itemPickTime
        self.ActivationOrderID = actOrdID
        if queues is None:
            self.Queues = []
        else:
            self.Queues = queues
        self.totalWeight = 0
        self.feasibleBatches = []
        self.feasibleBatchesDF = pd.DataFrame
        
    def AddQueue(self,queue):
        self.Queues.append(queue)

    def UpdateWeight(self):
        if len(self.Queues) == 0:
            self.totalWeight = 0
        else:
            weightList = []
            for i in range(len(self.Queues)):
                weightList.append(self.Queues[i].totalWeight)
        self.totalWeight = sum(weightList)

    def getFeasibleBatches(self): # returns: Dict{batchID, List[Orders]}:
        '''
        function that gets all feasible batches for a respective Output Station, depending on the orders that were assigned to it.
        '''

        # creating table with weights of orders
        weights = []
        orders = []
        for i in range(len(self.Queues)):
            weights.append(self.Queues[i].totalWeight)
            orders.append(i)

        weight_table = pd.DataFrame({'orders':orders, 'weights':weights})
        feasibleBatchesList = []
        target = []
        data = list(weight_table["orders"])
        cobotCapacity = 18 # spuld be drawn from dynamic variable (_demo.batch_weight)

        feasibleBatchesList = getFeasibleOrderCombinations(feasibleBatchesList, target, data, weight_table, cobotCapacity)

        self.feasibleBatches = feasibleBatchesList
        self.feasibleBatchesDF = pd.DataFrame({0: feasibleBatchesList})



class Elevator:
    def __init__(self,ID):
        self.ID = ID

class Tier:
    def __init__(self, ID, length, width, relPosX, relPosY, relPosZ):
        self.ID = ID
        self.Length = length
        self.Width = width
        self.RelPosX = relPosX
        self.RelPosY = relPosY
        self.RelPosZ = relPosZ
        
class Waypoint:
    def __init__(self, ID, posX, posY, tier, outStat, inStat, elevator, 
                 pod, depotStat, chargeStat, podStorLoc, isQueWaypoint, paths = None):
        self.ID = ID
        self.Position = np.array((float(posX),float(posY)))
        self.Tier = tier
        self.OutputStation = outStat
        self.InputStation = inStat
        self.Elevator = elevator
        self.Pod = pod
        self.PickLocation = depotStat
        self.ChargingStation = chargeStat
        self.PodStorageLocation = podStorLoc
        self.IsQueueWaypoint = isQueWaypoint
        if paths is None:
            self.Paths = []
        else:
            self.Paths = paths
        
class Semaphore:
    def __init__(self, ID, cap, guards = None):
        self.ID = ID
        self.Capacity = cap
        if guards is None:
            self.Guards = []
        else:
            self.Guards = guards
         
class Guard:
    def __init__(self, ID, fromNum, toNum, entry, barrier, semaphore):
        self.ID = ID
        self.From = fromNum
        self.To = toNum
        self.Entry = entry
        self.Barrier = barrier
        self.Semaphore = semaphore
          

#%% Classes for orders
#Class Order
class Order:
    def __init__(self,timeStamp):
        self.TimeStamp = timeStamp
        self.Positions = {}
        self.totalWeight = None

    def AddPosition(self, pos):
        self.Positions[pos.ItemDescID] = pos

    def getTotalWeight(self, totalWeightList):
        self.totalWeight = sum(totalWeightList)
        
#Class Item Bundle
class ItemBundle:
    def __init__(self,timeStamp,itemDesc,size):
        self.TimeStamp = timeStamp
        self.ItemDescription = itemDesc
        self.Size = size
        
#Class Item Description
class ItemDescription:
    def __init__(self,ID,type,weight,letter,color):
        self.ID = ID
        self.Type = type
        self.Weight = float(weight)
        self.Letter = letter
        self.Color = color
        self.ItemPodID = self.Color + '/' + self.Letter
        
#Container for item position and count
class ItemPosition:
    def __init__(self, ID, count):
        self.ID = ID
        self.Count = count

#Container for item position and count
class OrderItemPosition:
    def __init__(self, ID, count):
        self.ItemDescID = ID
        self.Count = count     


#Warehouse class, that holds all information about the warehouse


######### FUNCTIONS
def getFeasibleOrderCombinations(feasibleBatchesList, target, data, weight_table, cobotCapacity):
    for i in range(len(data)):
        new_target = copy.copy(target)
        new_target.append(data[i])
        new_data = data[i + 1:]
        weights = weight_table.loc[new_target]["weights"]
        if weights.sum() <= cobotCapacity:
            if new_target not in feasibleBatchesList:
                feasibleBatchesList.append(new_target)
        else:
            new_target = new_target[:-1]
        getFeasibleOrderCombinations(feasibleBatchesList, new_target, new_data, weight_table, cobotCapacity)
    return feasibleBatchesList
#######################


## WAREHOUSE CLASS
class Warehouse:
    def __init__(self, layoutFile, instanceFile, podInfoFile, podItemFile, orderFile):
        self.InstanceFile = instanceFile
        self.Bots = None
        self.Pods = None
        self.ChargingStations = None
        self.PickLocations = None
        self.InputStations = None
        self.Elevators = None
        self.OutputStations = None
        self.Tiers = None
        self.Waypoints = None
        self.Semaphores = None
        self.PodLayout = None
        self.ItemDescriptions = None
        self.Orders = None
        self.openOrders = None
        self.assignedOrders = None
        self.ItemBundles = None
        self.feasibleBatches = None
        
        #Import layout
        self.ImportLayout(layoutFile)
        
        #Import instance
        self.ImportInstance(instanceFile)
        
        #Import pod info
        self.ImportPods(instanceFile, podInfoFile, podItemFile)
        
        #Import oders
        self.ImportOrders(orderFile)
        self.openOrders = copy.deepcopy(self.Orders)

        # set assignedOrders
        self.assignedOrders = []

        self.ItemPodLocations = self.getPodsforItems()

        #self.openOrders.pop(-1)
        #self.openOrders.pop(-1)
        #self.openOrders.pop(-1)


    #Import functions
    #Import layout functions
    def ImportLayout(self, file): 
        #Parse xml file with order information
        tree = ET.parse(file)
        root = tree.getroot()
        
        #Create pod layout
        podLayout = PodLayout()
        
        #Loop over all items 
        for item in root:
            podLayout.SetAttribute(item.tag,item.text)
        
        self.PodLayout = podLayout    
     
    #Import instance

    def ImportInstance(self, file):
    
        tree = ET.parse(file)
        root = tree.getroot()
             
        #Import bots
        #Dictionary for all bots
        bots = {}
        for bot in root.iter('Bot'):
            botObj = Bot(bot.get('ID'),bot.get('PodTransferTime'),bot.get('MaxAcceleration'),bot.get('MaxDeceleration'),
                      bot.get('MaxDeceleration'),bot.get('TurnSpeed'),bot.get('CollisionPenaltyTime'),bot.get('X'), bot.get('Y'),
                      bot.get('Radius'), bot.get('Orientation'), bot.get('Tier'),bot.get('Capacity'), bot.get('OutputStation'))
            bots[bot.get('ID')] = botObj
        
        self.Bots = bots
        
        #Import charging stations
        #Dictionary for all charging stations
        chargeStations = {}
        for chargeStat in root.iter('Chargingstation'):
            chargeStatObj = ChargingStation(chargeStat.get('ID'), chargeStat.get('X'), chargeStat.get('Y'),
                                            chargeStat.get('Tier'))
            chargeStations [chargeStat.get('ID')] = chargeStatObj
            
        self.ChargingStations = chargeStations
        
        #Import depot stations
        #Dictionary for all depot stations
        picklocations = {}
        for depotStat in root.iter('PickLocation'):
            depotStatObj = PickLocation(depotStat.get('ID'), depotStat.get('Pod'), depotStat.get('X'), depotStat.get('Y'),
                                            depotStat.get('Radius'), depotStat.get('Orientation'), depotStat.get('Tier'))
            picklocations [depotStat.get('ID')] = depotStatObj
        
        self.PickLocations  = picklocations
        
        
        #Import elevators
        #TBD: No example file yet    
            
        #Import input stations
        #Dictionary for all input stations
            
        inputStations = {}
        for inputStat in root.iter('InputStation'):
            inID = 'InD' + inputStat.get('ID')
            inputStatObj = InputStation(inID, inputStat.get('X'), inputStat.get('Y'), inputStat.get('Radius'),
                                            inputStat.get('Tier'), inputStat.get('Capacity'), inputStat.get('ItemBundleTransferTime'),
                                            inputStat.get('ActivationOrderID'))
            
            inputStations[inID] = inputStatObj
        
            #Loop over queues
            for queue in root.iter('Queues'):
                info = queue.text
                info = info.strip()
                if info != "":
                    queueObj = Queue(queue.text)
                    inputStatObj.AddQueue(queueObj)
                
        self.InputStations = inputStations
        
        #Import output stations
        #Dictionary for all output stations
        outputStations = {}
        for outputStat in root.iter('OutputStation'):
            outID = 'OutD' + outputStat.get('ID')
            outputStatObj = OutputStation(outID, outputStat.get('X'), outputStat.get('Y'), outputStat.get('Radius'),
                                            outputStat.get('Tier'), outputStat.get('Capacity'), outputStat.get('ItemTransferTime'),
                                            outputStat.get('ItemPickTime'), outputStat.get('ActivationOrderID'))
            outputStations[outID] = outputStatObj
            #Loop over queues
            for queue in root.iter('Queues'):
                info = queue.text
                info = info.strip()
                if info != "":
                    queueObj = Queue(queue.text)
                    outputStatObj.AddQueue(queueObj)
         
        self.OutputStations = outputStations
            
        #Import tiers
        #Dictionary for all tiers
        tiers = {}
        
        for tier in root.iter('Tier'):
            tierObj = Tier(tier.get('ID'),tier.get('Length'),tier.get('Width'),tier.get('RelativePositionX'),
                           tier.get('RelativePositionY'),tier.get('RelativePositionZ'))
            tiers[tier.get('ID')] = tierObj
        
        self.Tiers = tiers
        
        #Import waypoints
        #Dictionary for all waypoints
        waypoints = {}  
        
        
        for waypoint in root.findall('Waypoints/Waypoint'):
            #Retrieve paths
            paths = []
            for pathWP in waypoint.findall('Paths/Waypoint'):
                paths.append(pathWP.text)
                
            waypointObj = Waypoint(waypoint.get('ID'), waypoint.get('X'), waypoint.get('Y'), waypoint.get('Tier'),
                                   waypoint.get('OutputStation'),waypoint.get('InputStation'),waypoint.get('Elevator'),
                                   waypoint.get('Pod'),waypoint.get('PickLocation'),waypoint.get('Chargingstation'),waypoint.get('PodStorageLocation'),
                                   waypoint.get('IsQueueWaypoint'), paths)    
            waypoints[waypoint.get('ID')] = waypointObj

                
        self.Waypoints = waypoints
        
        #Import semaphores
        #Dictionary for all semaphores
        semaphores={}
        for semaphore in root.findall('Semaphores/Semaphore'):
            #Guards
            guards = []
            
            for guard in semaphore.findall('Guards/Guard'):
                guardObj = Guard(None, guard.get('From'), guard.get('To'), guard.get('Entry'), guard.get('Barrier'), guard.get('Semaphore'))
                guards.append(guardObj)
                
            semaphoreObj = Semaphore(semaphore.get('ID'),semaphore.get('Capacity'), guards)
            semaphores[semaphore.get('ID')] = semaphoreObj
        self.Semaphores = semaphores
        
    # Import pod and pod infos
    def ImportPods(self, instanceFile, podInfoFile, podItemFile):
    
        #Import pods
        tree = ET.parse(instanceFile)
        root = tree.getroot()
             
        #Dictionary for all pods
        pods = {}
        for pod in root.iter('Pod'):
            podObj = Pod(podID = pod.get('ID'), posX = pod.get('X'), posY = pod.get('Y'), rad = pod.get('Radius'),
                         orientation = pod.get('Orientation'), tier = pod.get('Tier'), maxCapacity = pod.get('Capacity'))
            pods[pod.get('ID')] = podObj
    
        #Import pod infos 
        with open(podInfoFile) as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        
        for line in content:
             split=line.split(';')
             podID = split[0]
             #pos = split[1].split('/')
             cap = split[2].split(':')[1].split('/')
             resCap = split[3].split(':')[1].split('/')
             tag = split[4].split(':')
             refill = split[5].split(':')
             
             #Create one object per pod and store it in dictionary
             #PodID,PosX,PosY,UsedCap,MaxCap,ReservedCap,Tag,ReadyRefill
             #pod = Pod(split[0], pos[0], pos[1],cap[0],cap[1], resCap[0],tag[1],refill[1] in ['True'])
             
             #Find pod in list
             if podID in pods:
                 pods[podID].UsedCapacity = cap[0]
                 pods[podID].ReservedCapacity = resCap[0]
                 pods[podID].Tag = tag[1]
                 pods[podID].ReadyForRefill = refill[1] in ['True']
                 pods[podID].Items = []
    
        #Import Inventory of pods
        with open(podItemFile) as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        j = 0
        for line in content:
             split=line.split(';')
             
             for i in list(range(2,len(split))):
                 if split[i] != '':
                     item = split[i].split('/')
                     
                     color = item[0]
                     letter = item[1] 
                     count =  item [2]
                     
                     itemPos = ItemPosition(color + '/' + letter, count)
                     
                     #Add item to pod
                     podID = str(split[0])
                     
                     if podID in pods:
                         pods[podID].Items.append(itemPos)
                         j += 1
                     
        self.Pods = pods

    #  Import orders
    #Function for importing orders and itemDescription
    def ImportOrders(self, file):
        
        
        #Create dictionary that holds all item descriptions
        itemDescriptions = {}
        
        #List that holds all orders
        allOrders = []   
                            
        #Loop over item bundles
        itemBundles = []
        
        #Parse xml file with order information
        tree = ET.parse(file)
        root = tree.getroot()
             
        #Loop over item descriptions
        
        for itemDesc in root.iter('ItemDescription'):
            itemDescObj = ItemDescription(itemDesc.get('ID'),itemDesc.get('Type'),
                                          itemDesc.get('Weight'),itemDesc.get('Letter'),itemDesc.get('Color'))
            itemDescriptions[itemDesc.get('ID')] = itemDescObj
            
        
        #Loop over orders  
        for order in root.iter('Order'):
            #Create order object and append it to order list        
            orderObj = Order(order.get('TimeStamp')) 
            #Loop over positions in order

            totalWeightList = []

            for pos in order.iter('Position'):
                
                #TBD:
                #Object for item position
                orderItemPos = OrderItemPosition(pos.get('ItemDescriptionID'),pos.get('Count'))


                #Attach item position to order
                orderObj.AddPosition(orderItemPos)

                weightOfPosition = itemDescriptions[str(orderItemPos.ItemDescID)].Weight * int(orderItemPos.Count)

                totalWeightList.append(weightOfPosition)

                #Object for item position
            orderObj.getTotalWeight(totalWeightList)
            allOrders.append(orderObj)

        for itemBun in root.iter('ItemBundle'):
            itemBunObj = ItemBundle(itemBun.get('TimeStamp'),itemBun.get('ItemDescriptionID'),itemBun.get('Size')) 
            itemBundles.append(itemBunObj)
            
            
        self.ItemDescriptions = itemDescriptions
        self.Orders = allOrders
        self.ItemBundles = itemBundles

    def getPodsforItems(self):
        ''':key
        method to collect the pod in which each item is located. can be adapoted for a mixed storage policy where items can be contained in multiple pods.
        '''

        item_id_pod_id_dict = {} #initialize dictionary for item pod location
        for i in self.ItemDescriptions:
            itemPodID = self.ItemDescriptions[i].ItemPodID
            item_id_pod_id_dict[itemPodID] = []

            for j in self.Pods:
                for k in range(len(self.Pods[j].Items)):
                    if self.Pods[j].Items[k].ID == itemPodID:
                        item_id_pod_id_dict[itemPodID].append(j)


        return item_id_pod_id_dict  # a dictionary with item IDs as keys and a list of pod locations as value.

    def getItemsInBatch(self, batch, outputStationKey):
        '''
        calculates the route and travel time of each batch, based on the batch and the output station
        :return: travel time and route
        '''
        #INPUTS:
        Batch = [1,2]       ## draw this from function call

        # get list of order of items in batch
        # TODO: hier müssen wir die orders vom batch finden und die items in eine liste schreiben)
        # get orderpositions from orderQueue inside of OutputStation
        itemsInBatch = []
        for i in Batch:     # the orders in batch
            for j in self.OutputStations[outputStationKey].feasibleBatches:#Orders[i].Positions:
                itemsInBatch.append(self.Orders[i].Positions[str(j)])



        # accumulate items
        ItemsDict = {}
        for j in itemsInBatch:
            ItemsDict[j.ItemDescID] = 0
        for k in itemsInBatch:
            ItemsDict[k.ItemDescID] = int(ItemsDict[k.ItemDescID]) + int(k.Count)

        # get ItemPodID for ItemID
        ItemPodID = {}
        for i in ItemsDict:
            ItemPodID[i] = self.ItemDescriptions[i].ItemPodID

        # get the pod ID for each Item
        PodID = {}
        for i in ItemsDict:
            PodID[i] = self.ItemPodLocations[ItemPodID[i]]

        # collecting all information in dataframe
        batchItemsDF = pd.DataFrame({'items':ItemsDict.keys(), 'quantity':ItemsDict.values(), 'ItemPodID':ItemPodID.values(), 'PodID': PodID.values()})



    # getting a list of feasible batches of open orders in the warehouse
    def getFeasibleBatchesWH(self): # returns: Dict{batchID, List[Orders]}:
        '''
        function that gets all feasible batches for a respective Output Station, depending on the orders that were assigned to it.
        '''

        # creating table with weights of orders
        weights = []
        orders = []
        for i in range(len(self.openOrders)):
            weights.append(self.openOrders[i].totalWeight)
            orders.append(i)

        weight_table = pd.DataFrame({'orders':orders, 'weights':weights})
        feasibleBatchesList = []
        target = []
        data = list(weight_table["orders"])
        cobotCapacity = 18 # spuld be drawn from dynamic variable (_demo.batch_weight)

        feasibleBatchesList = getFeasibleOrderCombinations(feasibleBatchesList, target, data, weight_table, cobotCapacity)

        self.feasibleBatches = feasibleBatchesList
        self.feasibleBatchesDF = pd.DataFrame({0: feasibleBatchesList})






#Warehouse class, that holds all information about the warehouse
class WarehouseLite:
    def __init__(self):
        self.Bots = None
        self.Pods = None
        self.ChargingStations = None
        self.DepotStations = None
        self.InputStations = None
        self.Elevators = None
        self.OutputStations = None
        self.Tiers = None
        self.Waypoints = None
        self.Semaphores = None
        self.PodLayout = None
        self.ItemDescriptions = None
        self.Orders = None
        self.ItemBundles = None
        
        #Import oders
    def ImportInstance(self, file):
    
        tree = ET.parse(file)
        root = tree.getroot()
             
        #Import bots
        #Dictionary for all bots
        bots = {}
        for bot in root.iter('Bot'):
            botObj = Bot(bot.get('ID'),bot.get('PodTransferTime'),bot.get('MaxAcceleration'),bot.get('MaxDeceleration'),
                      bot.get('MaxDeceleration'),bot.get('TurnSpeed'),bot.get('CollisionPenaltyTime'),bot.get('X'), bot.get('Y'),
                      bot.get('Radius'), bot.get('Orientation'), bot.get('Tier'),bot.get('Capacity'), bot.get('OutputStation'))
            bots[bot.get('ID')] = botObj
        
        self.Bots = bots
        
        #Import charging stations
        #Dictionary for all charging stations
        chargeStations = {}
        for chargeStat in root.iter('Chargingstation'):
            chargeStatObj = ChargingStation(chargeStat.get('ID'), chargeStat.get('X'), chargeStat.get('Y'),
                                            chargeStat.get('Tier'))
            chargeStations [chargeStat.get('ID')] = chargeStatObj
            
        self.ChargingStations = chargeStations
        
        #Import depot stations
        #Dictionary for all depot stations
        depotStations = {}
        for depotStat in root.iter('PickLocation'):
            depotStatObj = PickLocation(depotStat.get('ID'), depotStat.get('Pod'), depotStat.get('X'), depotStat.get('Y'),
                                            depotStat.get('Radius'), depotStat.get('Orientation'), depotStat.get('Tier'))
            depotStations [depotStat.get('ID')] = depotStatObj
        
        self.DepotStations  = depotStations
        
        
        #Import elevators
        #TBD: No example file yet    
            
        #Import input stations
        #Dictionary for all input stations
            
        inputStations = {}
        for inputStat in root.iter('InputStation'):
            inID = 'InD' + inputStat.get('ID')
            inputStatObj = InputStation(inID, inputStat.get('X'), inputStat.get('Y'), inputStat.get('Radius'),
                                            inputStat.get('Tier'), inputStat.get('Capacity'), inputStat.get('ItemBundleTransferTime'),
                                            inputStat.get('ActivationOrderID'))
            
            inputStations[inID] = inputStatObj
        
            #Loop over queues
            for queue in root.iter('Queues'):
                info = queue.text
                info = info.strip()
                if info != "":
                    queueObj = Queue(queue.text)
                    inputStatObj.AddQueue(queueObj)
                
        self.InputStations = inputStations
        
        #Import output stations
        #Dictionary for all output stations
        outputStations = {}
        for outputStat in root.iter('OutputStation'):
            outID = 'OutD' + outputStat.get('ID')
            outputStatObj = OutputStation(outID, outputStat.get('X'), outputStat.get('Y'), outputStat.get('Radius'),
                                            outputStat.get('Tier'), outputStat.get('Capacity'), outputStat.get('ItemTransferTime'),
                                            outputStat.get('ItemPickTime'), outputStat.get('ActivationOrderID'))
            outputStations[outID] = outputStatObj
            #Loop over queues
            for queue in root.iter('Queues'):
                info = queue.text
                info = info.strip()
                if info != "":
                    queueObj = Queue(queue.text)
                    outputStatObj.AddQueue(queueObj)
         
        self.OutputStations = outputStations
            
        #Import tiers
        #Dictionary for all tiers
        tiers = {}
        
        for tier in root.iter('Tier'):
            tierObj = Tier(tier.get('ID'),tier.get('Length'),tier.get('Width'),tier.get('RelativePositionX'),
                           tier.get('RelativePositionY'),tier.get('RelativePositionZ'))
            tiers[tier.get('ID')] = tierObj
        
        self.Tiers = tiers
        
        #Import waypoints
        #Dictionary for all waypoints
        waypoints = {}  
        
        
        for waypoint in root.findall('Waypoints/Waypoint'):
            #Retrieve paths
            paths = []
            for pathWP in waypoint.findall('Paths/Waypoint'):
                paths.append(pathWP.text)
                
            waypointObj = Waypoint(waypoint.get('ID'), waypoint.get('X'), waypoint.get('Y'), waypoint.get('Tier'),
                                   waypoint.get('OutputStation'),waypoint.get('InputStation'),waypoint.get('Elevator'),
                                   waypoint.get('Pod'),waypoint.get('PickLocation'),waypoint.get('Chargingstation'),waypoint.get('PodStorageLocation'),
                                   waypoint.get('IsQueueWaypoint'), paths)    
            waypoints[waypoint.get('ID')] = waypointObj

                
        self.Waypoints = waypoints
        
        #Import semaphores
        #Dictionary for all semaphores
        semaphores={}
        for semaphore in root.findall('Semaphores/Semaphore'):
            #Guards
            guards = []
            
            for guard in semaphore.findall('Guards/Guard'):
                guardObj = Guard(None, guard.get('From'), guard.get('To'), guard.get('Entry'), guard.get('Barrier'), guard.get('Semaphore'))
                guards.append(guardObj)
                
            semaphoreObj = Semaphore(semaphore.get('ID'),semaphore.get('Capacity'), guards)
            semaphores[semaphore.get('ID')] = semaphoreObj
        self.Semaphores = semaphores

    #  Import orders
    #Function for importing orders
    def ImportOrders(self, file):
        
        #Create dictionary that holds all item descriptions
        itemDescriptions = {}
        
        #List that holds all orders
        allOrders = []   
                            
        #Loop over item bundles
        itemBundles = []
        
        #Parse xml file with order information
        tree = ET.parse(file)
        root = tree.getroot()
             
        #Loop over item descriptions
        
        for itemDesc in root.iter('ItemDescription'):
            itemDescObj = ItemDescription(itemDesc.get('ID'),itemDesc.get('Type'),
                                          itemDesc.get('Weight'),itemDesc.get('Letter'),itemDesc.get('Color'))
            itemDescriptions[itemDesc.get('ID')] = itemDescObj
            
        
        #Loop over orders  
        for order in root.iter('Order'):
            #Create order object and append it to order list        
            orderObj = Order(order.get('TimeStamp')) 
            #Loop over positions in order
            for pos in order.iter('Position'):
                
                #TBD:
                #Object for item position
                orderItemPos = OrderItemPosition(pos.get('ItemDescriptionID'),pos.get('Count'))
                
                #Attach item position to order
                orderObj.AddPosition(orderItemPos)
            
                #Object for item position
            allOrders.append(orderObj)   
                
        for itemBun in root.iter('ItemBundle'):
            itemBunObj = ItemBundle(itemBun.get('TimeStamp'),itemBun.get('ItemDescriptionID'),itemBun.get('Size')) 
            itemBundles.append(itemBunObj)
            
            
        self.ItemDescriptions = itemDescriptions
        self.Orders = allOrders
        self.ItemBundles = itemBundles

    # Import pod and pod infos
    def ImportPods(self, instanceFile, podInfoFile, podItemFile):
    
        #Import pods
        tree = ET.parse(instanceFile)
        root = tree.getroot()
             
        #Dictionary for all pods
        pods = {}
        for pod in root.iter('Pod'):
            podObj = Pod(podID = pod.get('ID'), posX = pod.get('X'), posY = pod.get('Y'), rad = pod.get('Radius'),
                         orientation = pod.get('Orientation'), tier = pod.get('Tier'), maxCapacity = pod.get('Capacity'))
            pods[pod.get('ID')] = podObj
    
        #Import pod infos 
        with open(podInfoFile) as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        
        for line in content:
             split=line.split(';')
             podID = split[0]
             #pos = split[1].split('/')
             cap = split[2].split(':')[1].split('/')
             resCap = split[3].split(':')[1].split('/')
             tag = split[4].split(':')
             refill = split[5].split(':')
             
             #Create one object per pod and store it in dictionary
             #PodID,PosX,PosY,UsedCap,MaxCap,ReservedCap,Tag,ReadyRefill
             #pod = Pod(split[0], pos[0], pos[1],cap[0],cap[1], resCap[0],tag[1],refill[1] in ['True'])
             
             #Find pod in list
             if podID in pods:
                 pods[podID].UsedCapacity = cap[0]
                 pods[podID].ReservedCapacity = resCap[0]
                 pods[podID].Tag = tag[1]
                 pods[podID].ReadyForRefill = refill[1] in ['True']
                 pods[podID].Items = []
    
        #Import Inventory of pods
        with open(podItemFile) as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        j = 0
        for line in content:
             split=line.split(';')
             
             for i in list(range(2,len(split))):
                 if split[i] != '':
                     item = split[i].split('/')
                     
                     color = item[0]
                     letter = item[1] 
                     count =  item [2]
                     
                     itemPos = ItemPosition(color + '/' + letter, count)
                     
                     #Add item to pod
                     podID = str(split[0])
                     
                     if podID in pods:
                         pods[podID].Items.append(itemPos)
                         j += 1
                     
        self.Pods = pods

    # Import pod and pod infos
    def ImportPodsFromLayout(self, instanceFile):
    
        #Import pods
        tree = ET.parse(instanceFile)
        root = tree.getroot()
             
        #Dictionary for all pods
        pods = {}
        for pod in root.iter('Pod'):
            podObj = Pod(podID = pod.get('ID'), posX = pod.get('X'), posY = pod.get('Y'), rad = pod.get('Radius'),
                         orientation = pod.get('Orientation'), tier = pod.get('Tier'), maxCapacity = pod.get('Capacity'))
            pods[pod.get('ID')] = podObj
    
        self.Pods = pods
