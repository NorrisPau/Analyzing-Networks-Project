#Tryout Nora

'''
What we want:
<root>
  <Collecting>
     <Split>
        <Bot ID="0">
           <Batch ID="1">
              <Orders>
                 <Order>OC2</Order>
              </Orders>
              ...
'''

print(_demo.warehouseInstance.BatchesDF)
for col in _demo.warehouseInstance.BatchesDF:
   print(col)
'''
Batch, BatchID, ItemsinBatch, ItemsinBatchDict
ItemCount, OrderCount. ItemPodID, PodIDs, StationsToVisit, shortestRoute_OutD0
travelTime_OutD0, travelTimeperOrder_OutD0, shortestRoute_OutD1, travelTime_OutD1, travelTimeperOrder_OutD1
'''
print(_demo.BatchAssignCobot_List)
# [[[2], [3, 7], [1], [9]], [[0], [4, 6], [5], [8]]]
# list of lists:
#list 1: Cobot 0
   # [[2], [3, 7], [1], [9]]
       #[2] = OC2 -> Batch in _demo.warehouseInstance.BatchesDF -> get BatchID
#list 2: Cobot 1

# Give Bot ID = 0 for list 1 and BotID = 1 for list 2


#Batches per Cobot
for i in _demo.BatchAssignCobot_List:
    print(i)
#cobot 0
_demo.BatchAssignCobot_List[0]
#cobot 1
_demo.BatchAssignCobot_List[1]


#Order per Batch per Cobot
for batch in _demo.BatchAssignCobot_List:
    for order in batch:
        print(order)


#Other Idea: Build a nested dicionary and turn dictionary into xml


root = ET.Element("root")
# first section "split" contains information about which orders are in which batches, and which batches are assigned to which station (=bot)
collecting = ET.SubElement(root, "Collecting")
split = ET.SubElement(collecting, 'Split')
#packingStationNames = # TODO: we don't divide by station but batches, so this would be batchesIds?
bot_id =




def writeToXML(self):
    # base element
    root = ET.Element("root")
    # first section "split" contains information about which orders are in which batches, and which batches are assigned to which station (=bot)
    collecting = ET.SubElement(root, "Collecting")
    split = ET.SubElement(collecting, 'Split')
    #packingStationNames = result.keys() #TODO
    # write each station as a sub-node of split
    for station in packingStationNames:
        Bot_ID = ET.SubElement(split, "Bot")
        Bot_ID.set("ID", station  # TODO: find this information in our class strcuture
        # filter the solution so it only contains batches for the right station
        stationSolution = result[station]
        # write each batch as a sub-node of Bot_ID
        batchID = 1
        for batch in stationSolution:
            Batch_ID = ET.SubElement(Bot_ID, "Batch")
        Batch_ID.set("ID", str(batchID))  # TODO: find this information in our class strcuture
        batchID += 1
        # write Orders as the sub-node of Batch_ID
        Orders = ET.SubElement(Batch_ID, "Orders")
        # write each order as sub-node of Batch_ID
        for order in batch["ordersInBatch"]:
            Order = ET.SubElement(Orders, "Order")
        Order.text = str(order)  # TODO: find this information in our class strcuture



        # first section "bots" contains detailed information about each bot (station)
        bots = ET.SubElement(collecting, "Bots")
        # write each station as a sub-node of bots
        for station in packingStationNames:
            Bot_ID = ET.SubElement(bots, "Bot")
        Bot_ID.set("ID", station)  # TODO: find this information in our class strcuture
        stationSolution = result[station]
        # batches are written in sub-node Batches of Bot_ID
        Batches = ET.SubElement(Bot_ID, "Batches")
        # write each batch as a sub-node of Bot_ID
        batchID = 1
        for batch in stationSolution:
            Batch_ID = ET.SubElement(Batches, "Batch")
        Batch_ID.set("BatchNumber", str(batchID))  # TODO: find this information in our class strcuture
        Batch_ID.set("Distance", str(batch["distance"]))  # TODO: find this information in our class strcuture
        Batch_ID.set("Weight", str(batch["weight"]))  # TODO: find this information in our class strcuture
        batchID += 1
        # for each batch, write two sub-nodes: itemsData, edges
        # first write ItemsData
        ItemsData = ET.SubElement(Batch_ID, "ItemsData")
        # ItemsData has a sub-node called Orders
        Orders = ET.SubElement(ItemsData, "Orders")
        # write each order as sub-node of Orders:
        for order in batch["ordersInBatch"]:
            Order = ET.SubElement(Orders, "Order")
        Order.set("ID", str(order))  # TODO: find this information in our class strcuture

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





print(_demo.TimeCobot_List)
# [381.4, 377.7]

print(_demo.TimePacker_List)
#[441.4, 437.7]