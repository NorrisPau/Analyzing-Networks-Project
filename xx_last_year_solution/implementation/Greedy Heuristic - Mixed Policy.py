# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:49:11 2020

@author: 93cha
"""

from instance_demo import *

itemAmount = 24
import pandas as pd
import numpy as np

#%%
def F_itemsDescInPod(itemDesc):
    for i in range(len(warehouseInstance.Pods[str(podID)].__dict__["Items"])):
        print(warehouseInstance.Pods[str(podID)].__dict__["Items"][i].ID)
        print(warehouseInstance.Pods[str(podID)].__dict__["Items"][i].Count)
        
#%% F_itemInfo
def F_itemInfoMix(itemID):
    item = warehouseInstance.ItemDescriptions[str(itemID)].__dict__
    itemDesc = item["Color"] + "/" + item["Letter"]
    item["Description"] = itemDesc
    item["PodLocation"] = {"Pod":[],"Amount":[]}
    return(item)
    
#%%
itemList = []

for i in range(itemAmount):
    itemList.append(F_itemInfoMix(i))
    
itemList = pd.DataFrame(itemList)

#%%
for podID in [*warehouseInstance.Pods]:
    print("In Pod " + podID +":")
    for i in range(len(warehouseInstance.Pods[str(podID)].__dict__["Items"])):
        itemDesc = warehouseInstance.Pods[str(podID)].__dict__["Items"][i].ID
        print(itemDesc)
        itemCount = warehouseInstance.Pods[str(podID)].__dict__["Items"][i].Count 
        print(itemCount)
        
        itemList[itemList.Description == itemDesc][["PodLocation"]].iat[0,0]["Pod"].append(podID)
        itemList[itemList.Description == itemDesc][["PodLocation"]].iat[0,0]["Amount"].append(itemCount)
        
    
        
        

        #%%
        
