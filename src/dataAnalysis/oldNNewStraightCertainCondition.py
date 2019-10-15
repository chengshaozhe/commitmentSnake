import pandas as pd
import matplotlib.pyplot as plt
import os
import pylab as pl
import numpy as np
from functools import reduce
from math import ceil,floor
from collections import Counter


def createAllCertainFormatFileList(filePath, fileFormat):
    filenameList = [os.path.join(filePath, relativeFilename) for relativeFilename in os.listdir(filePath)
                    if os.path.isfile(os.path.join(filePath, relativeFilename))
                    if os.path.splitext(relativeFilename)[1] in fileFormat]
    return filenameList


def cleanDataFrame(rawDataFrame):
    cleanConditionDataFrame = rawDataFrame[rawDataFrame.condition != 'None']
    cleanBeanEatenDataFrame = cleanConditionDataFrame[cleanConditionDataFrame.beanEaten != 0]
    return cleanBeanEatenDataFrame

def judgeStraightCondition(player,target):
    if player[0]==target[0] or player[1]==target[1] :
        straightCondition =True
    else:
        straightCondition = False
    return straightCondition




if __name__ == "__main__":
    resultsPath = os.path.abspath(os.path.join(os.getcwd(),"../..")) + '/Results/'
    fileFormat = '.csv'
    writerPath=resultsPath+str("modelCondition0NewBeanStraight.csv")
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    resultsDataFrame = pd.concat(resultsDataFrameList, sort=False)
    resultsDataFrame = cleanDataFrame(resultsDataFrame)
    condition = '0'
    bean = 2
    resultsDataFrameToCheck=resultsDataFrame[resultsDataFrame["condition"]==condition]
    trialNumber=resultsDataFrameToCheck.shape[0]
    resultsDataFrameToCheck=resultsDataFrameToCheck.reset_index(drop=True)
    beanEatList=[resultsDataFrameToCheck.iat[trialIndex,14] for trialIndex in range(trialNumber)]
    straightConditionTrial=list()

    for i in range(trialNumber):
        bean1Grid=[resultsDataFrameToCheck.iat[i,4],resultsDataFrameToCheck.iat[i,5]]
        bean2Grid=[resultsDataFrameToCheck.iat[i,6],resultsDataFrameToCheck.iat[i,7]]
        playerGrid=[resultsDataFrameToCheck.iat[i,8],resultsDataFrameToCheck.iat[i,9]]
        beanGrid=eval("bean"+str(bean)+"Grid")
        if judgeStraightCondition(playerGrid,beanGrid):
            straightConditionTrial.append(i)
    print(resultsDataFrameToCheck.shape[0])
    resultsDataFrameToCheck=resultsDataFrameToCheck.iloc[straightConditionTrial,]
    resultsDataFrameToCheck = resultsDataFrameToCheck.reset_index(drop=True)
    print(len(straightConditionTrial))
    print(resultsDataFrameToCheck.shape[0])
    resultsDataFrameToCheck.to_csv(writerPath)





