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

def judgeStraightCondition(player,target1,target2):
    if player[0]==target1[0] or playerGrid[0]==target2[0] or player[1]==target1[1] or playerGrid[1]==target2[1]:
        straightCondition =True
    else:
        straightCondition = False
    return straightCondition




if __name__ == "__main__":
    resultsPath = os.path.abspath(os.path.join(os.getcwd(),"../..")) + '/Results/'
    fileFormat = '.csv'
    writerPath=resultsPath+str("machineMaxDeleteStraightCondition.csv")
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    resultsDataFrame = pd.concat(resultsDataFrameList, sort=False)
    resultsDataFrame = cleanDataFrame(resultsDataFrame)
    trialNumber=resultsDataFrame.shape[0]
    # resultsDataFrame.index=[i for i in range(trialNumber)]
    resultsDataFrame=resultsDataFrame.reset_index(drop=True)
    beanEatList=[resultsDataFrame.iat[trialIndex,14] for trialIndex in range(trialNumber)]
    straightConditionTrial=list()
    for i in range(trialNumber):
        bean1Grid=[resultsDataFrame.iat[i,4],resultsDataFrame.iat[i,5]]
        bean2Grid=[resultsDataFrame.iat[i,6],resultsDataFrame.iat[i,7]]
        playerGrid=[resultsDataFrame.iat[i,8],resultsDataFrame.iat[i,9]]
        if judgeStraightCondition(playerGrid,bean1Grid,bean2Grid):
            straightConditionTrial.append(i)
    print(resultsDataFrame.shape[0])
    resultsDataFrame.drop(straightConditionTrial,inplace=True)
    resultsDataFrame = resultsDataFrame.reset_index(drop=True)
    print(len(straightConditionTrial))
    print(resultsDataFrame.shape[0])
    resultsDataFrame.to_csv(writerPath)





