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


def calculateAvoidCommitmnetZone(playerGrid,target1,target2):
    actionZone=[[min(playerGrid[0],target1[0],target2[0]),min(playerGrid[1],target1[1],target2[1])],
                [max(playerGrid[0], target1[0], target2[0]), max(playerGrid[1], target1[1], target2[1])]]
    actionZone=[(x,y) for x in list(range(actionZone[0][0], actionZone[1][0]+1,1)) for y in list(range(actionZone[0][1], actionZone[1][1]+1, 1))]
    avoidCommitmentZone=list()
    for point in actionZone:
       if np.linalg.norm(np.array(point) - np.array(target1), ord=1)== np.linalg.norm(np.array(point) - np.array(target2), ord=1):
           avoidCommitmentZone.append(point)
    if tuple(playerGrid) in avoidCommitmentZone:
        avoidCommitmentZone.remove(tuple(playerGrid))
    return  avoidCommitmentZone

def calculateAvoidCommitmentRatio(trajectory,zone):
    avoidCommitmentPath=list()
    for point in trajectory:
        if tuple(point) not in zone and len(avoidCommitmentPath)!=0:
            break
        if tuple(point) in zone:
            avoidCommitmentPath.append(point)
    avoidCommitmentRatio=len(avoidCommitmentPath)/len(trajectory)
    return avoidCommitmentRatio

if __name__ == "__main__":
    resultsPath = os.path.abspath(os.path.join(os.getcwd(),"../..")) + '/Results/'
    fileFormat = '.csv'
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    eatStraightBean=0
    straightEatOld=0
    averageAvoidCommitmentRatio=list()
    for resultsDataFrame in resultsDataFrameList:
        resultsDataFrame=cleanDataFrame(resultsDataFrame)
        trialNumber = resultsDataFrame.shape[0]
        avoidCommitmentRatioList = list()
        for i in range(trialNumber):
            bean1Grid=[resultsDataFrame.iat[i,4],resultsDataFrame.iat[i,5]]
            bean2Grid=[resultsDataFrame.iat[i,6],resultsDataFrame.iat[i,7]]
            playerGrid=[resultsDataFrame.iat[i,8],resultsDataFrame.iat[i,9]]
            trajectory=eval(resultsDataFrame.iat[i,11])
            avoidCommitmentZone=calculateAvoidCommitmnetZone(playerGrid,bean1Grid,bean2Grid)
            if len(avoidCommitmentZone)!=0:
                avoidCommitmentRatio=calculateAvoidCommitmentRatio(trajectory,avoidCommitmentZone)
                avoidCommitmentRatioList.append(avoidCommitmentRatio)
        averageAvoidCommitmentRatio.append(np.mean(np.array(avoidCommitmentRatioList)))
    averate=np.mean(averageAvoidCommitmentRatio)
    stdAvoidCommitmentRatio = np.std(np.array(averageAvoidCommitmentRatio))/np.sqrt(len(averageAvoidCommitmentRatio)-1)
    print(averate)
    print(stdAvoidCommitmentRatio)





