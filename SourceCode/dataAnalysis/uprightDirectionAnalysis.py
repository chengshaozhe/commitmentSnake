import pandas as pd
import matplotlib.pyplot as plt
import os
import pylab as pl
import numpy as np
from functools import reduce
from math import ceil,floor
from collections import Counter
import math


def createAllCertainFormatFileList(filePath, fileFormat):
    filenameList = [os.path.join(filePath, relativeFilename) for relativeFilename in os.listdir(filePath)
                    if os.path.isfile(os.path.join(filePath, relativeFilename))
                    if os.path.splitext(relativeFilename)[1] in fileFormat]
    return filenameList


def cleanDataFrame(rawDataFrame):
    cleanConditionDataFrame = rawDataFrame[rawDataFrame.condition != 'None']
    cleanBeanEatenDataFrame = cleanConditionDataFrame[cleanConditionDataFrame.beanEaten != 0]
    return cleanBeanEatenDataFrame

def computeAngleBetweenTwoVectors(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    lenthOfVector1 = np.sqrt(vector1.dot(vector1))
    lenthOfVector2 = np.sqrt(vector2.dot(vector2))
    cosAngle = vector1.dot(vector2) / (lenthOfVector1 * lenthOfVector2)
    if cosAngle<-1:
        cosAngle=-1
    elif cosAngle>1:
        cosAngle=1
    angle = math.degrees(np.arccos(cosAngle))
    return angle

def judgeUprightCondition(player,target,lastAction):
    vectorBetweenPlayerAndTarget=np.array(target)-np.array(player)
    angleBetweenPositionVectorAndActionVector=computeAngleBetweenTwoVectors(vectorBetweenPlayerAndTarget,np.array(lastAction))
    if angleBetweenPositionVectorAndActionVector==0:
        uprightCondition=True
    else:
        uprightCondition=False
    return uprightCondition




if __name__ == "__main__":
    resultsPath = os.path.abspath(os.path.join(os.getcwd(),"../..")) + '/Results/'
    fileFormat = '.csv'
    writerPath=resultsPath+str("humanUprightCondition.csv")
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    resultsDataFrame = pd.concat(resultsDataFrameList, sort=False)
    resultsDataFrame = cleanDataFrame(resultsDataFrame)
    bean = 2
    trialNumber=resultsDataFrame.shape[0]
    beanEatList=[resultsDataFrame.iat[trialIndex,14] for trialIndex in range(trialNumber)]
    straightConditionTrial=list()
    for i in range(1,trialNumber):
        bean1Grid=[resultsDataFrame.iat[i,4],resultsDataFrame.iat[i,5]]
        bean2Grid=[resultsDataFrame.iat[i,6],resultsDataFrame.iat[i,7]]
        playerGrid=[resultsDataFrame.iat[i,8],resultsDataFrame.iat[i,9]]
        lastAction=eval(resultsDataFrame.iat[i-1,12])[-1]
        beanGrid=eval("bean"+str(bean)+"Grid")
        if judgeUprightCondition(playerGrid,beanGrid,lastAction):
            straightConditionTrial.append(i)
    print(resultsDataFrame.shape[0])
    resultsDataFrame=resultsDataFrame.iloc[straightConditionTrial,]
    resultsDataFrame = resultsDataFrame.reset_index(drop=True)
    print(len(straightConditionTrial))
    print(resultsDataFrame.shape[0])
    resultsDataFrame.to_csv(writerPath)





