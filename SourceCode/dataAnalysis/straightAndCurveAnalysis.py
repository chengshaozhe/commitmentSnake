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

def calculateStraightRatio(target1,target2):
    curvePath=min(abs(target1[0]-target2[0]),abs(target1[1]-target2[1]))
    straightPath=max(abs(target1[0]-target2[0]),abs(target1[1]-target2[1]))
    straightRatio=-curvePath/straightPath
    return straightPath




if __name__ == "__main__":
    resultsPath = os.path.abspath(os.path.join(os.getcwd(),"../..")) + '/Results/'
    fileFormat = '.csv'
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    resultsDataFrame = pd.concat(resultsDataFrameList, sort=False)
    resultsDataFrame = cleanDataFrame(resultsDataFrame)
    trialNumber=resultsDataFrame.shape[0]
    participantsTypeList = ['machine' if 'machine' in name else 'Human' for name in resultsDataFrame['name']]
    resultsDataFrame['participantsType'] = participantsTypeList
    resultsDataFrame['goal'] = resultsDataFrame['goal']
    beanEatList=[resultsDataFrame.iat[trialIndex,14] for trialIndex in range(trialNumber)]
    eatStraightBean=0
    straightEatOld=0
    for i in range(trialNumber):
        bean1Grid=[resultsDataFrame.iat[i,4],resultsDataFrame.iat[i,5]]
        bean2Grid=[resultsDataFrame.iat[i,6],resultsDataFrame.iat[i,7]]
        playerGrid=[resultsDataFrame.iat[i,8],resultsDataFrame.iat[i,9]]
        straightRatioBean1=calculateStraightRatio(bean1Grid,playerGrid)
        straightRatioBean2=calculateStraightRatio(bean2Grid,playerGrid)
        beanEat=beanEatList[i]
        beanNotEat=[1,2]
        beanNotEat.remove(beanEat)
        beanNotEat=beanNotEat[0]
        if eval("straightRatioBean"+str(beanEat))>eval("straightRatioBean"+str(beanNotEat)):
            eatStraightBean=eatStraightBean+1
        if straightRatioBean2>straightRatioBean1 and beanEat==1:
            straightEatOld=straightEatOld+1
    eatStraightBeanPercentage=eatStraightBean/trialNumber
    straightEatOldPercentage=straightEatOld/trialNumber
    print(eatStraightBeanPercentage)
    print(straightEatOldPercentage)




