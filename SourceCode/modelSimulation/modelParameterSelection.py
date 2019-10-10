import pandas as pd
import matplotlib.pyplot as plt
import os
import pylab as pl
import numpy as np
from numpy import log

def createAllCertainFormatFileList(filePath,fileFormat,parameter):
    fileNameList=[os.path.join(filePath,relativeFilename) for relativeFilename in os.listdir(filePath)
        if os.path.isfile(os.path.join(filePath,relativeFilename))
        if os.path.splitext(relativeFilename)[1] in fileFormat ]
    specificParameterFileNameList=[file for file in fileNameList if parameter in file]
    return specificParameterFileNameList

def cleanDataFrame(rawDataFrame):
    cleanConditionDataFrame=rawDataFrame[rawDataFrame.condition != "None"]
    cleanBeanEatenDataFrame=cleanConditionDataFrame[cleanConditionDataFrame.beanEaten!=0]
    return cleanBeanEatenDataFrame

class SelectModelParameter:
    def __init__(self,modelResultsPath,modelResultsFormat):
        self.modelResultsPath=modelResultsPath
        self.modelResultsFormat=modelResultsFormat

    def __call__(self,parameter):
        resultsFilenameList = createAllCertainFormatFileList(self.modelResultsPath, self.modelResultsFormat)
        resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
        resultsDataFrame = pd.concat(resultsDataFrameList,sort=False)
        resultsDataFrame=cleanDataFrame(resultsDataFrame)
        participantsTypeList = ['machine' if 'machine' in name else 'Human' for name in resultsDataFrame['name']]
        resultsDataFrame['participantsType']=participantsTypeList
        resultsDataFrame['beanEaten']=resultsDataFrame['beanEaten']-1
        trialNumberEatNewDataFrame = resultsDataFrame.groupby(['name','condition','participantsType']).sum()['beanEaten']
        trialNumberTotalEatDataFrame = resultsDataFrame.groupby(['name','condition','participantsType']).count()['beanEaten']
        mergeConditionDataFrame = pd.DataFrame(trialNumberEatNewDataFrame.values/trialNumberTotalEatDataFrame.values,index=trialNumberTotalEatDataFrame.index,columns=['eatNewPercentage'])
        mergeConditionDataFrame['eatOldPercentage']=1 - mergeConditionDataFrame['eatNewPercentage']
        mergeParticipantsDataFrameMean = mergeConditionDataFrame.groupby(['condition','participantsType']).mean()
        mergeParticipantsDataFrameStandardError = mergeConditionDataFrame.groupby(['condition', 'participantsType']).std()
        mergeConditionDataFrame.groupby(['condition', 'participantsType'])
        drawEatOldDataFrameMean=mergeParticipantsDataFrameMean['eatOldPercentage'].unstack('participantsType')
        crossEntropy=(drawEatOldDataFrameMean.apply(lambda x:-x['Human']*log(x['machine']),axis=1)).sum()
        return crossEntropy

