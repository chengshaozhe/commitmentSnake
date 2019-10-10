import pandas as pd
import matplotlib.pyplot as plt
import os
import pylab as pl
import numpy as np
from collections import Counter
from Writer import WriteDataFrameToCSV



def createAllCertainFormatFileList(filePath, fileFormat):
    filenameList = [os.path.join(filePath, relativeFilename) for relativeFilename in os.listdir(filePath)
                    if os.path.isfile(os.path.join(filePath, relativeFilename))
                    if os.path.splitext(relativeFilename)[1] in fileFormat]
    return filenameList


def cleanDataFrame(rawDataFrame):
    cleanConditionDataFrame = rawDataFrame[rawDataFrame.condition != 'None']
    cleanBeanEatenDataFrame = cleanConditionDataFrame[cleanConditionDataFrame.beanEaten != 0]
    return cleanBeanEatenDataFrame

def calculateFirstIntentionStep(intentionList):
    goal1Step=float('inf')
    goal2Step=float('inf')
    if 1 in intentionList:
        goal1Step=intentionList.index(1)
    if 2 in intentionList:
        goal2Step=intentionList.index(2)
    firstIntentionStep=min(goal1Step,goal2Step)
    if goal1Step<goal2Step:
        firstIntention=1
    elif goal2Step<goal1Step:
        firstIntention=2
    else:
        firstIntention=0
    return firstIntentionStep,firstIntention


#
# if __name__ == "__main__":
#     resultsPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/Results/'
#     personResults = resultsPath + "firstIntentionStepResults.csv"
#     fileFormat = '.csv'
#     resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
#     resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
#     resultsDataFrame = pd.concat(resultsDataFrameList, sort=False)
#     # participantsTypeList = ['machine' if 'machine' in name else 'Human' for name in resultsDataFrame['name']]
#     conditionData = pd.Series(resultsDataFrame['condition'].values, index=list(range(resultsDataFrame.iloc[:, 0].size)))
#     # resultsDataFrame['participantsType'] = participantsTypeList
#     trialNumber = resultsDataFrame.shape[0]
#     # resultsDataFrame["firstIntetionStep"]=float("inf")
#     writerPath = resultsPath+'firstIntentionEveryCondition.csv'
#     writer = WriteDataFrameToCSV(writerPath)
#     firstIntentionList=list()
#     condition=[-5,-3,-1,0,1,3,5]
#     condition0=list()
#     condition2=list()
#     condition4=list()
#     condition5=list()
#     condition6=list()
#     condition8=list()
#     condition10=list()
#     conditionList=[resultsDataFrame.iat[trialIndex,2] for trialIndex in range(trialNumber)]
#     for trialIndex in range(trialNumber):
#         print(resultsDataFrame.iat[trialIndex, 13])
#         intentionList = eval(resultsDataFrame.iat[trialIndex, 13])
#         firstIntentionStep,firstIntention=calculateFirstIntentionStep(intentionList)
#         eval("condition"+str(eval(conditionList[trialIndex])+5)).append(firstIntention)
#         if firstIntentionStep==float('inf'):
#             resultsDataFrame.iat[trialIndex, 16] = firstIntentionStep
#         resultsDataFrame.iat[trialIndex, 15]=firstIntentionStep
#     # trialNumberEatNewDataFrame = resultsDataFrame.groupby(['name', 'condition', 'participantsType']).mean()["firstIntetionStep"]
#     # trialNumberTotalEatDataFrame = resultsDataFrame.groupby(['name', 'condition', 'participantsType']).count()["firstIntetionStep"]
#     for everyCondition in condition:
#         counter=dict(Counter(eval("condition"+str(everyCondition+5))))
#         conditionCounter=dict()
#         allGoal=[1,2]
#         for goal in allGoal:
#             if goal in counter.keys():
#                 conditionCounter[goal]=counter[goal]/len(eval("condition"+str(everyCondition+5)))
#             else:
#                 conditionCounter[goal]=0
#         intentionDF = pd.DataFrame(conditionCounter, index=[everyCondition])
#         writer(intentionDF)


if __name__ == "__main__":
    resultsPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/Results/'
    personResults = resultsPath + "firstIntentionStepResults.csv"
    fileFormat = '.csv'
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    writerPath = resultsPath+'firstIntentionEveryCondition.csv'
    writer = WriteDataFrameToCSV(writerPath)
    firstIntentionList=list()
    condition=[-5,-3,-1,0,1,3,5]
    condition0Mean=list()
    for resultsDataFrame in resultsDataFrameList:
        resultsDataFrame=cleanDataFrame(resultsDataFrame)
        condition0 = list()
        for trialIndex in range(resultsDataFrame.shape[0]):
            if resultsDataFrame.iat[trialIndex,2]=='0':
                intentionList = eval(resultsDataFrame.iat[trialIndex, 13])
                firstIntentionStep,firstIntention=calculateFirstIntentionStep(intentionList)
                condition0.append(firstIntention)
        counter=dict(Counter(condition0))
        conditionCounter=dict()
        allGoal=[1,2]
        for goal in allGoal:
            if goal in counter.keys():
                conditionCounter[goal]=counter[goal]/len(condition0)
            else:
                conditionCounter[goal]=0
        condition0Mean.append(conditionCounter[1])
    meanFirstIntention=np.mean(condition0Mean)
    stdFirstIntention= np.std(np.array(condition0Mean))/np.sqrt(len(condition0Mean)-1)
    print(meanFirstIntention)
    print(stdFirstIntention)
