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

def calculateFirstIntentionStep(data):
    goal1Step=float('inf')
    goal2Step=float('inf')
    intentionList=eval(data)
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
    return firstIntention


def cleanDataFrame(rawDataFrame):
    cleanConditionDataFrame = rawDataFrame[rawDataFrame.condition != 'None']
    cleanBeanEatenDataFrame = cleanConditionDataFrame[cleanConditionDataFrame.beanEaten != 0]
    return cleanBeanEatenDataFrame

def calculateFirstIntention(intentionList):
    try:
        target1Goal=intentionList.index(1)
    except ValueError as e:
        target1Goal=999
    try:
        target2Goal=intentionList.index(2)
    except ValueError as e:
        target2Goal=999
    if target1Goal<target2Goal:
        firstGoal=1
    elif target2Goal<target1Goal:
        firstGoal=2
    else:
        firstGoal=0
    return firstGoal

def judgeGoalChange(goalList):
    allGoal=[goalList[index] for index in range(len(goalList)-1) if goalList[index] != goalList[index+1]]
    try:
        if goalList[-1]!= allGoal[-1]:
            allGoal.append(goalList[-1])
        allGoalExcludeGoal0 = list(filter(lambda x: x != 0, allGoal))
        allGoalExcludeGoal0StrList=list(map(str,allGoalExcludeGoal0))
        allGoalExcludeGoal0Str=reduce(lambda x,y:x+y,allGoalExcludeGoal0StrList)
        goalOldSwitchToNewNumber=allGoalExcludeGoal0Str.count("12")
        goalNewSwitchToOldNumber=allGoalExcludeGoal0Str.count("21")
        goalSwitchNumber=goalOldSwitchToNewNumber+goalNewSwitchToOldNumber
    except IndexError as e:
        goalSwitchNumber=0
        goalOldSwitchToNewNumber=0
        goalNewSwitchToOldNumber=0
    return goalSwitchNumber,goalOldSwitchToNewNumber,goalNewSwitchToOldNumber



if __name__ == "__main__":
    resultsPath = os.path.abspath(os.path.join(os.getcwd(),"../..")) + '/Results/'
    fileFormat = '.csv'
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    resultsDataFrame = pd.concat(resultsDataFrameList, sort=False)
    resultsDataFrame = cleanDataFrame(resultsDataFrame)
    trialNumber=resultsDataFrame.shape[0]
    participantsTypeList = ['machine' if 'machine' in name else 'Human' for name in resultsDataFrame['name']]
    conditionData = pd.Series(resultsDataFrame['condition'].values, index=list(range(resultsDataFrame.iloc[:, 0].size)))
    resultsDataFrame['participantsType'] = participantsTypeList
    resultsDataFrame['goal'] = resultsDataFrame['goal']
    goalList=[eval(resultsDataFrame.iat[trialIndex,13]) for trialIndex in range(trialNumber)]
    conditionList=[resultsDataFrame.iat[trialIndex,2] for trialIndex in range(trialNumber)]
    goalSwitchNumberAllTrial=0
    goalOldSwitchToNewNumberAllTrial=0
    goalNewSwitchToOldNumberAllTrial=0
    conditionGoalNewSwitchToOld=list()
    conditionGoalOldSwitchToNew=list()
    firstIntention=list()
    for trialIndex in range(len(goalList)):
        goalSwitchOneTrial,goalOldSwitchToNewNumberOneTrial,goalNewSwitchToOldNumberOneTrial=judgeGoalChange(goalList[trialIndex])
        goalSwitchNumberAllTrial=goalSwitchNumberAllTrial+goalSwitchOneTrial
        goalOldSwitchToNewNumberAllTrial=goalOldSwitchToNewNumberAllTrial+goalOldSwitchToNewNumberOneTrial
        goalNewSwitchToOldNumberAllTrial=goalNewSwitchToOldNumberAllTrial+goalNewSwitchToOldNumberOneTrial
        if goalOldSwitchToNewNumberOneTrial !=0:
            if conditionList[trialIndex]=='-5':
                bean1Grid = [resultsDataFrame.iat[trialIndex, 4], resultsDataFrame.iat[trialIndex, 5]]
                bean2Grid = [resultsDataFrame.iat[trialIndex, 6], resultsDataFrame.iat[trialIndex, 7]]
                playerGrid = [resultsDataFrame.iat[trialIndex, 8], resultsDataFrame.iat[trialIndex, 9]]
                goal=eval(resultsDataFrame.iat[trialIndex,13])
                trajectory=resultsDataFrame.iat[trialIndex,11]
                firstIntention.append(calculateFirstIntention(goal))
                print("AOO")
            conditionGoalOldSwitchToNew.append(conditionList[trialIndex])
        if goalNewSwitchToOldNumberOneTrial !=0:
            conditionGoalNewSwitchToOld.append(conditionList[trialIndex])
    print(Counter(firstIntention))
    print(goalSwitchNumberAllTrial,goalOldSwitchToNewNumberAllTrial,goalNewSwitchToOldNumberAllTrial)
    print(Counter(conditionGoalOldSwitchToNew))
    print(Counter(conditionGoalNewSwitchToOld))

