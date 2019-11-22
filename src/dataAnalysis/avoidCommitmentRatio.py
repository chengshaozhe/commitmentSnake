import pandas as pd
import matplotlib.pyplot as plt
import os
import pylab as pl
import numpy as np
from functools import reduce
from math import ceil, floor
from collections import Counter
import itertools as it


def createAllCertainFormatFileList(filePath, fileFormat):
    filenameList = [os.path.join(filePath, relativeFilename) for relativeFilename in os.listdir(filePath)
                    if os.path.isfile(os.path.join(filePath, relativeFilename))
                    if os.path.splitext(relativeFilename)[1] in fileFormat]
    return filenameList


def cleanDataFrame(rawDataFrame):
    cleanConditionDataFrame = rawDataFrame[rawDataFrame.condition != 'None']
    cleanBeanEatenDataFrame = cleanConditionDataFrame[cleanConditionDataFrame.beanEaten != 0]
    return cleanBeanEatenDataFrame


# def calculateAvoidCommitmnetZone(playerGrid, target1, target2):
#     actionZone = [[min(playerGrid[0], target1[0], target2[0]), min(playerGrid[1], target1[1], target2[1])],
#                   [max(playerGrid[0], target1[0], target2[0]), max(playerGrid[1], target1[1], target2[1])]]
#     actionZone = [(x, y) for x in list(range(actionZone[0][0], actionZone[1][0] + 1, 1)) for y in list(range(actionZone[0][1], actionZone[1][1] + 1, 1))]
#     avoidCommitmentZone = list()
#     for point in actionZone:
#         if np.linalg.norm(np.array(point) - np.array(target1), ord=1) == np.linalg.norm(np.array(point) - np.array(target2), ord=1):
#             avoidCommitmentZone.append(point)
#     if tuple(playerGrid) in avoidCommitmentZone:
#         avoidCommitmentZone.remove(tuple(playerGrid))
#     return avoidCommitmentZone


def calculateAvoidCommitmentRatio(trajectory, zone):
    avoidCommitmentPath = list()
    for point in trajectory:
        if tuple(point) not in zone and len(avoidCommitmentPath) != 0:
            break
        if tuple(point) in zone:
            avoidCommitmentPath.append(point)
    avoidCommitmentRatio = len(avoidCommitmentPath) / len(trajectory)
    return avoidCommitmentRatio


def creatRect(coor1, coor2):
    vector = np.array(list(zip(coor1, coor2)))
    vector.sort(axis=1)
    rect = [(i, j) for i in range(vector[0][0], vector[0][1] + 1) for j in range(vector[1][0], vector[1][1] + 1)]
    return rect


def calculateAvoidCommitmnetZone(playerGrid, target1, target2):
    dis1 = np.linalg.norm(np.array(playerGrid) - np.array(target1), ord=1)
    dis2 = np.linalg.norm(np.array(playerGrid) - np.array(target2), ord=1)
    if dis1 == dis2:
        rect1 = creatRect(playerGrid, target1)
        rect2 = creatRect(playerGrid, target2)
        avoidCommitmentZone = list(set(rect1).intersection(set(rect2)))
        avoidCommitmentZone.remove(tuple(playerGrid))
    else:
        avoidCommitmentZone = []

    return avoidCommitmentZone


def calculateFirstIntentionStep(intentionList):
    goal1Step = len(intentionList)
    goal2Step = len(intentionList)
    if 1 in intentionList:
        goal1Step = intentionList.index(1)
    if 2 in intentionList:
        goal2Step = intentionList.index(2)
    firstIntentionStep = min(goal1Step, goal2Step)
    if goal1Step < goal2Step:
        firstIntention = 1
    elif goal2Step < goal1Step:
        firstIntention = 2
    else:
        firstIntention = 0
    return firstIntentionStep + 1


if __name__ == "__main__":
    resultsPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/Results/human'
    fileFormat = '.csv'
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    eatStraightBean = 0
    straightEatOld = 0
    averageAvoidCommitmentRatio = list()
    for resultsDataFrame in resultsDataFrameList:
        resultsDataFrame = cleanDataFrame(resultsDataFrame)
        trialNumber = resultsDataFrame.shape[0]
        avoidCommitmentRatioList = list()
        for i in range(trialNumber):
            bean1Grid = [resultsDataFrame.iat[i, 4], resultsDataFrame.iat[i, 5]]
            bean2Grid = [resultsDataFrame.iat[i, 6], resultsDataFrame.iat[i, 7]]
            playerGrid = [resultsDataFrame.iat[i, 8], resultsDataFrame.iat[i, 9]]
            trajectory = eval(resultsDataFrame.iat[i, 11])
            goal = eval(resultsDataFrame.iat[i, 13])

            firstIntentionStep = calculateFirstIntentionStep(goal)
            avoidCommitmentRatioList.append(firstIntentionStep / len(goal))

            # avoidCommitmentZone = calculateAvoidCommitmnetZone(playerGrid, bean1Grid, bean2Grid)
            # if len(avoidCommitmentZone) != 0:
            #     avoidCommitmentRatio = calculateAvoidCommitmentRatio(trajectory, avoidCommitmentZone)
            #     avoidCommitmentRatioList.append(avoidCommitmentRatio)

        averageAvoidCommitmentRatio.append(np.mean(np.array(avoidCommitmentRatioList)))
    averate = np.mean(averageAvoidCommitmentRatio)
    stdAvoidCommitmentRatio = np.std(np.array(averageAvoidCommitmentRatio)) / np.sqrt(len(averageAvoidCommitmentRatio) - 1)
    print(averate)
    print(stdAvoidCommitmentRatio)
