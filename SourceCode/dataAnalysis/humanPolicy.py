import pandas as pd
import matplotlib.pyplot as plt
import os
import pylab as pl
import numpy as np
import pickle


def createAllCertainFormatFileList(filePath, fileFormat):
    filenameList = [os.path.join(filePath, relativeFilename) for relativeFilename in os.listdir(filePath)
                    if os.path.isfile(os.path.join(filePath, relativeFilename))
                    if os.path.splitext(relativeFilename)[1] in fileFormat]
    return filenameList


def cleanDataFrame(rawDataFrame):
    cleanConditionDataFrame = rawDataFrame[rawDataFrame.condition != 'None']
    cleanBeanEatenDataFrame = cleanConditionDataFrame[cleanConditionDataFrame.beanEaten != 0]
    return cleanBeanEatenDataFrame


if __name__ == "__main__":
    resultsPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Results/'
    personResults = resultsPath + "personResults.csv"
    policyPath=resultsPath+"SingleWolfTwoSheepsGrid15Person.pickle"
    fileFormat = '.csv'
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    resultsDataFrame = pd.concat(resultsDataFrameList, sort=False)
    trialNumber = resultsDataFrame.shape[0]
    print(trialNumber)
    humanPolicy=dict()
    for trialIndex in range(trialNumber):
        actionList = eval(resultsDataFrame.iat[trialIndex, 12])
        trajectoryList=eval(resultsDataFrame.iat[trialIndex, 11])
        for everyStep in range(len(trajectoryList)-1):
            state2=(tuple(trajectoryList[everyStep]),((resultsDataFrame.iat[trialIndex, 6],resultsDataFrame.iat[trialIndex, 7]),
                                               (resultsDataFrame.iat[trialIndex, 4],resultsDataFrame.iat[trialIndex, 5])))
            state1=(tuple(trajectoryList[everyStep]),((resultsDataFrame.iat[trialIndex, 4],resultsDataFrame.iat[trialIndex, 5]),
                                               (resultsDataFrame.iat[trialIndex, 6],resultsDataFrame.iat[trialIndex, 7])))
            if state2 in humanPolicy.keys():
                state=state2
            else:
                state=state1
            humanPolicy[state]=dict([((1,0),0),((-1,0),0),((0,1),0),((0,-1),0)])
            humanPolicy[state][actionList[everyStep]]=humanPolicy[state][actionList[everyStep]]+1
    for state in humanPolicy.keys():
        frequency=0
        policy=humanPolicy[state]
        for action in humanPolicy[state].keys():
            frequency=frequency+humanPolicy[state][action]
        for action in humanPolicy[state].keys():
            humanPolicy[state][action]=humanPolicy[state][action]/frequency
    with open(policyPath, 'wb') as handle:
        pickle.dump(humanPolicy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(policyPath, 'rb') as handle:
        b = pickle.load(handle)


