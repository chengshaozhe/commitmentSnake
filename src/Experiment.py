#!/usr/bin/env python

# -*- coding: utf-8 -*-

import pygame as pg
import os
import pandas as pd
import collections as co
import numpy as np
import  pickle
from Visualization import DrawBackground, DrawNewState, DrawImage
from Controller import HumanController,ModelController,CheckBoundary
import UpdateWorld
from Writer import WriteDataFrameToCSV
from Trial import Trial
from math import ceil
from collections import Counter

def createAllCertainFormatFileList(filePath, fileFormat):
    filenameList = [os.path.join(filePath, relativeFilename) for relativeFilename in os.listdir(filePath)
                    if os.path.isfile(os.path.join(filePath, relativeFilename))
                    if os.path.splitext(relativeFilename)[1] in fileFormat]
    return filenameList


class Experiment():
    def __init__(self, trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath, \
                 minDistanceBetweenGrids,maxDistanceBetweenGrids,restImage,finishImage,restTrial):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.initialWorld = initialWorld
        self.updateWorld = updateWorld
        self.drawImage = drawImage
        self.resultsPath = resultsPath
        self.minDistanceBetweenGrids = minDistanceBetweenGrids
        self.maxDistanceBetweenGrids = maxDistanceBetweenGrids
        self.restImage=restImage
        self.finishImage=finishImage
        self.restTrial=restTrial

    def __call__(self, designValues):
        bean1Grid, bean2Grid, playerGrid,angle = self.initialWorld(self.minDistanceBetweenGrids,self.maxDistanceBetweenGrids)
        trialIndex=0
        while trialIndex<len(designValues):
            self.experimentValues["angle"]=angle
            results, bean1Grid, playerGrid = self.trial(bean1Grid, bean2Grid, playerGrid)
            response = self.experimentValues.copy()
            response.update(results)
            responseDF = pd.DataFrame(response, index=[trialIndex])
            self.writer(responseDF)
            bean2Grid, nextCondition,angle = self.updateWorld(bean1Grid, playerGrid,designValues[trialIndex])
            self.experimentValues["angle"]=angle
            self.experimentValues["condition"]=nextCondition
            designValues=UpdateWorld.adjustDesignValues(nextCondition, trialIndex, designValues)
            trialIndex=trialIndex+1
            #if trialIndex in self.restTrial:
                #self.drawImage(self.restImage)

        #self.drawImage(self.finishImage)



def main():
    resultsPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Results/'
    fileFormat = '.csv'
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    resultsDataFrame = pd.concat(resultsDataFrameList, sort=False)
    dimension = 15
    bounds = [0, 0, dimension - 1,dimension - 1]
    pg.init()
    screenWidth = 680
    screenHeight = 680
    screen = pg.display.set_mode((screenWidth, screenHeight))
    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)
    pg.event.set_allowed([pg.KEYDOWN, pg.QUIT])
    designValues=list(range(resultsDataFrame.shape[0]))
    updateWorld =UpdateWorld(resultsDataFrame,[4,5],[6,7],[8,9])
    humanController = HumanController(dimension)
    controller = humanController
    experimentValues = co.OrderedDict()
    drawBackground = DrawBackground(screen, dimension, leaveEdgeSpace, backgroundColor, lineColor, lineWidth,
                                    textColorTuple)
    checkBoundary = CheckBoundary([0, dimension-1 ], [0, dimension -1])
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    trial = Trial(controller, drawNewState, checkBoundary)
    experiment = Experiment(trial, experimentValues, updateWorld )
    experiment(designValues)



if __name__ == "__main__":
    main()
