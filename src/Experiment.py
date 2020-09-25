import pygame as pg
import os
import pandas as pd
import collections as co
import numpy as np
import pickle
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from src.UpdateWorld import *


class Experiment():
    def __init__(self, trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath,
                 minDistanceBetweenGrids, maxDistanceBetweenGrids, restImage, finishImage, restTrial):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.initialWorld = initialWorld
        self.updateWorld = updateWorld
        self.drawImage = drawImage
        self.resultsPath = resultsPath
        self.minDistanceBetweenGrids = minDistanceBetweenGrids
        self.maxDistanceBetweenGrids = maxDistanceBetweenGrids
        self.restImage = restImage
        self.finishImage = finishImage
        self.restTrial = restTrial

    def __call__(self, designValues):
        bean1Grid, bean2Grid, playerGrid, angle = self.initialWorld(self.minDistanceBetweenGrids, self.maxDistanceBetweenGrids)
        trialIndex = 0
        while trialIndex < len(designValues):
            self.experimentValues["angle"] = angle
            results, bean1Grid, playerGrid = self.trial(bean1Grid, bean2Grid, playerGrid)
            response = self.experimentValues.copy()
            response.update(results)
            responseDF = pd.DataFrame(response, index=[trialIndex])
            self.writer(responseDF)
            bean2Grid, nextCondition, angle = self.updateWorld(bean1Grid, playerGrid, designValues[trialIndex])
            self.experimentValues["angle"] = angle
            self.experimentValues["condition"] = nextCondition
            designValues = adjustDesignValues(nextCondition, trialIndex, designValues)
            trialIndex = trialIndex + 1
            if trialIndex in self.restTrial:
                self.drawImage(self.restImage)

        self.drawImage(self.finishImage)


def main():
    from Visualization import DrawBackground, DrawNewState, DrawImage
    from Controller import HumanController, ModelController, CheckBoundary
    import UpdateWorld
    from Writer import WriteDataFrameToCSV
    from Trial import Trial

    dimension = 15
    bounds = [0, 0, dimension - 1, dimension - 1]
    condition = [-5, -3, -1, 0, 1, 3, 5]
    minDistanceBetweenGrids = max(condition) + 1
    maxDistanceBetweenGrids = UpdateWorld.calculateMaxDistanceOfGrid(bounds) - minDistanceBetweenGrids
    block = 15
    initialWorld = UpdateWorld.InitialWorld(bounds)
    updateWorld = UpdateWorld.UpdateWorld(bounds, condition, minDistanceBetweenGrids, maxDistanceBetweenGrids)
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
    softmaxBeta = -1
    episilonGreedy = 0.8
    pg.event.set_allowed([pg.KEYDOWN, pg.QUIT])
    picturePath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Pictures/'
    resultsPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Results/'
    policyPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/machinePolicy/'
    humanController = HumanController(dimension)
    policyFile = open(policyPath + "noise0commitSnakeGoalGird15_policy.pkl", "rb")
    policy = pickle.load(policyFile)
    modelController = ModelController(policy, dimension, softmaxBeta, episilonGreedy)
    controller = humanController
    experimentValues = co.OrderedDict()
    experimentValues["name"] = input("Please enter your name:").capitalize()
    experimentValues["condition"] = 'None'
    writerPath = resultsPath + experimentValues["name"] + '.csv'
    writer = WriteDataFrameToCSV(writerPath)
    introductionImage = pg.image.load(picturePath + 'introduction.png')
    restImage = pg.image.load(picturePath + 'rest.png')
    finishImage = pg.image.load(picturePath + 'finish.png')
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))
    drawBackground = DrawBackground(screen, dimension, leaveEdgeSpace, backgroundColor, lineColor, lineWidth,
                                    textColorTuple)
    checkBoundary = CheckBoundary([0, dimension - 1], [0, dimension - 1])
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawImage = DrawImage(screen)
    designValues = UpdateWorld.createDesignValues(condition * 3, block)
    restTrial = list(range(0, len(designValues), len(condition) * 15))
    trial = Trial(controller, drawNewState, checkBoundary)
    experiment = Experiment(trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath,
                            minDistanceBetweenGrids, maxDistanceBetweenGrids, restImage, finishImage, restTrial)
    drawImage(introductionImage)
    experiment(designValues)


if __name__ == "__main__":
    main()
