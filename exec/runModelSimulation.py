import pygame as pg
import os
import collections as co
import numpy as np
import pickle
import sys
import math
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from src.Visualization import DrawBackground, DrawNewState, DrawImage
from src.Controller import HumanController, CheckBoundary
from src.UpdateWorld import *
from src.Trial import Trial
from src.Experiment import MoldelExperiment
from src.Writer import WriteDataFrameToCSV


def main():
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
    softmaxBeta = 2.5
    episilonGreedy = 1
    pg.event.set_allowed([pg.KEYDOWN, pg.QUIT])

    picturePath = os.path.join(os.path.join(DIRNAME, '../..'), '/pictures/')
    resultsPath = os.path.join(os.path.join(DIRNAME, '../..'), '/results/')
    policyPath = os.path.join(os.path.join(DIRNAME, '../..'), 'machinePolicy/')
    humanController = HumanController(dimension)
    policyFile = open(policyPath + "noise0commitSnakeGoalGird15_policy.pkl", "rb")
    policy = pickle.load(policyFile)
    modelController = ModelController(policy, dimension, softmaxBeta, episilonGreedy)
    controller = modelController

    numberOfMachineRun = 20
    for i in range(numberOfMachineRun):
        experimentValues = co.OrderedDict()
        experimentValues["name"] = 'machineEpisilon' + str(episilonGreedy) + "_" + str(i)
        experimentValues["condition"] = 'None'
        writerPath = resultsPath + experimentValues["name"] + '.csv'
        writer = WriteDataFrameToCSV(writerPath)
        drawBackground = DrawBackground(screen, dimension, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
        checkBoundary = CheckBoundary([0, dimension - 1], [0, dimension - 1])
        drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
        drawImage = DrawImage(screen)
        designValues = UpdateWorld.createDesignValues(condition * 3, block)
        trial = Trial(controller, drawNewState, checkBoundary)
        experiment = Experiment(trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath, minDistanceBetweenGrids, maxDistanceBetweenGrids)
        experiment(designValues)


if __name__ == "__main__":
    main()
