#!/usr/bin/env python

# -*- coding: utf-8 -*-

import numpy as np
import pygame as pg
import random
import Visualization

class HumanController():
	def __init__(self,gridSize):
		self.actionDict = {pg.K_UP: (0, -1), pg.K_DOWN: (0, 1), pg.K_LEFT: (-1, 0), pg.K_RIGHT: (1, 0)}
		self.actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
		self.gridSize = gridSize

	def __call__(self,targetPositionA,targetPositionB,playerPosition):
		action = [0, 0]
		pause = True
		while pause:
			for event in pg.event.get():
				if event.type == pg.KEYDOWN:
					if event.key in self.actionDict.keys():
						action = self.actionDict[event.key]
						newPlayerGrid = tuple(np.add(playerPosition, action))
						pause = False
		return newPlayerGrid, action

class CheckBoundary():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position):
        adjustedX, adjustedY = position
        if position[0] >= self.xMax:
            adjustedX = self.xMax
        if position[0] <= self.xMin:
            adjustedX =  self.xMin
        if position[1] >= self.yMax:
            adjustedY =  self.yMax
        if position[1] <= self.yMin:
            adjustedY = self.yMin
        checkedPosition =(adjustedX, adjustedY)
        return checkedPosition

def calculateSoftmaxProbability(probabilityList,beita):
	newProbabilityList=list(np.divide(np.exp(np.multiply(beita,probabilityList)),np.sum(np.exp(np.multiply(beita,probabilityList)))))
	return newProbabilityList

class ModelController():
	def __init__(self,policy,gridSize,softmaxBeta):
		self.policy=policy
		self.gridSize=gridSize
		self.softmaxBeta=softmaxBeta
		self.actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]

	def __call__(self,targetPositionA,targetPositionB,playerPosition):
		try:
			policyForCurrentStateDict = self.policy[(playerPosition, (targetPositionA, targetPositionB))]
		except KeyError as e:
			policyForCurrentStateDict = self.policy[(playerPosition, (targetPositionA, targetPositionB))]
		if self.softmaxBeta < 0:
			actionMaxList = [action for action in policyForCurrentStateDict.keys() if
							 policyForCurrentStateDict[action] == np.max(list(policyForCurrentStateDict.values()))]
			action = random.choice(actionMaxList)
		else:
			actionProbability = np.divide(list(policyForCurrentStateDict.values()),
										  np.sum(list(policyForCurrentStateDict.values())))
			softmaxProbabilityList = calculateSoftmaxProbability(list(actionProbability), self.softmaxBeta)
			action = list(policyForCurrentStateDict.keys())[
				list(np.random.multinomial(1, softmaxProbabilityList)).index(1)]
		aimePlayerGrid = tuple(np.add(playerGrid, action))
		pg.time.delay(0)
		return aimePlayerGrid, action

if __name__=="__main__":
	pg.init()
	screenWidth=720
	screenHeight=720
	screen=pg.display.set_mode((screenWidth,screenHeight))
	gridSize=20
	leaveEdgeSpace=2
	lineWidth=2
	backgroundColor=[188,188,0]
	lineColor=[255,255,255]
	targetColor=[255,50,50]
	playerColor=[50,50,255]
	targetRadius=10
	playerRadius=10
	targetPositionA=[5,5]
	targetPositionB=[15,5]
	playerPosition=[10,15]
	currentScore=5
	textColorTuple=(255,50,50)
	stopwatchEvent = pg.USEREVENT + 1
	stopwatchUnit=10
	pg.time.set_timer(stopwatchEvent, stopwatchUnit)
	finishTime=90000
	currentStopwatch=32000
	softmaxBeita=20

	drawBackground=Visualization.DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
	drawNewState=Visualization.DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)

	getHumanAction = HumanController(gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime)
	# newProbabilityList=calculateSoftmaxProbability([0.5,0.3,0.2],20)
	# print(newProbabilityList)
	import pickle
	policy=pickle.load(open("SingleWolfTwoSheepsGrid15.pkl","rb"))
	getModelAction = ModelController(policy, gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime, softmaxBeita)

	# [playerNextPosition,action,newStopwatch]=getHumanAction(targetPositionA, targetPositionB, playerPosition, currentScore, currentStopwatch)
	[playerNextPosition,action,newStopwatch]=getModelAction(targetPositionA, targetPositionB, playerPosition, currentScore, currentStopwatch)
	print(playerNextPosition,action,newStopwatch)

	pg.quit()

