import numpy as np
import pygame as pg
from pygame import time
import collections as co
import pickle
from Visualization import DrawBackground,DrawNewState,DrawImage
from Controller import HumanController,ModelController,CheckBoundary
import UpdateWorld



def inferGoal(originGrid, aimGrid, targetGridA, targetGridB):
    pacmanBean1aimDisplacement = np.linalg.norm(np.array(targetGridA) - np.array(aimGrid), ord=1)
    pacmanBean2aimDisplacement = np.linalg.norm(np.array(targetGridB) - np.array(aimGrid), ord=1)
    pacmanBean1LastStepDisplacement = np.linalg.norm(np.array(targetGridA) - np.array(originGrid), ord=1)
    pacmanBean2LastStepDisplacement = np.linalg.norm(np.array(targetGridB) - np.array(originGrid), ord=1)
    bean1Goal = pacmanBean1LastStepDisplacement - pacmanBean1aimDisplacement
    bean2Goal = pacmanBean2LastStepDisplacement - pacmanBean2aimDisplacement
    if bean1Goal > bean2Goal:
        goal = 1
    elif bean1Goal < bean2Goal:
        goal = 2
    else:
        goal = 0
    return goal

class Trial():
    def __init__(self,controller,drawNewState,checkBoundary):
        self.controller=controller
        self.drawNewState=drawNewState
        self.checkBoundary=checkBoundary

    def checkEaten(self,bean1Grid, bean2Grid, humanGrid):
        if np.linalg.norm(np.array(humanGrid) - np.array(bean1Grid), ord=1)==0:
            eatenFlag=[True,False]
        elif np.linalg.norm(np.array(humanGrid) - np.array(bean2Grid), ord=1) == 0:
            eatenFlag=[False,True]
        else:
            eatenFlag=[False,False]
        return eatenFlag

    def checkTerminationOfTrial(self,action,eatenFlag):
        if np.any(eatenFlag)==True or action==pg.QUIT :
            pause=False
        else:
            pause=True
        return pause

    def __call__(self,bean1Grid,bean2Grid,playerGrid):
        initialPlayerGrid = playerGrid
        initialTime = time.get_ticks()
        reactionTime = list()
        actionList=list()
        goalList=list()
        trajectory = [initialPlayerGrid]
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP,pg.QUIT])
        results=co.OrderedDict()
        self.drawNewState(bean1Grid,bean2Grid,initialPlayerGrid)
        stepCount=0
        playerGrid, action= self.controller(bean1Grid, bean2Grid, playerGrid)
        goal = inferGoal(trajectory[-1], playerGrid, bean1Grid, bean2Grid)
        goalList.append(goal)
        eatenFlag = self.checkEaten(bean1Grid, bean2Grid, playerGrid)
        realPlayerGrid=self.checkBoundary(playerGrid)
        self.drawNewState(bean1Grid,bean2Grid,realPlayerGrid)
        reactionTime.append(time.get_ticks() - initialTime)
        trajectory.append(list(realPlayerGrid))
        actionList.append(action)
        stepCount=stepCount+1
        pause = self.checkTerminationOfTrial(action, eatenFlag)
        while pause:
            playerGrid, action = self.controller(bean1Grid, bean2Grid, realPlayerGrid)
            goal = inferGoal(trajectory[-1], playerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            eatenFlag = self.checkEaten(bean1Grid, bean2Grid, playerGrid)
            realPlayerGrid = self.checkBoundary(playerGrid)
            self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid)
            reactionTime.append(time.get_ticks() - reactionTime[-1])
            trajectory.append(list(realPlayerGrid))
            actionList.append(action)
            stepCount = stepCount + 1
            pause = self.checkTerminationOfTrial(action, eatenFlag)
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])
        results["bean1GridX"] = bean1Grid[0]
        results["bean1GridY"] = bean1Grid[1]
        results["bean2GridX"] = bean2Grid[0]
        results["bean2GridY"] = bean2Grid[1]
        results["playerGridX"] = initialPlayerGrid[0]
        results["playerGridY"] = initialPlayerGrid[1]
        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(actionList)
        results["goal"]=str(goalList)
        if True in eatenFlag:
            results["beanEaten"] = eatenFlag.index(True)+1
            oldGrid=eval('bean'+str(eatenFlag.index(False)+1)+'Grid')
        else:
            results["beanEaten"] = 0
            oldGrid=None
        return results,oldGrid,playerGrid







