import pandas as pd
import matplotlib.pyplot as plt
import os
import pylab as pl
import numpy as np
from numpy import log

def createAllCertainFormatFileList(filePath,fileFormat):
	filenameList=[os.path.join(filePath,relativeFilename) for relativeFilename in os.listdir(filePath)
		if os.path.isfile(os.path.join(filePath,relativeFilename))
		if os.path.splitext(relativeFilename)[1] in fileFormat]
	return filenameList

def cleanDataFrame(rawDataFrame):
	cleanConditionDataFrame=rawDataFrame[rawDataFrame.condition != "None"]
	cleanBeanEatenDataFrame=cleanConditionDataFrame[cleanConditionDataFrame.beanEaten!=0]
	return cleanBeanEatenDataFrame

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

if __name__=="__main__":
    resultsPath = os.path.abspath(os.path.join(os.getcwd(),"../.." )) + '/Results/maxModel'
    fileFormat = '.csv'
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    resultsDataFrame = pd.concat(resultsDataFrameList,sort=False)
    resultsDataFrame=cleanDataFrame(resultsDataFrame)
    participantsTypeList = ['machine' if 'machine' in name else 'Human' for name in resultsDataFrame['name']]
    conditionData= pd.Series(resultsDataFrame['condition'].values, index=list(range(resultsDataFrame.iloc[:,0].size)))
    resultsDataFrame['participantsType']=participantsTypeList
    resultsDataFrame['firstIntentionStep']=resultsDataFrame['goal'].apply(calculateFirstIntentionStep)
    trialNumberConsistencyDataFrame = resultsDataFrame.groupby(['name', 'condition', 'participantsType']).mean()["firstIntentionStep"]
    mergeParticipantsDataFrame=pd.DataFrame(trialNumberConsistencyDataFrame,index=trialNumberConsistencyDataFrame.index,columns=['firstIntentionStep'])
    mergeParticipantsDataFrameMean = mergeParticipantsDataFrame.groupby(['condition','participantsType']).mean()
    mergeParticipantsDataFrameError = mergeParticipantsDataFrame.groupby(['condition','participantsType']).std()
    mergeParticipantsDataFrame.groupby(['condition', 'participantsType'])
    orderedCondition=['-5','-3','-1','0','1','3','5']
    drawEatOldDataFrameMean = mergeParticipantsDataFrameMean['firstIntentionStep'].unstack('participantsType')
    drawEatOldDataFrameMean['orderedCondition']=drawEatOldDataFrameMean.index
    drawEatOldDataFrameMean['orderedCondition']=drawEatOldDataFrameMean['orderedCondition'].astype('category')
    drawEatOldDataFrameMean['orderedCondition'].cat.reorder_categories(orderedCondition,inplace=True)
    drawEatOldDataFrameMean.sort_values('orderedCondition',inplace=True)
    drawEatOldDataFrameError=mergeParticipantsDataFrameError['firstIntentionStep'].unstack('participantsType')
    ax=drawEatOldDataFrameMean.plot.bar(yerr=drawEatOldDataFrameError,color=['lightsalmon', 'lightseagreen'],ylim=[0.0,1.1],width=0.8)
    pl.xticks(rotation=0)
    plt.yticks(np.arange(0, 6, 0.5))
    ax.set_xlabel('Distance(new - old)',fontweight='bold')
    ax.set_ylabel('first intention step',fontweight='bold')
    plt.show()

	# crossEntropy=(drawConsistencyDataFrameMean.apply(lambda x:-x['Human']*log(x['machine']),axis=1)).sum()
	# print(crossEntropy)



