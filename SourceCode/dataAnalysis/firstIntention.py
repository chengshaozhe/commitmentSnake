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
    resultsPath = os.path.abspath(os.path.join(os.getcwd(),"../.." )) + '/Results/'
    fileFormat = '.csv'
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    resultsDataFrame = pd.concat(resultsDataFrameList,sort=False)
    resultsDataFrame=cleanDataFrame(resultsDataFrame)
    participantsTypeList = ['machine' if 'machine' in name else 'Human' for name in resultsDataFrame['name']]
    conditionData= pd.Series(resultsDataFrame['condition'].values, index=list(range(resultsDataFrame.iloc[:,0].size)))
    resultsDataFrame['participantsType']=participantsTypeList
    resultsDataFrame['firstIntention']=resultsDataFrame['goal'].apply(calculateFirstIntentionStep)
    resultsDataFrame['firstIntention']=resultsDataFrame['firstIntention']-1
    resultsDataFrame['beanEaten']=resultsDataFrame['beanEaten']-1
    resultsDataFrame["firstIntentionConsist"]=resultsDataFrame["firstIntention"]==resultsDataFrame["beanEaten"]
    trialNumberConsistencyDataFrame = resultsDataFrame.groupby(['name', 'condition', 'participantsType']).sum()["firstIntentionConsist"]
    trialNumberTotalEatDataFrame = resultsDataFrame.groupby(['name', 'condition', 'participantsType']).count()[
        "firstIntentionConsist"]
    print(trialNumberTotalEatDataFrame)

    mergeConditionDataFrame = pd.DataFrame(trialNumberConsistencyDataFrame.values / trialNumberTotalEatDataFrame.values,
                                           index=trialNumberTotalEatDataFrame.index, columns=['firstIntentionConsistency'])
    mergeParticipantsDataFrameMean = mergeConditionDataFrame.groupby(['condition','participantsType']).mean()
    mergeParticipantsDataFrameStandardError = mergeConditionDataFrame.groupby(['condition', 'participantsType']).std()
    mergeConditionDataFrame.groupby(['condition', 'participantsType'])
    drawConsistencyDataFrameMean=mergeParticipantsDataFrameMean['firstIntentionConsistency'].unstack('participantsType')
    orderedCondition=['-5','-3','-1','0','1','3','5']
    drawConsistencyDataFrameMean['orderedCondition']=drawConsistencyDataFrameMean.index
    drawConsistencyDataFrameMean['orderedCondition']=drawConsistencyDataFrameMean['orderedCondition'].astype('category')
    drawConsistencyDataFrameMean['orderedCondition'].cat.reorder_categories(orderedCondition,inplace=True)
    drawConsistencyDataFrameMean.sort_values('orderedCondition',inplace=True)
    drawConsistencyDataFrameError=mergeParticipantsDataFrameStandardError['firstIntentionConsistency'].unstack('participantsType')
    ax=drawConsistencyDataFrameMean.plot.bar(yerr=drawConsistencyDataFrameError,color=['lightsalmon', 'lightseagreen'],ylim=[0.0,1.1],width=0.8)
    pl.xticks(rotation=0)
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('Distance(new - old)',fontweight='bold')
    ax.set_ylabel('consistency of first intention and final goal',fontweight='bold')
    plt.legend(loc='right')
    plt.show()

	# crossEntropy=(drawConsistencyDataFrameMean.apply(lambda x:-x['Human']*log(x['machine']),axis=1)).sum()
	# print(crossEntropy)

    # mergeConditionDataFrame = pd.DataFrame(trialNumberConsistencyDataFrame.values / trialNumberTotalEatDataFrame.values,
    #                                        index=trialNumberTotalEatDataFrame.index,
    #                                        columns=['firstIntentionConsistency'])
    # print(mergeConditionDataFrame)
    # mergeParticipantsDataFrameMean = mergeConditionDataFrame.groupby([ 'participantsType']).mean()
    # mergeParticipantsDataFrameStandardError = mergeConditionDataFrame.groupby([ 'participantsType']).std()
    # mergeConditionDataFrame.groupby([ 'participantsType'])
    # print(mergeConditionDataFrame)
    # drawConsistencyDataFrameMean = mergeParticipantsDataFrameMean['firstIntentionConsistency']
    # drawConsistencyDataFrameError = mergeParticipantsDataFrameStandardError['firstIntentionConsistency']
    # ax = drawConsistencyDataFrameMean.plot.bar(yerr=drawConsistencyDataFrameError,color=['lightsalmon', 'lightseagreen'],
    #                                             width=0.5)
    # pl.xticks(rotation=0)
    # plt.yticks(np.arange(0, 1.1, 0.1))
    # ax.set_xlabel('Distance(new - old)', fontweight='bold')
    # ax.set_ylabel('consistency of first intention and final goal', fontweight='bold')
    # plt.show()
