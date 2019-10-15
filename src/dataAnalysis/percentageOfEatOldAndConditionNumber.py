import pandas as pd
import matplotlib.pyplot as plt
import os
import pylab as pl
import numpy as np
from numpy import log


def createAllCertainFormatFileList(filePath, fileFormat):
    filenameList = [os.path.join(filePath, relativeFilename) for relativeFilename in os.listdir(filePath)
                    if os.path.isfile(os.path.join(filePath, relativeFilename))
                    if os.path.splitext(relativeFilename)[1] in fileFormat]
    return filenameList


def cleanDataFrame(rawDataFrame):
    cleanConditionDataFrame = rawDataFrame[rawDataFrame.condition != "None"]
    cleanBeanEatenDataFrame = cleanConditionDataFrame[cleanConditionDataFrame.beanEaten != 0]
    return cleanBeanEatenDataFrame


if __name__ == "__main__":
    resultsPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/Results/'
    fileFormat = '.csv'
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    resultsDataFrame = pd.concat(resultsDataFrameList, sort=False)
    resultsDataFrame = cleanDataFrame(resultsDataFrame)
    participantsTypeList = ['machine' if 'machine' in name else 'Human' for name in resultsDataFrame['name']]
    conditionData = pd.Series(resultsDataFrame['condition'].values, index=list(range(resultsDataFrame.iloc[:, 0].size)))
    resultsDataFrame['participantsType'] = participantsTypeList
    resultsDataFrame['beanEaten'] = resultsDataFrame['beanEaten'] - 1
    trialNumberEatNewDataFrame = resultsDataFrame.groupby(['name', 'condition', 'participantsType']).sum()['beanEaten']
    trialNumberTotalEatDataFrame = resultsDataFrame.groupby(['name', 'condition', 'participantsType']).count()['beanEaten']
    mergeConditionDataFrame = pd.DataFrame(trialNumberEatNewDataFrame.values / trialNumberTotalEatDataFrame.values, index=trialNumberTotalEatDataFrame.index, columns=['eatNewPercentage'])
    mergeConditionDataFrame['eatOldPercentage'] = 1 - mergeConditionDataFrame['eatNewPercentage']
    mergeParticipantsDataFrameMean = mergeConditionDataFrame.groupby(['condition', 'participantsType']).mean()
    mergeParticipantsDataFrameStandardError = mergeConditionDataFrame.groupby(['condition', 'participantsType']).std()
    mergeConditionDataFrame.groupby(['condition', 'participantsType'])
    drawEatOldDataFrameMean = mergeParticipantsDataFrameMean['eatOldPercentage'].unstack('participantsType')
    print(drawEatOldDataFrameMean)

    orderedCondition = ['-5', '-3', '-1', '0', '1', '3', '5']
    # orderedCondition = [-5,-3,-1,0,1,3,5]
    drawEatOldDataFrameMean['orderedCondition'] = drawEatOldDataFrameMean.index
    drawEatOldDataFrameMean['orderedCondition'] = drawEatOldDataFrameMean['orderedCondition'].astype('category')
    drawEatOldDataFrameMean['orderedCondition'].cat.reorder_categories(orderedCondition, inplace=True)
    drawEatOldDataFrameMean.sort_values('orderedCondition', inplace=True)
    drawEatOldDataFrameError = mergeParticipantsDataFrameStandardError['eatOldPercentage'].unstack('participantsType')
    ax = drawEatOldDataFrameMean.plot.bar(yerr=drawEatOldDataFrameError, color=['lightsalmon', 'lightseagreen'], ylim=[0.0, 1.1], width=0.8)
    plt.xticks(rotation=0)
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('Distance(new - old)', fontweight='bold')
    ax.set_ylabel('Percentage of Eat Old', fontweight='bold')
    plt.show()
    condition = ['-5', '-3', '-1', '0', '1', '3', '5']
    # conditionFrequency=resultsDataFrame['condition'].value_counts()
    # conditionFrequency.reindex(index=condition)
    # conditionFrequency.plot(kind='bar')
    # plt.show()
    # crossEntropy=(drawEatOldDataFrameMean.apply(lambda x:-x['Human']*log(x['machine']),axis=1)).sum()
    # print(crossEntropy)
