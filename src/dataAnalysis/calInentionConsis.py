import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
from dataAnalysis import *


def isEatOld(beanEaten):
    if beanEaten == 1:
        return True
    else:
        return False


def cleanDataFrame(rawDataFrame):
    cleanConditionDataFrame = rawDataFrame[rawDataFrame.condition != "None"]
    cleanBeanEatenDataFrame = cleanConditionDataFrame[cleanConditionDataFrame.beanEaten != 0]
    return cleanBeanEatenDataFrame


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '../..'), 'results')
    commitmentRatioList = []
    stdList = []
    participants = ['human', 'maxModel']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)

        df = cleanDataFrame(df)

        # df.to_csv("all.csv")
        # print(df.head(6))

        nubOfSubj = len(df["name"].unique())
        statDF = pd.DataFrame()
        print('participant', participant, nubOfSubj)

        df["eatOld"] = df.apply(lambda x: isEatOld(x['beanEaten']), axis=1)

        statDF['eatOldRatio'] = df.groupby('condition')["eatOld"].mean().sort_values()
        statDF['eatOldRatioSE'] = df.groupby('condition')["eatOld"].apply(calculateSE)
        print(statDF)

       # statDF.to_csv("statDF.csv")
        print('eatOldRatio', np.mean(statDF['eatOldRatio']))
        print('')
        commitmentRatioList.append(statDF['eatOldRatio'].tolist())
        stdList.append(statDF['eatOldRatioSE'].tolist())

    condition = ['-5', '-3', '-1', '0', '1', '3', '5']
    x = np.arange(len(condition))
    totalWidth, n = 0.6, len(participants)
    width = totalWidth / n
    x = x - (totalWidth - width) / 2
    for i in range(len(commitmentRatioList)):
        plt.bar(x + width * i, commitmentRatioList[i],yerr=stdList[i], width=width, label=participants[i])
    plt.xticks(x, condition)

    plt.xlabel('Distance(new - old)')
    plt.ylim((0, 1))
    plt.legend(loc='best')
    plt.title('commit to old ratio')
    plt.show()
