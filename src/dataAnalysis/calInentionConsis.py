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


def judgeStraightCondition(player, target1, target2):
    if player[0] == target1[0] or player[0] == target2[0] or player[1] == target1[1] or player[1] == target2[1]:
        straightCondition = True
    else:
        straightCondition = False
    return straightCondition


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '../..'), 'results')
    commitmentRatioList = []
    stdList = []
    participants = ['human', 'softMaxBeta0.5']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        nubOfSubj = len(df["name"].unique())
        print('participant', participant, nubOfSubj)
        # print(df.columns)

        df = cleanDataFrame(df)
        df['straightCondition'] = df.apply(lambda x: judgeStraightCondition((x['playerGridX'], x['playerGridY']), (x['bean1GridX'], x['bean1GridY']), (x['bean2GridX'], x['bean2GridY'])), axis=1)

        df = df[df['straightCondition'] != 1]

        statDF = pd.DataFrame()
        df["eatOld"] = df.apply(lambda x: isEatOld(x['beanEaten']), axis=1)

        df.condition = df.apply(lambda x: eval(x['condition']), axis=1)
        statDF['eatOldRatio'] = df.groupby('condition')["eatOld"].mean().sort_index()
        statDF['eatOldRatioSE'] = df.groupby('condition')["eatOld"].apply(calculateSE)
        # print(statDF)

       # statDF.to_csv("statDF.csv")
        print('eatOldRatio', np.mean(statDF['eatOldRatio']))
        print('')
        commitmentRatioList.append(statDF['eatOldRatio'])
        stdList.append(statDF['eatOldRatioSE'])

    condition = sorted(set(df.condition))
    x = np.arange(len(condition))
    totalWidth, n = 0.6, len(participants)

    labels = ['human', 'RL agent']
    width = totalWidth / n
    x = x - (totalWidth - width) / 2
    for i in range(len(commitmentRatioList)):
        plt.bar(x + width * i, commitmentRatioList[i], yerr=stdList[i], width=width, label=labels[i])
    plt.xticks(x, condition)

    plt.xlabel('Distance Difference(new - old)')
    plt.ylabel('Eat Old Ratio')
    plt.ylim((0, 1))
    plt.legend(loc='best')
    plt.title('Commitment to Future')
    plt.show()
