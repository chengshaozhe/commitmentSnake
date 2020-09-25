import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
from collections import Counter


def calculateSE(data):
    standardError = np.std(data, ddof=1) / np.sqrt(len(data) - 1)
    return standardError


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '../..'), 'results')
    statsList = []
    stdList = []
    # participants = ['human', 'softmaxBeta0.5', 'rewardVariance5','rewardVariance10', 'rewardVariance30' ,'rewardVariance50']

    participants = ['human','softmaxBeta0.1', 'softmaxBeta2.5', 'max']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)

        df['totalStep'] = df.apply(lambda x: len(eval(x['trajectory'])), axis=1)

        # df.to_csv("all.csv")
        nubOfSubj = len(df["name"].unique())
        print(participant, nubOfSubj)

        # dfExpTrail = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] != 'none')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'midLine')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'straightLine')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'straightLine') & (df['intentionedDisToTargetMin'] == 2)]

        # dfExpTrail = df[(df['areaType'] == 'straightLine') | (df['areaType'] == 'midLine') & (df['distanceDiff'] == 0)]
        # dfExpTrail = df[(df['areaType'] != 'none')]
        # dfExpTrail = df[(df['areaType'] == 'expRect') & (df['areaType'] != 'rect')]

        # dfExpTrail = df[df['noiseNumber'] != 'special']
        dfExpTrail = df

        statDF = pd.DataFrame()
        # statDF['firstIntentionStep'] = dfExpTrail.groupby('name')["firstIntentionStep"].mean()
        statDF['totalStep'] = dfExpTrail.groupby('name')["totalStep"].mean()

        # print('firstIntentionStep', np.mean(statDF['firstIntentionStep']))
        print('')

        stats = statDF.columns
        statsList.append([np.mean(statDF[stat]) for stat in stats])
        stdList.append([calculateSE(statDF[stat]) for stat in stats])

    xlabels = ['totalStep']
    labels = participants
    x = np.arange(len(xlabels))
    totalWidth, n = 0.1, len(participants)
    width = totalWidth / n
    x = x - (totalWidth - width) / 2
    for i in range(len(statsList)):
        plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=labels[i])
    plt.xticks(x, xlabels)
    # plt.ylim((0, 10))
    plt.legend(loc='best')

    plt.title('totalStep')
    plt.show()
