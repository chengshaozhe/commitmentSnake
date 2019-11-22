import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
from collections import Counter
from dataAnalysis import *


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '../..'), 'results')
    statsList = []
    stdList = []
    participants = ['human', 'maxModel']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        nubOfSubj = len(df["name"].unique())
        statDF = pd.DataFrame()

        # df.to_csv("all.csv")
        # print(df.head(6))

        df['avoidCommitmentZone'] = df.apply(lambda x: calculateAvoidCommitmnetZone([x['playerGridX'], x['playerGridY']], [x['bean1GridX'], x['bean1GridY']], [x['bean2GridX'], x['bean2GridY']]), axis=1)

        dfExpTrial = df[df['condition'] == '0']
        # dfExpTrial = df

        dfExpTrial['avoidCommitmentRatio'] = dfExpTrial.apply(lambda x: calculateAvoidCommitmentRatio(eval(x['trajectory']), x['avoidCommitmentZone']), axis=1)
        statDF['avoidCommitmentRatio'] = dfExpTrial.groupby('name')["avoidCommitmentRatio"].mean()

        dfExpTrial['firstIntentionRatio'] = dfExpTrial.apply(lambda x: calculateFirstIntentionRatio(eval(x['goal'])), axis=1)
        statDF['firstIntentionRatio'] = dfExpTrial.groupby('name')["firstIntentionRatio"].mean()

        # df.to_csv("all.csv")
        # print(df.head(6))
        print(nubOfSubj)

        print('avoidCommitmentRatio', np.mean(statDF['avoidCommitmentRatio']))
        print('firstIntentionRatio', np.mean(statDF['firstIntentionRatio']))

        statsList.append([np.mean(statDF['firstIntentionRatio']), np.mean(statDF['avoidCommitmentRatio'])])
        stdList.append([calculateSE(statDF['firstIntentionRatio']),calculateSE(statDF['avoidCommitmentRatio'])])

    xlabels = ['firstIntentionRatio', 'avoidCommitmentAreaRatio']
    labels = participants
    x = np.arange(len(xlabels))
    totalWidth, n = 0.6, len(xlabels)
    width = totalWidth / n
    x = x - (totalWidth - width) / 2
    for i in range(len(statsList)):
        plt.bar(x + width * i, statsList[i],yerr =stdList[i] , width=width, label=labels[i])
    plt.xticks(x, xlabels)
    plt.ylim((0, 0.5))
    plt.legend(loc='best')
    plt.title('avoidCommitmentRatio distance=0')
    plt.show()
