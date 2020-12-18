import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
from dataAnalysis import *
import researchpy


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
    participants = ['human', 'softmaxBeta2.5']
    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]

    df = pd.concat(dfList)
    df['participantsType'] = ['machine' if 'max' in name else 'Human' for name in df['name']]
    df['straightCondition'] = df.apply(lambda x: judgeStraightCondition((x['playerGridX'], x['playerGridY']), (x['bean1GridX'], x['bean1GridY']), (x['bean2GridX'], x['bean2GridY'])), axis=1)

    df = df[df['straightCondition'] != 1]

    df["eatOld"] = df.apply(lambda x: isEatOld(x['beanEaten']), axis=1)
    df.condition = df.apply(lambda x: eval(x['condition']), axis=1)

    resultDf = df[df['condition'] == 0]
    # resultDf = df
    # crosstab, res = researchpy.crosstab(resultDf['participantsType'], resultDf['eatOld'], test="chi-square")
    # print(crosstab)
    # print(res)

    statDF = pd.DataFrame()
    statDF['eatOldRatio'] = resultDf.groupby('name')["eatOld"].mean()
    statDF = statDF.reset_index()
    statDF['participantsType'] = ['machine' if 'max' in name else 'Human' for name in statDF['name']]

    humanDf = statDF[statDF['participantsType'] == 'Human']
    machineDf = statDF[statDF['participantsType'] == 'machine']

    statDFList = [humanDf['eatOldRatio'].tolist(), machineDf['eatOldRatio'].tolist()]

    des, res = researchpy.ttest(pd.Series(statDFList[0], name='human'), pd.Series(statDFList[1], name='RL'))
    print(des)
    print(res)
    print('======')
    from pingouin import ttest
    print(ttest(statDFList[0], statDFList[1]).round(2))


# viz all condition sum
    VIZ = 1
    if VIZ:
        xlabels = ['Human', 'RL Agent']
        x = np.arange(len(xlabels))
        totalWidth, n = 1, len(xlabels)

        labels = ['Human', 'RL Agent']
        width = totalWidth / n
        x = x - (totalWidth - width) / 2

        commitmentRatioList = df.groupby('participantsType')["eatOld"].mean().tolist()
        stdList = df.groupby('participantsType')["eatOld"].apply(calculateSE)

        plt.bar(x + width, commitmentRatioList, yerr=stdList, width=width, color=['#E24A33', '#348ABD'])
        fontSize = 16
        plt.xticks(x + width, xlabels, fontsize=fontSize, color='black')
        plt.yticks(fontsize=fontSize, color='black')

        plt.ylabel('Reach Old Ratio', fontsize=fontSize, color='black')
        # plt.ylim((0, 1))
        plt.legend(loc='best')
        plt.title('Commitment to Future: All conditions Merged', fontsize=fontSize, color='black')
        plt.show()
