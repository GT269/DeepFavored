import sys, os, json, re, random, warnings
from tqdm import tqdm
import pandas as pd
import numpy as np

# ===========================
#          Tools
# ===========================
def shuffle(X, Y=None):
    np.random.seed(13)
    indices = np.random.permutation(len(X))
    X = X[indices]
    if Y is None:
        return X
    else:
        Y = Y[indices]
        return X, Y

def ReadFiles(inPath, files):
    if os.path.isfile(inPath):
        files.append(inPath)
    if os.path.isdir(inPath):
        for term in os.listdir(inPath):
            files = ReadFiles(inPath+'/'+term, files)
    return files

def RandomSample(dataArray, sampleNum):
    dataArray = shuffle(dataArray)
    sampleTotalNum = dataArray.shape[0]
    if sampleTotalNum < sampleNum:
        info = 'There are %i muts, but %i muts are needed, thus tiggering randomly sampling with replacement.' % (sampleTotalNum, sampleNum)
        warnings.warn(info, UserWarning)
        dataArrayNew = list(dataArray)
        diff = sampleNum - sampleTotalNum
        for i in range(diff):
            oneSample = random.choice(dataArray)
            dataArrayNew.append(oneSample)
        dataArray = np.array(dataArrayNew)
    else:
        dataArray = dataArray[:sampleNum]
    return dataArray

# ===========================
#          Read data
# ===========================
def ReadData(inPath, features, mutNum):
    if 'favored' not in inPath:
        features = [feature+'_norm' for feature in features]
    X = []
    files = ReadFiles(inPath, files=[])
    for file in files:
        with open(file, 'r') as fr:
            rows = fr.readlines()
        headerCols = rows[0].rstrip('\n').split('\t')
        featureIdxList = []
        for feature in features:
            featureIdxList.append(headerCols.index(feature))
        for row in rows[1:]:
            cols = row.rstrip('\n').split('\t')
            vals = []
            for idx in featureIdxList:
                vals.append(float(cols[idx]))
            X.append(vals)
    X = np.array(X)
    X = X.astype(np.float32)
    X = RandomSample(X, mutNum)
    return X

def GenerateLabel(rowsNum, positive_set, labelType):
    labels = []
    if labelType in ['BinaryLabelVector', '2D_binaryVector', '2dBinaryVector']:
        if positive_set:
            one_label = [1.0,0.0]
        else:
            one_label = [0.0,1.0]
        labels = [one_label]*rowsNum
    labels = np.array(labels)
    return labels

def LoadTrainData(trainDataPath, argsDict, labelType):
    print('Load train data...')
    ### Read data
    favMutData = ReadData(trainDataPath+'/favored_mutations', features=argsDict['componentStats'], mutNum=argsDict['favMutNum'])
    hitchNeutMutData = ReadData(trainDataPath+'/hitchhiking_neutral_mutations', features=argsDict['componentStats'], mutNum=argsDict['hitchNeutMutNum'])
    ordNeutMutData = ReadData(trainDataPath+'/ordinary_neutral_mutations', features=argsDict['componentStats'], mutNum=argsDict['ordNeutMutNum'])
    
    ### Generate label
    favMutLabel = GenerateLabel(rowsNum=favMutData.shape[0], positive_set=True, labelType=labelType)
    hitchMutLabel = GenerateLabel(rowsNum=hitchNeutMutData.shape[0], positive_set=False, labelType=labelType)
    ordNeutMutLabel = GenerateLabel(rowsNum=ordNeutMutData.shape[0], positive_set=False, labelType=labelType)
    
    ### Compose train data
    # fav muts and hitch muts
    X1 = np.vstack((favMutData,hitchNeutMutData))
    Y1 = np.vstack((favMutLabel,hitchMutLabel))

    # fav muts and ordinary neut muts
    X2 = np.vstack((favMutData,ordNeutMutData))
    Y2 = np.vstack((favMutLabel,ordNeutMutLabel))

    X1, Y1 = shuffle(X1, Y1)
    X2, Y2 = shuffle(X2, Y2)

    return X1, Y1, X2, Y2