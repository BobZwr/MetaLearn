import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import auc

def check(label):
    # decide whether to adopt the ith data
    if sum('VF/VT' in i for i in label) > 20:
        return True
    return False

def read_data( filename ):
    '''
     *** Description ***

     Read the data in filename
     Return the dictionary Data & Label grouped by the pid
    '''

    Data = dict()
    Label = dict()
    data = np.load('data/' + filename + '_data.npy')
    pid = np.load('data/' + filename + '_pid.npy')
    label = np.load('data/' + filename + '_label.npy', allow_pickle=True)
    name = np.unique(pid)
    for i in name:
        if not check(label[pid == i]):
            continue
        Data[i] = data[pid == i]
        Label[i] = label[pid == i]
    return Data, Label


def create_sets(train_size):
    '''
    *** Description ***

    Create the sets for pre_training and fine_tune.
    Train means pre_training, and test means fine_tune.
    '''

    Data = []
    TmpLabel = []
    files = ['cudb', 'mitdb', 'vfdb']
    for i in files:
        data, label = read_data(i)
        Data.extend(data.values())
        TmpLabel.extend(label.values())
    Data = np.array(Data, dtype=object)
    Label = []

    #encode Label
    for i in TmpLabel:
        Label.append(list(int('VF/VT' in j) for j in i))
    Label = np.array(Label)
    index = np.random.permutation(len(Label))
    Data = Data[index]
    Label = Label[index]

    ## scale data
    for i in range(len(Data)):
        tmp_data = Data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        Data[i] = (tmp_data - tmp_mean) / tmp_std

    XTrain, XTest, YTrain, YTest = train_test_split(Data, Label, train_size=train_size, random_state=0)
    return XTrain, XTest, YTrain, YTest


def pair_shuffle(data, label):
    index = np.random.permutation(len(label))
    return data[index], label[index]

def FilterNwaysKshots(data, label, N, train_shots, test_shots = 1, remain = False):
    '''
    ** Description **
    randomly pick N-way K-shot data from the whole set
    :param data: data
    :param label: label
    :param N: number of classes
    :param train_shots: number of shots each class in train_set
    :param test_shots: number of shots each class in test_set
    :param remain: whether to return the remaining data
    :return:train_set, test_set, (remaining data)
    '''
    # data, label must be ndarray
    name = np.unique(label)
    np.random.shuffle(name)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    remain_x = []
    remain_y = []
    for i in name[0 : N]:
        is_name = label == i
        l, d = label[is_name], data[is_name]
        if not len(l) >= train_shots + test_shots:
            raise IndexError("dataprocess: FilterNwaysKshots: we lack some class of data")
        index = np.random.permutation(len(l))
        train_y.extend(l[index[: train_shots]])
        train_x.extend(d[index[: train_shots]])
        test_x.extend(d[index[train_shots : train_shots + test_shots]])
        test_y.extend(l[index[train_shots : train_shots + test_shots]])
        remain_x.extend(d[index[train_shots + test_shots : ]])
        remain_y.extend(l[index[train_shots + test_shots : ]])

    if remain == False:
        return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

    for i in name[N : ]:
        remain_x.extend(data[label == i])
        remain_y.extend(label[label == i])
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), np.array(remain_x), np.array(remain_y)

def calc_rate(prob, label, threshold):
    all_number = len(prob)
    TP = FP = FN = TN = 0
    for i in range(all_number):
        if  prob[i] > threshold:
            if label[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if label[i] == 0:
                TN += 1
            else:
                FN += 1
    accracy = (TP + FP) / all_number
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    TPR = TP / (TP + FN)
    TNR = 0 if TN == 0 else TN / (FP + TN)
    FNR = 0 if FN == 0 else FN / (TP + FN)
    FPR = FP / (FP + TN)
    return accracy, precision, TPR, TNR, FNR, FPR


def roc_curve(prob, label):
    '''
    ** Description **
    Draw roc curve and calculate the AUC
    **
    :param prob: prob of positive class
    :param label: corresponding label
    :return: AUC
    '''
    threshold_vaule = sorted(prob)
    threshold_num = len(threshold_vaule)
    accracy_array = np.zeros(threshold_num)
    precision_array = np.zeros(threshold_num)
    TPR_array = np.zeros(threshold_num)
    FPR_array = np.zeros(threshold_num)
    for thres in range(threshold_num):
        accracy, precision, TPR, _, _, FPR = calc_rate(prob, label, threshold_vaule[thres])
        accracy_array[thres] = accracy
        precision_array[thres] = precision
        TPR_array[thres] = TPR
        FPR_array[thres] = FPR
    AUC = auc(FPR_array, TPR_array)
    # plt.plot(FPR_array, TPR_array)
    # plt.title('roc')
    # plt.xlabel('FPR_array')
    # plt.ylabel('TPR_array')
    # plt.show()
    return AUC