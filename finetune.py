import numpy as np
from sklearn.model_selection import train_test_split
import numpy.random
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


from net1d import Net1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import MAML
import dataprocess as dp

def fine_tune(model, data, label, lr, classes, n_epoch, train_size = 5, batch_size = 1):
    '''
    ** Description **
    The main part of fine tune(quickly adaptation)
    :param model:
    :param data:
    :param label:
    :param lr:
    :param classes:
    :param n_epoch:
    :param train_size:
    :param batch_size:
    :return:
    '''
    data = np.expand_dims(data, 1)
    train_data, train_label, valid_data, valid_label, test_data, test_label = dp.FilterNwaysKshots(data = data,
                                                                                                   label = label,
                                                                                                   N = classes,
                                                                                                   train_shots = train_size,
                                                                                                   test_shots = train_size,
                                                                                                   remain = True)
    dataset_train = MyDataset(train_data, train_label)
    dataset_valid = MyDataset(valid_data, valid_label)
    dataset_test = MyDataset(test_data, test_label)
    trainset = DataLoader(dataset_train, batch_size = batch_size)
    validset = DataLoader(dataset_valid, batch_size = batch_size)
    testset = DataLoader(dataset_test, batch_size = batch_size)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)
    #train
    model.train()
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):
        # train
        model.train()
        #prog_iter = tqdm(trainset, desc="Training", leave=False)
        for batch_idx, batch in enumerate(trainset):

            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(_)

    #test
    model.eval()
    prog_iter_test = tqdm(testset, desc="Testing", leave=False)
    pred_prob = []
    acc_label = []
    acc = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter_test):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            pred = F.softmax(pred, dim = 1)
            pred_prob.append(pred[:, 1])
            acc_label.extend(list(int(i) for i in input_y))
    AUC = dp.roc_curve(pred_prob, acc_label)

    return AUC

def fine_tune2(model, data, label, lr, classes, n_epoch, train_size = 5, batch_size = 1):
    '''
    ** Description **
    The main part of fine tune(quickly adaptation)

    :param model:
    :param data:
    :param label:
    :param lr:
    :param classes:
    :param n_epoch:
    :param train_size:
    :param batch_size:
    :return:
    '''
    data = np.expand_dims(data, 1)
    train_data, train_label, valid_data, valid_label, test_data, test_label = dp.FilterNwaysKshots(data = data,
                                                                                                   label = label,
                                                                                                   N = classes,
                                                                                                   train_shots = train_size,
                                                                                                   test_shots = train_size,
                                                                                                   remain = True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    print(len(train_data))
    for tt in tqdm(range(n_epoch), desc="epoch", leave=False):
        dataset_train = MyDataset(train_data[:3]+train_data[5:8], train_label[:3]+train_label[5:8])
        dataset_valid = MyDataset(train_data[3:5]+train_data[8:], train_label[3:5]+train_label[8:])
        trainset = DataLoader(dataset_train, batch_size=batch_size)
        validset = DataLoader(dataset_valid, batch_size=batch_size)
        sum_loss = 0
        # train
        learner = model.clone()
        for _ in tqdm(range(1), desc='update_num', leave=False, colour='white'):
            learner.train()
            # train
            prog_iter = tqdm(trainset, desc='Epoch', leave=False, colour='yellow')
            for batch_idx, batch in enumerate(prog_iter):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = learner(input_x)
                loss = loss_func(pred, input_y)
                learner.adapt(loss=loss / batch_size)
            # test
            learner.eval()
            test_task = tqdm(validset, desc="Test")
            for batch_idx, batch in enumerate(test_task):
                x, y = tuple(t.to(device) for t in batch)
                pred = model(x)
                loss = loss_func(pred, y)
                pred = pred.argmax(dim=1)
                sum_loss += loss / batch_size
        sum_loss /= train_size
        optimizer.zero_grad()
        sum_loss.backward()
        optimizer.step()
        scheduler.step(tt)

    dataset_train = MyDataset(train_data, train_label)
    dataset_valid = MyDataset(valid_data, valid_label)
    dataset_test = MyDataset(test_data, test_label)
    trainset = DataLoader(dataset_train, batch_size = batch_size)
    validset = DataLoader(dataset_valid, batch_size = batch_size)
    testset = DataLoader(dataset_test, batch_size = batch_size)

    #train
    model.train()
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):
        # train

        #prog_iter = tqdm(trainset, desc="Training", leave=False)
        for batch_idx, batch in enumerate(trainset):

            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(_)

    #test
    model.eval()
    prog_iter_test = tqdm(testset, desc="Testing", leave=False)
    pred_prob = []
    acc_label = []
    acc = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter_test):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            pred = F.softmax(pred, dim = 1)
            pred_prob.append(pred[:, 1])
            acc_label.extend(list(int(i) for i in input_y))
    AUC = dp.roc_curve(pred_prob, acc_label)

    return AUC
def test_para(para, data, label):
    AUC_array = []
    AUC_array2 = []
    for i in tqdm(range(len(data)), desc = para + '_Task'):
        model = torch.load(para + '.pkl')
        auc = fine_tune(model=model, data=np.array(adapt_data[i]), label=np.array(adapt_label[i]), lr=1e-3, classes=2,
                        n_epoch= 10 , train_size = 5)
        if para == 'metalearning':
            auc1 = fine_tune2(model=model, data=np.array(adapt_data[i]), label=np.array(adapt_label[i]), lr=1e-3, classes=2,
                        n_epoch=5, train_size = 5)
            AUC_array2.append(auc1)
        AUC_array.append(auc)

    print('***para:' + para)
    print(AUC_array)
    print("traditional:%f, new:%f" % (sum(AUC_array),sum(AUC_array2)))
    if para == 'metalearning':
        return [sum(AUC_array)/len(AUC_array), sum(AUC_array2)/len(AUC_array2)]
    else:
        return sum(AUC_array) / len(AUC_array)

if __name__ == '__main__':
    adapt_data = np.load('data/adapt_data.npy', allow_pickle=True)
    adapt_label = np.load('data/adapt_label.npy', allow_pickle=True)
    netmodel = ['metalearning', 'traditional', 'MTL']

    result = {}
    for i in netmodel:
        mean = []
        for j in range(10):
            auc = test_para(para = i, data = adapt_data, label = adapt_label)
            mean.append(auc)
        result[i] = mean

    print(mean)
    exit(0)