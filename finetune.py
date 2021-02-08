import numpy as np
from sklearn.model_selection import train_test_split
import numpy.random
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import os

from net1d import Net1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import MAML
import dataprocess as dp

##TODO: Cooperative inference (see VT/VF Paper)

def test_para(para, data, label, update):
    AUC_array = []
    AUC_array2 = []
    ACC_array = []
    ACC_array2 = []
    for i in tqdm(range(len(data)), desc = para + '_Task'):
        train_index = np.random.permutation(len(adapt_data[i]))[ : 10]
        auc = acc = auc1 = acc1 = 0
        for lr in [1e-3, 5e-3, 1e-2]:
            model = torch.load(para + '.pkl')
            if para == 'metalearning':
                model = model.model
            temauc, temacc = fine_tune(model=model, data=np.array(adapt_data[i]), label=np.array(adapt_label[i]), lr=lr,  classes=2,
                            n_epoch= update , train_size = 5, train_index = train_index)
            auc = max(auc, temauc)
            acc = max(acc, temacc)
            model = torch.load(para + '.pkl')
            if para == 'metalearning':
                model = model.model
            if para == 'metalearning':
                temauc, temacc = fine_tune2(model=model, data=np.array(adapt_data[i]), label=np.array(adapt_label[i]), lr=lr, classes=2,
                            n_epoch= update, train_size = 5, train_index = train_index)
                auc1 = max(auc1, temauc)
                acc1 = max(acc1, temacc)
        AUC_array2.append(auc1)
        ACC_array2.append(acc1)
        AUC_array.append(auc)
        ACC_array.append(acc)

    if para == 'metalearning':
        return sum(AUC_array)/len(AUC_array), sum(AUC_array2)/len(AUC_array2), sum(ACC_array) / len(ACC_array), sum(ACC_array2) / len(ACC_array2)
    else:
        return sum(AUC_array) / len(AUC_array), sum(ACC_array) / len(ACC_array), 0, 0


def fine_tune(model, data, label, lr, classes, n_epoch, train_size = 5, batch_size = 1, train_index = list(range(10))):
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
    train_data = data[train_index]#data[0:70:7]
    train_label = label[train_index]#label[0:70:5]
    test_data = data
    test_label = label
    dataset_train = MyDataset(train_data, train_label)
    #dataset_valid = MyDataset(valid_data, valid_label)
    dataset_test = MyDataset(test_data, test_label)
    trainset = DataLoader(dataset_train, batch_size = batch_size)
    #validset = DataLoader(dataset_valid, batch_size = batch_size)
    testset = DataLoader(dataset_test, batch_size = batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)
    loss_func = torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')#cuda:7' if torch.cuda.is_available() else "cpu")
    model.to(device)
    #train
    model.train()
    step = 1;
    while step <= n_epoch:
#    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):
        # train
        sum_loss = 0
        model.train()
        #prog_iter = tqdm(trainset, desc="Training", leave=False)
        for batch_idx, batch in enumerate(trainset):

            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss
        print(sum_loss)
        if (sum_loss <= 0.0005 and step > 20) or (sum_loss <= 0.002 and step > 50) or (sum_loss <= 0.01 and step > 100) or (sum_loss <= 0.1 and step > 150):
            break
        step += 1
        scheduler.step(step)

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
            acc += sum(pred.argmax() == input_y)
            acc_label.extend(list(int(i) for i in input_y))
    AUC = dp.roc_curve(pred_prob, acc_label)
    acc = acc / len(test_label)
    return AUC, acc

def MTLfine_tune(model, data, label, lr, classes, n_epoch, train_size = 5, batch_size = 1):
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

    optimizer = optim.Adam(model.parameters(), lr= 5e-3, weight_decay=1e-4)
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
            pred = model.base_learner(model.encoder(input_x))
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
            pred = model.base_learner(model.encoder(input_x))
            pred = F.softmax(pred, dim = 1)
            pred_prob.append(pred[:, 1])
            acc_label.extend(list(int(i) for i in input_y))
            acc += sum(pred.argmax() == input_y)
    acc = acc / len(test_label)
    AUC = 1#dp.roc_curve(pred_prob, acc_label)

    return AUC, acc

def fine_tune2(model, data, label, lr, classes, n_epoch, train_size = 5, batch_size = 1, train_index = list(range(10))):
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
    train_data = data[train_index]#data[0:70:7]
    train_label = label[train_index]#label[0:70:5]
    test_data = data
    test_label = label
    optimizer = optim.Adam(model.parameters(), lr=lr , weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')#'cuda:7' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for tt in tqdm(range(10), desc="epoch", leave=False):
        dataset_train = MyDataset(train_data[ : 6], train_label[ : 6])
        dataset_valid = MyDataset(train_data[6 : ], train_label[6 : ])
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
                sum_loss += loss
        sum_loss /= train_size
        optimizer.zero_grad()
        sum_loss.backward()
        optimizer.step()
        scheduler.step(tt)

    dataset_train = MyDataset(train_data, train_label)
    #dataset_valid = MyDataset(valid_data, valid_label)
    dataset_test = MyDataset(test_data, test_label)
    trainset = DataLoader(dataset_train, batch_size = batch_size)
    #validset = DataLoader(dataset_valid, batch_size = batch_size)
    testset = DataLoader(dataset_test, batch_size = batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    step = 1

    #train
    model.train()
    #for _ in tqdm(range(n_epoch), desc="epoch", leave=False):
        # train
    while step <= n_epoch:
        step += 1
        sum_loss = 0
        #prog_iter = tqdm(trainset, desc="Training", leave=False)
        for batch_idx, batch in enumerate(trainset):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss
        print(step,"::",sum_loss)
        if (sum_loss <= 0.0005 and step > 20) or (sum_loss <= 0.002 and step > 50) or (
                sum_loss <= 0.01 and step > 100) or (sum_loss <= 0.1 and step > 150):
            break
        #if loss < torch.tensor()
        scheduler.step(step)
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
            acc += sum(pred.argmax() == input_y)
            pred_prob.append(pred[:, 1])
            acc_label.extend(list(int(i) for i in input_y))
    acc = acc / len(test_label)
    AUC = dp.roc_curve(pred_prob, acc_label)

    return AUC, acc

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    adapt_data = np.load('data/adapt_data.npy', allow_pickle=True)
    adapt_label = np.load('data/adapt_label.npy', allow_pickle=True)
    netmodel = ['metalearning']#, 'traditional']

    mean = []
    accu = []
    result = {}
    for i in range(0):
        for i in tqdm(range(len(adapt_data)), desc = 'MTL_Task'):
            model = torch.load('MTL.pkl').cuda(7)
            auc, acc = MTLfine_tune(model=model, data=np.array(adapt_data[i]), label=np.array(adapt_label[i]), lr=5e-3,
                            classes=2, n_epoch=10, train_size=5)
            mean.append(auc)
            accu.append(acc)
    result['MTLauc'] = mean
    result['MTLacc'] = accu
    for i in netmodel:
        mean = []
        accu = []
        mean1 = []
        accu1 = []
        for j in range(5):
            auc, auc1, acc, acc1 = test_para(para = i, data = adapt_data, label = adapt_label, update=200)
            mean.append(auc)
            accu.append(acc)
            if auc1 != 0:
                mean1.append(auc1)
                accu1.append(acc1)
        result[i+'auc'] = mean
        result[i+'acc'] = accu
        result[i+'newauc'] = mean1
        result[i+'newacc'] = accu1
    print(result)
    exit(0)