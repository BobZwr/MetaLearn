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
import copy
import dataprocess as dp
import MAML
import args
import MTL

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()

def MAMLtrain(model, xset, yset, lr, shots, tasks, update, ways = 2, first_order = True, batch_size = 1):
    '''
    ** Description
        the main function to pre_train using meta learning
    :param model: model
    :param xset: data set
    :param yset: label set
    :param lr: learning rate
    :param ways: the number of classification, which is 2 in this project(VT/VF Or Not)
    :param shots: the size of data for each task
    :param tasks: the number of tasks
    :param update: the number of updating for each meta task
    :return:
    '''

    # TODO: try Mini-Batch Meta Tasks(1) and Mini-Batch in each task(2)
    xset, yset = dp.pair_shuffle(data = xset, label = yset)
    train_x = xset[ : len(xset)*3 // 4 - 1]
    train_y = yset[ : len(yset)*3 // 4 - 1]
    valid_x = xset[len(xset)*3 // 4 : ]
    valid_y = yset[len(yset)*3 // 4 : ]
    #Pre_Train
    #MetaX = dp.SelectNTasks(train_x, train_y, tasks)
      #TODO:(1)

    model = MAML.MAML(model, lr, first_order)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    loss_func = torch.nn.CrossEntropyLoss()

    MetaTaskIndex = np.random.permutation(len(train_x))[0: tasks]
    for task_index in tqdm(list(i for i in range(tasks)), desc = 'Task', position = 1):
        data = train_x[MetaTaskIndex[task_index]]
        label = train_y[MetaTaskIndex[task_index]]
        label = np.array(label)
        # XTrain, XTest, YTrain, YTest = train_test_split(train_x[MetaTaskIndex],     ## TODO: (2)
        #                                                 train_y[MetaTaskIndex],
        #                                                 train_size=
        #                                                 )

        # sample shots (may be replaced by mini-batch)

        train_data, train_label, test_data, test_label = dp.FilterNwaysKshots(data,
                                           label,
                                           N = ways,
                                           train_shots = batch_size * shots,
                                           test_shots= batch_size)
        train_data = np.expand_dims(train_data, 1)
        test_data = np.expand_dims(test_data, 1)
        sum_loss = 0.0
        dataset = MyDataset(train_data, train_label)
        dataset_test = MyDataset(test_data, test_label)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

        learner = model.clone()
        for _ in tqdm(range(update), desc = 'update_num', leave = False, colour = 'white'):
            learner.train()
            # train
            prog_iter = tqdm(dataloader, desc = 'Epoch', leave = False, colour = 'yellow')
            for batch_idx, batch in enumerate(prog_iter):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = learner(input_x)
                loss = loss_func(pred, input_y)
                learner.adapt(loss=loss / batch_size, first_order=first_order)

            print("train finish-----------------------")
            # test
            learner.eval()
            test_task = tqdm(dataloader_test, desc="Test")
            for batch_idx, batch in enumerate(test_task):
                x, y = tuple(t.to(device) for t in batch)
                pred = model(x)
                loss = loss_func(pred, y)
                pred = pred.argmax(dim=1)
                sum_loss += loss / batch_size

        sum_loss /= update * shots
        optimizer.zero_grad()
        sum_loss.backward()
        optimizer.step()
        scheduler.step(task_index)
    torch.save(model, 'metalearning.pkl')
    return

def TRADITION(model, xset, yset, lr, shots, update, batch_size = 1, ways = 2 ):
    '''
    ** Description **
    the main part to pre train the model using traditional method
    :param model: model
    :param xset: data
    :param yset: label
    :param lr: learning rate
    :param shots: number of shots according to meta learning
    :param update: the number to update the model during training
    :param batch_size: batch size
    :return:
    '''
    xset, yset = dp.pair_shuffle(data=xset, label=yset)
    train_x = xset[: len(xset) * 3 // 4 - 1]
    train_y = yset[: len(yset) * 3 // 4 - 1]
    valid_x = xset[len(xset) * 3 // 4:]
    valid_y = yset[len(yset) * 3 // 4:]

    device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss()
    train_data, train_label, _, _ = dp.FilterNwaysKshots(data = np.array(train_x[0]),
                                                         label = np.array(train_y[0]),
                                                         N = ways,
                                                         train_shots = 2 * batch_size * shots,
                                                         test_shots = 0)

    for i in range(1, len(train_x)):
        tmp_data, tmp_label, _, _ = dp.FilterNwaysKshots(data = np.array(train_x[i]),
                                                         label = np.array(train_y[i]),
                                                         N = ways,
                                                         train_shots = 2 * batch_size * shots,
                                                         test_shots = 0)
        train_data = np.append(train_data, tmp_data,axis=0)
        train_label = np.append(train_label, tmp_label)
    print(train_data)
    train_data = np.expand_dims(train_data, 1)

    dataset = MyDataset(train_data, train_label)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # train
    model.train()
    for _ in tqdm(range(update), desc='n_update', leave=False, colour='white'):
        prog_iter = tqdm(dataloader, desc='Epoch', leave=False, colour='Yellow')
        for batch_idx, batch in enumerate(prog_iter):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(_)

    torch.save(model, 'traditional.pkl')
    torch.save(model.Conv, 'preMTL.pkl')
    return

def MTLTrain(xset, yset, args = args.MTL):
    xset, yset = dp.pair_shuffle(data = xset, label = yset)
    train_x = xset[ : len(xset)*3 // 4 - 1]
    train_y = yset[ : len(yset)*3 // 4 - 1]
    valid_x = xset[len(xset)*3 // 4 : ]
    valid_y = yset[len(yset)*3 // 4 : ]

    model = MTL.MtlLearner(lr=args['inlr'], update_step=args['update'])
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)

    tasks = args['tasks']
    ways = args['ways']
    batch_size = args['batch_size']
    shots = args['shots']

    #load pretrained model without FC classifier
    model_dict = model.state_dict()
    pretrained_dict = torch.load('preMTL.pkl')
    pretrained_dict = {'model.Conv.'+k : v for k, v in pretrained_dict.state_dict().items()}
    pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, model.encoder.parameters())},
                            {'params': model.base_learner.parameters(), 'lr': 1e-3}], lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss()
    MetaTaskIndex = np.random.permutation(len(train_x))[0: tasks]
    for task_index in tqdm(list(i for i in range(args['tasks'])), desc = 'Task', position = 1):

        data = train_x[MetaTaskIndex[task_index]]
        label = train_y[MetaTaskIndex[task_index]]
        label = np.array(label)
        # XTrain, XTest, YTrain, YTest = train_test_split(train_x[MetaTaskIndex],     ## TODO: (2)
        #                                                 train_y[MetaTaskIndex],
        #                                                 train_size=
        #                                                 )

        # sample shots (may be replaced by mini-batch)

        train_data, train_label, test_data, test_label = dp.FilterNwaysKshots(data,
                                           label,
                                           N = ways,
                                           train_shots = batch_size * shots,
                                           test_shots= batch_size)
        train_data = np.expand_dims(train_data, 1)
        test_data = np.expand_dims(test_data, 1)
        dataset = MyDataset(train_data, train_label)
        dataset_test = MyDataset(test_data, test_label)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

        loss = model(dataloader, dataloader_test) / shots

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(task_index)
    torch.save(model, 'MTL.pkl')
    return


if __name__ == '__main__':
    train_data = np.load('data/train_data.npy', allow_pickle = True)
    train_label = np.load('data/train_label.npy', allow_pickle = True)
    model = Net1D(
        in_channels=1,
        base_filters=128,
        ratio=1.0,
        filter_list=[128, 64, 64, 32, 32],
        m_blocks_list=[2, 2, 2, 2, 2],
        kernel_size=16,
        stride=2,
        groups_width=32,
        verbose=False,
        n_classes=2)
    torch.save(model, 'raw.pkl')

    model_test = Net1D(
        in_channels=1,
        base_filters=128,
        ratio=1.0,
        filter_list=[128, 64, 64, 32, 32],
        m_blocks_list=[2, 2, 2, 2, 2],
        kernel_size=16,
        stride=2,
        groups_width=32,
        verbose=False,
        n_classes=2)

    model_test.load_state_dict(model.state_dict())
    MAMLtrain(model=model_test, xset = train_data, yset = train_label, lr = 1e-3, shots=5, tasks= 25, update = 10)

    model_test.load_state_dict(model.state_dict())
    TRADITION(model_test, xset = train_data, yset = train_label, lr = 1e-3, shots = 5, update = 10)

    model_test.load_state_dict(model.state_dict())
    MTLTrain(xset=train_data, yset=train_label)
    print('save success')
