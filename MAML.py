from torch.autograd import grad
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import traceback
from tqdm import tqdm

import util
from util import update_module, clone_module
from net1d import MyDataset
from args import maml
import dataprocess as dp

class BaseLearner(nn.Module):
    def __init__(self, module=None):
        super(BaseLearner, self).__init__()
        self.model = module

    def __getattr__(self, attr):
        return super(BaseLearner, self).__getattr__(attr)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def MAML_update(model, lr, grads):
    # ** Description **
    #
    # MAML update on model using gradients(grads) & learning_rate(lr)
    #
    # **

    if grads is not None:
        params = list(model.parameters())
        if not len(list(params)) == len(grads):
            warn = 'WARNING:MAML_update(): Parameters and gradients should have same length, but we get {} & {}'.format(len(params), len(grads))
        for p, g in zip(params, grads):
            if g is not None:
                p.update = - lr * g
    return update_module(model)

class MetaLearner(BaseLearner):
    # ** Description
    #
    # Inner-loop Learner
    #
    # **

    def __init__(self, model, lr = None, first_order = None):
        super(MetaLearner, self).__init__()
        self.model = model
        self.lr = maml['innerlr'] if lr == None else lr
        self.first_order = maml['first_order'] if first_order == None else first_order

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def adapt(self, loss, first_order = None):
        # ** Description
        #
        # Takes a gradient step on the loss and updates the cloned parameters in place
        #
        # first_order: Default to self.first_order
        #
        # **
        if first_order == None:
            first_order = self.first_order
        grads = grad(loss,
                     self.model.parameters(),
                     retain_graph = not first_order,
                     create_graph = not first_order,
                     allow_unused = True
                     )
        # try:
        #
        # except RuntimeError:
        #     traceback.print_exc()
        #     print("MAML_adapt:something wrong with the autograd_backward")

        self.model = MAML_update(self.model, self.lr, grads)

    def clone(self, first_order = None):
        # ** Description
        #
        # Returns a MAML-wrapped copy of the module whose parameters and buffers
        # are 'torch.clone'd from the original module
        #
        # **

        if first_order == None:
            first_order = self.first_order

        return MetaLearner(model=clone_module(self.model),
                           lr = self.lr,
                           first_order = first_order
                           )


class MAML(nn.Module):
    # ** Description
    #
    # implementation of MAML.
    # first_order(bool, optional, default = False) - Whether to use first-order approximation
    #
    # **

    def __init__(self, model, data, label):
        super(MAML, self).__init__()
        self.model = model

        self.train_x = data[ : len(data) * 3 // 4 - 1]
        self.train_y = label[ : len(label) * 3 // 4 - 1]
        self.valid_x = data[len(data) * 3 // 4 : ]
        self.valid_y = data[len(data) * 3 // 4 : ]

        self.innerlr = maml['innerlr']
        self.outerlr = maml['outerlr']
        self.update = maml['update']
        self.first_order = maml['first_order']
        self.batch_size = maml['batch_size']
        self.ways = maml['ways']
        self.shots = maml['shots']

        self.device = torch.device(maml['device'] if torch.cuda.is_available() else 'cpu')


    def SampleTask(self):
        index = np.random.random_integers(low = 0, high = len(self.train_y) - 1)
        data = self.train_x[index]
        label = self.train_y[index]
        label = np.array(label)
        train_data, train_label, test_data, test_label = dp.FilterNwaysKshots(data,
                                           label,
                                           N = self.ways,
                                           train_shots = self.batch_size * self.shots,
                                           test_shots= self.batch_size * self.shots)
        train_data = np.expand_dims(train_data, 1)
        test_data = np.expand_dims(test_data, 1)
        dataset = MyDataset(train_data, train_label)
        dataset_test = MyDataset(test_data, test_label)
        dataloader = DataLoader(dataset, batch_size = self.batch_size)
        dataloader_test = DataLoader(dataset_test, batch_size = self.batch_size)

        return dataloader, dataloader_test

    def metatrain(self):
        # ** Description
        #   Meta-Train phase
        #
        # **

        dataloader, dataloader_test = self.SampleTask()
        learner = self.model.clone()
        init_entro = util.calc_entropy(self.train_x, learner, self.device)
        loss_func = torch.nn.CrossEntropyLoss()
        sum_loss = 0.0
        for _ in tqdm(range(self.update), desc='update_num', leave=False, colour='white'):
            learner.train()
            for batch_idx, batch in enumerate(dataloader):
                input_x, input_y = tuple(t.to(self.device) for t in batch)
                pred = learner(input_x)
                loss = loss_func(pred, input_y)
                learner.adapt(loss=loss / self.batch_size, first_order= self.first_order)

            print("train finish-----------------------")
            # test
            learner.eval()
            for batch_idx, batch in enumerate(dataloader_test):
                x, y = tuple(t.to(self.device) for t in batch)
                pred = self.model(x)
                loss = loss_func(pred, y)
                sum_loss += loss / self.batch_size

        final = util.calc_entropy(self.train_x, learner, device=self.device)
        sum_loss = sum_loss / self.shots + (final - init_entro) * 0.5

        return sum_loss

    def valid_per_task(self, data, label):
        data = np.expand_dims(data, 1)
        train_data, train_label, valid_data, valid_label, test_data, test_label = dp.FilterNwaysKshots(data=data,
                                                                                                       label=label,
                                                                                                       N=self.ways,
                                                                                                       train_shots=self.batch_size * self.shots,
                                                                                                       test_shots=0,
                                                                                                       remain=True)
        dataset_train = MyDataset(train_data, train_label)
        dataset_test = MyDataset(test_data, test_label)
        trainset = DataLoader(dataset_train, batch_size = self.batch_size)
        testset = DataLoader(dataset_test, batch_size = self.batch_size)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        loss_func = torch.nn.CrossEntropyLoss()

        # train
        for _ in tqdm(range(maml['valid_step']), desc="epoch", leave=False):
            # train
            self.model.train()
            # prog_iter = tqdm(trainset, desc="Training", leave=False)
            for batch_idx, batch in enumerate(trainset):
                input_x, input_y = tuple(t.to(self.device) for t in batch)
                pred = self.model(input_x)
                loss = loss_func(pred, input_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step(_)

        # test
        self.model.eval()
        prog_iter_test = tqdm(testset, desc="Testing", leave=False)
        pred_prob = []
        acc_label = []
        acc = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):
                input_x, input_y = tuple(t.to(self.device) for t in batch)
                pred = self.model(input_x)
                pred = F.softmax(pred, dim=1)
                pred_prob.append(pred[:, 1])
                acc += sum(pred.argmax() == input_y)
                acc_label.extend(list(int(i) for i in input_y))
        AUC = dp.roc_curve(pred_prob, acc_label)
        acc = acc / len(test_label)
        return AUC, acc