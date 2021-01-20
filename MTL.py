import torch
import torch.nn as nn
import os.path as osp
import os
import torch.nn.functional as F
from net1d import  Net1D
from args import Network

class BaseLearner(nn.Module):
    """The class for inner loop."""

    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([Network['n_classes'], self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(Network['n_classes']))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars


class MtlLearner(nn.Module):
    """The class for outer loop."""

    def __init__(self, lr, update_step, mode='meta'):
        super().__init__()
        self.mode = mode
        if mode == 'meta':
            self.model = Net1D(
                in_channels=Network['in_channels'],
                base_filters=Network['base_filters'],
                ratio=Network['ratio'],
                filter_list=Network['filter_list'],
                m_blocks_list=Network['m_blocks_list'],
                kernel_size=Network['kernel_size'],
                stride = Network['stride'],
                groups_width=Network['groups_width'],
                verbose = False,
                n_classes=Network['n_classes'],
                mtl= True
            )
        else:
            self.model = Net1D(
                in_channels=Network['in_channels'],
                base_filters=Network['base_filters'],
                ratio=Network['ratio'],
                filter_list=Network['filter_list'],
                m_blocks_list=Network['m_blocks_list'],
                kernel_size=Network['kernel_size'],
                stride = Network['stride'],
                groups_width=Network['groups_width'],
                verbose = False,
                n_classes=Network['n_classes']

            )
        self.encoder = self.model.Conv
        self.pre_fc = self.model.dense
        self.update_lr = lr
        self.update_step = update_step
        z_dim = Network['filter_list'][-1]
        self.base_learner = BaseLearner(z_dim)

    def forward(self, inp):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of MTL model.
        """
        if self.mode == 'pre':
            return self.pretrain_forward(inp)
        elif self.mode == 'meta':
            data_shot, label_shot, data_query = inp
            return self.meta_forward(data_shot, label_shot, data_query)
        elif self.mode == 'preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        """The function to forward pretrain phase.
        Args:
          inp: input images.
        Returns:
          the outputs of pretrain model.
        """
        return self.pre_fc(self.encoder(inp))

    def meta_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-train phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        #logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            #logits_q = self.base_learner(embedding_query, fast_weights)
        logits_q = self.base_learner(embedding_query, fast_weights)
        return logits_q

    def preval_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-validation during pretrain phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, 100):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)
        return logits_q


