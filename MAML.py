from torch.autograd import grad
import torch.nn as nn
import traceback

from util import update_module, clone_module

class BaseLearner(nn.Module):
    def __init__(self, module=None):
        super(BaseLearner, self).__init__()
        self.model = module

    def __getattr__(self, attr):
        try:
            return super(BaseLearner, self).__getattr__(attr)
        except AttributeError:
            return getattr(self.__dict__['_modules']['module'], attr)

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

class MAML(BaseLearner):
    # ** Description
    #
    # implementation of MAML.
    # first_order(bool, optional, default = False) - Whether to use first-order approximation
    #
    # **

    def __init__(self, model, lr, first_order = False):
        super(MAML, self).__init__()
        self.model = model
        self.lr = lr
        self.first_order = first_order

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

        if first_order is None:
            first_order = self.first_order
        return MAML(model=clone_module(self.model),
                    lr = self.lr,
                    first_order = self.first_order
                    )