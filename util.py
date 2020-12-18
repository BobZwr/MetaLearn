import torch
import torch.nn as nn

def clone_module(module):
    # ** Description
    #
    # Create a copy of module, whose parameters, submodules are
    # created using PyTorch's torch.clone
    #
    # TODO: decide use shallow copy or deep copy!
    # **

    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._modules = clone._modules.copy()

    if hasattr(clone, '_parameters'):
        for key in module._parameters:
            if module._parameters[key] is not None:
                param = module._parameters[key]
                cloned = param.clone()
                clone._parameters[key] = cloned

    if hasattr(clone, '_modules'):
        for key in clone._modules:
            clone._modules[key] = clone_module(
                module._modules[key]
            )
    return clone



def update_module(module, updates = None):
    #"""
    #** Description **
    #Update the paramaters of a module using GD.
    #
    #"""

    if not updates == None:          ## in this case, we won't meet this case
        params = list(module.parameters())
        if len(updates) != len(list(params)):
           warn = 'WARNING:update_module(): Paramaters and updates should have same length, but we get {} & {}'.format(len(params), len(updates))
           print(warn)
        for p, g in zip(params, updates):
            p.update = g

    #Update the params
    for key in module._parameters:
        value = module._parameters[key]
        if value is not None and hasattr(value, 'update') and p.update is not None:
            module.parameters[key] = value + value.update

    #recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(module._modules[module_key])

    return module


