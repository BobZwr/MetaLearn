

maml = {
    'inner_lr': 1e-3,
    'outer_lr': 1e-3,
    'shots' : 5,  # shots, choice = [1,5]
    'tasks' : 20,
    'update' : 20,
    'ways' : 2,
    'first_order' : True,
    'batch_size' : 1
}

pretrain = {
    'lr' : 1e-3,
    'shots' : 5,
    'update' : 20,
    'batch_size' : 1,
    'ways' : 2
}

MTL = {
    'prelr' : 1e-3,
    'inlr' : 1e-3,
    'outlr' : 1e-3,
    'shots' : 5,
    'tasks' : 20,
    'preupdate' : 10,
    'update' : 10,
    'ways' : 2,
    'first_order' : True,
    'batch_size' : 1
}

Network = {
    'in_channels' : 1,
    'base_filters' : 128,
    'ratio' : 1.0,
    'filter_list' : [128, 64, 64, 32, 32],
    'm_blocks_list' : [2,2,2,2,2],
    'kernel_size' : 16,
    'stride' : 2,
    'groups_width' : 32,
    'verbose' : False,
    'n_classes' : 2
}