from yacs.config import CfgNode as CN

_C = CN()

_C.hyperparameters = CN()
_C.hyperparameters.num_epochs = 50
_C.hyperparameters.learning_rate = 0.1
_C.hyperparameters.batch_size = 32
_C.hyperparameters.neurons_per_layer = [64, 10]
_C.hyperparameters.momentum_gamma = 0.9

_C.settings = CN()
_C.settings.shuffle_data = True
_C.settings.use_improved_weight_init = False
_C.settings.use_improved_sigmoid = False
_C.settings.use_momentum = False

def get_cfg_defaults():
    return _C.clone()