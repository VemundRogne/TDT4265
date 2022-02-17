""" Loads parameters from the experiments folder with YACS

Saves results as pickled arrays
Loads if any are available
"""

import utils
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer
import pickle
from run_experiments_config import get_cfg_defaults
import os


def load_data():
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    # Calc mean and std
    mean, std = utils.calc_mean_std(X_train) # Only looking at training set when calculating mean, std

    # Load dataset
    X_train = pre_process_images(X_train, mean=mean, std=std)
    X_val = pre_process_images(X_val, mean=mean, std=std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    return X_train, X_val, Y_train, Y_val


def run_training(cfg):
    X_train, X_val, Y_train, Y_val = load_data()

    model = SoftmaxModel(
        cfg.hyperparameters.neurons_per_layer,
        cfg.settings.use_improved_sigmoid,
        cfg.settings.use_improved_weight_init
    )

    trainer = SoftmaxTrainer(
        cfg.hyperparameters.momentum_gamma,
        cfg.settings.use_momentum,
        model,
        cfg.hyperparameters.learning_rate,
        cfg.hyperparameters.batch_size,
        cfg.settings.shuffle_data,
        X_train, Y_train, X_val, Y_val
    )

    return trainer.train(cfg.hyperparameters.num_epochs)


def _load_cfg(path_to_yaml):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(path_to_yaml)
    cfg.freeze()
    return cfg


def _save_experiment_data(path_to_yaml, experiment_data):
    path_to_file = path_to_yaml.split('.yaml')[0] + ".pkl"
    with open(path_to_file, "wb") as f:
        pickle.dump(experiment_data, f)

def _load_experiment_data(path_to_yaml):
    path_to_file = path_to_yaml.split('.yaml')[0] + ".pkl"
    with open(path_to_file, "rb") as f:
        return pickle.load(f)


def run_experiment(path_to_yaml):
    print(f"Running experiment: {path_to_yaml}")
    try:
        experiment_data = _load_experiment_data(path_to_yaml)
        print("Experiment has been run, returning saved data")
        return experiment_data
    except Exception as e:
        print(e)
        print("Probably just not run the experiment yet;)")
    
    cfg = _load_cfg(path_to_yaml)
    print(cfg)
    train_history, val_history = run_training(cfg)
    return {
        "train_history": train_history,
        "val_history": val_history,
        "cfg": cfg
    }


def run_all_experiments(base_path):
    experiment_yamls = []
    for file in os.listdir(base_path):
        if file.endswith(".yaml"):
            experiment_yamls.append(base_path + file)
 
    for experiment_yaml in experiment_yamls:
        print(experiment_yaml)
        experiment_data = run_experiment(experiment_yaml)
        _save_experiment_data(experiment_yaml, experiment_data)


if __name__ == '__main__':
#    path = "experiments/2c.yaml"
    #experiment_data = run_experiment("experiments/2c.yaml")
#    _save_experiment_data("experiments/2c.yaml", experiment_data)
    #print(experiment_data['val_history'])
    run_all_experiments("experiments/")