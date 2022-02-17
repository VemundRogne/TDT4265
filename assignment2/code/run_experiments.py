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
import matplotlib.pyplot as plt


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
        cfg.settings.use_improved_weight_init,
        cfg.settings.use_uniform_improved_weight_init,
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
        last_key = list(experiment_data['val_history']['accuracy'])[-1]
        print("last accuracy:", experiment_data['val_history']['accuracy'][last_key])
        print("Experiment has been run, returning saved data")
        #print("last accuracy:", experiment_data['val_history']['accuracy'][-1])
        return experiment_data
    except FileNotFoundError as e:
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


def plot_experiment(path_to_yaml):
    experiment_data = run_experiment(path_to_yaml)
    plt.figure(figsize=(16,9))
    plt.subplot(2, 1, 1)
    utils.plot_loss(
        experiment_data['train_history']['loss'],
        npoints_to_average = 10,
        label="Train history loss"
    )
    utils.plot_loss(
        experiment_data['val_history']['loss'],
        npoints_to_average = 10,
        label="Validation history loss"
    )
    plt.legend()
    plt.subplot(2, 1, 2)
    utils.plot_loss(
        experiment_data['val_history']['accuracy'],
        npoints_to_average=10,
        label="validation accuracy"
    )
    plt.legend()
    plt.title(path_to_yaml)
    save_name = path_to_yaml.split('.')[0] + ".png"
    plt.savefig(save_name)


def compare_experiment(path_to_yamls, labels, ylims = None):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    experiment_datas = [run_experiment(path) for path in path_to_yamls]
    
    # Plot loss
    for (exp, label) in zip(experiment_datas, labels):
        utils.plot_loss_ax(
            ax1,
            exp['train_history']['loss'],
            npoints_to_average=10,
            label="Train history loss"+label
        )
    ax1.legend()

    # Plot validation accuracy
    for (exp, label) in zip(experiment_datas, labels):
        utils.plot_loss_ax(
            ax2,
            exp['val_history']['accuracy'],
            npoints_to_average=1,
            label="Val history" + label
        )
    ax2.legend()

    if ylims:
        ax1.set_ylim(ylims[0])
        ax2.set_ylim(ylims[1])

    return fig

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
    #plot_experiment("experiments/2c.yaml")
    #plot_experiment("experiments/3a.yaml")
    #plot_experiment("experiments/3b.yaml")
    #plot_experiment("experiments/3c.yaml")
    #plot_experiment("experiments/4a.yaml")
    #plot_experiment("experiments/4b.yaml")
    #plot_experiment("experiments/4d.yaml")
    #plot_experiment("experiments/4e.yaml")
    fig = compare_experiment(
        ["experiments/4a.yaml", "experiments/3c.yaml", "experiments/4b.yaml"],
        labels=[" 32 neurons", " 64 neurons", " 128 neurons"],
        ylims = [(-0.1, 1), (0.96, 1)]
    )
    plt.savefig("neuron_count_comparison.png")

    compare_experiment(
        ["experiments/3c.yaml", "experiments/4d.yaml"],
        labels=[" [64, 10]", "[59, 59, 10]"],
        ylims = [(-0.1, 1), (0.96, 1)]
    )
    plt.savefig("one_vs_two_hidden_layers")

    compare_experiment(
        ["experiments/3a.yaml", "experiments/3b.yaml", "experiments/3c.yaml"],
        labels=[" 3a", " 3b", " 3c"],
        ylims=[(-0.1, 2),(0.95, 1)]
    )
    plt.savefig("Adding tricks")

    compare_experiment(
        ["experiments/3c.yaml", "experiments/4e.yaml"],
        labels=[" 3c", "4e"],
        ylims = [(-0.1, 1), (0.96, 1)]
    )
    plt.savefig("deeeeep_network")
    plt.show()

    #fig = compare_experiment(
        #[f"experiments/4{character}.yaml" for character in ['a', 'b', 'd', 'e']],
        #labels = [" a", " b", " d", " e"]
    #)
    #plt.show()