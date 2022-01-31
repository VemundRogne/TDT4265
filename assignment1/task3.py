from assignment1.task2 import FIGURE_DIRECTORY
import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)

FIGURE_DIRECTORY = "figures/"

def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    outputs = model.forward(X)
    y_pred = np.argmax(outputs, axis=1)
    y_true = np.argmax(targets, axis=1)

    accuracy = np.mean((np.isclose(y_pred, y_true)).astype(int)) # Convert bool to int so accuracy can be summed
    
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        outputs = self.model.forward(X_batch)

        # Calculate and do gradient step
        self.model.zero_grad() # reset gradient
        self.model.backward(X_batch, outputs, Y_batch)

        # Take gradient descent step
        self.model.w -= self.learning_rate * self.model.grad 

        # Calculate loss
        loss = cross_entropy_loss(Y_batch, outputs)

        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val

def plot_weights(models):
    # Plots the weights of the models and the lambda values
    n_classes = 10
    n_models = len(models)

    for i_model, model in enumerate(models):
        for i_class in range(n_classes):
            plt.subplot(n_models, n_classes, i_class + 1 + n_classes*i_model)
            weight = model.w[:-1, i_class].reshape((28, 28)) # Don't plot bias (last column of weights)
            plt.imshow(weight, cmap="gray")
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

    fig = plt.gcf()
    fig.set_tight_layout(True)
    plt.show()


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    # plt.ylim([0.2, .6])

    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig(FIGURE_DIRECTORY + "task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(FIGURE_DIRECTORY + "task3b_softmax_train_accuracy.png")

    # Train a model with L2 regularization (task 4b)
    model1 = SoftmaxModel(l2_reg_lambda=2.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)

    # Plotting of softmax weights (Task 4b)
    models_to_plot = [model, model1]
    plot_weights(models_to_plot)
    # plt.imsave("task4b_softmax_weight.png", weight, cmap="gray")

    # Training models with different L2 regularizations (task4c)
    l2_lambdas = [2, .2, .02, .002]
    l2_models = [SoftmaxModel(l2_reg_lambda=val) for val in l2_lambdas]
    l2_train_histories = []
    l2_val_histories = []
    for _model in l2_models:
        trainer = SoftmaxTrainer(
            _model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        _train_history, _val_history = trainer.train(num_epochs)
        l2_train_histories.append(_train_history)
        l2_val_histories.append(_val_history)
    
    # Plot softmax weghts:
    plot_weights(l2_models)

    plt.figure()
    # Plot validation accuracies
    for i in range(len(l2_models)):
        utils.plot_loss(l2_val_histories[i]['accuracy'], f"Validation Accuracy: lambda={l2_lambdas[i]}")
        plt.xlabel("Number of training steps")
        plt.ylabel("Accuracy")
    plt.legend()
    #plt.savefig(FIGURE_DIRECTORY + "task4c_l2_reg_accuracy.png")

    # Task 4d - Plotting of the l2 norm for each weight

    plt.savefig(FIGURE_DIRECTORY + "task4d_l2_reg_norms.png")

    plt.show()