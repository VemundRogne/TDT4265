import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
from numpy import linalg as LA
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

    # Convert bool to int so accuracy can be summed
    accuracy = np.mean((np.isclose(y_pred, y_true)).astype(int))

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
        self.model.zero_grad()  # reset gradient
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


def plot_weights(models, show=True, save_path="figures/task"):
    # Plots the weights of the models and the lambda values
    n_classes = 10

    n_cols = n_classes
    n_rows = len(models)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 4))

    # Plot the weights
    for i_model, model in enumerate(models):
        for i_class in range(n_classes):
            # Don't plot bias (last column of weights)
            weight = model.w[:-1, i_class].reshape((28, 28))
            ax = axs[i_model, i_class]
            ax.imshow(weight, cmap="gray")
            ax.set_axis_off()
        # axs[i_model, 0].set_xlabel(f"lambda = {model.l2_reg_lambda}")

    plt.subplots_adjust(wspace=0, hspace=0)

    if show:
        fig.show()
        # input("Press <enter> to continue")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")


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
    plt.figure()
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig(FIGURE_DIRECTORY + "task3b_softmax_train_loss.png")

    plt.figure()
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
    # plot_weights(models_to_plot, save_path=FIGURE_DIRECTORY+"task4b_softmax_weight.png")

    # plt.imsave("task4b_softmax_weight.png", weight, cmap="gray")


    # Training models with different L2 regularizations (task4c)
    l2_lambdas = [2, .2, .02, .002]

    training_results = []

    for i, l2_lambda in enumerate(l2_lambdas):
        # Initialize model
        model = SoftmaxModel(l2_lambda)
        # Train model
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)

        weight_norm = np.linalg.norm(model.w, ord=2)

        training_results.append({
            "model": model,
            "l2_lambda": l2_lambda,
            "val_history": val_history,
            "weight_norm": weight_norm
        })
    
    # Train and plot validation accuracy for different regularization parameters
    plt.figure()
    for result in training_results:
        val_history = result["val_history"]
        l2_lambda = result["l2_lambda"]

        utils.plot_loss(val_history["accuracy"],
                        f"lambda={l2_lambda}")

    plt.ylim([.73, .93])
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation accuracy")
    # plt.savefig(FIGURE_DIRECTORY + "task4c_l2_reg_accuracy.png")

    weight_norms = [res["weight_norm"] for res in training_results]
    plt.figure()
    plt.plot(l2_lambdas, weight_norms)
    plt.ylim(0, 2.5)
    plt.xlabel("Regularization parameter lambda")
    plt.ylabel("L2 norm of weights")
    #plt.savefig(FIGURE_DIRECTORY + "task4d_l2_reg_norms.png")
    plt.show()
