import numpy as np
import utils
import typing
np.random.seed(1)

from tqdm.auto import trange


def pre_process_images(X: np.ndarray, mean=None, std=None):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    assert mean is not None and std is not None, \
        "Mean and std to normalize with should be given as argument!"
    
    X_normalized = (X - mean) / std
    
    batch_size = X_normalized.shape[0]
    X_preprocessed = np.hstack([X_normalized, np.ones((batch_size, 1))])

    return X_preprocessed


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    loss_vec = -np.sum(targets * np.log(outputs), axis=1)  # Sum over classes
    loss = np.mean(loss_vec)

    return loss


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785 # None
        self.use_improved_sigmoid = use_improved_sigmoid

        self.z = [] # List of weighted inputs at each layer:
        # size z[k] = [batch_size, n_neurons_layer_k]
        self.a = [] # List of activations at each layer
        # size a[k] = [batch_size, n_neurons_layer_k]

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            
            if use_improved_weight_init:
                # Initialize weights to uniform random in range [-1, 1] (task 2c)
                w = np.random.uniform(low=-1.0, high=1.0, size=w_shape)

                # Task 3a: Initialize using fan-in
                fan_in = size
                std = 1 / np.sqrt(fan_in) # Standard 
                w = np.random.normal(loc=0.0, scale=std, size=w_shape) 
            else:
                # Initialize weights to zeroes
                w = np.zeros(w_shape)
            
            self.ws.append(w)
            prev = size

        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...

        self.z = [] # List of weighted inputs at each layer:
        # size z[k] = [batch_size, n_neurons_layer_k]
        self.a = [] # List of activations at each layer
        # size a[k] = [batch_size, n_neurons_layer_k]

        # Weighted input for input layer is just the image
        self.z.append(X)

        # Run activations for the first layer
        if self.use_improved_sigmoid:
            activations = 1.7159 * np.tanh(2 * X / 3)
        else:
            activations = 1 / (1 + np.exp(-X))
        
        self.a.append(activations)

        prev_activations = activations

        # Iterate weights and activations for the next layers
        last_layer_idx = len(self.ws) - 1
        for i, layer_w in enumerate(self.ws):
            # Apply weights
            z = prev_activations @ layer_w
            #z = layer_w.T @ prev_activations.T

            # Different activations functions for hidden and last layer
            if i < last_layer_idx:
                # Hidden layers use sigmoid
                if self.use_improved_sigmoid:
                    activations = 1.7159 * np.tanh(2 * z / 3)
                else:
                    activations = 1 / (1 + np.exp(-z))
            else:
                # Last layer uses softmax
                activations = np.exp(z) / np.sum(np.exp(z), axis=1)[:, np.newaxis]

            self.z.append(z)
            self.a.append(activations)
            prev_activations = activations

        y = activations # Activations of last layer
        return y

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        batch_size = X.shape[0]
        deltas = []
        
        # Calculate deltas of last layer
        #self.grad = -X.T @ (targets - outputs) * (1 / batch_size)
        
        #z_derivative = - self.z[-1].T @ (targets-outputs)*(1/batch_size)
        deltas.append(-targets + outputs)


        for i in range(len(self.ws)):
            # We want to 1-index, because we are running backwards
            i += 1

            # Activation-derivative
            z_derivative = self._get_sigmoid_derivative(self.z[-(i+1)])

            # Deltas does not run backwards, so we go back to 0-indexing for             
            deltas.append((self.ws[-i] @ deltas[i-1].T).T * z_derivative)


        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = []
        for i in range(len(self.ws)):
            # Iterate through the network, from input to output.
            # Note that deltas is backwards, and needs to be iterated -(i+2) in relation to a
            # Also note that we use @ to sum over all and then use 1/batch to calculate the mean.
            self.grads.append((1 / batch_size) * (self.a[i].T @ deltas[-(i+2)]))

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        """
        Sets the gradients to None, but doesn't affect the saved hidden layer outputs
        """
        self.grads = [None for i in range(len(self.ws))]
    
    def _get_sigmoid_derivative(self, z):
        if self.use_improved_sigmoid:
            # Derivative of LeCun's improved sigmoid
            sigmoid_derivative = 4.57572 / np.square(np.exp(-2 * z/3) + np.exp(2 * z/3))
        else:
            # Use normal sigmoid
            activation = 1 / (1 + np.exp(-z))
            sigmoid_derivative = activation * (1 - activation)
        
        return sigmoid_derivative


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    n_examples = Y.shape[0]

    Y_one_hot = np.zeros((n_examples, num_classes))

    for i in range(n_examples):
        Y_one_hot[i, int(Y[i, 0])] = 1  # Maybe index error?

    return Y_one_hot


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3

    #for layer_idx, w in enumerate(model.ws):
    #    for i in range(w.shape[0]):
    #        for j in range(w.shape[1]):

    for layer_idx in trange(len(model.ws), desc="Iterating over layers"):
        w = model.ws[layer_idx]
        for i in trange(w.shape[0], desc="Iterating over shape[0]"):
            for j in trange(w.shape[1], desc="Iterating over shape[1]", leave=False):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2 + 0.0001,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    mean, std = utils.calc_mean_std(X_train)
    X_train = pre_process_images(X_train, mean=mean, std=std) # Edited from given code
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
