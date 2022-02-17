import numpy as np
import utils
from task2a import one_hot_encode, pre_process_images, SoftmaxModel, gradient_approximation_test


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    # Calc mean and std
    mean, std = utils.calc_mean_std(X_train) # Only looking at training set when calculating mean, std

    # Load dataset
    X_train = pre_process_images(X_train, mean=mean, std=std)
    X_val = pre_process_images(X_val, mean=mean, std=std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    # Modify your network here
    neurons_per_layer = [64, 64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
