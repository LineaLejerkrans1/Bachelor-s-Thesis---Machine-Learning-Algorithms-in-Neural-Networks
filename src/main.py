import numpy as np
import json
import sys
import matplotlib.pyplot as plt

import tensorflow as tf
mnist = tf.keras.datasets.mnist


from network2 import Network  
from network2 import load_data_wrapper

def plot_accuracy_and_cost(training_cost, evaluation_accuracy):
    """Plot the training and evaluation accuracy and cost over epochs."""
    evaluation_accuracy_percent = [x / 100 for x in evaluation_accuracy]

    
    # Plot the cost during training and evaluation
    plt.figure(figsize=(12, 6))

    # Cost plot (Training vs Evaluation)
    plt.subplot(1, 2, 1)
    plt.plot(training_cost, label="Training Cost", color="b")
    plt.legend(fontsize=14)
    #if evaluation_cost:
    #    plt.plot(evaluation_cost, label="Evaluation Cost", linestyle="--", color="r")
    plt.title("Cost on training data", fontsize = 16)
    plt.xlabel("Epoch", fontsize = 14)
    plt.ylabel("Cost", fontsize = 14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True)
    plt.legend()

    # Accuracy plot (Training vs Evaluation)
    plt.subplot(1, 2, 2)
    plt.plot(evaluation_accuracy_percent, label="Evaluation Accuracy", color="b")
    plt.legend(fontsize=14)
    #if evaluation_accuracy:
    #    plt.plot(evaluation_accuracy, label="Evaluation Accuracy", linestyle="--", color="r")
    plt.title("Accuracy on validation data", fontsize = 16)
    plt.xlabel("Epoch", fontsize = 14)
    plt.ylabel("Accuracy [%]", fontsize = 14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# Load the network from a file
def load_network(filename):
    """Load a neural network from a file and return the instance."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Retrieve the cost function (e.g., Cross-Entropy Loss or another)
    cost = getattr(sys.modules[__name__], data["cost"])  
    
    # Create a network with the loaded data
    net = Network(data["sizes"], cost=cost)
    
    # Set the weights and biases from the saved data
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    
    return net


# load training history from a file
def load_training_history(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return (
        data.get("training_cost", []),
        data.get("training_accuracy", []),
        data.get("evaluation_cost", []),
        data.get("evaluation_accuracy", [])
    )


# load the mnist data
def load_data_wrapper():
    # 60 000 training images, 10 000 test images (28x28)
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # converts 28×28 image to a 784×1 column vector
    # normalizes the value to between 0-1
    train_images = [x.reshape(784, 1) / 255.0 for x in train_images]
    test_images = [x.reshape(784, 1) / 255.0 for x in test_images]

    # combines each image and its label into a tuple
    def vectorized_result(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    training_data = list(zip(train_images, [vectorized_result(y) for y in train_labels])) # (for backprop)
    test_data = list(zip(test_images, test_labels)) 

    return training_data, None, test_data  




def main():
    # load data
    training_data, _, test_data = load_data_wrapper()

    # create a network
    net = Network([784, 30, 10])
    net.default_weight_initializer()

    # train the network
    training_cost, training_accuracy, evaluation_cost, evaluation_accuracy = net.SGD(
        training_data[:2000], 10, 10, 0.5, 5,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True
    )

    # plot cost and accuaracy
    plot_accuracy_and_cost(training_cost, evaluation_accuracy)

    # Save everything
    #net.save("Regularization.json", training_cost, training_accuracy, evaluation_cost, evaluation_accuracy)

    """# reload and replot:
    training_cost, training_accuracy, evaluation_cost, evaluation_accuracy = load_training_history("file1.json")
    plot_accuracy_and_cost(training_cost, evaluation_accuracy)
    """


if __name__ == "__main__":
    main()