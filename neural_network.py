import numpy as np
import math
import matplotlib.pyplot as plt

class Module(object):
    def __init__(self):
        self.gradInput = None
        self.output = None

    def forward(self, *input):
        """
        Defines the computation performed at every call.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *input):
        """
        Defines the computation performed at every call.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError


class MSE(Module):
    """
    This implementation of the mean squared loss assumes that the data comes as a 2-dimensional array
    of size (batch_size, num_classes) and the labels as a vector of size (batch_size)
    """
    def __init__(self, num_classes=10):
        super(MSE, self).__init__()
        self.num_classes = num_classes

    def make_target(self, x, labels):
        target = np.zeros([x.shape[0], self.num_classes])
        for i in range(x.shape[0]):
            target[i, labels[i]] = 1

        return target

    def forward(self, x, labels):
        target = self.make_target(x, labels)
        self.output = np.sum((target-x)**2, axis=1)
        return np.mean(self.output)

    def backward(self, x, labels):
        target = self.make_target(x, labels)
        self.gradInput = -2*(target - x) / x.shape[0]
        return self.gradInput


class Linear(Module):
    """
    The input is supposed to have two dimensions (batch_size, in_features)
    """
    def __init__(self, in_features, out_features, weights=[], biases=[], bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        std = math.sqrt(2/in_features) # He init because we have ReLU.
        if weights==[]:
            self.weight = std*np.random.randn(out_features, in_features)
        else :
            assert weights.shape == (out_features, in_features), "pas les bonnes dimensions de weights"
            self.weights = weights

        if biases == []:
            self.bias = np.zeros(out_features)
        else :
            assert biases.shape == (len(out_features),), "pas les bonnes dimensions de biases"

    def forward(self, x):
        self.output = np.dot(x, self.weight.transpose()) + self.bias[None, :]
        return self.output

    def backward(self, x, gradOutput):
        self.gradInput = np.dot(gradOutput, self.weight)
        self.gradWeight = np.dot(gradOutput.transpose(), x)
        self.gradBias = np.sum(gradOutput, axis=0)
        return self.gradInput

    def gradientStep(self, lr):
        self.weight = self.weight - lr*self.gradWeight
        self.bias = self.bias - lr*self.gradBias


class ReLU(Module):
    def __init__(self, bias=True):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.output = x.clip(0)
        return self.output

    def backward(self, x, gradOutput):
        self.gradInput = (x>0) * gradOutput
        return self.gradInput


class SimpleMLP(Module):
    """
    This class is a simple example of a neural network, composed of two
    linear layers, with a ReLU non-linearity in the middle
    """
    def __init__(self, in_dimension=784, hidden_dimension=64, non_linearity = ReLU, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = Linear(in_dimension, hidden_dimension)
        self.non_lin1 = non_linearity()
        self.fc2 = Linear(hidden_dimension, num_classes)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.non_lin1.forward(x)
        x = self.fc2.forward(x)
        return x

    def backward(self, x, gradient):
        gradient = self.fc2.backward(self.non_lin1.output, gradient)
        gradient = self.non_lin1.backward(self.fc1.output, gradient)
        gradient = self.fc1.backward(x, gradient)
        return gradient

    def gradientStep(self, lr):
        self.fc2.gradientStep(lr)
        self.fc1.gradientStep(lr)

class DoubleMLP(Module):
    """
    This class is an example of a neural network, composed of three
    linear layers, with ReLU non-linearities in the middle
    """
    def __init__(self, in_dimension=784, hidden_dimension_1=64, hidden_dimension_2=32, num_classes=10):
        super(DoubleMLP, self).__init__()
        self.fc1 = Linear(in_dimension, hidden_dimension_1)
        self.relu1 = ReLU()
        self.fc2 = Linear(hidden_dimension_1, hidden_dimension_2)
        self.relu2 = ReLU()
        self.fc3 = Linear(hidden_dimension_2, num_classes)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        return x

    def backward(self, x, gradient):
        gradient = self.fc3.backward(self.relu2.output, gradient)
        gradient = self.relu2.backward(self.fc2.output, gradient)
        gradient = self.fc2.backward(self.relu1.output, gradient)
        gradient = self.relu1.backward(self.fc1.output, gradient)
        gradient = self.fc1.backward(x, gradient)
        return gradient

    def gradientStep(self, lr):
        self.fc3.gradientStep(lr)
        self.fc2.gradientStep(lr)
        self.fc1.gradientStep(lr)

class DeepMLP(Module):
    """
    This class allows to define a MLP with a variable number of layers and hidden features and ReLU layers in between.

    """
    def __init__(self, hidden_features=[64], in_dimension=784, num_classes=10):
        super(DeepMLP, self).__init__()
        self.layers = []  # List to store all layers (Linear and ReLU)

        # Input layer
        self.layers.append(Linear(in_dimension, hidden_features[0]))
        self.layers.append(ReLU())

        # Hidden layers
        for i in range(len(hidden_features) - 1):
            self.layers.append(Linear(hidden_features[i], hidden_features[i + 1]))
            self.layers.append(ReLU())

        # Output layer
        self.layers.append(Linear(hidden_features[-1], num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x, gradient):
        for i in range (len(self.layers)-1,0, -1):
            layer = self.layers[i]
            previous_layer= self.layers[i-1]
            gradient = layer.backward(previous_layer.output, gradient)  # Backpropagate through the current layer
        gradient = self.layers[0].backward(x, gradient)  # Backpropagate through the input layer
        return gradient

    def gradientStep(self, lr):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.gradientStep(lr)

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, x, gradOutput):
        self.gradInput = gradOutput * self.output * (1 - self.output)
        return self.gradInput

class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        self.output = np.where(x > 0, x, x * self.negative_slope)
        return self.output

    def backward(self, x, gradOutput):
        self.gradInput = np.where(x > 0, gradOutput, gradOutput * self.negative_slope)
        return self.gradInput

class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, x, gradOutput):
        self.gradInput = gradOutput * (1 - self.output**2)
        return self.gradInput


def train_iter(model, loss, batch_data, batch_labels, lr):
  """
  effectue une itéation de la descente de gradient sur model
  """
  predicted_labels = model.forward(batch_data)
  loss_value = loss.forward(predicted_labels, batch_labels) # training loss
  loss_grad = loss.backward(predicted_labels, batch_labels)
  model_grad = model.backward(batch_data, loss_grad)
  model.gradientStep(lr)
  return loss_value

def evaluate(model, loss, data, labels):
  """
  evalue model en renvoyant la loss et la précision
  """
  predicted_labels = model.forward(data)
  predicted_classes = np.argmax(predicted_labels, axis =1)
  differences = abs(predicted_classes - labels)
  accuracy = 1- np.sum(differences > 0)/len(differences)

  loss_value = loss.forward(predicted_labels, labels)
  return loss_value, accuracy


def train_epoch(model, loss, data, labels, val_data, val_labels, lr, batch_size):
  nb_iterations = len(data) // batch_size
  remaining_values = len(data) % batch_size
  count_iterations = 0
  train_losses = []
  val_losses = []
  val_accuracies = []

  # Training for a whole epoch
  indexes = np.arange(len(data))
  np.random.shuffle(indexes)

  for i in range(nb_iterations):
    count_iterations += 1
    batch_data = data[indexes[i * batch_size : (i + 1) * batch_size]]
    batch_labels = labels[indexes[i * batch_size : (i + 1) * batch_size]]

    training_loss = train_iter(model, loss, batch_data, batch_labels, lr)

    # every 10 iterations we use the model on the validation set and add the losses and accuracy to lists
    if count_iterations % 10 == 0:
      val_loss, val_accuracy = evaluate(model, loss, val_data, val_labels)
      train_losses.append(training_loss)
      val_losses.append(val_loss)
      val_accuracies.append(val_accuracy)

  if remaining_values > 0:
    count_iterations += 1
    batch_data = data[indexes[len(data) - remaining_values :]]
    batch_labels = labels[indexes[len(data) - remaining_values :]]
    training_loss = train_iter(model, loss, batch_data, batch_labels, lr)

    if count_iterations % 10 == 0:
        val_loss, val_accuracy = evaluate(model, loss, val_data, val_labels)
        train_losses.append(training_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

  return train_losses, val_losses, val_accuracies

def train(model, loss, train_data, train_labels, val_data, val_labels, lr, batch_size, epochs):

    # We store values in the following lists every 10 iterations
    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []

    for epoch in range(epochs):
        train_losses, val_losses, val_accuracies = train_epoch(model, loss, train_data, train_labels, val_data, val_labels, lr, batch_size)

        all_train_losses.extend(train_losses)
        all_val_losses.extend(val_losses)
        all_val_accuracies.extend(val_accuracies)

    iterations = range(0, len(all_train_losses) * 10, 10)  # We store values every 10 iterations

    plt.plot(iterations, all_train_losses, color='blue', label='Training Loss')
    plt.plot(iterations, all_val_losses, color='red', label='Validation Loss')
    plt.plot(iterations, all_val_accuracies, color='green', label='Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training and Validation Performance')
    plt.legend()
    plt.show()

def get_rd_param(model, loss, train_data, train_labels, val_data, val_labels, lr, batch_size, epochs):
    """
    La fonction calcule le nombre de paramètres du modèle, en choisi 2 aléatoirement et stoque dans une liste les changements au cours des itérations
    Une fois le modèle entraîné, on calcule les valeurs de retour de la fonction en faisant varié les paramètres choisis 
    """
    return 0
