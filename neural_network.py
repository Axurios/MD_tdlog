import numpy as np
import math
import matplotlib.pyplot as plt
import random as rd

class Module(object):
    def __init__(self):
        self.gradInput = None
        self.output = None

    def forward(self, *input):
        """
        Définit le calcul effectué à chaque appel.
        Doit être remplacé par toutes les sous-classes.
        """
        raise NotImplementedError

    def backward(self, *input):
        """
        Définit le calcul effectué à chaque appel.
        Doit être remplacé par toutes les sous-classes.
        """
        raise NotImplementedError


class MSE(Module):
    """
    Cette implémentation de la perte quadratique moyenne suppose que les données sont fournies sous forme
    d'un tableau 2D de taille (batch_size, num_classes) et que les étiquettes sont fournies sous forme
    d'un vecteur de taille (batch_size).
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
    
    def forward_grid(self, x, labels):
        rows, cols = x.shape
        result = np.empty((rows, cols), dtype=object)

        for i in range(rows):
            for j in range(cols):
                result[i,j] = self.forward(x[i,j],labels)
        
        return result

    def backward(self, x, labels):
        target = self.make_target(x, labels)
        self.gradInput = -2*(target - x) / x.shape[0]
        return self.gradInput


class Linear(Module):
    """
    L'entrée est censée avoir deux dimensions (batch_size, in_features).
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        std = math.sqrt(2/in_features) 
        self.weight = std*np.random.randn(out_features, in_features)

        self.bias = np.zeros(out_features)

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
    
    def var_rd_param(self, delta, nb_val):
        """
        La fonction choisit aléatoirement 2 valeurs dans weights et renvoie un plan de dimensions nb_val*nb_val 
        où les paramètres choisis varient de +/- delta autour de la valeur initiale.
        On renvoie également les coordonnées des poids qui ont été modifiés dans la matrice des poids
        """
        selected_line_1, selected_line_2 =  rd.randint(0,self.out_features-1),rd.randint(0,self.out_features-1)
        selected_col_1, selected_col_2 =  rd.randint(0,self.in_features-1),rd.randint(0,self.in_features-1)

        while selected_line_1==selected_line_2 and selected_col_1==selected_col_2:
            selected_line_2 =  rd.randint(0,self.out_features-1)
            selected_col_2 =  rd.randint(0,self.in_features-1)
        coord_w1 = (selected_line_1,selected_col_1)
        coord_w2 = (selected_line_2,selected_col_2)
        w1=self.weight[coord_w1]
        w2=self.weight[coord_w2]
        w1_range = np.linspace(w1-delta, w1+delta, nb_val)
        w2_range = np.linspace(w2-delta, w2+delta, nb_val)
        W1, W2 = np.meshgrid(w1_range,w2_range)

        return coord_w1, coord_w2, W1, W2

    def forward_modif_val(self, x, V1, V2, coord_w1, coord_w2):
        """
        forward mais avec les poids w1 et w2 qui changent en prenant les valeurs V1 et V2 d'une meshgrid
        """
        rows, cols = V1.shape
        result = np.empty((rows, cols), dtype=object)

        for i in range(rows):
            for j in range(cols):
                v1 = V1[i, j]
                v2 = V2[i, j]
                self.weight[coord_w1]=v1
                self.weight[coord_w2]=v2
                result[i,j] = self.forward(x)

        return result



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
    Cette classe est un exemple simple de réseau de neurones, composé de deux
    couches linéaires, avec une non-linéarité ReLU au milieu.
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

    def forward_grid(self,x, delta, nb_val):
        """
        On choisit au hasard l'une des couches linéaires dans laquelle on va faire varier 2 poids au hasard avec la fonction var_rd_param.
        On calcule les prédictions du réseau de neuronnes lorsque les 2 poids prennent les valeurs du plan renvoyé par var_rd_param.
        Cette fonction permettra de visualiser les variations de la loss lorsqu'on fait varier les 2 poids choisis
        """

        rd_int = rd.randint(0,1)
        
        if rd_int == 0: # on change 2 paramètres dans la couche linéaire 1
            coord_w1, coord_w2, W1, W2 = self.fc1.var_rd_param(delta,nb_val)
            x = self.fc1.forward_modif_val(x,W1,W2,coord_w1,coord_w2)
            rows, cols = W1.shape
            for i in range(rows):
                for j in range(cols):
                    x[i,j] = self.non_lin1.forward(x[i,j])
                    x[i,j] = self.fc2.forward(x[i,j])
            return x, W1, W2
        
        else : # on change 2 paramètres dans la couche linéaire 1
            coord_w1, coord_w2, W1, W2 = self.fc2.var_rd_param(delta,nb_val)
            x = self.fc1.forward(x)
            x = self.non_lin1.forward(x) 
            x = self.fc2.forward_modif_val(x,W1,W2,coord_w1,coord_w2)
            return x, W1, W2
            

class DoubleMLP(Module):
    """
    Cette classe est un exemple de réseau de neurones, composé de trois
    couches linéaires, avec des non-linéarités ReLU au milieu.
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
    Cette classe permet de définir un MLP avec un nombre variable de couches
    et de caractéristiques cachées, ainsi que des couches ReLU entre elles.
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
        for i in range(len(self.layers)-1,0, -1):
            layer = self.layers[i]
            previous_layer= self.layers[i-1]
            gradient = layer.backward(previous_layer.output, gradient)  # Backpropagation dans chaque couche
        gradient = self.layers[0].backward(x, gradient)  # Backpropagation dans la première couche
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
  effectue une itéation de la descente de gradient sur modèle
  """
  predicted_labels = model.forward(batch_data)
  loss_value = loss.forward(predicted_labels, batch_labels) # training loss
  loss_grad = loss.backward(predicted_labels, batch_labels)
  model_grad = model.backward(batch_data, loss_grad)
  model.gradientStep(lr)
  return loss_value

def evaluate(model, loss, data, labels):
  """
  évalue model en renvoyant la loss et la précision
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

  # Training pour 1 epoch
  indexes = np.arange(len(data))
  np.random.shuffle(indexes)

  for i in range(nb_iterations):
    count_iterations += 1
    batch_data = data[indexes[i * batch_size : (i + 1) * batch_size]]
    batch_labels = labels[indexes[i * batch_size : (i + 1) * batch_size]]

    training_loss = train_iter(model, loss, batch_data, batch_labels, lr)

    # Toutes les 10 itérations, nous utilisons le modèle sur l'ensemble de validation 
    # et ajoutons les pertes et la précision aux listes.

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

def train(model, loss, train_data, train_labels, val_data, val_labels, lr=1e-2, batch_size=16, epochs=1):

    # On stocke les valeurs dans les listes suivantes toutes les 10 itérations
    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []

    for epoch in range(epochs):
        train_losses, val_losses, val_accuracies = train_epoch(model, loss, train_data, train_labels, val_data, val_labels, lr, batch_size)

        all_train_losses.extend(train_losses)
        all_val_losses.extend(val_losses)
        all_val_accuracies.extend(val_accuracies)

    iterations = range(0, len(all_train_losses) * 10, 10) 

    plt.plot(iterations, all_train_losses, color='blue', label='Training Loss')
    plt.plot(iterations, all_val_losses, color='red', label='Validation Loss')
    plt.plot(iterations, all_val_accuracies, color='green', label='Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training and Validation Performance')
    plt.legend()
    plt.show()

data = np.load("mini_mnist.npz")

train_data = data["train_data"]
train_labels = data["train_labels"]
test_data = data["test_data"]
test_labels = data["test_labels"]

N_val = int(0.1 * len(train_data))
val_data = train_data[-N_val:]
val_labels = train_labels[-N_val:]

N_train = len(train_data) - N_val
train_data = train_data[:N_train]
train_labels = train_labels[:N_train]

N_test = test_data.shape[0]
