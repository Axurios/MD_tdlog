import numpy as np
import matplotlib.pyplot as plt
import neural_network 

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

print(f'n_train={N_train}, n_val={N_val}, n_test={N_test}')

# Check that data makes sense
plt.figure()
plt.imshow(train_data[0, :].reshape(28,28))
print(train_labels[0])
plt.show()

def lin_layer(x,w,a):
    """
    simule le passage des données x à travers une couche linéaire 
    w est la liste des poids (weights)
    et a celle des biais
    """
    assert len(x)==len(w)==len(a), "les données ne sont pas toutes de même longueur"
    return x*w+a

# pouvoir changer de fonction de loss
# vérifier en fonction de la fonction de loss la taille des tableau en entrée et en sortie pour que ce soit compatible pour les opérations 
# implémenter des tests 
# avec matplotlib on peut afficher des trucs très spécifiques donc ça vaut peut-être le coup de se renseigner en fonction de ce qu'on veut
# ici on a que 2 vecteurs en paramètres donc ça marche mais il va falloir faire des projections ensuite, 
# permettre de choisir les paramètres que l'on veut afficher -> peut-être en fonction des gradients
# réfléchir à l'intégration au programme final 
# en vrai c'est plutôt cool 
# faire la documentation -> expliquer fonctions + dire pk on a choisi matplotlib 

# on a peu près boltzmann mais il faut extraire totalement boltzmann 

def RMSE(w_range,a_range,x,NN):
    """
    Calcule la RMSE de x pour le réseau de neuronnes NN sur tout l'espace 2D w_range*a_range
    """
    
    W, A = np.meshgrid(w_range, a_range)
    y = NN(x,W,A)
    RMSE = np.square(x-y)
    return W, A, RMSE


x = np.random.rand(100)
w_range = np.linspace(-2, 2, 100)
a_range = np.linspace(-2, 2, 100)
NN=lin_layer

# Compute the loss landscape
W, A, RMSE = RMSE(w_range, a_range, x,NN)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, A, RMSE, cmap='viridis', alpha=0.8)
ax.set_xlabel('W')
ax.set_ylabel('A')
ax.set_zlabel('RMSE')
ax.set_title('RMSE en 3D')
plt.show()