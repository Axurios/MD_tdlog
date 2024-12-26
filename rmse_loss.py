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

model = neural_network.SimpleMLP()
loss=neural_network.MSE()
neural_network.train(model, loss, train_data, train_labels, val_data,val_labels)
y,W1, W2= model.forward_grid(val_data,2,100)
RMSE = loss.forward_grid(y,val_labels)


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W1, W2, RMSE, cmap='viridis', alpha=0.8)
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('RMSE')
ax.set_title('RMSE en 3D')
plt.show()