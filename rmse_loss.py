import numpy as np
import matplotlib.pyplot as plt

def lin_layer(x,w,a):
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

def RMSE(w_range,a_range,x):
    W, A = np.meshgrid(w_range, a_range)
    y = lin_layer(x,W,A)
    RMSE = np.square(x-y)
    return W, A, RMSE

x = np.random.rand(100)
w_range = np.linspace(-2, 2, 100)
a_range = np.linspace(-2, 2, 100)

# Compute the loss landscape
W, A, RMSE = RMSE(w_range, a_range, x)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, A, RMSE, cmap='viridis', alpha=0.8)
ax.set_xlabel('W')
ax.set_ylabel('A')
ax.set_zlabel('RMSE')
ax.set_title('RMSE en 3D')
plt.show()