# file to experiment and code here the scientific aspect, not used directly in the project, prototypage



# ░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░▒▓████████▓▒░       ░▒▓██████▓▒░ ░▒▓██████▓▒░       
# ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░          ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      
# ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░          ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      
# ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░          ░▒▓█▓▒▒▓███▓▒░▒▓█▓▒░░▒▓█▓▒░      
# ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░          ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      
# ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░          ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      
# ░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░           ░▒▓██████▓▒░ ░▒▓██████▓▒░       
                                                                                           
                                                                                           
# ░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░                                
# ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░                               
# ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░                               
# ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░                               
# ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░                               
# ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░                               
# ░▒▓███████▓▒░ ░▒▓██████▓▒░ ░▒▓█████████████▓▒░░▒▓█▓▒░░▒▓█▓▒░                               
                                                                                           

# -------------------------------------------------------------------------------------------------------------------------
import pickle
import numpy as np
import os
from ase import Atoms
from typing import TypedDict, List, Dict
from scipy.stats import entropy#, boltzmann
from scipy.constants import Boltzmann
import matplotlib.pyplot as plt


class MDData(TypedDict):
    """Structure du dictionnaire de données"""

    atoms: List[Atoms]
    energies: List[float]


class Theta(TypedDict):
    """Structure du dictionnaire de \theta"""

    coef: np.ndarray
    intercept: float


##########################################################
current_dir = os.path.dirname(os.path.abspath(__file__))
path_data = os.path.join(current_dir, "Data/NP_1200K_desc.pkl")
path_theta = os.path.join(current_dir, "Data/theta.pkl")
##########################################################


def DotProductDesc(theta: Theta, desc_type: np.ndarray) -> np.ndarray:
    if len(desc_type.shape) > 2:
        return np.tensordot(theta["coef"], desc_type, axes=(0, 2))
    else:
        return desc @ theta["coef"] + theta["intercept"]


# Lit le fichier binaire .pkl contenant les data
md_data: Dict[str, MDData] = pickle.load(open(path_data, "rb"))
theta: Theta = pickle.load(open(path_theta, "rb"))

keys = list(md_data.keys())
#tkeys = list(theta.keys)
# print(md_data[keys[0]].keys())
# print(type(md_data[keys[0]]['atoms'][0]))
# print(md_data[keys[0]]['atoms'][1])
# print(md_data[keys[0]]['atoms'][1].get_array("forces"))
#print(md_data[keys[0]]['atoms'][1].get_positions())

print(len(keys))

#print('pot',[md_data[keys[x]]['energies'] for x in range(35)])

#print(md_data[keys[0]]['atoms'][-1])

#print('okok',md_data[keys[0]]['atoms'][-1].get_array('milady-descriptors-forces').shape)

print(theta['coef'])
#print('ouioui', )

# for k in keys :
#     md_data[k]['atoms'][-1].calc = LennardJones()
#     data = md_data[k]['atoms'][-1].get_potential_energies()
#     sorted,cdf_data = cdf(data)
#     plt.plot(cdf_data, sorted)
# plt.show()

E_tot_ml_list = []
G_list = []
grad_list = []
f_list = []
for key, val in md_data.items():
    # liste contenant des objets Atoms
    atoms = val["atoms"]
    # liste contenant les energies associée
    energies = val["energies"]
    # Lecture des descripteurs...
    for ats, ene in zip(atoms, energies):
        # descripteurs D \in R^{M \times D}
        desc = ats.get_array("milady-descriptors")
        G_list.append(desc)
        # gradient des descripteurs \nabla D \in R^{M \times D \times 3}
        grad_desc = ats.get_array("milady-descriptors-forces")
        grad_list.append(grad_desc)
        # positions des atomes \in R^{M \times 3}
        position = ats.positions
        # forces sur les atomes \in R^{M \times 3}
        f = ats.get_array("forces")
        f_list.append(f)
        # print(f'desc dim = {desc.shape}, grad desc dim = {grad_desc.shape}, position dim = {position.shape}, forces dim = {f.shape}') # noqa:
        
        # évaluation de l'énergie par le modèle linéaire
        e_ml = DotProductDesc(theta, desc)
        E_tot_ml = np.sum(e_ml)
        E_tot_ml_list.append(E_tot_ml)
        # évaluation des forces par le modèle linéaire
        f_ml = DotProductDesc(theta, grad_desc)


def cdf(data: np.ndarray) -> np.ndarray:
    """Compute the cumulative distribution function."""
    sorted_data = np.sort(data)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf_values


def theoretical_boltzmann_cdf(energies, T, N=1000):
    """
    Calculate the Boltzmann CDF using scipy.stats.boltzmann.
    :param energies: Array-like, range of discrete energy values to evaluate the CDF.
    :param T: Temperature in Kelvin.
    :param N: Number of discrete states (truncation).
    :return: Tuple (energy_points, cdf_values).
    """
    beta = 1 / (Boltzmann * T)
    print(energies * beta * 1.60218e-19)
    cdf_value = np.array(1 - np.exp(energies * beta))
    print(cdf_value)
    return energies, cdf_value
    # # Boltzmann constant in eV/K
    # Boltzmann = 8.617333262145e-5  # eV/K


def boltzmann_cdf(E, T, kb=Boltzmann):
    """
    Cumulative distribution function for Boltzmann distribution
    E: Energy levels (in Joules)
    T: Temperature (in Kelvin)
    kb: Boltzmann constant (in J/K)
    """
    beta = 1 / (kb * T)
    return 1 - np.exp(-beta * E)


# Create energy values in electron volts (eV) then convert to Joules
# 1 eV = 1.602176634e-19 Joules

all_energies = [ene for key, val in md_data.items() for ene in val["energies"]]
# Convert energies to SI units for consistency
energies_ = (-1) * np.array(
    all_energies
)  # * 1.60218e-19  # decomment to convert from eV to Joules
E_tot_ml_array = (-1) * np.array(
    E_tot_ml_list
)  # * 1.60218e-19  # decomment to Convert predicted energies from eV to Joules

print("E_tot_ml range:", E_tot_ml_array.min(), E_tot_ml_array.max())
print("Original energies range:", energies_.min(), energies_.max())

# Calculate CDFs
energies_sorted, energies_cdf = cdf(energies_)
E_tot_ml_sorted, E_tot_ml_cdf = cdf(E_tot_ml_array)
print(E_tot_ml_array.max())


eV_to_J = 1.602176634e-19
E_eV = np.linspace(0, 1, len(energies_sorted))  # Energy range from 0 to 1 eV

E = E_eV * eV_to_J  # Convert to Joules
Tb = 2000
bolcdf = boltzmann_cdf(E, Tb)
ene_length = abs((energies_.max()) - (energies_.min()))
E_transformed = (E_eV) * ene_length + energies_.min()

plt.figure(figsize=(10, 6))
plt.plot(E_transformed, bolcdf, label=f"boltzmann cdf à {Tb}", color="purple")
plt.plot(E_tot_ml_sorted, E_tot_ml_cdf, label="E_tot_ml CDF", color="orange")
plt.plot(energies_sorted, energies_cdf, label="Energy CDF", color="blue")
# plt.plot(energies_, bolt_cdf, label=f'Theoretical Boltzmann CDF (T={Tb}K)', #  color='red', linestyle='--')

plt.xlabel("Energy (eV)")
plt.ylabel("CDF")
plt.title("Comparison of Energy Distributions")
plt.legend()
plt.grid(True)
plt.show()


def KL_boltzmann(energies: List[float], temperature: float) -> float:
    """
    Evaluate how Boltzmannian the energy distribution is. Parameters:
    - energies: List of energy values from the dataset.
    - temperature: Temperature in Kelvin at which the distribution should be evaluated. # noqa:
    Returns :
    - A score representing the closeness to a Boltzmann distribution.
      Lower values indicate a closer match.
    """
    k_B = Boltzmann  # Boltzmann constant in J/K (SI units)
    # Convert energies to SI units if needed (assuming they're in eV)
    energies_J = np.array(energies) * 1.60218e-19  # eV to Joules
    # Compute log-Boltzmann weights for numerical stability
    log_boltzmann_weights = -energies_J / (k_B * temperature)

    # Log-Sum-Exp Trick
    max_log_weight = np.max(
        log_boltzmann_weights
    )  # Maximum value for numerical stability
    log_boltzmann_probs = log_boltzmann_weights - (
        max_log_weight + np.log(np.sum(np.exp(log_boltzmann_weights - max_log_weight)))
    )
    # Convert back to probabilities
    boltzmann_probs = np.exp(log_boltzmann_probs)

    # Estimate the empirical energy distribution :
    # (interpolating an histogram, might be useful to look into more details here
    hist, bin_edges = np.histogram(energies_J, bins="auto", density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Interpolate the empirical histogram onto the Boltzmann probabilities
    empirical_probs = np.interp(energies_J, bin_centers, hist)

    # Normalization
    empirical_probs /= empirical_probs.sum()
    # Avoid zero probabilities for KL divergence calculation
    empirical_probs = np.clip(empirical_probs, a_min=1e-12, a_max=None)
    boltzmann_probs = np.clip(boltzmann_probs, a_min=1e-12, a_max=None)
    # Compute the Kullback-Leibler (KL) divergence between the distributions
    kl_divergence = entropy(empirical_probs, boltzmann_probs)
    return kl_divergence


# Evaluate Boltzmannian nature at a specific temperature
temperature = 200  # Example temperature in Kelvin
boltzmann_score_dataset = KL_boltzmann(all_energies, temperature)
boltzmann_score_simulation = KL_boltzmann(E_tot_ml_array, temperature)
print(f"Boltzmannian score (KL divergence): {boltzmann_score_dataset}")
print(f"Boltzmannian score (KL divergence): {boltzmann_score_simulation}")


## Get boltzmann from initial data :
# best Kolmogorov-Smirnov test score (max likelyhood conditionned to boltzmann) :
# Calculate CDFs
# energies_sorted, energies_cdf = cdf(energies_)
# E_tot_ml_sorted, E_tot_ml_cdf = cdf(E_tot_ml_array)

# eV_to_J = 1.602176634e-19
# E_eV = np.linspace(0, 1, 1000)  # Energy range from 0 to 1 eV

# E = E_eV * eV_to_J  # Convert to Joules

Tb = 2000
# print( np.shape(energies_cdf))
# print( np.shape(E))
ene_length = abs((energies_.max()) - (energies_.min()))
E_transformed = (E_eV) * ene_length + energies_.min()


def boltCDF(T):
    return boltzmann_cdf(E, T)


# bolcdf = boltzmann_cdf(E, Tb)
def objective(t):
    return np.sum( abs(boltCDF(t) - energies_cdf) )

from scipy.optimize import minimize
result = minimize(objective, x0=100)

result = minimize(objective, x0=2000)
print(f"optimal temperature is {result.x[0]}")
plt.figure(figsize=(10, 6))
plt.plot(
    E_transformed,
    boltzmann_cdf(E, result.x[0]),
    label=f"boltzmann cdf à {result.x[0]}",
    color="purple",
)
plt.plot(E_tot_ml_sorted, E_tot_ml_cdf, label="E_tot_ml CDF", color="orange")
plt.plot(energies_sorted, energies_cdf, label="Energy CDF", color="blue")
plt.xlabel("Energy (eV)")
plt.ylabel("CDF")
plt.title("Comparison of Energy Distributions")
plt.legend()
plt.grid(True)
plt.show()


# kl distance between solution, fisher distance
# distance between parameters ?
# theta* for kl, for rmse, for fisher.
# packaging python project
# find best boltzmann cdf to your dataset.

# # paysage sur les données
# # paysage sur les données
def create_theta_fisher(gradDesc_list: list, gradU_list: list) -> Theta:
   """
   Creates a Theta instance using Fisher's method
   
   Args:
       G: Input Xx3NxD (as list)
       gradU_list: Gradient list
   
   Returns:
       Theta: TypedDict with coef and intercept
   """
   #original_shape = np.array(gradDesc_list).shape
   reshaped = [g.reshape(260, -1) for g in gradDesc_list]
   gradDescT_list = [g.T for g in reshaped]
   print(gradDescT_list[0].shape, "/////")
   print(reshaped[0].shape)
   GT_G = [np.matmul(g, g_t) for g_t, g in zip(gradDescT_list, reshaped)]
   # GT_G = np.matmul(G_array.T, G_array)
   print(len(GT_G), "len gtg")
   print(GT_G[0].shape, "first elem of gtg")
   #a = [np.mean(m) for m in GT_G]
   a = np.mean(GT_G, axis=0)
   
   # Calculate second expectancy (b)
   GT_gradU = [np.matmul(gradU, g_t) for g_t, gradU in zip(gradDescT_list, reshaped)]
   # GT_gradU = np.matmul(G_array.T, gradU_array)
   print(len(GT_gradU), "len gtg")
   print( GT_gradU[0].shape, "first elem of gtg")
   b = np.mean(GT_gradU, axis=0)
   
   # Solve the system
   # print(len(a))
   print(a.shape)
   # print(len(b))
   print(b.shape)
   coef = np.linalg.solve(a, b)
   new_theta: Theta = {
       "coef": coef,
       "intercept": 0.0
   }
   
   return new_theta
# def create_theta_fisher(gradDesc_list: list, gradU_list: list) -> Theta:
#     """
#     Creates a Theta instance using Fisher's method
    
#     Args:
#         G: Input Xx3NxD (as list)
#         gradU_list: Gradient list
    
#     Returns:
#         Theta: TypedDict with coef and intercept
#     """
#     gradDescT_list = [g.T for g in gradDesc_list]
#     GT_G = [np.matmul(g_t, g) for g_t, g in zip(gradDescT_list, gradDesc_list)]
#     # GT_G = np.matmul(G_array.T, G_array)
#     # print("info")
#     print(len(GT_G), "len gtg")
#     print( GT_G[0].shape, "first elem of gtg")
#     #a = [np.mean(m) for m in GT_G]
#     a = np.mean(GT_G, axis=0)
    
#     # Calculate second expectancy (b)
#     GT_gradU = [np.matmul(g_t, gradU) for g_t, gradU in zip(gradDescT_list, gradU_list)]
#     # GT_gradU = np.matmul(G_array.T, gradU_array)
#     #b = [np.mean(gU) for gU in GT_gradU]
#     print(len(GT_gradU), "len gtg")
#     print( GT_gradU[0].shape, "first elem of gtg")
#     b = np.mean(GT_gradU, axis=0)
    
#     # Solve the system
#     # print(len(a))
#     print(a.shape)
#     # print(len(b))
#     print(b.shape)
#     coef = np.linalg.solve(a, b)
#     # coef = [y * 1/x for x, y in zip(a, b)]
#     print(coef[0].shape)
#     # Create new Theta instance
#     theta_star_coeff_array = np.mean(np.array(coef), axis=1)
#     new_coef = theta_star_coeff_array.tolist()
#     new_theta: Theta = {
#         "coef": new_coef,
#         "intercept": 0.0
#     }
    
#     return new_theta


def create_theta_to_theta_star(theta: Theta, theta_star: Theta,n):
   t_list = np.linspace(0, 1, n) 
   # print(t_list)
   theta_list = [] 
   for t in t_list :
       print(t)
       theta_coeff_array = np.array(theta["coef"])
       theta_star_coeff_array = np.mean(np.array(theta_star["coef"]), axis=1)
       new_theta_coef_array = (1-t)*theta_coeff_array + t*theta_star_coeff_array
       new_theta_coef = new_theta_coef_array.tolist()

       new_theta_inter = (1-t)*theta["intercept"] + t*theta_star["intercept"]
       new_theta: Theta = {
       "coef": new_theta_coef,
       "intercept": new_theta_inter 
       }
       theta_list.append(new_theta)
   return theta_list


theta_fish = create_theta_fisher(grad_list, f_list)
print(theta_fish)

def plot_loss_theta_to_theta_star(theta_list):
   print("longeur theta_list", len(theta_list))
   for i, current_theta in enumerate(theta_list) :
       print(i)
       E_tot_current_theta_list = []
       for ats in atoms :
           # descripteurs D \in R^{M \times D}
           desc = ats.get_array("milady-descriptors")
           e_ml = DotProductDesc(current_theta, desc)
           E_tot_ml = np.sum(e_ml)
           E_tot_current_theta_list.append(E_tot_ml)

       E_tot_ml_array = (-1) * np.array(E_tot_current_theta_list)
       E_tot_ml_sorted, E_tot_ml_cdf = cdf(E_tot_ml_array)
       plt.plot(E_tot_ml_sorted, E_tot_ml_cdf, label=f"CDF theta n°{i}", alpha = 0.3)
   plt.legend()
   plt.show()
# :)
# now compare theta fisher and theta normal
theta_list = create_theta_to_theta_star(theta, theta_fish, 4)
print(len(theta_list), "LEN THETAS")

def thetaMSE(theta: Theta, theta_star: Theta):
   # print(t_list)
   arr1 = np.array(theta["coef"])
   arr2 = np.array(theta_star["coef"])

   # Compute the Mean Squared Error
   print(arr1,"STOP")
   print((arr1).max())
   print(arr2)
   print((arr2).max())
   mse = np.mean((arr1 - arr2) ** 2) #np.mean(arr2, axis=1))
   return mse

print(thetaMSE(theta, theta_fish))
print("ok")


E_tot_ml_list2 = []
E_tot_ml_list3 = []

for key, val in md_data.items():
    atoms = val["atoms"]
    energies = val["energies"]
    # Lecture des descripteurs...
    for ats, ene in zip(atoms, energies):
        # descripteurs D \in R^{M \times D}
        desc = ats.get_array("milady-descriptors")
        # évaluation de l'énergie par le modèle linéaire
        e_ml = DotProductDesc(theta, desc)
        E_tot_ml = np.sum(e_ml)
        E_tot_ml_list2.append(E_tot_ml)
        
        e_ml3 = DotProductDesc(theta_fish, desc)
        E_tot_ml3 = np.sum(e_ml3)
        E_tot_ml_list3.append(E_tot_ml3)
        
all_energies = [ene for key, val in md_data.items() for ene in val["energies"]]
energies_ = (-1) * np.array(all_energies)
E_tot_ml_array2 = (-1) * np.array(E_tot_ml_list2)
E_tot_ml_array3 = (-1) * np.array(E_tot_ml_list3)
print(min(E_tot_ml_array3))

# Calculate CDFs
energies_sorted, energies_cdf = cdf(energies_)
E_tot_ml_sorted2, E_tot_ml_cdf2 = cdf(E_tot_ml_array2)
E_tot_ml_sorted3, E_tot_ml_cdf3 = cdf(E_tot_ml_array3)

plt.figure(figsize=(10, 6))
# plt.plot(E_transformed, bolcdf, label=f"boltzmann cdf à {Tb}", color="purple")
plt.plot(E_tot_ml_sorted2, E_tot_ml_cdf2, label="E_tot_ml CDF", color="orange")
# plt.plot(E_tot_ml_sorted3, E_tot_ml_cdf3, label="E_tot_ml CDF", color="green")
plt.plot(energies_sorted, energies_cdf, label="Energy CDF", color="blue")


# E_tot_fish = []
# for ats in atoms :
# # descripteurs D \in R^{M \times D}
#    desc = ats.get_array("milady-descriptors")
#    e_ml = DotProductDesc(theta, desc)    
#    E_tot_ml = np.sum(e_ml)
#    E_tot_fish.append(E_tot_ml)

# E_tot_ml_array = (-1) * np.array(E_tot_fish)
# E_tot_ml_sorted, E_tot_ml_cdf = cdf(E_tot_ml_array)
# plt.plot(E_tot_ml_sorted, E_tot_ml_cdf, label=f"CDF theta fisher", alpha = 0.3)
plt.show()

# print(G_list[0]-G_list[1])
# theta_fish = create_theta_fisher(G_list, f_list)


# def plot_loss_theta_to_theta_star(theta_list):
#     print("longeur theta_list", len(theta_list))
#     for i, current_theta in enumerate(theta_list) :
#         print(i)
#         E_tot_current_theta_list = []
#         for ats in atoms :
#             # descripteurs D \in R^{M \times D}
#             desc = ats.get_array("milady-descriptors")
#             e_ml = DotProductDesc(current_theta, desc)
#             E_tot_ml = np.sum(e_ml)
#             E_tot_current_theta_list.append(E_tot_ml)

#         E_tot_ml_array = (-1) * np.array(E_tot_current_theta_list)
#         E_tot_ml_sorted, E_tot_ml_cdf = cdf(E_tot_ml_array)
#         plt.plot(E_tot_ml_sorted, E_tot_ml_cdf, label=f"CDF theta n°{i}", alpha = 0.3)
#     plt.legend()
#     plt.show()
# # :)
# # now compare theta fisher and theta normal
# theta_list = create_theta_to_theta_star(theta, theta_fish, 4)
# print(len(theta_list), "LEN THETAS")



# def thetaMSE(theta: Theta, theta_star: Theta):
#     # print(t_list)
#     arr1 = np.array(theta["coef"])
#     arr2 = np.array(theta_star["coef"])

#     # Compute the Mean Squared Error
#     print(arr1,"STOP")
#     print((arr1).max())
#     print(arr2)
#     print((arr2).max())
#     mse = np.mean((arr1 - arr2) ** 2) #np.mean(arr2, axis=1))
#     return mse

# thetaMSE(theta, theta_list[0])
# print(thetaMSE(theta, theta_fish))


# E_tot_fish = []
# for ats in atoms :
# # descripteurs D \in R^{M \times D}
#     desc = ats.get_array("milady-descriptors")
#     e_ml = DotProductDesc(theta, desc)    
#     E_tot_ml = np.sum(e_ml)
#     E_tot_fish.append(E_tot_ml)

# E_tot_ml_array = (-1) * np.array(E_tot_fish)
# E_tot_ml_sorted, E_tot_ml_cdf = cdf(E_tot_ml_array)
# plt.plot(E_tot_ml_sorted, E_tot_ml_cdf, label=f"CDF theta fisher", alpha = 0.3)
# plt.show()
# #






