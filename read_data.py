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
print(md_data[keys[0]].keys())
print(type(md_data[keys[0]]['atoms'][0]))
print(md_data[keys[0]]['atoms'][1])
print(md_data[keys[0]]['atoms'][1].get_array("forces"))
#print(md_data[keys[0]]['atoms'][1].get_positions())

E_tot_ml_list = []
for key, val in md_data.items():
    # liste contenant des objets Atoms
    atoms = val["atoms"]
    # liste contenant les energies associée
    energies = val["energies"]
    # Lecture des descripteurs...
    for ats, ene in zip(atoms, energies):
        # descripteurs D \in R^{M \times D}
        desc = ats.get_array("milady-descriptors")
        # gradient des descripteurs \nabla D \in R^{M \times D \times 3}
        grad_desc = ats.get_array("milady-descriptors-forces")
        # positions des atomes \in R^{M \times 3}
        position = ats.positions
        # forces sur les atomes \in R^{M \times 3}
        f = ats.get_array("forces")
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


# def boltzmann_cdf(E, T, kb=Boltzmann):
#     E_center = E/(abs(np.max(E) - np.min(E)))
#     E_norm = E_center - abs(np.max(E_center) - np.min(E_center))/2
#     print(np.max(E_norm), np.min(E_norm))
#     eV_to_J = 1.602176634e-19
#     E_norm = (E_norm+ 1.2) * eV_to_J
#     beta = 1/(kb*T)
#     return (1 - np.exp(-beta*(E_norm)))
# Create energy values in electron volts (eV) then convert to Joules
# 1 eV = 1.602176634e-19 Joules
# E_eV = np.linspace(0, 1, 1000)  # Energy range from 0 to 1 eV
# E = E_eV * eV_to_J
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


eV_to_J = 1.602176634e-19
E_eV = np.linspace(0, 1, len(energies_sorted))  # Energy range from 0 to 1 eV

E = E_eV * eV_to_J  # Convert to Joules
Tb = 2000
bolcdf = boltzmann_cdf(E, Tb)
ene_length = abs((energies_.max()) - (energies_.min()))
E_transformed = (E_eV) * ene_length + energies_.min()
# print("Bolt energies range:", E_transformed.min(), E_transformed.max())
# Calculate theoretical Boltzmann CDF
# Tb = 200  # Temperature in Kelvin # Tb = 300
# # theoretical_energies, theoretical_cdf = theoretical_boltzmann_cdf(energies_, Tb)
# bolt_cdf = boltzmann_cdf(energies_, Tb)
# Plot the CDFs


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
    Returns:
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


# # Objective function to minimize the difference between empirical and theoretical CDF
# def objective_function(params, energies_sorted, empirical_cdf_vals):
#     T = params[0]  # Temperature
#     if T <= 0:  # Ensure valid temperature
#         return np.inf
#     # Generate the theoretical CDF for the current temperature
#     theoretical_cdf = boltzmann_cdf(energies_sorted, T)
#     # Compute the sum of squared differences between the empirical and theoretical CDFs
#     return np.sum((theoretical_cdf - empirical_cdf_vals)**2)

# # Generate empirical CDF values
# empirical_cdf_vals = energies_cdf

# # Initial guess for temperature
# initial_guess = [1000]  # Temperature in Kelvin
# from scipy.optimize import minimize
# # Minimize the objective function
# result = minimize(
#     objective_function, initial_guess,
#     args=(energies_sorted, empirical_cdf_vals),
#     bounds=[(1e-6, None)],  # Temperature must be positive
#     method='L-BFGS-B'
# )

# # Optimal temperature
# T_opt = result.x[0]

# # Generate theoretical CDF with the optimized temperature
# theoretical_cdf = boltzmann_cdf(energies_sorted, T_opt)

# # Plotting the CDFs
# plt.figure(figsize=(8, 6))
# plt.plot(energies_sorted, empirical_cdf_vals, label='Empirical CDF', color='blue')
# plt.plot(energies_sorted, theoretical_cdf, label=f'Optimized Boltzmann CDF (T={T_opt:.2f} K)', color='orange')
# plt.xlabel('Energy')
# plt.ylabel('CDF')
# plt.title('Empirical vs Optimized Boltzmann CDF')
# plt.legend()
# plt.grid()
# plt.show()


# # EN TRAVAUX : (pas fini)
# def KL_boltzmann(energies: List[float], temperature: float) -> float:
#     """
#     Evaluate how Boltzmannian the energy distribution is.
#     Parameters:
#     - energies: List of energy values from the dataset.
#     - temperature: Temperature in Kelvin at which the distribution should be evaluated. # noqa:

#     Returns:
#     - A score representing the closeness to a Boltzmann distribution.
#       Lower values indicate a closer match.
#     """
#     k_B = Boltzmann  # Boltzmann constant in J/K (SI units)
#     # Convert energies to SI units if needed (assuming they're in eV)
#     energies_J = np.array(energies) * 1.60218e-19  # eV to Joules
#     # Compute log-Boltzmann weights for numerical stability
#     log_boltzmann_weights = -energies_J / (k_B * temperature)

#     # Log-Sum-Exp Trick
#     max_log_weight = np.max(log_boltzmann_weights)  # Maximum value for numerical stability
#     log_boltzmann_probs = log_boltzmann_weights - (max_log_weight + np.log(np.sum(np.exp(log_boltzmann_weights - max_log_weight))))
#     # Convert back to probabilities
#     boltzmann_probs = np.exp(log_boltzmann_probs)

#     # Estimate the empirical energy distribution :
#     # (interpolating an histogram, might be useful to look into more details here
#     hist, bin_edges = np.histogram(energies_J, bins='auto', density=True)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     # Interpolate the empirical histogram onto the Boltzmann probabilities
#     empirical_probs = np.interp(energies_J, bin_centers, hist)

#     # Normalization
#     empirical_probs /= empirical_probs.sum()
#     # Avoid zero probabilities for KL divergence calculation
#     empirical_probs = np.clip(empirical_probs, a_min=1e-12, a_max=None)
#     boltzmann_probs = np.clip(boltzmann_probs, a_min=1e-12, a_max=None)
#     # Compute the Kullback-Leibler (KL) divergence between the distributions
#     kl_divergence = entropy(empirical_probs, boltzmann_probs)
#     return kl_divergence


# # Evaluate Boltzmannian nature at a specific temperature
# temperature = 200  # Example temperature in Kelvin
# boltzmann_score_dataset = KL_boltzmann(all_energies, temperature)
# boltzmann_score_simulation = KL_boltzmann(e_ml, temperature)
# print(f"Boltzmannian score (KL divergence): {boltzmann_score_dataset}")
# print(f"Boltzmannian score (KL divergence): {boltzmann_score_simulation}")
# # k_B = 1.380649e-23  # Boltzmann constant in J/K
# # boltzmann_weights = np.exp(-energies_si / (k_B * temperature))
# # boltzmann_weights /= np.sum(boltzmann_weights)  # Normalize weights
# # Generate Boltzmann distribution (sorted for CDF plotting)
# # sorted_energies, boltzmann_cdf = cdf(energies_si)
# #energies_sorted, energies_cdf = cdf(energies_)
# #e_ml_sorted, e_ml_cdf = cdf(e_ml_)

# # def boltzmann_cdf(energy, T):
# #     """
# #     Calculate the Boltzmann CDF for a given temperature T using the log-sum-exp trick.
# #     :param energy: Array of energy values
# #     :param T: Temperature in Kelvin
# #     :return: Array of CDF values
# #     """
# #     kT = Boltzmann * T
# #     scaled_energy = -energy / kT
# #     # Compute log-sum-exp for normalization
# #     # max_scaled_energy = np.max(scaled_energy)
# #     #log_sum_exp = max_scaled_energy + np.log(np.sum(np.exp(scaled_energy - max_scaled_energy)))
# #     # Calculate the normalized CDF
# #     # log_cdf = np.log(np.cumsum(np.exp(scaled_energy - max_scaled_energy)))  # Cumulative sum in log space
# #     # log_cdf -= log_sum_exp  # Normalize in log space
# #     # Return the CDF in standard space
# #     cdf = np.exp(scaled_energy)
# #     return cdf
