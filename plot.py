from dataHolder import DataHolder
import numpy as np
from typing import Tuple
from scipy.constants import Boltzmann
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def cdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Cumulative Distribution Function (CDF).
    Args: data (np.ndarray): Input data array
    Returns: Tuple of sorted data and CDF values
    """
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


def boltzmann_cdf(E, T, kb=Boltzmann):
    """
    Cumulative distribution function for Boltzmann distribution
    E: Energy levels (in Joules)
    T: Temperature (in Kelvin)
    kb: Boltzmann constant (in J/K)
    """
    beta = 1/(kb*T)
    return 1 - np.exp(-beta*E)


def CDF_plot(Data : DataHolder):
    # Get energy distributions
    energies_, E_tot_ml_array = Data.get_energy_distributions()
    print(E_tot_ml_array)

    # Compute CDFs
    energies_sorted, energies_cdf = cdf(energies_)
    E_tot_ml_sorted, E_tot_ml_cdf = cdf(E_tot_ml_array)

    E_eV = np.linspace(0, 1, len(energies_sorted))  # Energy range from 0 to 1 eV
    eV_to_J = 1.602176634e-19
    E = E_eV * eV_to_J  # Convert to Joules
    Tb = 2000
    # Prepare energy range for Boltzmann CDF
    ene_length = abs((energies_.max()) - (energies_.min()))
    E_transformed = (E_eV)*ene_length + energies_.min()

    # Compute Boltzmann CDF
    bolcdf = boltzmann_cdf(E, Tb)
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Plot data
    ax.plot(E_transformed, bolcdf, label=f"Boltzmann CDF at {Tb}K", color='purple')
    # ax.plot(E_tot_ml_sorted, E_tot_ml_cdf, label='E_tot_ml CDF', color='orange')
    ax.plot(energies_sorted, energies_cdf, label='Energy CDF', color='blue')

    # Labeling
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('CDF')
    ax.set_title('Comparison of Energy Distributions')
    ax.legend()
    ax.grid(True)

    return fig  # Return the figure object
