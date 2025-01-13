from dataHolder import DataHolder
import numpy as np
from typing import Tuple
from scipy.constants import Boltzmann
from matplotlib.figure import Figure
from fisher import fisher_theta
#from ase.calculators.lj import LennardJones

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


def CDF_plot2(Data : DataHolder, descstr : str):

    keys  = list(Data.md_data.keys())   #getting list of experiences
    fig = Figure()
    ax = fig.add_subplot()

    for k in keys :
        data = np.dot(Data.theta['coef'],np.array(Data.md_data[k]['atoms'][-1].get_array(descstr)).transpose()) #computing energies
        sorted,cdf_data = cdf(data) # making it a distribution
        ax.plot(sorted, cdf_data)   # plotting
    return fig

def CDF_fisher(Data : DataHolder, descstr : str,  gradstr : str, forcestr : str, beta : float = 1):
    
    keys  = list(Data.md_data.keys())   #getting list of experiences
    fig = Figure()
    ax = fig.add_subplot()
    
    fishertheta = fisher_theta(Data, gradstr, forcestr, beta) # computing the new parameters based on fisher
    
    for k in keys : 
        data = np.dot(fishertheta,np.array(Data.md_data[k]['atoms'][-1].get_array(descstr)).transpose())    #computing energies
        sorted,cdf_data = cdf(data) # making it a distribution
        ax.plot(sorted, cdf_data)   # plotting
    return fig