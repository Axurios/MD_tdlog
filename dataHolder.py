import os
import pickle
import numpy as np
from ase import Atoms
from typing import List, Dict, TypedDict
from scipy.stats import entropy
from scipy.constants import Boltzmann

class MDData(TypedDict):
    """Structure of MD Data"""
    atoms: List[Atoms]
    energies: List[float]

class Theta(TypedDict):
    """Structure of θ"""
    coef: np.ndarray
    intercept: float


class DataHolder:
    """Class to manage data loading and processing."""
    def __init__(self):
        self.md_data: Dict[str, MDData] = {}
        self.theta: Theta = {}
        self.E_tot_ml_list: List[float] = []
        self.all_energies: List[float] = []

    def load_md_data(self, file_path: str):
        try:
            with open(file_path, "rb") as f:
                self.md_data = pickle.load(f)
            self._extract_energies()
        except Exception as e:
            raise ValueError(f"Error loading MD data: {e}")

    def load_theta(self, file_path: str):
        try:
            with open(file_path, "rb") as f:
                self.theta = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Error loading Theta data: {e}")

    def compute_predicted_energies(self):
        """Compute predicted energies using θ."""
        if not self.theta or not self.md_data:
            raise ValueError("Theta or MD data not loaded.")
        self.E_tot_ml_list.clear()
        for key, val in self.md_data.items():
            for ats in val['atoms']:
                desc = ats.get_array('milady-descriptors')
                e_ml = np.tensordot(self.theta['coef'], desc, axes=(0, 2))
                self.E_tot_ml_list.append(np.sum(e_ml))
        return self.E_tot_ml_list

    def compute_kl_divergence(self, temperature: float):
        """Calculate the Boltzmannian score (KL divergence)."""
        energies = np.array(self.all_energies) * 1.60218e-19
        return self._kl_divergence(energies, temperature)

    def _extract_energies(self):
        self.all_energies = [
            ene for key, val in self.md_data.items() for ene in val['energies']
        ]

    @staticmethod
    def _kl_divergence(energies, temperature):
        k_B = Boltzmann
        log_boltzmann_weights = -energies / (k_B * temperature)
        max_log_weight = np.max(log_boltzmann_weights)
        log_boltzmann_probs = log_boltzmann_weights - (
            max_log_weight + np.log(np.sum(np.exp(log_boltzmann_weights - max_log_weight)))
        )
        boltzmann_probs = np.exp(log_boltzmann_probs)

        hist, bin_edges = np.histogram(energies, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        empirical_probs = np.interp(energies, bin_centers, hist)
        empirical_probs /= empirical_probs.sum()
        empirical_probs = np.clip(empirical_probs, a_min=1e-12, a_max=None)
        boltzmann_probs = np.clip(boltzmann_probs, a_min=1e-12, a_max=None)

        return entropy(empirical_probs, boltzmann_probs)