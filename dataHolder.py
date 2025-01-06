import pickle
import numpy as np
from ase import Atoms
from typing import TypedDict, List, Dict, Tuple


class MDData(TypedDict):
    """Structure of MD Data dictionary"""
    atoms: List[Atoms]
    energies: List[float]


class Theta(TypedDict):
    """Structure of θ dictionary"""
    coef: np.ndarray
    intercept: float


class DataHolder:
    """Class to manage data loading and processing."""

    def __init__(self):
        self.md_data: Dict[str, MDData] = {}
        self.md_data_loaded = False
        self.theta: Theta = {}
        self.theta_loaded = False
        self.E_tot_ml_list: List[float] = []
        self.all_energies: List[float] = []
        self.plot_data = None
        self.metadata = None

    def load_md_data(self, file_path: str):
        """
        Load Molecular Dynamics data from a pickle file.
        Args:
            file_path (str): Path to the pickle file containing MD data
            descriptor (str) : key for descriptor in atoms
        Returns:
            dict: Metadata about the loaded MD data
        """
        with open(file_path, "rb") as f:
            self.md_data = pickle.load(f)

        check_md_format(self.md_data)

        # extract list of attributes in atoms
        first_item = list(self.md_data.keys())[0]
        dict = self.md_data[first_item]['atoms'][0].arrays
        list_of_attributes = list(dict.keys())

        # Extract energies
        self._extract_energies()

        self.md_data_loaded = True
        return list_of_attributes

    def load_theta(self, file_path: str):
        """
        Load Theta parameters from a pickle file.
        Args: file_path (str): Path to the pickle file containing Theta parameters
        Returns: dict: Metadata about the loaded Theta parameters
        """
        
        with open(file_path, "rb") as f:
            self.theta = pickle.load(f)
            # Validate Theta data structure
        if (
            not isinstance(self.theta, dict)
            or "coef" not in self.theta
            or "intercept" not in self.theta
        ):
            raise ValueError("Invalid Theta data format")

        self.theta_loaded = True

    def _extract_energies(self):
        """Extract energies from MD data."""
        self.all_energies = [
            ene for key, val in self.md_data.items() for ene in val["energies"]
        ]
    
    def get_energy_distributions(self) -> Tuple[np.ndarray,
                                                np.ndarray,
                                                np.ndarray,
                                                np.ndarray]:
        # Collect all original energies
        all_energies = [ene for key, val in self.md_data.items() for ene in val["energies"]]
        # Convert energies and apply sign change
        energies_ = (-1) * np.array(all_energies)  # * self.eV_to_J  # Uncomment to convert to Joules
        E_tot_ml_array = (-1) * np.array(self.E_tot_ml_list)  # * self.eV_to_J  # Uncomment to convert to Joules

        return energies_, E_tot_ml_array


def check_md_format(data):  
    if (not isinstance(data, dict)):
        raise ValueError("data is not pickle dictionnary")
    for key, val in data.items():
        if ('atoms' not in val or "energies" not in val):
            raise ValueError(f"Invalid MD data for key {key}.")
        if not all(type(atoms) is Atoms for atoms in val['atoms']):
            raise ValueError("'atoms' elements are not Atoms element from ase")