import pickle
import numpy as np
#import os
from ase import Atoms
from typing import TypedDict, List, Dict, Tuple
#from scipy.stats import entropy
#from scipy.constants import Boltzmann
#import matplotlib.pyplot as plt


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
        try:
            with open(file_path, "rb") as f:
                self.md_data = pickle.load(f)

            first_item = list(self.md_data.keys())[0]
            dict = self.md_data[first_item]['atoms'][0].arrays
            list_of_attributes = list(dict.keys())

            # Extract energies
            self._extract_energies()
            # Prepare metadata for display
            # self.metadata = self._get_md_metadata(descriptor)

        except Exception as e:
            raise ValueError(f"Error loading MD data: {e}")
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

            # Prepare metadata for display
        #self.metadata = self._get_theta_metadata()
        self.theta_loaded = True


    def _get_md_metadata(self, descriptor):
        """
        Generate metadata about the loaded MD data.
        Returns: dict: Metadata about MD data
        """
        # If no data is loaded
        if not self.md_data:
            return {"summary": "No MD data loaded", "total_datasets": 0, "details": []}

        # Prepare detailed metadata
        metadata = {"total_datasets": len(self.md_data), "details": []}

        for key, val in self.md_data.items():
            dataset_info = {
                "key": key,
                "num_configurations": len(val["atoms"]),
                "energy_range": (min(val["energies"]), max(val["energies"])),
                "first_config_details": {
                    "num_atoms": len(val["atoms"][0]) if val["atoms"] else 0,
                    "first_energy": val["energies"][0] if val["energies"] else None,
                    "descriptor_shape": (
                        val["atoms"][0].get_array(descriptor).shape
                        if val["atoms"]
                        else None
                    ),
                },
            }
            metadata["details"].append(dataset_info)

        # Add summary
        metadata["summary"] = (
            f"Loaded {metadata['total_datasets']} datasets\n"
            f"Total configurations: {sum(len(val['atoms']) for val in self.md_data.values())}\n"
            f"Energy range across all datasets: {min(self.all_energies)} to {max(self.all_energies)} eV"
        )

        return metadata


    def _get_theta_metadata(self):
        """
        Generate metadata about the loaded Theta parameters.
        Returns: dict: Metadata about Theta parameters
        """
        # If no theta data is loaded
        if not self.theta:
            return {
                "summary": "No Theta data loaded",
                "coefficient_shape": None,
                "intercept": None,
            }

        return {
            "summary": "Theta parameters loaded successfully",
            "coefficient_shape": self.theta["coef"].shape,
            "intercept": self.theta["intercept"],
            "coefficient_details": {
                "min_value": np.min(self.theta["coef"]),
                "max_value": np.max(self.theta["coef"]),
                "mean_value": np.mean(self.theta["coef"]),
            },
        }


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


def check_md_format(data, check_in_atoms= []):  
    if (not isinstance(data, dict)) :
        raise ValueError("data is not pickle dictionnary")
    for key, val in data.items():
        if ('atoms' not in val or "energies" not in val):
            raise ValueError(f"Invalid MD data for key {key}.")
        if not all(type(atoms) ==  Atoms for atoms in val['atoms']):
            raise ValueError(f"'atoms' elements are not Atoms element from ase")
    atom1 = data[list(data.keys())[0]]['atoms'][0]
    for array in check_in_atoms: 
        if not atom1.has(array) :
            raise ValueError(f"Invalid data for atoms in MD : no descriptor named {array} found")

