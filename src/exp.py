# # can be deleted ?
# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import List, Dict, Tuple
# from ase import Atoms
# from dataclasses import dataclass
# from scipy.stats import entropy
# from scipy.constants import Boltzmann

# @dataclass
# class Theta:
#     """Represents the model parameters for linear descriptor-based energy calculation."""
#     coef: np.ndarray
#     intercept: float = 0.0

# @dataclass
# class MDDataEntry:
#     """Represents a single molecular dynamics data entry."""
#     atoms: List[Atoms]
#     energies: List[float]

# class MolecularDynamicsAnalyzer:
#     """A class to analyze molecular dynamics data using linear descriptor-based models."""
#     def __init__(self, data_path: str, theta_path: str):
#         """
#         Initialize the analyzer with data and model parameters.
#         Args:
#             data_path (str): Path to the pickle file containing MD data
#             theta_path (str): Path to the pickle file containing model parameters
#         """
#         self.md_data: Dict[str, MDDataEntry] = self._load_data(data_path)
#         self.theta: Theta = self._load_theta(theta_path)
#         self.E_tot_ml_list: List[float] = []
#         self.eV_to_J = 1.602176634e-19
    
#     @staticmethod
#     def _load_data(path: str) -> Dict[str, MDDataEntry]:
#         """Load molecular dynamics data from a pickle file."""
#         try:
#             return pickle.load(open(path, 'rb'))
#         except Exception as e:
#             print(f"Error loading data: {e}")
#             return {}
    
#     @staticmethod
#     def _load_theta(path: str) -> Theta:
#         """Load model parameters from a pickle file."""
#         try:
#             return pickle.load(open(path, 'rb'))
#         except Exception as e:
#             print(f"Error loading theta: {e}")
#             return Theta(coef=np.array([]))
    
#     def dot_product_desc(self, desc_type: np.ndarray) -> np.ndarray:
#         """Calculate dot product of descriptors with model coefficients."""
#         if len(desc_type.shape) > 2:
#             return np.tensordot(self.theta["coef"], desc_type, axes=(0, 2))
#         else:
#             return desc_type @ self.theta["coef"] + self.theta["intercept"]
    
#     def analyze_molecular_dynamics(self) -> None:
#         """Analyze molecular dynamics data by calculating ML energies and forces."""
#         self.E_tot_ml_list = []  # Reset the list
#         for key, val in self.md_data.items():
#             atoms = val['atoms']
#             # liste contenant les energies associée
#             energies = val['energies']
            
#             for ats, ene in zip(atoms, energies):
#                 # Descriptors
#                 desc = ats.get_array('milady-descriptors')
#                 # Descriptor gradients
#                 grad_desc = ats.get_array('milady-descriptors-forces')
                
#                 # ML energy evaluation
#                 e_ml = self.dot_product_desc(desc)
#                 E_tot_ml = np.sum(e_ml)
#                 self.E_tot_ml_list.append(E_tot_ml)
    
#     def get_energy_distributions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         """
#         Compute energy distributions and CDFs.
#         Returns:
#             Tuple containing:
#             - All original energies
#             - ML predicted energies
#             - Sorted original energies
#             - Sorted ML energies
#         """
#         # Collect all original energies
#         all_energies = [ene for key, val in self.md_data.items() for ene in val["energies"]]
        
#         # Convert energies and apply sign change
#         energies_ = (-1) * np.array(all_energies)  # * self.eV_to_J  # Uncomment to convert to Joules
#         E_tot_ml_array = (-1) * np.array(self.E_tot_ml_list)  # * self.eV_to_J  # Uncomment to convert to Joules
        
#         return energies_, E_tot_ml_array
    
#     @staticmethod
#     def cdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Compute Cumulative Distribution Function (CDF).
#         Args:
#             data (np.ndarray): Input data array
#         Returns:
#             Tuple of sorted data and CDF values
#         """
#         sorted_data = np.sort(data)
#         cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
#         return sorted_data, cdf
    
#     @staticmethod
#     def boltzmann_cdf(E, T, kb=Boltzmann):
#         """
#         Cumulative distribution function for Boltzmann distribution
#         E: Energy levels (in Joules)
#         T: Temperature (in Kelvin)
#         kb: Boltzmann constant (in J/K)
#         """
#         beta = 1/(kb*T)
#         return 1 - np.exp(-beta*E)
    
#     def plot_energy_distributions(self, temperature: float = 2000) -> None:
#         """
#         Plot energy distributions and CDFs.
#         Args:
#             temperature (float): Temperature for Boltzmann CDF calculation
#         """
#         # Get energy distributions
#         energies_, E_tot_ml_array = self.get_energy_distributions()
        

#         # Compute CDFs
#         energies_sorted, energies_cdf = self.cdf(energies_)
#         E_tot_ml_sorted, E_tot_ml_cdf = self.cdf(E_tot_ml_array)
        
#         E_eV = np.linspace(0, 1, len(energies_sorted))  # Energy range from 0 to 1 eV
#         eV_to_J = 1.602176634e-19
    

#         E = E_eV * eV_to_J  # Convert to Joules
#         Tb = 2000
        
#         # Prepare energy range for Boltzmann CDF
#         ene_length = abs((energies_.max()) - (energies_.min()))
#         E_transformed = (E_eV)*ene_length  +energies_.min() 


        
#         # Compute Boltzmann CDF
#         bolcdf = self.boltzmann_cdf(E, Tb)
#         # bolcdf = self.boltzmann_cdf(E_transformed * self.eV_to_J, temperature)
        
#         # Plotting
#         plt.figure(figsize=(10, 6))
#         plt.plot(E_transformed, bolcdf, label=f"Boltzmann CDF at {temperature}K", color='purple')
#         plt.plot(E_tot_ml_sorted, E_tot_ml_cdf, label='E_tot_ml CDF', color='orange')
#         plt.plot(energies_sorted, energies_cdf, label='Energy CDF', color='blue')
        
#         plt.xlabel('Energy (eV)')
#         plt.ylabel('CDF')
#         plt.title('Comparison of Energy Distributions')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
        
#         # Print energy ranges for reference
#         print("E_tot_ml range:", E_tot_ml_array.min(), E_tot_ml_array.max())
#         print("Original energies range:", energies_.min(), energies_.max())

# def main():
#     # Determine paths relative to the script location
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     path_data = os.path.join(current_dir, 'Data/NP_1200K_desc.pkl')
#     path_theta = os.path.join(current_dir, 'Data/theta.pkl')
    
#     # Create and run the analyzer
#     analyzer = MolecularDynamicsAnalyzer(path_data, path_theta)
#     analyzer.analyze_molecular_dynamics()
    
#     # Plot energy distributions
#     analyzer.plot_energy_distributions(temperature=2000)

# if __name__ == "__main__":
#     main()