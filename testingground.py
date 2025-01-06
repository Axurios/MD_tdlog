import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.constants import Boltzmann
from scipy.optimize import minimize_scalar
from typing import List, Tuple, Dict, Optional

def compute_energy_distribution(
    energies: List[float], 
    temperature: Optional[float] = None, 
    plot: bool = False
) -> Dict[str, float]:
    """
    Compute and analyze energy distribution characteristics.
    
    Parameters:
    -----------
    energies : List[float]
        List of energy values to analyze
    temperature : float, optional
        Specific temperature to evaluate Boltzmann distribution (default: optimized)
    plot : bool, optional
        Whether to generate visualization of distribution (default: False)
    
    Returns:
    --------
    Dict containing various energy distribution metrics
    """
    # Convert energies to numpy array and to Joules if needed
    energies_J = np.array(energies) * 1.60218e-19  # Convert eV to Joules if necessary
    
    # Cumulative Distribution Function (CDF)
    def cdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the cumulative distribution function."""
        sorted_data = np.sort(data)
        cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        return sorted_data, cdf_values
    
    # Boltzmann CDF
    def boltzmann_cdf(E: np.ndarray, T: float, kb: float = Boltzmann) -> np.ndarray:
        """
        Cumulative distribution function for Boltzmann distribution
        
        Parameters:
        -----------
        E : np.ndarray
            Energy levels (in Joules)
        T : float
            Temperature (in Kelvin)
        kb : float, optional
            Boltzmann constant (default: scipy.constants.Boltzmann)
        
        Returns:
        --------
        np.ndarray: Cumulative distribution values
        """
        beta = 1 / (kb * T)
        return 1 - np.exp(-beta * E)
    
    # Optimize temperature to match empirical distribution
    def objective_temperature(T: float) -> float:
        """
        Objective function to minimize difference between empirical 
        and Boltzmann CDFs
        
        Parameters:
        -----------
        T : float
            Temperature to evaluate
        
        Returns:
        --------
        float: Sum of absolute differences between CDFs
        """
        # Normalize energy range
        energy_range = np.linspace(0, 1, len(energies_J))
        E = energy_range * (energies_J.max() - energies_J.min()) + energies_J.min()
        
        # Compute CDFs
        _, empirical_cdf = cdf(energies_J)
        theoretical_cdf = boltzmann_cdf(E, T)
        
        return np.sum(np.abs(empirical_cdf - theoretical_cdf))
    
    # Optimize temperature if not provided
    if temperature is None:
        result = minimize_scalar(objective_temperature, bounds=(10, 5000), method='bounded')
        optimal_temperature = result.x
    else:
        optimal_temperature = temperature
    
    # Compute KL divergence to assess Boltzmannian nature
    def kl_boltzmann(energies: np.ndarray, T: float) -> float:
        """
        Evaluate Boltzmannian nature of energy distribution
        
        Parameters:
        -----------
        energies : np.ndarray
            Energy values
        T : float
            Temperature to evaluate
        
        Returns:
        --------
        float: KL divergence score (lower is more Boltzmann-like)
        """
        # Compute log-Boltzmann weights
        log_boltzmann_weights = -energies / (Boltzmann * T)
        
        # Log-Sum-Exp Trick for numerical stability
        max_log_weight = np.max(log_boltzmann_weights)
        log_boltzmann_probs = log_boltzmann_weights - (
            max_log_weight + np.log(np.sum(np.exp(log_boltzmann_weights - max_log_weight)))
        )
        boltzmann_probs = np.exp(log_boltzmann_probs)
        
        # Compute empirical distribution
        hist, bin_edges = np.histogram(energies, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        empirical_probs = np.interp(energies, bin_centers, hist)
        
        # Normalization and avoiding zero probabilities
        empirical_probs /= empirical_probs.sum()
        empirical_probs = np.clip(empirical_probs, a_min=1e-12, a_max=None)
        boltzmann_probs = np.clip(boltzmann_probs, a_min=1e-12, a_max=None)
        
        return entropy(empirical_probs, boltzmann_probs)
    
    # Visualization if requested
    if plot:
        plt.figure(figsize=(10, 6))
        
        # Compute various CDFs
        energy_sorted, energy_cdf = cdf(energies_J)
        
        # Create normalized energy range for Boltzmann CDF
        energy_range = np.linspace(0, 1, len(energies_J))
        E = energy_range * (energies_J.max() - energies_J.min()) + energies_J.min()
        
        # Plot CDFs
        plt.plot(energy_sorted, energy_cdf, label='Empirical CDF', color='blue')
        plt.plot(E, boltzmann_cdf(E, optimal_temperature), 
                 label=f'Boltzmann CDF (T={optimal_temperature:.2f}K)', 
                 color='red', linestyle='--')
        
        plt.xlabel('Energy (J)')
        plt.ylabel('Cumulative Probability')
        plt.title('Energy Distribution Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Compute KL divergence at the optimized temperature
    kl_divergence = kl_boltzmann(energies_J, optimal_temperature)
    
    # Return comprehensive results
    return {
        'optimal_temperature': optimal_temperature,
        'kl_divergence': kl_divergence,
        'mean_energy': np.mean(energies_J),
        'std_energy': np.std(energies_J),
        'min_energy': np.min(energies_J),
        'max_energy': np.max(energies_J)
    }

# Example usage
def main():
    # Simulated energy data
    sample_energies = np.random.normal(0, 1, 1000)
    
    # Compute energy distribution
    results = compute_energy_distribution(sample_energies, plot=True)
    print("Energy Distribution Analysis Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()