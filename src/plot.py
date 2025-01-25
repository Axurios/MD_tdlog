from .dataHolder import DataHolder
import numpy as np
from typing import Tuple
from scipy.constants import Boltzmann
from matplotlib.figure import Figure
from .fisher import fisher_theta


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
        #Data.md_data[k]['atoms'][-1].calc = LennardJones()
        data = np.dot(fishertheta,np.array(Data.md_data[k]['atoms'][-1].get_array(descstr)).transpose())
        sorted,cdf_data = cdf(data)
        ax.plot(sorted, cdf_data)
    return fig




def CDF_inverse(P, data):
  # Sort the sample to use for inverse CDF (quantile function)
    sorted_data = np.sort(data)
    # Calculate the index corresponding to the quantile
    int_list = [int(np.floor(p * np.shape(sorted_data)[0])) for p in P]
    #index = int(np.floor(p * np.shape(sorted_data)[0]))
    # Return the value from the sorted sample at that index
    return sorted_data[int_list]



def Q_KS(lambda_):
    # Survival function of Kolmogorov distribution
    terms = [(-1)**(k-1) * np.exp(-2 * (k**2) * (lambda_**2)) for k in range(1, 100)]
    return 2 * sum(terms)


def CDF(t, data_sample):
  selected_sample = np.extract(data_sample <= t, data_sample)
  return len(selected_sample) / len(data_sample)


def ks_plot(Data : DataHolder, descstr : str,  gradstr : str, forcestr : str, beta : float = 1):
    keys  = list(Data.md_data.keys())
    fig = Figure()
    ax = fig.add_subplot()
    fishertheta = fisher_theta(Data, gradstr, forcestr, beta)
    #ax.set_yscale('log')
    for i,k in enumerate(keys):
        if i == 0: 
            # Generating CDFs
            fisher_data = np.dot(fishertheta,np.array(Data.md_data[k]['atoms'][-1].get_array(descstr)).transpose()) 
            data = np.dot(Data.theta['coef'],np.array(Data.md_data[k]['atoms'][-1].get_array(descstr)).transpose())
            fisher_sorted, fisher_cdf_data = cdf(fisher_data)
            sorted_data, cdf_data = cdf(data)

            # Create the union of the sorted data
            union_list = np.sort(np.unique(np.concatenate((sorted_data, fisher_sorted))))

            # Interpolating CDFs onto the union list
            if len(sorted_data) > 0 and len(fisher_sorted) > 0:
                cdf1_interpolated = np.interp(union_list, sorted_data, cdf_data)
                cdf2_interpolated = np.interp(union_list, fisher_sorted, fisher_cdf_data)

                # Compute the Kolmogorov-Smirnov statistic
                gap = np.abs(cdf1_interpolated - cdf2_interpolated)
                statistic = np.max(gap)
                max_index = np.argmax(gap)
                y1, y2 = cdf1_interpolated[max_index], cdf2_interpolated[max_index]
                stat_sign = np.sign(y1-y2)

                # Plotting
                ax.plot(fisher_sorted, fisher_cdf_data, label="Fisher CDF", color="blue")
                ax.plot(sorted_data, cdf_data, label="Data CDF", color="orange")
                ax.vlines(union_list[max_index], y1, y2, color="red", linestyle="--", label="KS Gap")

                # Annotate the KS statistic
                ax.annotate(
                    f"KS Statistic: {statistic:.3f}",
                    xy=(union_list[max_index], (y1 + y2) / 2),
                    xytext=(union_list[max_index], max(y1, y2) + 0.1),
                    arrowprops=dict(facecolor='black', arrowstyle="->"),
                )
                n1, n2 = len(sorted_data), len(fisher_sorted)
                n_eff = (n1*n2) / (n1+n2)
                # Calculate lambda and p_value
                lambda_ = np.sqrt(n_eff) * statistic  # D is the KS statistic
                p_value = Q_KS(lambda_)

                print({"statistique": statistic, "p_value": p_value, "stat_location": round(union_list[max_index], 2), "stat_sign": stat_sign})
            else:
                raise ValueError("One of the input CDF arrays is empty or too small.")
#    return fig

            data = np.dot(fishertheta,np.array(Data.md_data[k]['atoms'][-1].get_array(descstr)).transpose())    #computing energies
            sorted,cdf_data = cdf(data) # making it a distribution
            ax.plot(sorted, cdf_data)   # plotting
    return fig

