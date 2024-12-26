import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import Boltzmann

ceil = 0.1

def boltzmann(E, T, Z):
    return (np.exp(-E/(Boltzmann*T))/Z)

E_data = np.linspace(0, 100, 1000)
y_data =  np.exp(-E_data/(Boltzmann*290))/30 + np.random.normal(0, 0.05, E_data.size)

params, covariance = curve_fit(boltzmann, E_data, y_data, p0=[300, 10])

T_fit, Z_fit = params
print(f"Paramètres ajustés : T = {T_fit:.2f}, Z = {Z_fit:.2f}")

# Générer la courbe ajustée
y_fit = boltzmann(E_data, *params)

E_data_boltzmann = []
y_data_boltzmann = []
for i in range (len(y_data)) : 
    if np.argmin(abs(y_fit - y_data[i])) < ceil:
        y_data_boltzmann.append(y_data[i])
        E_data_boltzmann.append(E_data[i])


# Affichage des résultats
plt.scatter(E_data, y_data, label="Données", color="blue", alpha=0.6)
plt.plot(E_data, y_fit, color="red")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()