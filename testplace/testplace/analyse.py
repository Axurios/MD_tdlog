import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np
from typing import Tuple



# Define line styles for different methods
line_styles = {
    'default': '-',
    'mixed': '--',
    'fisher': ':',
    'bfgs': '-.'  # or any style you like
}


# Updated patterns to capture epot, ekin, etot
patterns = {
    'bfgs': re.compile(r'^BFGS:\s*(\d+)\s[\d:]+\s+([-0-9.eE]+)'),  # Still captures only etot
    'default': re.compile(r'\[default\]\s*step\s*(\d+)\s+epot\s+([-0-9.eE]+)\s+ekin\s+([-0-9.eE]+)\s+etot\s+([-0-9.eE]+)'),
    'mixed': re.compile(r'\[mixed\]\s*step\s*(\d+)\s+epot\s+([-0-9.eE]+)\s+ekin\s+([-0-9.eE]+)\s+etot\s+([-0-9.eE]+)'),
    'fisher': re.compile(r'\[fisher\]\s*step\s*(\d+)\s+epot\s+([-0-9.eE]+)\s+ekin\s+([-0-9.eE]+)\s+etot\s+([-0-9.eE]+)')
}

# New structure: {folder: {method: [(step, epot, ekin, etot)]}}
data = defaultdict(lambda: defaultdict(list))
processed_paths = []

# Walk and collect data
for root, _, files in os.walk('.'):
    for file in files:
        if file.startswith('out_run'):
            folder = os.path.relpath(root, '.')
            filepath = os.path.join(root, file)
            processed_paths.append(os.path.abspath(filepath))

            with open(filepath) as f:
                for line in f:
                    for method, pattern in patterns.items():
                        match = pattern.search(line)
                        if match:
                            step = int(match.group(1))
                            if method == 'bfgs':
                                etot = float(match.group(2))
                                data[folder][method].append((step, None, None, etot))
                            else:
                                epot = float(match.group(2))
                                ekin = float(match.group(3))
                                etot = float(match.group(4))
                                data[folder][method].append((step, epot, ekin, etot))

# Sort each method's data by step
for folder in data:
    for method in data[folder]:
        data[folder][method].sort()


# Colors per folder
colors = plt.cm.tab10.colors
folder_colors = {folder: colors[i % len(colors)] for i, folder in enumerate(data)}

# Energy type line styles (can adjust as needed)
energy_styles = {
    'epot': ':',
    'ekin': '--',
    'etot': '-'
}

plt.figure(figsize=(12, 7))

for folder, methods in data.items():
    color = folder_colors[folder]
    for method, points in methods.items():
        if method == 'bfgs':  # Skip bfgs
            continue
        steps, epots, ekins, etots = zip(*points)

        # Plot each energy component with a distinct linestyle
        for energy_type, values, style in zip(
            ['epot', 'ekin', 'etot'],
            [epots, ekins, etots],
            [energy_styles['epot'], energy_styles['ekin'], energy_styles['etot']]
        ):
            label = f"{folder} - {method} - {energy_type}"
            plt.plot(steps, values, linestyle=style, color=color)

# Build legend for line styles (for energy types)
custom_lines = []
for energy_type, style in energy_styles.items():
    custom_lines.append(plt.Line2D([], [], linestyle=style, color='black', label=energy_type))

# Add legend
plt.xlabel('Step')
plt.ylabel('Energy')
plt.title('Energy Components vs Step for Different Methods')
plt.legend(ncol=2, fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()



# ---------- Zoomed-In Plot ----------
plt.figure(figsize=(12, 7))

for folder, methods in data.items():
    color = folder_colors[folder]
    for method, points in methods.items():
        if method == 'bfgs':
            continue

        # Filter for steps <= 10000
        zoomed_points = [(s, ep, ek, et) for s, ep, ek, et in points if s <= 10000]
        if not zoomed_points:
            continue

        steps, epots, ekins, etots = zip(*zoomed_points)

        # Plot each energy component with unique linestyle
        for energy_type, values, style in zip(
            ['epot', 'ekin', 'etot'],
            [epots, ekins, etots],
            [energy_styles['epot'], energy_styles['ekin'], energy_styles['etot']]
        ):
            label = f"{folder} - {method} - {energy_type}"
            plt.plot(steps, values, linestyle=style, color=color)

# Create dummy lines for legend to show energy type styles
for energy_type, style in energy_styles.items():
    plt.plot([], [], style, color='black', label=energy_type)

plt.xlabel('Step')
plt.ylabel('Energy')
plt.title('Zoomed: Energy Components vs Step (Steps ≤ 10,000)')
plt.grid(True)
plt.tight_layout()
plt.legend(ncol=2, fontsize='small')
plt.show()






import random
import matplotlib.pyplot as plt

# Pick a random folder from data
selected_folder, methods = random.choice(list(data.items()))

plt.figure(figsize=(12, 7))
color = folder_colors.get(selected_folder, 'black')

for method, points in methods.items():
    if method == 'bfgs':
        continue

    steps, epots, ekins, etots = zip(*points)

    # Plot each energy component
    for energy_type, values, style in zip(
        ['epot', 'ekin', 'etot'],
        [epots, ekins, etots],
        [energy_styles['epot'], energy_styles['ekin'], energy_styles['etot']]
    ):
        label = f"{method} - {energy_type}"
        plt.plot(steps, values, linestyle=style, color=color)

# Add legend explanation for energy types
for energy_type, style in energy_styles.items():
    plt.plot([], [], style, color='black', label=energy_type)

plt.xlabel('Step')
plt.ylabel('Energy')
plt.title(f"Random Folder: {selected_folder} — Energy Components per Method")
plt.grid(True)
plt.legend(ncol=2, fontsize='small')
plt.tight_layout()
plt.show()
print("random done")


# -----
# The target folder you want to plot
target_path = r'.\f32_maxD3_lr0.005_ep1000_bs50_it3_nbf32_co3.0_tr1400_val200_fw1.0_runtr1000_runval100_steps1000000_temp500_r12'
target_folder = os.path.normpath(target_path).replace('\\', '/')
# We only care about the end of the path
target_folder_end = '/'.join(target_folder.split('/')[-1:])  # or use more parts like [-2:] if you want more context

# Find the matching folder key in data
matching_folder = None
for folder_key in data.keys():
    folder_key_norm = folder_key.replace('\\', '/')
    if folder_key_norm.endswith(target_folder_end):
        matching_folder = folder_key
        break

if matching_folder is None:
    print(f"Folder ending with '{target_folder_end}' not found in data keys.")
else:
    plt.figure(figsize=(10, 6))
    color = folder_colors.get(matching_folder, 'black')
    methods = data[matching_folder]
    for method, points in methods.items():
        if method == 'bfgs':  # skip bfgs
            continue

        steps, epots, ekins, etots = zip(*points)

        # Plot each energy component with different line styles
        for energy_type, values, style in zip(
            ['epot', 'ekin', 'etot'],
            [epots, ekins, etots],
            [energy_styles['epot'], energy_styles['ekin'], energy_styles['etot']]
        ):
            label = f"{method} - {energy_type}"
            plt.plot(steps, values, linestyle=style, color=color)

    # Add legend explanation for energy types (black color to distinguish)
    for energy_type, style in energy_styles.items():
        plt.plot([], [], linestyle=style, color='black', label=energy_type)

    plt.xlabel('Step')
    plt.ylabel('Energy (etot, epot, ekin)')
    plt.title(f"Folder: {matching_folder} (all methods except bfgs)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------
import re
import matplotlib.pyplot as plt
import random

# Group folders by temperature
temp_groups = {}
for folder, methods in data.items():
    match = re.search(r'temp(\d+)', folder)
    if match:
        temp = int(match.group(1))
        temp_groups.setdefault(temp, []).append((folder, methods))

# Plot a separate figure for each temperature
for temp, folders in temp_groups.items():
    plt.figure(figsize=(10, 6))
    for folder, methods in folders:
        color = folder_colors.get(folder, 'black')
        for method, points in methods.items():
            if method == 'bfgs':
                continue
            steps, energies = zip(*points)
            plt.plot(steps, energies, line_styles[method], color=color, alpha=0.7)

    # Add legend for methods
    for method, style in line_styles.items():
        if method == 'bfgs':
            continue
        plt.plot([], [], style, color='black', label=method)

    plt.xlabel('Step')
    plt.ylabel('Energy (etot)')
    plt.title(f"Energy Convergence at Temp = {temp}K")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
##




# Write processed file paths to a text file
with open('processed_files.txt', 'w') as f:
    for path in processed_paths:
        f.write(path + '\n')


"""
class TimeOfFailureAnalysis : 
    def __init__(self, name_simu : str,
                 init_kinetic : np.ndarray,
                 init_pot : np.ndarray,
                 init_tot : np.ndarray,
                 positions_array : np.ndarray,
                 temperature : float) : 
        
        self.name_sim = name_simu
        self.failure = False
        self.kB = 8.6173303e-5

        # buffer for energetic quantities
        self.kinetic = init_kinetic
        self.pot = init_pot
        self.tot = init_tot

        # should be 3D (N,3,N_buffer)
        self.positions = positions_array

        self.temperature = temperature

    def check_kinetic_failure(self, kinetic : float,
                              positions : np.ndarray, 
                              nb_sigma : float = 1.0) -> bool : 
        
        # fluctuations
        sigma_energy = np.sqrt(1.5*self.positions.shape[0])*self.kB*self.temperature
        
        # failure => atoms are moving free
        if (np.abs(self.kinetic-kinetic) < nb_sigma*sigma_energy).all() : 
            self.failure = True

        self.kinetic[:-1] = self.kinetic[1:]
        self.kinetic[-1] = kinetic

        self.positions[:,:,:-1] = self.positions[:,:,1:]
        self.positions[:,:,-1] = positions

        return self.failure
    
    def check_total_energy_failure(self, 
                                   total_energy : float,
                                   nb_sigma : float = 3.0) -> bool :
        # fluctuations
        sigma_energy = np.sqrt(1.5*self.positions.shape[0])*self.kB*self.temperature
        
        # failure => atoms are moving free
        if (np.abs(self.tot - total_energy) > nb_sigma*sigma_energy).all() : 
            self.failure = True

        self.tot[:-1] = self.tot[1:]
        self.kinetic[-1] = total_energy

        return self.failure
    
    def analysis_failure(self, kinetic_energy : float,
                         total_energy : float,
                         positions : np.ndarray) -> Tuple[bool, str] :
        bool_kin = self.check_kinetic_failure(kinetic_energy,
                                              positions)
        bool_tot = self.check_total_energy_failure(total_energy)

        if bool_kin and not bool_tot : 
            return True, 'kinetic_failure'
        
        if not bool_kin and bool_tot : 
            return True, 'total_energy_failure'
        
        if bool_kin and bool_tot : 
            return True, 'kinetic_and_total_energy_failure'

        else : 
            return False, 'no_failure'
"""

# import re

# keys = list(data.keys())

# # Example: extract temperature and run index from each key
# for key in keys:
#     temp_match = re.search(r'temp(\d+)', key)
#     run_match = re.search(r'_r(\d+)', key)
#     co_match = re.search(r'co([0-9.]+)', key)

#     if temp_match and run_match and co_match:
#         temp = int(temp_match.group(1))
#         run = int(run_match.group(1))
#         co = float(co_match.group(1))
#         print(f"Temperature: {temp}, Run: {run}, co: {co}")
