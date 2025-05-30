import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import re

kB = 8.617333262145e-5  # eV/K (Boltzmann constant)

def parse_hyperparams(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return {child.tag: child.text for child in root}

def match_criteria(hyperparams, criteria):
    for key, value in criteria.items():
        if str(hyperparams.get(key)) != str(value):
            return False
    return True

def parse_out_file(out_file):
    steps, epot, ekin, etot = [], [], [], []
    with open(out_file, 'r') as f:
        for line in f:
            match = re.match(r"\[.*\] step\s+(\d+)\s+epot\s+([-.\d]+)\s+ekin\s+([-.\d]+)\s+etot\s+([-.\d]+)", line)
            if match:
                step = int(match.group(1))
                steps.append(step)
                epot.append(float(match.group(2)))
                ekin.append(float(match.group(3)))
                etot.append(float(match.group(4)))
    return steps, epot, ekin, etot

def gather_data(base_dir, criteria):
    data = {'default': [], 'mixed': [], 'fisher': []}
    for run_dir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, run_dir)
        xml_path = os.path.join(full_path, 'hyperparams.xml')
        if not os.path.isfile(xml_path):
            continue
        try:
            hyperparams = parse_hyperparams(xml_path)
        except Exception:
            continue
        if not match_criteria(hyperparams, criteria):
            continue
        for run_type in ['default', 'mixed', 'fisher']:
            out_file = os.path.join(full_path, f'out_run_{run_type}')
            if os.path.isfile(out_file):
                steps, epot, ekin, etot = parse_out_file(out_file)
                data[run_type].append((steps, epot, ekin, etot))
    return data

def compute_temperature(ekin_list, n_atoms):
    ekin_array = np.array(ekin_list)
    mean_ekin = np.mean(ekin_array)
    temp = (2.0 / 3.0) * mean_ekin / (n_atoms * kB)
    return temp

def plot_energies(data, min_steps=1, max_steps=None, n_atoms=9):
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    run_types = ['default', 'mixed', 'fisher']
    labels = ['Potential Energy', 'Kinetic Energy', 'Total Energy']

    if max_steps is None:
        max_steps = 0
        for run_list in data.values():
            for steps, *_ in run_list:
                if steps:
                    max_steps = max(max_steps, max(steps))

    # Plot time series
    for i, run_type in enumerate(run_types):
        for j, energy_type in enumerate(['epot', 'ekin', 'etot']):
            ax = axes[i, j]
            for steps, epot, ekin, etot in data[run_type]:
                energy_data = {'epot': epot, 'ekin': ekin, 'etot': etot}[energy_type]
                filtered_steps = []
                filtered_energy = []
                for k, s in enumerate(steps):
                    if min_steps <= s <= max_steps:
                        filtered_steps.append(s)
                        filtered_energy.append(energy_data[k])
                ax.plot(filtered_steps, filtered_energy, alpha=0.7)
            if i == 0:
                ax.set_title(labels[j])
            if j == 0:
                ax.set_ylabel(run_type.capitalize())
            ax.set_xlabel("Step")
            ax.set_xlim([min_steps, max_steps])

    # Plot trajectory-wise statistics
    for j, run_type in enumerate(run_types):
        epot_var, ekin_temp, etot_var = [], [], []
        for steps, epot, ekin, etot in data[run_type]:
            steps = np.array(steps)
            mask = (steps >= min_steps) & (steps <= max_steps)
            epot_filtered = np.array(epot)[mask]
            ekin_filtered = np.array(ekin)[mask]
            etot_filtered = np.array(etot)[mask]
            epot_var.append(np.var(epot_filtered))
            etot_var.append(np.var(etot_filtered))
            ekin_temp.append(compute_temperature(ekin_filtered, n_atoms))

        # Plot per-trajectory values
        axes[3, 0].plot(range(len(epot_var)), epot_var, 'o-', label=run_type)
        axes[3, 1].plot(range(len(ekin_temp)), ekin_temp, 'o-', label=run_type)
        axes[3, 2].plot(range(len(etot_var)), etot_var, 'o-', label=run_type)

    axes[3, 0].set_ylabel("Var(Potential)")
    axes[3, 0].set_yscale("log")    
    axes[3, 1].set_ylabel("Temperature (K)")
    axes[3, 2].set_ylabel("Var(Total)")
    axes[3, 2].set_yscale("log")
    for ax in axes[3]:
        ax.set_xlabel("Trajectory Index")
        ax.legend()

    plt.tight_layout()
    plt.show()

# Example usage:
base_dir = 'runs'
criteria = {
    'cutoff': '3.0',
    'temperature': '500'
}
min_steps = 1
max_steps = 100000
n_atoms = 9  # ethanol = C2H5OH
data = gather_data(base_dir, criteria)
plot_energies(data, min_steps=min_steps, max_steps=max_steps, n_atoms=n_atoms)
