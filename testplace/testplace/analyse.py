import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Define line styles for different methods
line_styles = {
    'default': '-',
    'mixed': '--',
    'fisher': ':',
    'bfgs': '-.'  # or any style you like
}

# Define method regex patterns
patterns = {
    'bfgs': re.compile(r'^BFGS:\s*(\d+)\s[\d:]+\s+([-0-9.eE]+)'),
    'default': re.compile(r'\[default\]\s*step\s*(\d+)\s.*etot\s*([-0-9.eE]+)'),
    'mixed': re.compile(r'\[mixed\]\s*step\s*(\d+)\s.*etot\s*([-0-9.eE]+)'),
    'fisher': re.compile(r'\[fisher\]\s*step\s*(\d+)\s.*etot\s*([-0-9.eE]+)')
}

# Data structure: {folder: {method: [(step, energy)]}}
data = defaultdict(lambda: defaultdict(list))
processed_paths = []

# Walk and collect data
for root, _, files in os.walk('.'):
    for file in files:
        if file.startswith('out_run'):
            folder = os.path.relpath(root, '.')  # folder as hyperparam set
            filepath = os.path.join(root, file)
            processed_paths.append(os.path.abspath(filepath))  # Save full path


            with open(filepath) as f:
                for line in f:
                    for method, pattern in patterns.items():
                        match = pattern.search(line)
                        if match:
                            step = int(match.group(1))
                            energy = float(match.group(2))
                            data[folder][method].append((step, energy))

# Sort each method's data by step
for folder in data:
    for method in data[folder]:
        data[folder][method].sort()

# Assign a color to each folder
colors = plt.cm.tab10.colors  # 10 distinct colors
folder_colors = {folder: colors[i % len(colors)] for i, folder in enumerate(data)}

# Plotting
plt.figure(figsize=(10, 6))

for folder, methods in data.items():
    color = folder_colors[folder]
    for method, points in methods.items():
        if method == 'bfgs':  # skip bfgs
            continue
        steps, energies = zip(*points)
        label = f"{folder} - {method}"
        plt.plot(steps, energies, line_styles[method], color=color)

# Create dummy lines for legend to show method styles
for method, style in line_styles.items():
    if method == 'bfgs':  # skip bfgs
        continue
    plt.plot([], [], style, color='black', label=method)


plt.xlabel('Step')
plt.ylabel('Energy (etot)')
plt.title('Energy vs Step for Different Methods')
plt.legend() 
plt.grid(True)
plt.tight_layout()
plt.show()


# ---------- Zoomed-In Plot ----------
plt.figure(figsize=(10, 6))
for folder, methods in data.items():
    color = folder_colors[folder]
    for method, points in methods.items():
        if method == 'bfgs':  # skip bfgs
            continue
        # Only include steps <= 100
        zoomed_points = [(s, e) for s, e in points if s <= 10000]
        if not zoomed_points:
            continue
        steps, energies = zip(*zoomed_points)
        label = f"{folder} - {method}"
        plt.plot(steps, energies, line_styles[method], color=color)

# Create dummy lines for legend to show method styles
for method, style in line_styles.items():
    if method == 'bfgs':  # skip bfgs
        continue
    plt.plot([], [], style, color='black', label=method)



plt.xlabel('Step')
plt.ylabel('Energy (etot)')
plt.title('Zoomed: Energy vs Step (First 100 Steps of 1000)')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# # ------------
# MAX_STEPS_ZOOM = 1000   # max points for zoomed plot
# MAX_ENERGY = 0.1      # max allowed energy value to include curve
# MIN_ENERGY = -1*MAX_ENERGY

# plt.figure(figsize=(10, 6))

# for folder, methods in data.items():
#     color = folder_colors[folder]
#     for method, points in methods.items():
#         if method == 'bfgs':  # skip bfgs
#             continue
#         zoomed_points = [(s, e) for s, e in points if s <= 100]
#         if not zoomed_points:
#             continue
        
#         # Filter out curves with any energy > MAX_ENERGY
#         if any(e > MAX_ENERGY for _, e in zoomed_points):
#             continue
#         if any(e < MIN_ENERGY for _, e in zoomed_points):
#             continue
        
#         # if len(zoomed_points) > MAX_STEPS_ZOOM:
#         #     continue  # skip if too many points
        
#         steps, energies = zip(*zoomed_points)
#         label = f"{folder} - {method}"
#         plt.plot(steps, energies, line_styles[method], color=color)

# # Create dummy lines for legend to show method styles
# for method, style in line_styles.items():
#     if method == 'bfgs':  # skip bfgs
#         continue
#     plt.plot([], [], style, color='black', label=method)

# plt.xlabel('Step')
# plt.ylabel('Energy (etot)')
# plt.title('Zoomed: Energy vs Step (First 100 Steps, Filtered by Energy)')
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
# plt.show()






# -----
import random

# Pick a random folder from data.items()
selected_folder, methods = random.choice(list(data.items()))

plt.figure(figsize=(10, 6))
color = folder_colors.get(selected_folder, 'black')
for method, points in methods.items():
    if method == 'bfgs':  # skip bfgs
        continue
    steps, energies = zip(*points)
    plt.plot(steps, energies, line_styles[method], color=color)

for method, style in line_styles.items():
    if method == 'bfgs':  # skip bfgs
        continue
    plt.plot([], [], style, color='black', label=method)


plt.xlabel('Step')
plt.ylabel('Energy (etot)')
plt.title(f"Random folder: {selected_folder} (all methods available in it)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----
# The target folder you want to plot
target_folder = r'MD_tdlog\testplace\testplace\runs\f32_maxD3_lr0.005_ep1000_bs50_it3_nbf32_co3.0_tr1400_val200_fw1.0_runtr1000_runval100_steps1000000_temp500_r12'

# Since data keys are relative paths, make sure the path format matches your data keys.
# You might want to normalize both to forward slashes or backslashes to be sure.

# Normalize to forward slashes for matching (optional, depends on how your keys look)
target_folder_norm = target_folder.replace('\\', '/')

# Find the matching folder key in data (some tolerance if keys differ by ./ or ./some/...)
matching_folder = None
for folder_key in data.keys():
    folder_key_norm = folder_key.replace('\\', '/')
    if folder_key_norm == target_folder_norm:
        matching_folder = folder_key
        break

if matching_folder is None:
    print(f"Folder '{target_folder}' not found in data keys.")
else:
    plt.figure(figsize=(10, 6))
    color = folder_colors.get(matching_folder, 'black')
    methods = data[matching_folder]
    for method, points in methods.items():
        if method == 'bfgs':  # skip bfgs
            continue
        steps, energies = zip(*points)
        plt.plot(steps, energies, line_styles[method], color=color, label=method)

    # # Legend for plotted methods
    # for method, style in line_styles.items():
    #     if method == 'bfgs':
    #         continue
    #     plt.plot([], [], style, color='black', label=method)

    plt.xlabel('Step')
    plt.ylabel('Energy (etot)')
    plt.title(f"Folder: {matching_folder} (all methods except bfgs)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------
plt.figure(figsize=(10, 6))
for folder, methods in data.items():
    color = folder_colors[folder]
    # Only process the 'default' method
    if 'default' not in methods:
        continue
    points = methods['default']
    # Only include steps <= 10000 (as you wrote)
    zoomed_points = [(s, e) for s, e in points if s <= 10000]
    if not zoomed_points:
        continue
    steps, energies = zip(*zoomed_points)
    label = f"{folder} - default"
    plt.plot(steps, energies, line_styles['default'], color=color)

# Legend for just 'default'
plt.plot([], [], line_styles['default'], color='black', label='default')

plt.xlabel('Step')
plt.ylabel('Energy (etot)')
plt.title('Zoomed: Energy vs Step (First 10000 Steps)')
plt.grid(True)
plt.tight_layout()
plt.legend()
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
