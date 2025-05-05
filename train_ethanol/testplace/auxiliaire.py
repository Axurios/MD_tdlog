import os
import shutil
import numpy as np
from config_managers import H5Manager, XMLManager

# Definit hyperparameters de base
hyperparams = {
    "features" : 32,
    "max_degree" : 2,
    "num_iterations" : 3,
    "num_basis_functions" : 32,
    "cutoff" : 3.0,
    "num_train" : 200,
    "num_valid" : 25,
    "num_epochs" : 20,  # short for testing; increase as needed
    "learning_rate" : 0.01,
    "forces_weight" : 1.0,
    "batch_size" : 20
}


# Abbreviate keys for folder naming
key_abbrev = {
    "features": "f",
    "max_degree": "maxD",
    "learning_rate": "lr",
    "num_epochs": "ep",
    "batch_size": "bs",
    "num_iterations": "it",
    "num_basis_functions": "nbf",
    "cutoff": "co",
    "num_train": "tr",
    "num_valid": "val",
    "forces_weight": "fw",
}

from itertools import product

def run_all_combinations(hyperparam_options, use_xml=True):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    inter_file = os.path.join(base_dir, "inter.py")
    run_file = os.path.join(base_dir, "run.py")

    # Get all hyperparameter keys and their value combinations
    keys = list(hyperparam_options.keys())
    combinations = list(product(*[hyperparam_options[k] for k in keys]))

    for combo in combinations:
        # print(combo)
        # Build the hyperparams dict from this combo
        hyperparams = dict(zip(keys, combo))

        # Create a unique folder name based on the hyperparameters
        # folder_name = "_".join(f"{k}{v}" for k, v in hyperparams.items())
        folder_name = "_".join(f"{key_abbrev[k]}{v}" for k, v in hyperparams.items())

        # print(folder_name)
        target_folder = os.path.join(base_dir, "runs", folder_name)
        os.makedirs(target_folder, exist_ok=True)

        # Save hyperparameters
        if use_xml:
            xml_path = os.path.join(target_folder, "hyperparams.xml")
            xml_writer = XMLManager(xml_path, mode="writing")
            xml_writer.generate_xml(hyperparams)
        else:
            h5_path = os.path.join(target_folder, "hyperparams.h5")
            h5_writer = H5Manager(h5_path, mode="writing")
            h5_writer.add_or_update_data("Hyperparameters", {
                key: np.string_(val) if isinstance(val, str) else val
                for key, val in hyperparams.items()
            })
            h5_writer.close()

        # Copy inter.py and run.py into the folder
        shutil.copy2(inter_file, os.path.join(target_folder, "inter_copy.py"))
        shutil.copy2(run_file, os.path.join(target_folder, "run_copy.py"))

        print(f"Created run in {target_folder}")

# Example usage
if __name__ == "__main__":
    hyperparam_options = {
        "features": [16, 32],
        "max_degree": [2, 3],
        "learning_rate": [0.001, 0.01],
        "num_epochs": [10],
        "batch_size": [20],

        "num_iterations" : [3],
        "num_basis_functions" : [32],
        "cutoff" : [3.0],
        "num_train" : [200],
        "num_valid" : [25],
        "forces_weight" : [1.0],
    }
    run_all_combinations(hyperparam_options, use_xml=True)
    # will create all possibilities from the cross product space of the hyperparams given









    
# # Choose XML or HDF5 format
# use_xml = True  # Set to False for HDF5
# base_dir = os.path.dirname(os.path.abspath(__file__))

# if use_xml:
#     path = os.path.join(base_dir, "hyperparams.xml")
#     # path = "hyperparams.xml"
#     xml_writer = XMLManager(path, mode="writing")
#     xml_writer.generate_xml(hyperparams)
# else:
#     # path = "hyperparams.h5"
#     path = os.path.join(base_dir, "hyperparams.xml")
#     h5_writer = H5Manager(path, mode="writing")
#     h5_writer.add_or_update_data("Hyperparameters", {
#         key: np.string_(val) if isinstance(val, str) else val
#         for key, val in hyperparams.items()
#     })
#     h5_writer.close()





# # Get current script directory
# base_dir = os.path.dirname(os.path.abspath(__file__))

# # Define source file and target folder
# inter_file = os.path.join(base_dir, "inter.py")
# run_file = os.path.join(base_dir, "run.py")
# target_folder = os.path.join(base_dir, "intercept_backup")

# # Create the folder if it doesn't exist
# os.makedirs(target_folder, exist_ok=True)

# inter_copy = os.path.join(target_folder, "inter_copy.py")
# shutil.copy2(inter_file, inter_copy)

# run_copy = os.path.join(target_folder, "run_copy.py")
# shutil.copy2(run_copy, run_copy)
