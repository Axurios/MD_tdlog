import os
import shutil
import numpy as np
from config_managers import H5Manager, XMLManager
import subprocess

def submit_job_in_folder(target_folder):
    # Save current directory
    original_dir = os.getcwd()

    try:
        # Go to the target folder
        os.chdir(target_folder)
        print(f"Changed to {target_folder}")

        # Submit the job
        result = subprocess.run(["sbatch", "jsub"], capture_output=True, text=True)

        # Show output
        print("STDOUT:", result.stdout.strip())
        print("STDERR:", result.stderr.strip())

    finally:
        # Go back to original directory
        os.chdir(original_dir)
        print(f"Returned to {original_dir}")



def write_jsub(job_name="gpu-test", time_limit="05:00:00", filename="jsub"):
    content = f"""#!/bin/bash
#SBATCH -A aih@v100               # account to charge
#SBATCH -C v100-32g 
#SBATCH --job-name={job_name}         # name of job
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p4, gpu_p5)
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
##SBATCH --hint=nomultithread 



module purge


which python > path_test
module load python/3.11.5
module load cuda/12.4.1
module load cudnn/9.8.0.87-cuda


python -u inter_copy.py  > out_train
python -u run_fisher_copy.py  > out_run_fisher
python -u run_mixed_copy.py  > out_run_mixed
python -u run_default_copy.py  > out_run_default
"""

    with open(filename, "w") as f:
        f.write(content)
    #print(f"Wrote SLURM job script to '{filename}'")


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
    "batch_size" : 20,
    "run_num_train":900,
    "run_num_valid":100,
    "timestep_fs": 1.0,
    "num_steps" : 1000000,
    "temperature" : 1000
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
    "run_num_train": "runtr",
    "run_num_valid": "runval",
    "timestep_fs": "tfs",
    "num_steps" : "steps",
    "temperature": "temp"
}



from itertools import product

def run_all_combinations(hyperparam_options, use_xml=True):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    inter_file = os.path.join(base_dir, "inter.py")
    run_fisher_file = os.path.join(base_dir, "run_fisher.py")
    run_mixed_file = os.path.join(base_dir, "run_mixed.py")
    run_default_file = os.path.join(base_dir, "run_default.py")
    config_manager_file = os.path.join(base_dir, "config_managers.py")
    # Absolute path to the source file
    source_db_file = os.path.join(base_dir, "md17_ethanol.npz")
    # Automatically extract just the file name
    db_filename = os.path.basename(source_db_file)  # => "md17_ethanol.npz"

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
        shutil.copy2(run_fisher_file, os.path.join(target_folder, "run_fisher_copy.py"))
        shutil.copy2(run_mixed_file, os.path.join(target_folder, "run_mixed_copy.py"))
        shutil.copy2(run_default_file, os.path.join(target_folder, "run_default_copy.py"))
        shutil.copy2(config_manager_file, os.path.join(target_folder, "config_managers.py"))
        target_db_file = os.path.join(target_folder, db_filename)
        # Create symlink or copy
        if not os.path.exists(target_db_file):
            os.symlink(source_db_file, target_db_file)
            # or use shutil.copy2(source_db_file, target_db_file) to copy

        print(f"Created run in {target_folder}")
        write_jsub(job_name="my-test-job", time_limit="06:30:00", filename=target_folder+'/jsub')
        submit_job_in_folder(target_folder)        

# Example usage
if __name__ == "__main__":
    hyperparam_options = {
        "features": [16, 32],
        "max_degree": [3],
        "learning_rate": [0.001, 0.01],
        "num_epochs": [800],
        "batch_size": [50],

        "num_iterations" : [3],
        "num_basis_functions" : [32],
        "cutoff" : [3.0],
        "num_train" : [1400],
        "num_valid" : [200],
        "forces_weight" : [1.0],
        "run_num_train":[1000],
        "run_num_valid":[100],
        "num_steps":[1000000],
        "temperature" : [1000]
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
