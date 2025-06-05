import os
import shutil
import numpy as np
from config_managers import H5Manager, XMLManager, MetaXML
import subprocess
import argparse

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
#SBATCH -A aih@a100               # account to charge
#SBATCH -C a100 
#SBATCH --job-name={job_name}         # name of job
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p4, gpu_p5)
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task=8            # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
##SBATCH --hint=nomultithread 



module purge


which python > path_test
module load python/3.11.5
module load cuda/12.4.1
module load cudnn/9.8.0.87-cuda


python -u inter_copy.py  > out_train
python -u run_fisher_copy.py  > out_run_fisher
#python -u run_mixed_copy.py  > out_run_mixed
cp out_run_fisher out_run_mixed 
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
    "num_calib" : 200, 
    "batch_size" : 20,   #<---until here concern train
    "run_num_train":900, # this has no impact on run 
    "run_num_valid":100, # this has no impact on run 
    "timestep_fs": 0.5,
    "num_steps" : 100000,
    "temperature" : 1000,
    "repeat" : 30,
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
    "num_calib": "ncb", 
    "forces_weight": "fw",
    "run_num_train": "runtr",
    "run_num_valid": "runval",
    "timestep_fs": "tfs",
    "num_steps" : "steps",
    "temperature": "temp",
    "repeat":"r"
}



from itertools import product

def write_all_combinations(hyperparam_options, use_xml=True):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    inter_file = os.path.join(base_dir, "inter_lowmem.py")
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

    list_folder = []

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
        list_folder.append(target_folder)

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
        write_jsub(job_name="myjob", time_limit="09:30:00", filename=target_folder+'/jsub')

    meta_xml = MetaXML(os.path.join(base_dir,'meta.xml'),
                       list_folder=list_folder,
                       mode='writing')
    meta_xml.write_xml()
    return 

def manage_launch_calculations(nb_calc) : 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    meta_xml = MetaXML(os.path.join(base_dir,'meta.xml'),
                       list_folder=[],
                       mode='reading')
    dic_path = meta_xml.parse_xml()
    
    # Select all jobs if nb_calc < 0
    if nb_calc < 0:
        nb_calc = len([v for v in dic_path.values() if not v.get("launch", False)])

    path2launch, dic_path_update = meta_xml.select_calculation_to_run(dic_path, nb_calc)

    #path2launch, dic_path_update = meta_xml.select_calculation_to_run(dic_path, nb_calc)
    #for path in path2launch : 
    #    try : 
    #        submit_job_in_folder(path)
    #    except : 
    #        print(f'Problem to launch {path}')
    #        dic_path_update[path]['launch'] = False
            
    
    for path in path2launch:
        try:
            submit_job_in_folder(path)
        except Exception as e:
            print(f'Problem to launch {path}: {e}')
            dic_path_update[path]['launch'] = False        

    meta_xml.update_xml(dic_path_update)
    meta_xml.write_xml()
    return 

parser = argparse.ArgumentParser('ToF')
parser.add_argument('-m','--mode',default="build")
args = parser.parse_args()
mode = args.mode




# Example usage
if __name__ == "__main__":
    hyperparam_options = {
        "features": [32,64],
        "max_degree": [2],
        "num_iterations" : [3],
        "num_basis_functions" : [32],
        "cutoff" : [3.0],
        "num_train" : [500, 1000],
        "num_valid" : [200],
        "num_epochs": [2000],
        "learning_rate": [0.01],
        "forces_weight" : [0.1],
        "num_calib" : [200, 2000], 
        "batch_size": [50],
        "timestep_fs": [0.5],
        "run_num_train":[1000],
        "run_num_valid":[100],
        "timestep_fs": [0.5],
        "num_steps":[1000000],
        "temperature" : [500, 1000],
        "repeat": np.arange(20).tolist()
    }

    nb_calc = -1

    if mode == 'build' :
        write_all_combinations(hyperparam_options, use_xml=True)

    elif mode == 'launch' : 
        manage_launch_calculations(nb_calc)
