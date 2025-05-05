import os
import shutil
import numpy as np
from config_managers import H5Manager, XMLManager

# Define hyperparameters
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

# Choose XML or HDF5 format
use_xml = True  # Set to False for HDF5
base_dir = os.path.dirname(os.path.abspath(__file__))

if use_xml:
    path = os.path.join(base_dir, "hyperparams.xml")
    # path = "hyperparams.xml"
    xml_writer = XMLManager(path, mode="writing")
    xml_writer.generate_xml(hyperparams)
else:
    # path = "hyperparams.h5"
    path = os.path.join(base_dir, "hyperparams.xml")
    h5_writer = H5Manager(path, mode="writing")
    h5_writer.add_or_update_data("Hyperparameters", {
        key: np.string_(val) if isinstance(val, str) else val
        for key, val in hyperparams.items()
    })
    h5_writer.close()


print("ok")





# Get current script directory
base_dir = os.path.dirname(os.path.abspath(__file__))


# Define source file and target folder
source_file = os.path.join(base_dir, "inter.py")
target_folder = os.path.join(base_dir, "intercept_backup")

# Create the folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# Define destination file path
destination_file = os.path.join(target_folder, "inter_copy.py")


print("ok")
# Copy the file
shutil.copy2(source_file, destination_file)

print(f"Copied {source_file} to {destination_file}")