import pytest
import sys
import os
import pickle
import numpy as np
from ase import Atoms
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataHolder import DataHolder


## Tests to see if loaded data are correctly held by DataHolder
    # A helper function to create and save valid MD data to a file
def create_valid_md_data_file(tmp_path):
    """Create a temporary file containing valid MD data."""
    atoms = [Atoms("H2"), Atoms("O2"), Atoms("CO2")]
    energies = [1.23, 2.34, 3.45]
    valid_md_data = {
        "descriptor1": {"atoms": atoms, "energies": energies},
        "descriptor2": {"atoms": atoms, "energies": [4.56, 5.67, 6.78]},
    }
    md_file = tmp_path / "md_data.pkl"
    with open(md_file, "wb") as f:
        pickle.dump(valid_md_data, f)
    return md_file, valid_md_data

# Test for DataHolder loading of valid MD data
def test_load_md_data_valid(tmp_path):
    """Test loading valid MD data."""
    # Use the helper function to create the valid MD data file
    md_file, expected_md_data = create_valid_md_data_file(tmp_path)
    # Initialize DataHolder and load MD data
    data = DataHolder()
    list_of_attributes = data.load_md_data(md_file)
    # Verify that MD data is loaded and attributes are extracted
    assert data.md_data == expected_md_data
    assert data.md_data_loaded is True



# A helper function to create and save an invalid MD data file
def create_invalid_md_data_file(tmp_path):
    """Create a temporary file containing invalid MD data."""
    invalid_md_data = {
        "descriptor1": {"atoms": [1, 2, 3], "energies": [1.23, 2.34]},  # Invalid 'atoms' elements
    } # missing "descriptor2"
    md_file = tmp_path / "invalid_md_data.pkl"
    with open(md_file, "wb") as f:
        pickle.dump(invalid_md_data, f)
    return md_file

# Test for invalid MD data (molecular dynamics data)
def test_load_md_data_invalid(tmp_path):
    """Test loading invalid MD data."""
    # Use the helper function to create the invalid MD data file
    md_file = create_invalid_md_data_file(tmp_path)
    data = DataHolder()
    # Verify that loading invalid MD data raises a ValueError
    with pytest.raises(ValueError):
        data.load_md_data(md_file)




## Tests to check how well DataHolder handles valid and invalid theta_data
# Helper function to create and save valid Theta data
def create_valid_theta_data_file(tmp_path):
    """Create a temporary file containing valid Theta data."""
    valid_theta_data = {"coef": np.array([1.2, 2.3]), "intercept": 0.5}
    theta_file = tmp_path / "theta_data.pkl"
    with open(theta_file, "wb") as f:
        pickle.dump(valid_theta_data, f)
    return theta_file, valid_theta_data

# Helper function to create and save invalid Theta data
def create_invalid_theta_data_file(tmp_path):
    """Create a temporary file containing invalid Theta data."""
    invalid_theta_data = {"coef": [1.2, 2.3]}  # Missing 'intercept'
    theta_file = tmp_path / "invalid_theta_data.pkl"
    with open(theta_file, "wb") as f:
        pickle.dump(invalid_theta_data, f)
    return theta_file, invalid_theta_data


# Test for loading Theta data
def test_load_theta_data(tmp_path):
    """Test loading Theta data with both valid and invalid files."""
    # Create valid and invalid Theta data files
    valid_theta_file, valid_theta_data = create_valid_theta_data_file(tmp_path)
    invalid_theta_file, invalid_theta_data = create_invalid_theta_data_file(tmp_path)
    # Initialize DataHolder
    data = DataHolder()
    # Simulate selecting the invalid file and verify that it raises a ValueError
    with pytest.raises(ValueError):
        data.load_theta(invalid_theta_file)

    # Load the valid Theta file and verify the data
    data.load_theta(valid_theta_file)
    assert data.theta_loaded is True
    assert data.theta.keys() == valid_theta_data.keys()  # Check that the keys match
    for key in data.theta:
        if isinstance(data.theta[key], (list, tuple, np.ndarray)):  # If value is an array
            assert (data.theta[key] == valid_theta_data[key]).all()  # Use `.all()` for array comparison
        else:
            assert data.theta[key] == valid_theta_data[key]  # Direct comparison for non-array values

    