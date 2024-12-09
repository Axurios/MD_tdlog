#import os
import pickle
import pytest
from ase import Atoms
#from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QApplication
#from QtGui import GUI  # Replace with your actual module name
import numpy as np
from dataHolder import DataHolder


# Create a QApplication instance for testing GUI
@pytest.fixture(scope="module")
def app():
    return QApplication([])


# Fixture to create sample MD data
@pytest.fixture
def sample_md_data():
    """
    Fixture providing sample MD data for testing.
    Each 'atoms' entry is an ASE Atoms object with a 'descriptor' array.
    """
    # Create an Atoms object
    atoms1 = Atoms(positions=[[0, 0, 0], [1, 1, 1]], symbols=["H", "H"])
    atoms2 = Atoms(positions=[[0, 0, 0], [2, 2, 2]], symbols=["O", "O"])

    # Add a 'descriptor' array to each Atoms object
    atoms1.new_array('descriptor', np.array([[1.0, 0.5], [0.2, 0.8]]))
    atoms2.new_array('descriptor', np.array([[0.3, 0.7], [0.6, 0.4]]))
    # Sample MD data structure
    return {
        "dataset1": {
            "atoms": [atoms1, atoms2],  #add str(1) in list to test if it raises Error for non atoms value
            "energies": [-0.5, 0.5],
        }
    }

# Fixture to create sample Theta data
@pytest.fixture
def sample_theta_data():
    return {
        "coeff": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "intercept": 0.5,
    }

@pytest.fixture
def sample_theta_data2():
    return {
        "coef": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "intercept": 0.5,
    }


# Test for loading MD data
def test_load_md_data(app, sample_md_data, tmp_path):
    # Create a temporary file for MD data
    md_file = tmp_path / "md_data.pkl"
    with open(md_file, "wb") as f:
        pickle.dump(sample_md_data, f)

    data = DataHolder()

    with pytest.raises(ValueError):
        data.load_md_data(md_file, 'descriptor2')

    data.load_md_data(md_file, 'descriptor')


# Test for loading Theta data
def test_load_theta_data(app, sample_theta_data,sample_theta_data2, tmp_path):
    # Create a temporary file for Theta data
    theta_file = tmp_path / "theta_data.pkl"
    with open(theta_file, "wb") as f:
        pickle.dump(sample_theta_data, f)

    theta_file2 = tmp_path / "theta_data2.pkl"
    with open(theta_file2, "wb") as f:
        pickle.dump(sample_theta_data2, f)

    data = DataHolder()

    # Simulate selecting the file
    with pytest.raises(ValueError):
        data.load_theta(theta_file)

    data.load_theta(theta_file2)


