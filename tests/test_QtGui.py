import os
import sys
import pickle
import pytest
from ase import Atoms
#from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QApplication
#from QtGui import GUI  # Replace with your actual module name
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataHolder import DataHolder


# Create a QApplication instance for testing GUI
@pytest.fixture(scope="module")
def app():
    return QApplication([])


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


