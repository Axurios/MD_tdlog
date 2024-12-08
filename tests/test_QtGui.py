import os
import pickle
import pytest
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QApplication
from QtGui import GUI  # Replace with your actual module name


# Create a QApplication instance for testing GUI
@pytest.fixture(scope="module")
def app():
    return QApplication([])


# Fixture to create sample MD data
@pytest.fixture
def sample_md_data():
    return {
        "dataset1": {
            "atoms": [{"id": 1, "coords": [0, 0, 0]}],
            "energies": [-0.5, 0.0, 0.5],
        }
    }


# Fixture to create sample Theta data
@pytest.fixture
def sample_theta_data():
    return {
        "coef": [[0.1, 0.2], [0.3, 0.4]],
        "intercept": 0.5,
    }


# Test for loading MD data
def test_load_md_data(app, sample_md_data, tmp_path):
    # Create a temporary file for MD data
    md_file = tmp_path / "md_data.pkl"
    with open(md_file, "wb") as f:
        pickle.dump(sample_md_data, f)

    gui = GUI()
    gui.data.load_md_data = MagicMock()  # Mock the actual data loading
    gui.display_md_data = MagicMock()  # Mock display function

    # Simulate selecting the file
    with patch(
        "PyQt5.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(md_file), "")
    ):
        gui.select_file("md")

    # Verify the data was loaded and display was updated
    gui.data.load_md_data.assert_called_once_with(str(md_file))
    gui.display_md_data.assert_called_once()


# Test for loading Theta data
def test_load_theta_data(app, sample_theta_data, tmp_path):
    # Create a temporary file for Theta data
    theta_file = tmp_path / "theta_data.pkl"
    with open(theta_file, "wb") as f:
        pickle.dump(sample_theta_data, f)

    gui = GUI()
    gui.data.load_theta = MagicMock()  # Mock the actual data loading
    gui.display_theta_data = MagicMock()  # Mock display function

    # Simulate selecting the file
    with patch(
        "PyQt5.QtWidgets.QFileDialog.getOpenFileName",
        return_value=(str(theta_file), ""),
    ):
        gui.select_file("theta")

    # Verify the data was loaded and display was updated
    gui.data.load_theta.assert_called_once_with(str(theta_file))
    gui.display_theta_data.assert_called_once()


# Test for handling invalid files
def test_load_invalid_file(app, tmp_path):
    # Create an invalid file
    invalid_file = tmp_path / "invalid_data.pkl"
    invalid_file.write_text("This is not a valid pickle file.")

    gui = GUI()

    # Simulate selecting the invalid file
    with patch(
        "PyQt5.QtWidgets.QFileDialog.getOpenFileName",
        return_value=(str(invalid_file), ""),
    ):
        with pytest.raises(ValueError, match="Error loading MD data"):
            gui.select_file("md")
