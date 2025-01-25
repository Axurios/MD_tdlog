import os
import pickle
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QListWidget, QHBoxLayout, QMessageBox, QFileDialog
import flax.serialization

import functools
import os
import urllib.request
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import e3x
import pickle
# Disable future warnings.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class NNManager:
    def __init__(self):
        """ Initialize the Neural Network Manager."""
        self.nn_models = {}  # Dictionary to store models with names as keys
        self.current_nn = None  # Currently selected neural network
        self.model_dir = "Model"  # Default folder for saving and loading models

        # Create the model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, model_name, model_data):
        """Save the model and its parameters to the 'Model' directory."""
        try:
            # Save model parameters as a binary file
            serialized_params = flax.serialization.to_bytes(model_data["params"])
            params_file_path = os.path.join(self.model_dir, f"{model_name}_params.bin")
            with open(params_file_path, "wb") as f:
                f.write(serialized_params)
            print(f"Serialized model parameters saved to: {params_file_path}")

            # Save the complete model checkpoint as a .pkl file
            checkpoint_file_path = os.path.join(self.model_dir, f"{model_name}_checkpoint.pkl")
            with open(checkpoint_file_path, "wb") as f:
                pickle.dump(model_data, f)
            print(f"Model checkpoint saved to: {checkpoint_file_path}")

            # Update the loaded models
            self.nn_models[model_name] = model_data
            self.current_nn = model_data
            QMessageBox.information(None, "Model Saved", f"Model '{model_name}' saved successfully.")
        except Exception as e:
            QMessageBox.critical(None, "Save Error", f"Failed to save model: {e}")

    def load_nn_model(self, file_path):
        """Load a neural network model checkpoint from a .pkl file."""
        try:
            with open(file_path, "rb") as f:
                model_data = pickle.load(f)

            model_name = os.path.basename(file_path).replace("_checkpoint.pkl", "")
            self.nn_models[model_name] = model_data
            self.current_nn = model_data  # Set the loaded model as the current model
            QMessageBox.information(None, "Model Loaded", f"Successfully loaded model: {model_name}")
        except Exception as e:
            QMessageBox.critical(None, "Load Error", f"Failed to load model: {e}")

    def select_nn_file(self):
        """Allow the user to select a neural network model file."""
        try:
            file_name, _ = QFileDialog.getOpenFileName(None, "Select Neural Network Model", self.model_dir, "Pickle Files (*.pkl)")
            if file_name:
                self.load_nn_model(file_name)
            else:
                QMessageBox.warning(None, "No File Selected", "You didn't select a file.")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error loading neural network model: {e}")

    def get_current_model(self):
        """Get the current neural network model."""
        if self.current_nn is not None:
            return self.current_nn
        else:
            QMessageBox.warning(None, "No Model Loaded", "No model is currently loaded.")
            return None

    def list_models(self):
        """List all loaded models."""
        return list(self.nn_models.keys())



class NNManagerDialog(QDialog):
    def __init__(self, nn_manager):
        """Initialize the Neural Network Manager Dialog."""
        super().__init__()
        self.nn_manager = nn_manager
        self.setWindowTitle("Neural Network Manager")
        self.setMinimumSize(400, 300)

        # Layouts
        layout = QVBoxLayout()

        # Label and List of Models
        self.label = QLabel("Loaded Models:")
        self.model_list = QListWidget()
        layout.addWidget(self.label)
        layout.addWidget(self.model_list)

        # Buttons
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Model")
        self.save_button = QPushButton("Save Current Model")
        self.switch_button = QPushButton("Switch Model")
        self.close_button = QPushButton("Close")
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.switch_button)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

        # Set layout
        self.setLayout(layout)

        # Populate the model list
        self.update_model_list()

        # Connect buttons
        self.load_button.clicked.connect(self.load_model)
        self.save_button.clicked.connect(self.save_model)
        self.switch_button.clicked.connect(self.switch_model)
        self.close_button.clicked.connect(self.close)

    def update_model_list(self):
        """Update the list of loaded models."""
        self.model_list.clear()
        model_list = self.nn_manager.list_models()
        print(f"Updating model list: {model_list}")  # Debug: Print model names
        self.model_list.addItems(model_list)


    def load_model(self):
        """Load a new model."""
        try:
            self.nn_manager.select_nn_file()
            self.update_model_list()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def save_model(self):
        """Save the current model."""
        current_model = self.nn_manager.get_current_model()
        if current_model is None:
            return  # If no model is loaded, return

        # Prompt the user for a model name
        model_name, ok = QFileDialog.getSaveFileName(self, "Save Model As", self.nn_manager.model_dir, "Pickle Files (*.pkl)")
        if ok and model_name:
            try:
                # Prepare model data and save
                self.nn_manager.save_model(model_name, current_model)
                self.update_model_list()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {e}")

    def switch_model(self):
        """Switch to the selected model."""
        selected_model = self.model_list.currentItem()
        if selected_model is None:
            QMessageBox.warning(self, "No Model Selected", "Please select a model to switch.")
            return

        model_name = selected_model.text()
        self.nn_manager.switch_model(model_name)


class MessagePassingModel(nn.Module):
  features: int = 32
  max_degree: int = 2
  num_iterations: int = 3
  num_basis_functions: int = 8
  cutoff: float = 5.0
  max_atomic_number: int = 118  # This is overkill for most applications.


  def energy(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
    # 1. Calculate displacement vectors.
    positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
    positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
    displacements = positions_src - positions_dst  # Shape (num_pairs, 3).

    # 2. Expand displacement vectors in basis functions.
    basis = e3x.nn.basis(  # Shape (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
      displacements,
      num=self.num_basis_functions,
      max_degree=self.max_degree,
      radial_fn=e3x.nn.reciprocal_bernstein,
      cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
    )

    # 3. Embed atomic numbers in feature space, x has shape (num_atoms, 1, 1, features).
    x = e3x.nn.Embed(num_embeddings=self.max_atomic_number+1, features=self.features)(atomic_numbers)

    # 4. Perform iterations (message-passing + atom-wise refinement).
    for i in range(self.num_iterations):
      # Message-pass.
      if i == self.num_iterations-1:  # Final iteration.
        # Since we will only use scalar features after the final message-pass, we do not want to produce non-scalar
        # features for efficiency reasons.
        y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(x, basis, dst_idx=dst_idx, src_idx=src_idx)
        # After the final message pass, we can safely throw away all non-scalar features.
        x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
      else:
        # In intermediate iterations, the message-pass should consider all possible coupling paths.
        y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
      y = e3x.nn.add(x, y)

      # Atom-wise refinement MLP.
      y = e3x.nn.Dense(self.features)(y)
      y = e3x.nn.silu(y)
      y = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(y)

      # Residual connection.
      x = e3x.nn.add(x, y)

    # 5. Predict atomic energies with an ordinary dense layer.
    element_bias = self.param('element_bias', lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number+1))
    atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x)  # (..., Natoms, 1, 1, 1)
    atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))  # Squeeze last 3 dimensions.
    atomic_energies += element_bias[atomic_numbers]

    # 6. Sum atomic energies to obtain the total energy.
    energy = jax.ops.segment_sum(atomic_energies, segment_ids=batch_segments, num_segments=batch_size)

    # To be able to efficiently compute forces, our model should return a single output (instead of one for each
    # molecule in the batch). Fortunately, since all atomic contributions only influence the energy in their own
    # batch segment, we can simply sum the energy of all molecules in the batch to obtain a single proxy output
    # to differentiate.
    return -jnp.sum(energy), energy  # Forces are the negative gradient, hence the minus sign.

  @nn.compact
  def __call__(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments=None, batch_size=None):
    if batch_segments is None:
      batch_segments = jnp.zeros_like(atomic_numbers)
      batch_size = 1

    # Since we want to also predict forces, i.e. the gradient of the energy w.r.t. positions (argument 1), we use
    # jax.value_and_grad to create a function for predicting both energy and forces for us.
    energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)
    (_, energy), forces = energy_and_forces(atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)

    return energy, forces