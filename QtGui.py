import os
import pickle
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QSlider, QTextEdit, 
    QFileDialog, QMessageBox, QHBoxLayout, QSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # noqa:
matplotlib.use("Qt5Agg")
# from ase import Atoms
# from typing import TypedDict, List, Dict
from dataHolder import DataHolder


class GUI(QWidget):
    def __init__(self):
        super().__init__()

        # Set window title and initialize variables
        self.setWindowTitle("MD Long-term stability")
        
        self.data = DataHolder()
        
        
        # self.file_name = "No file selected"
        # self.init_aspect_ratio = None  # To store the initial aspect ratio
        # Create the main layout

        # Initialize UI components
        self.initUI()

        self.add_percentage_slider()
        self.add_plot()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # File selection section
        file_section = QHBoxLayout()
        
        # MD Data File selection
        md_file_layout = QVBoxLayout()
        self.md_file_label = QLabel("No MD file selected", self)
        self.md_file_label.setFont(QFont("Calibri", 10))
        self.md_file_button = QPushButton("Select MD data file", self)
        self.md_file_button.clicked.connect(lambda: self.select_file('md'))
        md_file_layout.addWidget(self.md_file_label)
        md_file_layout.addWidget(self.md_file_button)
        
        # Theta File selection
        theta_file_layout = QVBoxLayout()
        self.theta_file_label = QLabel("No Theta file selected", self)
        self.theta_file_label.setFont(QFont("Calibri", 10))
        self.theta_file_button = QPushButton("Select Theta file", self)
        self.theta_file_button.clicked.connect(lambda: self.select_file('theta'))
        theta_file_layout.addWidget(self.theta_file_label)
        theta_file_layout.addWidget(self.theta_file_button)
        
        # Add both file selection layouts to the file section
        file_section.addLayout(md_file_layout)
        file_section.addLayout(theta_file_layout)
        self.layout.addLayout(file_section)

        # Temperature selection
        temp_section = QHBoxLayout()
        temp_label = QLabel("Temperature (K):")
        self.temp_spinbox = QSpinBox()
        self.temp_spinbox.setRange(100, 5000)
        self.temp_spinbox.setValue(2000)
        temp_section.addWidget(temp_label)
        temp_section.addWidget(self.temp_spinbox)
        self.compute_button = QPushButton("Compute Distribution", self)
        self.compute_button.clicked.connect(self.compute_and_plot_distribution)
        self.compute_button.setEnabled(False)
        temp_section.addWidget(self.compute_button)
        self.layout.addLayout(temp_section)

        # Data display
        self.data_display = QTextEdit(self)
        self.data_display.setReadOnly(True)
        self.data_display.setFixedHeight(150)
        self.layout.addWidget(self.data_display)


    # def initUI(self):
    #     self.layout = QVBoxLayout()
    #     self.setLayout(self.layout)
    #     # Label to display the selected file name
    #     self.file_label = QLabel("No file selected", self)
    #     self.file_label.setFont(QFont("Calibri", 10))
    #     self.layout.addWidget(self.file_label)

    #     # Button to open file dialog
    #     self.button = QPushButton("Select data file", self)
    #     self.button.clicked.connect(self.select_file)
    #     self.layout.addWidget(self.button)

        # TextEdit widget to display the head of the CSV data
        # self.data_display = QTextEdit(self)
        # self.data_display.setReadOnly(True)
        # self.data_display.setFixedHeight(150)
        # self.layout.addWidget(self.data_display)


    def add_percentage_slider(self):
        # Slider to select a percentage (1 to 100)
        self.percentage_slider = QSlider(Qt.Horizontal, self)
        self.percentage_slider.setRange(1, 100)
        self.percentage_slider.setValue(50)
        self.percentage_slider.setTickPosition(QSlider.TicksBelow)
        self.percentage_slider.setTickInterval(10)
        self.layout.addWidget(self.percentage_slider)

    def add_plot(self):
        self.figure_width, self.figure_height = 6, 4 
        self.figure = Figure(figsize=(self.figure_width, self.figure_height), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.init_aspect_ratio = self.figure_width / self.figure_height
                              
    def add_plot2(self):
        # Initial dimensions for the figure
        self.figure_width, self.figure_height = 5, 4  # in inches
        # Create the Matplotlib plot area
        self.figure = Figure(figsize=(self.figure_width, self.figure_height), dpi=100) # noqa:
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        # Calculate and store the initial aspect ratio
        self.init_aspect_ratio = self.figure_width / self.figure_height
        # Draw the initial plot
        self.plot_graph()

    def resizeEvent(self, event):
        # Get the new width of the window
        new_width = self.canvas.width() / self.figure.dpi
        # Calculate the new height to maintain aspect ratio
        new_height = new_width / self.init_aspect_ratio

        # Resize the figure
        self.figure.set_size_inches(new_width, new_height, forward=True)
        self.canvas.draw()

        # Pass the event to the base class for normal processing
        super().resizeEvent(event)

    def select_file(self, file_type):
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {file_type.upper()} File", "", "Pickle Files (*.pkl)"
        )

        if not file_path:
            return  # User canceled the file dialog

        try:
            if file_type == 'md':
                self.data.load_md_data(file_path)
                self.md_file_label.setText(os.path.basename(file_path))
                self.display_md_data()
            elif file_type == 'theta':
                self.data.load_theta(file_path)
                self.theta_file_label.setText(os.path.basename(file_path))
                self.display_theta_data()
            else:
                raise ValueError("Invalid file type")
            
            # Enable compute button if both data are loaded
            if self.data.md_data and self.data.theta:
                self.data._extract_energies()
                self.compute_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load {file_type.upper()} data: {str(e)}")


    # def select_file(self):
    #     file_path, _ = QFileDialog.getOpenFileName(self,"Select File", "", "Data Files (*.csv *.pkl);;CSV Files (*.csv);;Pickle Files (*.pkl)" )
    
        # if file_path:
        #     self.file_name = os.path.basename(file_path)
        #     self.file_label.setText(self.file_name)  # Update label with file name
            
            # # Check file extension and load accordingly
            # file_extension = os.path.splitext(file_path)[1].lower()
            # if file_extension == '.csv':
            #     self.load_data(file_path)
            # elif file_extension == '.pkl':
            #     self.load_pickle_data(file_path)

    def compute_and_plot_distribution(self):
        """Compute energy distributions and CDFs."""
        if not self.data.md_data:
            raise ValueError("MD data not loaded.")
        if not self.data.theta:
            raise ValueError("Theta parameters not loaded.")
        if not self.data.all_energies:
            raise ValueError("No energies extracted. Call _extract_energies() first.")
        if not self.data.E_tot_ml_list:
            raise ValueError("Predicted energies not computed. Call compute_predicted_energies() first.")
        try:
            temperature = self.temp_spinbox.value()
            plot_data = self.data._extract_energies(temperature)

            # Clear previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Plot distributions
            ax.plot(plot_data['E_transformed'], plot_data['bolcdf'], 
                    label=f'Boltzmann CDF at {temperature}K', color='purple')
            ax.plot(plot_data['E_tot_ml_sorted'], plot_data['E_tot_ml_cdf'], 
                    label='Predicted Energy CDF', color='orange')
            ax.plot(plot_data['energies_sorted'], plot_data['energies_cdf'], 
                    label='Original Energy CDF', color='blue')

            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Cumulative Distribution')
            ax.set_title('Energy Distribution Comparison')
            ax.legend()
            ax.grid(True)

            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Computation Error", str(e))

    def load_md_data(self, file_path: str):
        """Load MD data from a file."""
        try:
            with open(file_path, "rb") as f:
                loaded_data = pickle.load(f)
                if not isinstance(loaded_data, dict):
                    raise ValueError("Invalid MD data format: Expected a dictionary.")
                # Ensure all required fields are present
                for key, val in loaded_data.items():
                    if not isinstance(val, dict) or 'atoms' not in val or 'energies' not in val:
                        raise ValueError(f"Invalid MD data for key {key}.")
                self.md_data = loaded_data
                self._extract_energies()
        except Exception as e:
            raise ValueError(f"Error loading MD data: {e}")

    def load_theta(self, file_path: str):
        """Load Theta data from a file."""
        try:
            with open(file_path, "rb") as f:
                loaded_theta = pickle.load(f)
                if not isinstance(loaded_theta, dict) or 'coef' not in loaded_theta or 'intercept' not in loaded_theta:
                    raise ValueError("Invalid Theta data format: Expected keys 'coef' and 'intercept'.")
                # Validate coefficient and intercept shapes
                if not isinstance(loaded_theta['coef'], np.ndarray) or loaded_theta['coef'].ndim != 2:
                    raise ValueError("Theta 'coef' must be a 2D numpy array.")
                if not np.isscalar(loaded_theta['intercept']):
                    raise ValueError("Theta 'intercept' must be a scalar.")
                self.theta = loaded_theta
        except Exception as e:
            raise ValueError(f"Error loading Theta data: {e}")

    def display_md_data(self):
        if not self.data.md_data:
            self.data_display.setPlainText("No MD data loaded.")
            return
        
        # Get metadata
        metadata = self.data.metadata
        
        # Format detailed display
        display_text = f"MD Data Summary:\n{metadata['summary']}\n\n"
        print("ok")
        for dataset in metadata['details']:
            display_text += (
                f"Dataset Key: {dataset['key']}\n"
                f"Configurations: {dataset['num_configurations']}\n"
                f"Energy Range: {dataset['energy_range'][0]:.4f} to {dataset['energy_range'][1]:.4f} eV\n"
                f"First Configuration:\n"
                f"  - Number of Atoms: {dataset['first_config_details']['num_atoms']}\n"
                f"  - First Energy: {dataset['first_config_details']['first_energy']:.4f} eV\n"
                f"  - Descriptor Shape: {dataset['first_config_details']['descriptor_shape']}\n\n"
            )
        
        self.data_display.setPlainText(display_text)

    def display_theta_data(self):
        if not self.data.theta:
            self.data_display.setPlainText("No Theta data loaded.")
            return
        
        # Get metadata
        metadata = self.data.metadata
        print("okok")
        # print(metadata)
        display_text = (
            "Theta Parameters:\n"
            f"{metadata['summary']}\n\n"
            f"Coefficient Shape: {metadata['coefficient_shape']}\n"
            f"Intercept: {metadata['intercept']:.4f}\n\n"
            "Coefficient Details:\n"
            f"  - Minimum Value: {metadata['coefficient_details']['min_value']:.4e}\n"
            f"  - Maximum Value: {metadata['coefficient_details']['max_value']:.4e}\n"
            f"  - Mean Value: {metadata['coefficient_details']['mean_value']:.4e}"
        )
        
        self.data_display.setPlainText(display_text)
    # def load_data(self, file_path):
    #     try:
    #         # Load the .csv file
    #         df = pd.read_csv(file_path)
    #         # Get the head of the dataframe
    #         data_head = df.head().to_string(index=False)
    #         # Update the text widget with the data head
    #         self.data_display.setPlainText(data_head)
    #     except Exception as e:
    #         # If there is an error, display it in the text widget
    #         self.data_display.setPlainText(f"Error loading file: {e}")
    
    # def load_pickle_data(self, file_path):
    #     try:
    #         # Load the .pkl file
    #         with open(file_path, 'rb') as file:
    #             data = pickle.load(file)
            
    #         # Check if it's md_data or theta
    #         if isinstance(data, dict) and any(isinstance(v, dict) and 'atoms' in v for v in data.values()):
    #             # This is md_data
    #             display_text = []
    #             for key, val in data.items():
    #                 atoms = val['atoms']
    #                 energies = val['energies']
                    
    #                 # Show information about first few entries
    #                 display_text.append(f"\nDataset: {key}")
    #                 display_text.append(f"Number of configurations: {len(atoms)}")
                    
    #                 # Show details of first configuration
    #                 if len(atoms) > 0:
    #                     first_atoms = atoms[0]
    #                     display_text.append("\nFirst configuration details:")
    #                     display_text.append(f"Number of atoms: {len(first_atoms)}")
    #                     display_text.append(f"Energy: {energies[0]:.6f}")
                        
    #                     # Get descriptor information
    #                     desc = first_atoms.get_array('milady-descriptors')
    #                     grad_desc = first_atoms.get_array('milady-descriptors-forces')
    #                     display_text.append(f"Descriptor shape: {desc.shape}")
    #                     display_text.append(f"Gradient descriptor shape: {grad_desc.shape}")
                        
    #             self.data_display.setPlainText("\n".join(display_text))
                
    #         elif isinstance(data, dict) and 'coef' in data and 'intercept' in data:
    #             # This is theta data
    #             display_text = [
    #                 "Theta Parameters:",
    #                 f"Coefficient shape: {data['coef'].shape}",
    #                 f"Intercept: {data['intercept']:.6f}"
    #             ]
    #             self.data_display.setPlainText("\n".join(display_text))
                
    #         else:
    #             self.data_display.setPlainText("Unknown pickle file format")
                
    #     except Exception as e:
    #         self.data_display.setPlainText(f"Error loading file: {str(e)}")

    def plot_graph(self):
        # Example data for plotting
        X = [1, 2, 3, 4, 5, 6, 7]
        Y = [2, 3, 1, 6, 3, 4, 0]

        # Plot on the matplotlib canvas
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(X, Y)
        self.canvas.draw()
