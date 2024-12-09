import os
#import pickle
#import pandas as pd
#import numpy as np
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QSlider,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QHBoxLayout,
    QSpinBox,
    QInputDialog,
    QLineEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)  # noqa:

matplotlib.use("Qt5Agg")
from ase import Atoms
from typing import TypedDict, List, Dict
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
        self.md_file_button.setFixedSize(200, 50)
        self.md_file_button.clicked.connect(lambda: self.select_file("md"))
        md_file_layout.addWidget(self.md_file_label)
        md_file_layout.addWidget(self.md_file_button)

        # Theta File selection
        theta_file_layout = QVBoxLayout()
        self.theta_file_label = QLabel("No Theta file selected", self)
        self.theta_file_label.setFont(QFont("Calibri", 10))
        self.theta_file_button = QPushButton("Select Theta file", self)
        self.theta_file_button.setFixedSize(200, 50)
        self.theta_file_button.clicked.connect(lambda: self.select_file("theta"))
        theta_file_layout.addWidget(self.theta_file_label)
        theta_file_layout.addWidget(self.theta_file_button)

        # Add both file selection layouts to the file section
        file_section.addLayout(md_file_layout)
        file_section.addLayout(theta_file_layout)
        self.layout.addLayout(file_section)

        self.descriptor_input = QLineEdit(self)
        self.descriptor_input.setPlaceholderText("Enter descriptor name")
        self.descriptor_input.setFixedSize(200, 50)
        md_file_layout.addWidget(self.descriptor_input)

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
        self.figure = Figure(
            figsize=(self.figure_width, self.figure_height), dpi=100
        )  # noqa:
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
            if file_type == "md":
                descriptor, ok = QInputDialog.getText(None, "Input", "Enter your descriptor string:")
                self.data.load_md_data(file_path, str(descriptor))
                self.md_file_label.setText(os.path.basename(file_path))
                self.display_md_data()
            elif file_type == "theta":
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
            QMessageBox.critical(
                self, "Error", f"Failed to load {file_type.upper()} data: {str(e)}"
            )


    def compute_and_plot_distribution(self):
        """Compute energy distributions and CDFs."""
        if not self.data.md_data:
            raise ValueError("MD data not loaded.")
        if not self.data.theta:
            raise ValueError("Theta parameters not loaded.")
        if not self.data.all_energies:
            raise ValueError("No energies extracted. Call _extract_energies() first.")
        if not self.data.E_tot_ml_list:
            raise ValueError(
                "Predicted energies not computed. Call compute_predicted_energies() first."
            )
        try:
            temperature = self.temp_spinbox.value()
            plot_data = self.data._extract_energies(temperature)

            # Clear previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Plot distributions
            ax.plot(
                plot_data["E_transformed"],
                plot_data["bolcdf"],
                label=f"Boltzmann CDF at {temperature}K",
                color="purple",
            )
            ax.plot(
                plot_data["E_tot_ml_sorted"],
                plot_data["E_tot_ml_cdf"],
                label="Predicted Energy CDF",
                color="orange",
            )
            ax.plot(
                plot_data["energies_sorted"],
                plot_data["energies_cdf"],
                label="Original Energy CDF",
                color="blue",
            )

            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel("Cumulative Distribution")
            ax.set_title("Energy Distribution Comparison")
            ax.legend()
            ax.grid(True)

            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Computation Error", str(e))

    
    def display_md_data(self):
        if not self.data.md_data:
            self.data_display.setPlainText("No MD data loaded.")
            return

        # Get metadata
        metadata = self.data.metadata

        # Format detailed display
        display_text = f"MD Data Summary:\n{metadata['summary']}\n\n"
        print("ok")
        for dataset in metadata["details"]:
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


    def plot_graph(self):
        # Example data for plotting
        X = [1, 2, 3, 4, 5, 6, 7]
        Y = [2, 3, 1, 6, 3, 4, 0]

        # Plot on the matplotlib canvas
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(X, Y)
        self.canvas.draw()

