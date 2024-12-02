import os
import pickle
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QSlider, QTextEdit, QFileDialog, QMessageBox
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
        # Label to display the selected file name
        self.file_label = QLabel("No file selected", self)
        self.file_label.setFont(QFont("Calibri", 10))
        self.layout.addWidget(self.file_label)

        # Button to open file dialog
        self.button = QPushButton("Select data file", self)
        self.button.clicked.connect(self.select_file)
        self.layout.addWidget(self.button)

        # TextEdit widget to display the head of the CSV data
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

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self,"Select File", "", "Data Files (*.csv *.pkl);;CSV Files (*.csv);;Pickle Files (*.pkl)" )
    
        if file_path:
            self.file_name = os.path.basename(file_path)
            self.file_label.setText(self.file_name)  # Update label with file name
            
            # Check file extension and load accordingly
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.csv':
                self.load_data(file_path)
            elif file_extension == '.pkl':
                self.load_pickle_data(file_path)

    def load_data(self, file_path):
        try:
            # Load the .csv file
            df = pd.read_csv(file_path)
            # Get the head of the dataframe
            data_head = df.head().to_string(index=False)
            # Update the text widget with the data head
            self.data_display.setPlainText(data_head)
        except Exception as e:
            # If there is an error, display it in the text widget
            self.data_display.setPlainText(f"Error loading file: {e}")
    
    def load_pickle_data(self, file_path):
        try:
            # Load the .pkl file
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            
            # Check if it's md_data or theta
            if isinstance(data, dict) and any(isinstance(v, dict) and 'atoms' in v for v in data.values()):
                # This is md_data
                display_text = []
                for key, val in data.items():
                    atoms = val['atoms']
                    energies = val['energies']
                    
                    # Show information about first few entries
                    display_text.append(f"\nDataset: {key}")
                    display_text.append(f"Number of configurations: {len(atoms)}")
                    
                    # Show details of first configuration
                    if len(atoms) > 0:
                        first_atoms = atoms[0]
                        display_text.append("\nFirst configuration details:")
                        display_text.append(f"Number of atoms: {len(first_atoms)}")
                        display_text.append(f"Energy: {energies[0]:.6f}")
                        
                        # Get descriptor information
                        desc = first_atoms.get_array('milady-descriptors')
                        grad_desc = first_atoms.get_array('milady-descriptors-forces')
                        display_text.append(f"Descriptor shape: {desc.shape}")
                        display_text.append(f"Gradient descriptor shape: {grad_desc.shape}")
                        
                self.data_display.setPlainText("\n".join(display_text))
                
            elif isinstance(data, dict) and 'coef' in data and 'intercept' in data:
                # This is theta data
                display_text = [
                    "Theta Parameters:",
                    f"Coefficient shape: {data['coef'].shape}",
                    f"Intercept: {data['intercept']:.6f}"
                ]
                self.data_display.setPlainText("\n".join(display_text))
                
            else:
                self.data_display.setPlainText("Unknown pickle file format")
                
        except Exception as e:
            self.data_display.setPlainText(f"Error loading file: {str(e)}")

    def plot_graph(self):
        # Example data for plotting
        X = [1, 2, 3, 4, 5, 6, 7]
        Y = [2, 3, 1, 6, 3, 4, 0]

        # Plot on the matplotlib canvas
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(X, Y)
        self.canvas.draw()
