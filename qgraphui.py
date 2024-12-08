import os
import matplotlib
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QTextEdit,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from dataHolder import DataHolder

matplotlib.use("Qt5Agg")


class GUI(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize window properties
        self.setWindowTitle("ATK Application")
        self.setGeometry(200, 200, 800, 600)

        # Create a DataHolder instance
        self.data_holder = DataHolder()

        # Initialize UI
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # File selection label and button
        self.file_label = QLabel("No file selected")
        self.file_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.file_label)

        self.file_button = QPushButton("Select Data File")
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        # Data display
        self.data_display = QTextEdit(self)
        self.data_display.setReadOnly(True)
        self.data_display.setFixedHeight(150)
        layout.addWidget(self.data_display)

        # Button to display plot
        self.plot_button = QPushButton("Display Plot")
        self.plot_button.clicked.connect(self.display_plot)
        layout.addWidget(self.plot_button)

        # Matplotlib Canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "", "Data Files (*.pkl)"
        )
        if not file_path:
            return

        self.file_label.setText(os.path.basename(file_path))
        try:
            if "md_data" in file_path:
                self.data_holder.load_md_data(file_path)
                self.data_display.setPlainText("MD data loaded successfully.")
            elif "theta" in file_path:
                self.data_holder.load_theta(file_path)
                self.data_display.setPlainText("Theta data loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def display_plot(self):
        try:
            self.data_holder.compute_predicted_energies()

            # Example plot with data
            ax = self.figure.add_subplot(111)
            ax.clear()
            ax.plot(self.data_holder.all_energies, label="All Energies")
            ax.plot(self.data_holder.E_tot_ml_list, label="Predicted Energies")
            ax.legend()
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display plot: {e}")
