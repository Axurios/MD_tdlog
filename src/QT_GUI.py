from .dataHolder import DataHolder
from .NNmanager import NNManager
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QDialog,
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QMessageBox,
    QMenu,
    QLineEdit
)
from PyQt5.QtCore import (
    QRect,
)
from PyQt5.QtGui import (
    QGuiApplication,
    QIcon,
)
from .button_function import (
    select_md_file,
    select_theta_file,
    compute_and_plot_distribution,
    compute_theta_of_fischer,

    compute_ks_test,
    nn_import_button,
    select_loss,
    select_nn,
    plot_rmse
)

from .neural_network import MSELoss, MAELoss, CrossEntropyLoss, HingeLoss, SimpleMLP, DoubleMLP

import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
     FigureCanvasQTAgg as FigureCanvas)

# ### get screen dimensions#######
app = QApplication([])
# Get the primary screen
screen = QGuiApplication.primaryScreen()
# Get screen dimensions
screen_geometry = screen.geometry()
sw = screen_geometry.width()
sh = screen_geometry.height()
window_width = sw//2
window_height = sh//2
app.quit()
################################


#####specs of buttons and boxes#####
x_box1 = window_width//20
box1_width = window_width//6
box1_height = box1_width//4
y_box1 = box1_height//2

x_line2 = x_box1*2 + box1_width
####################################


class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.create_ui()
            self.data = DataHolder()
            self.nn_manager = NNManager()
            self.loss_landscape_window = LossLandscapeWindow(self)
            
            # Initialize state to track file uploads
            self.md_file_uploaded = False
            self.theta_file_uploaded = False
                   
        def create_ui(self):
            self.setWindowTitle("main page")
            self.setGeometry(sw//4, sh//4, window_width, window_height)
            self.setFixedSize(window_width, window_height)
            self.setWindowIcon(QIcon("images/cea.png"))
            self.initlayout()
            self.show()

        def initlayout(self):
            self.create_buttons()
            self.create_labels()
            self.create_choices()
            self.create_imput_boxes()

        def create_buttons(self):

            # create button to add file of Molecular Dynamic
            self.btn1 = QPushButton("Select MD Data File", self)
            self.btn1.setGeometry(QRect(x_box1, y_box1, box1_width,
                                        box1_height))
            self.btn1.clicked.connect(lambda: select_md_file(self))

            # button pour importer Theta
            self.btn2 = QPushButton("Select Theta File", self)
            self.btn2.setGeometry(QRect(x_box1, window_height//2,
                                        box1_width, box1_height))
            self.btn2.clicked.connect(lambda: select_theta_file(self))
            
            # compute button
            self.btn3 = QPushButton("Energy Distribution", self)
            self.btn3.setGeometry(QRect(window_width-box1_width-x_box1,
                                        window_height-box1_height-y_box1,
                                        box1_width, box1_height))
            self.btn3.clicked.connect(lambda: compute_and_plot_distribution(self))
            #self.btn3.setEnabled(False)

            # compute theta of Fisher
            self.btn4 = QPushButton("Compute Fisher", self)
            self.btn4.setGeometry(QRect(window_width-box1_width-x_box1,
                                        window_height-box1_height*2-y_box1*2,
                                        box1_width, box1_height))
            self.btn4.clicked.connect(lambda: compute_theta_of_fischer(self))
            self.btn4.setEnabled(False)


            # Compute kolmogorov smirnov test
            self.btn5 = QPushButton("Compute ks test", self)
            self.btn5.setGeometry(QRect(window_width-box1_width-x_box1,
                                        window_height-box1_height*3-y_box1*3,
                                        box1_width, box1_height))
            self.btn5.clicked.connect(lambda: compute_ks_test(self))
            self.btn5.setEnabled(False)


            # launch neural network manager
            self.nn_button = QPushButton("Manage Neural Networks", self)
            self.nn_button.setGeometry(QRect(window_width-box1_width-x_box1,
                                        window_height-box1_height*4-y_box1*4,
                                        box1_width, box1_height))
            self.nn_button.clicked.connect(lambda: nn_import_button(self))
            # self.layout.addWidget(self.nn_button)  

            
            self.btn6 = QPushButton("Loss Landscape", self)
            self.btn6.setGeometry(QRect(x_box1,
                                        window_height-box1_height-y_box1,
                                        box1_width, box1_height))
            self.btn6.clicked.connect(self.open_loss_landscape)
            

        def create_labels(self):
            self.lbl1 = QLabel("Path/to/MD_File", self)
            self.lbl1.setGeometry(QRect(x_line2, y_box1,
                                        box1_width*4, box1_height))
            

            #########invariable label
            self.lbl11 = QLabel("Descriptor Name :", self)
            self.lbl11.setGeometry(QRect(x_box1, y_box1*2 + box1_height,
                                         box1_width*2, box1_height))

            self.lbl12 = QLabel("Force Name :", self)
            self.lbl12.setGeometry(QRect(x_box1, y_box1*3 + box1_height*2,
                                         box1_width*2, box1_height))
            
            self.lbl13 = QLabel("Descriptor Gradient Name :", self)
            self.lbl13.setGeometry(QRect(x_box1, y_box1*4 + box1_height*3,
                                         box1_width*2, box1_height))
            ##########

            self.lbl2 = QLabel("Path/to/Theta_File", self)
            self.lbl2.setGeometry(QRect(x_line2, window_height//2,
                                        box1_width*4, box1_height))
            
            self.lbl22 = QLabel("Temperature in Kelvin", self)
            self.lbl22.setGeometry(QRect(x_box1, window_height//2 + box1_height + y_box1,
                                          box1_width, box1_height))
            
        def create_choices(self):

            self.choice1 = QComboBox(self)
            self.choice1.setGeometry(QRect(x_line2, y_box1*2 + box1_height,
                                           int(box1_width*1.5), box1_height))
            self.choice2 = QComboBox(self)
            self.choice2.setGeometry(QRect(x_line2, y_box1*3 + box1_height*2,
                                           int(box1_width*1.5), box1_height))
            self.choice3 = QComboBox(self)
            self.choice3.setGeometry(QRect(x_line2, y_box1*4 + box1_height*3,
                                           int(box1_width*1.5), box1_height))
        
        def update_buttons(self):
            """Enable or disable buttons based on file upload status."""
            enable = self.data.md_data_loaded and self.data.theta_loaded
            # self.btn3.setEnabled(enable)
            self.btn4.setEnabled(enable)
            self.btn5.setEnabled(enable)

        def open_loss_landscape(self):
            """Open the Loss Landscape window."""
            self.hide()  # Close the main window
            self.loss_landscape_window.show()  # Show the new window

        def create_imput_boxes(self):
            self.input1 = QLineEdit(self)
            self.input1.setGeometry(QRect(x_line2, window_height//2 + box1_height + y_box1,
                                          box1_width, box1_height))
            self.input1.setPlaceholderText("Temperature in K")

            
        def show_plot(self, fig):
            plot_window = PlotWindow(fig)
            plot_window.exec_()

        def show_error(self, ErrorMessage):
            error_dialog = QMessageBox(self)
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Error")
            error_dialog.setText("An error has occurred!")
            error_dialog.setInformativeText(ErrorMessage)
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()  # Display the error dialog



class LossLandscapeWindow(QMainWindow):
    """Window for loss landscape functionalities."""
    def __init__(self, main_window):
        super().__init__()
        self.loss = None
        self.nn = None
        self.plot = None
        self.main_window = main_window  # Reference to the main window
        self.create_ui()
        #self.load_data()
        

    def create_ui(self):
            self.setWindowTitle("Loss Landscape")
            self.setGeometry(sw//4, sh//4, window_width, window_height)
            self.setFixedSize(window_width, window_height)
            self.setWindowIcon(QIcon("cea.png"))
            self.initlayout()

    def initlayout(self):
            self.create_buttons()
            self.plot_fig()
    
    def load_data(self):
        data = np.load("Data/mini_mnist.npz")

        self.train_data = data["train_data"]
        self.train_labels = data["train_labels"]
        self.test_data = data["test_data"]
        self.test_labels = data["test_labels"]

        N_val = int(0.1 * len(self.train_data))
        self.val_data = self.train_data[-N_val:]
        self.val_labels = self.train_labels[-N_val:]

        N_train = len(self.train_data) - N_val
        self.train_data = self.train_data[:N_train]
        self.train_labels = self.train_labels[:N_train]
    
    def create_buttons(self):
         # loss landscape 
        self.btn_loss = QPushButton("Select Loss", self)
        self.btn_loss.setGeometry(QRect(x_box1,
                                        y_box1,
                                        box1_width, box1_height))
        self.btn_loss.setMenu(self.loss_options())

        self.btn_nn = QPushButton("Select Model", self)
        self.btn_nn.setGeometry(QRect(int(x_box1 + 1.2*box1_width),
                                        y_box1,
                                        box1_width, box1_height))
        self.btn_nn.setMenu(self.nn_options())

        self.btn_plot = QPushButton("Plot Loss Landscape", self)
        self.btn_plot.setGeometry(QRect(int(x_box1 + 2.4*box1_width),
                                        y_box1,
                                        box1_width, box1_height))
        self.btn_plot.clicked.connect(lambda : plot_rmse(self))

        self.btn_return = QPushButton("Main Window", self)
        self.btn_return.setGeometry(QRect(x_box1,
                                        window_height-box1_height-y_box1,
                                        box1_width, box1_height))
        self.btn_return.clicked.connect(self.return_to_main_window)
    
    def loss_options(self):
            """Create a drop-down menu for the button select loss."""
            menu = QMenu()

            # Add menu options
            menu.addAction("MSE", lambda: select_loss(self, MSELoss))
            menu.addAction("MAE", lambda: select_loss(self, MAELoss))
            menu.addAction("Hinge", lambda: select_loss(self, HingeLoss))
            menu.addAction("Cross Entropy", lambda: select_loss(self, CrossEntropyLoss))

            return menu
    
    def nn_options(self):
            """Create a drop-down menu for the button select nn."""
            menu = QMenu()

            # Add menu options
            menu.addAction("SimpleMLP", lambda: select_nn(self, SimpleMLP))
            menu.addAction("DoubleMLP", lambda: select_nn(self, DoubleMLP))

            return menu

    def plot_fig(self):
        """Create the 3D plot canvas."""

        if self.plot is None:
            # Create an empty 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.set_title("Empty Loss Landscape")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.grid(True)
        else:
            # Use the provided plot
            fig = self.plot

        # Create the FigureCanvas for embedding the plot
        self.canvas = FigureCanvas(fig)
        self.canvas.setParent(self)  # Set this window as the canvas's parent
        # Position and size of the canvas
        self.canvas.setGeometry(int(window_width/10), 
                                y_box1 + 2*box1_height, 
                                8*(int(window_width/10)), 
                                int(window_height - (y_box1 + 5*box1_height) ))  # (x, y, width, height)
    
    def return_to_main_window(self):
        """Return to the main window."""
        self.hide()  # Close the current window
        self.main_window.show()  # Show the main window


class PlotWindow(QDialog):
    def __init__(self, fig: matplotlib.figure.Figure):
        super().__init__()
        self.setWindowTitle("Plot Window")
        self.setGeometry(sw//4, sh//4, window_width, window_height)
        self.setFixedSize(window_width, window_height)

        # Set the passed figure
        self.figure = fig
        self.canvas = FigureCanvas(self.figure)

        # Add the canvas to the dialog's layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Redraw the canvas to show the plot
        self.canvas.draw()