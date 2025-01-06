from dataHolder import DataHolder
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QDialog,
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)
from PyQt5.QtCore import (
    QRect,
)
from PyQt5.QtGui import (
    QGuiApplication,
    QIcon,
)
from button_function import (
    select_md_file,
    select_theta_file,
    compute_and_plot_distribution
)
import matplotlib.figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

#### get screen dimensions#######
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
        
        def create_ui(self):
            self.setWindowTitle("main page")
            self.setGeometry(sw//4, sh//4, window_width, window_height)
            self.setFixedSize(window_width, window_height)
            self.setWindowIcon(QIcon("cea.png"))
            self.initlayout()
            self.show()

        def initlayout(self):
            self.create_buttons()
            self.create_labels()
            self.create_choices()

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
            
        def show_plot(self, fig):
            plot_window = PlotWindow(fig)
            plot_window.exec_()


class PlotWindow(QDialog):
    def __init__(self, fig: matplotlib.figure.Figure):
        super().__init__()
        self.setWindowTitle("Plot Window")
        self.setGeometry(sw//4, sh//4, window_width, window_height)
        self.setFixedSize(window_width, window_height)

        # Set the passed figure
        self.figure = fig
        self.canvas = FigureCanvas(self.figure)  # Create a canvas for the figure
        
        # Add the canvas to the dialog's layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Redraw the canvas to show the plot
        self.canvas.draw()

    