from PyQt5.QtWidgets import QFileDialog
from plot import CDF_plot2, CDF_fisher
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
     FigureCanvasQTAgg as FigureCanvas)
from neural_network import MSELoss, MAELoss, CrossEntropyLoss, HingeLoss, SimpleMLP, DoubleMLP, train
from scipy.constants import Boltzmann


def select_md_file(self):
    """ Function used for load MD_file in UI"""
    try:
        #open file
        file_name, _ = QFileDialog.getOpenFileName(self, "Select MD Data File",
                                                   "", "All Files (*)")
        if file_name:
            # getting strings and file path
            choices = self.data.load_md_data(file_name)
            self.lbl1.setText(f"MD File Path : {file_name}")

            #update choices of selection boxes
            self.choice1.clear()
            self.choice2.clear()
            self.choice3.clear()
            self.choice1.addItems(choices)
            self.choice2.addItems(choices)
            self.choice3.addItems(choices)
        else:
            self.show_error("Couldn't Load MD_Data")

    #if fail, clear all part in connection with MD_Data
    except Exception as e:
        self.lbl1.setText("Path/to/MD_File")
        self.choice1.clear()
        self.choice2.clear()
        self.choice3.clear()
        self.data.md_data = None
        self.data.md_data_loaded = False
        self.show_error(f"{e}, clearing all prevous data on MD_data")


def select_theta_file(self):
    """ Function used for load MD_file in UI"""
    try:
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Theta File",
                                                   "", "All Files (*)")
        if file_name:
            self.lbl2.setText(f"Theta File Path : {file_name}")
            self.data.load_theta(file_name)
        else:
            self.show_error("Couldn't Load Theta")

    #if fail, clear all part in connection with Theta
    except Exception as e:
        self.lbl2.setText("Path/to/Theta_File")
        self.data.theta = None
        self.data.theta_loaded = False
        self.show_error(f"{e}, clearing all prevous data on Theta")


def compute_and_plot_distribution(self):
    try :
        #ploting distribution of each simulation ending (of each key) with the energy distribution of theta_file
        if (self.data.md_data_loaded and self.data.theta_loaded):
            fig = CDF_plot2(self.data,self.choice1.currentText())
            self.show_plot(fig)
        elif not self.data.md_data_loaded:
            self.show_error("No MD_Data file")
        else:
            self.show_error("No Theta file")
            
    except Exception as e:
        self.show_error(f"{e}, Probable error of input strings or Data errors.")

def compute_theta_of_fischer(self):
    try : 
        #verify if strings are chosen correctly

        descriptorstring = self.choice1.currentText()
        gradientstring = self.choice3.currentText()
        forcestring  = self.choice2.currentText()

        text = self.input1.text()
        temperature = int(text) if text else 300

        if descriptorstring == gradientstring :
            self.show_error("2 Same parameters. Verify the parameters")
        elif descriptorstring == forcestring :
            self.show_error("2 Same parameters. Verify the parameters")
        elif gradientstring ==  forcestring :
            self.show_error("2 Same parameters. Verify the parameters")
        elif temperature <= 0 :
            self.show_error("Temperature must be positive")

        #ploting energy distribution with parameters computed based on fisher

        elif (self.data.md_data_loaded and self.data.theta_loaded):
            to_beta = 1/(Boltzmann*temperature)
            fig  = CDF_fisher(self.data, descriptorstring, gradientstring, forcestring, beta = to_beta)    
            self.show_plot(fig)
        elif not self.data.md_data_loaded:
            self.show_error("No MD_Data file")
        else:
            self.show_error("No Theta file")
    except Exception as e:
        self.show_error(f"{e}, Probable error of input strings or Data errors.")


def select_loss(self, loss_type):
    self.loss = loss_type()
    print("loss selected")

def select_nn(self, nn_type):
    self.nn = nn_type()
    print("nn selected")

def plot_rmse(self):

    train(self.nn, self.loss, self.train_data, self.train_labels, self.val_data,self.val_labels, epochs=1,show=False)
    y,W1, W2= self.nn.forward_grid(self.val_data,2,100)
    RMSE = self.loss.forward_grid(y,self.val_labels)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W1, W2, RMSE, cmap='viridis', alpha=0.8)
    ax.set_xlabel('W1')
    ax.set_ylabel('W2')
    ax.set_zlabel('loss')
    ax.set_title('loss en 3D')
    self.plot = fig
    self.canvas.figure = fig
    self.canvas.draw()
    print("calcul terminé")
