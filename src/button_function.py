from matplotlib.backends.backend_qt5agg import (
     FigureCanvasQTAgg as FigureCanvas)
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import matplotlib.pyplot as plt
from .plot import CDF_plot2, CDF_fisher, ks_plot
from .NNmanager import NNManagerDialog
from .neural_network import MSELoss, MAELoss, CrossEntropyLoss, HingeLoss, SimpleMLP, DoubleMLP, train
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


# Plotting the different scientific computations and statistical test
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

        if not self.data.md_data_loaded:
            self.show_error("No MD_Data file")
        elif not self.data.theta_loaded:
            self.show_error("No Theta file")

        elif descriptorstring == gradientstring :
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
        else:
            self.show_error("Unknown Error")
    except Exception as e:
        self.show_error(f"{e}, Probable error of input strings or Data errors.")




def compute_ks_test(self):
    try : 
        #verify if strings are chosen correctly

        descriptorstring = self.choice1.currentText()
        gradientstring = self.choice3.currentText()
        forcestring  = self.choice2.currentText()

        text = self.input1.text()
        temperature = int(text) if text else 300

        if not self.data.md_data_loaded:
            self.show_error("No MD_Data file")
        elif not self.data.theta_loaded:
            self.show_error("No Theta file")

        elif descriptorstring == gradientstring :
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
            fig  = ks_plot(self.data,self.choice1.currentText(), self.choice3.currentText(), self.choice2.currentText(), 1 )
            self.show_plot(fig)
        else:
            self.show_error("Unknown Error")
    except Exception as e:
        self.show_error(f"{e}, Probable error of input strings or Data errors.")



# Neural Network Manager
# importing NN model :
def nn_import_button(self):
    """Open the Neural Network Manager Dialog."""
    nn_manager_dialog = NNManagerDialog(self.nn_manager)
    nn_manager_dialog.exec_()





def select_loss(self, loss_type):
    self.loss = loss_type()
    self.btn_loss.setText(self.loss.name)

def select_nn(self, nn_type):
    self.nn = nn_type()
    self.btn_nn.setText(self.nn.name)

def plot_rmse(self):
    if self.nn is None :
        self.show_error("Please select a Neural Network")
    
    elif self.loss is None :
        self.show_error("Please select a loss function")
    
    else:
        train(self.nn, self.loss, self.train_data, self.train_labels, self.val_data,self.val_labels, epochs=1,show=False)
        y,W1, W2= self.nn.forward_grid(self.val_data,2,100)
        RMSE = self.loss.forward_grid(y,self.val_labels)

        # Create a 3D plot
        self.canvas.figure.clear()  
        self.canvas.draw()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(W1, W2, RMSE, cmap='viridis', alpha=0.8)
        ax.set_xlabel('W1')
        ax.set_ylabel('W2')
        ax.set_zlabel(self.loss.name)
        ax.set_title('Loss Landscape')
        self.plot = fig
        self.canvas.figure = fig
        self.canvas.draw()  
        #print("calcul terminé")
