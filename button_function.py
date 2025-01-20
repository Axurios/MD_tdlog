from PyQt5.QtWidgets import QFileDialog
from plot import CDF_plot2, CDF_fisher
from neural_network import MSELoss, MAELoss, CrossEntropyLoss, HingeLoss


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

        if descriptorstring == gradientstring :
            self.show_error("2 Same parameters. Verify the parameters")
        elif descriptorstring == forcestring :
            self.show_error("2 Same parameters. Verify the parameters")
        elif gradientstring ==  forcestring :
            self.show_error("2 Same parameters. Verify the parameters")

        #ploting energy distribution with parameters computed based on fisher

        elif (self.data.md_data_loaded and self.data.theta_loaded):
            fig  = CDF_fisher(self.data, descriptorstring, gradientstring, forcestring, 1 )    
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