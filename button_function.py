from PyQt5.QtWidgets import QFileDialog
from plot import CDF_plot, CDF_plot2, CDF_fisher


def select_md_file(self):
    """ Function used for load MD_file in UI"""
    try:
        file_name, _ = QFileDialog.getOpenFileName(self, "Select MD Data File",
                                                   "", "All Files (*)")
        if file_name:
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
    """ploting distribution of each simulation ending (of each key)"""
    if (self.data.md_data_loaded and self.data.theta_loaded):
        fig = CDF_plot2(self.data,self.choice1.currentText())
        self.show_plot(fig)
    elif not self.data.md_data_loaded:
        self.show_error("No MD_Data file")
    else:
        self.show_error("No Theta file")

def compute_theta_of_fischer(self):
    fig  = CDF_fisher(self.data,self.choice1.currentText(), self.choice3.currentText(), self.choice2.currentText(), 1 )
    self.show_plot(fig)
