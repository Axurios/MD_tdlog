from PyQt5.QtWidgets import QFileDialog
from plot import CDF_plot


def select_md_file(self):
    file_name, _ = QFileDialog.getOpenFileName(self, "Select MD Data File", "", "All Files (*)")
    if file_name:
        self.lbl1.setText(f"MD File Path : {file_name}")
        choices = self.data.load_md_data(file_name)
        self.choice1.addItems(choices)
        self.choice2.addItems(choices)
        self.choice3.addItems(choices)


def select_theta_file(self):
    file_name, _ = QFileDialog.getOpenFileName(self, "Select Theta File", "", "All Files (*)")
    if file_name:
        self.lbl2.setText(f"Theta File Path : {file_name}")
        self.data.load_theta(file_name)


def compute_and_plot_distribution(self):
    if (self.data.md_data_loaded and self.data.theta_loaded):
        fig = CDF_plot(self.data)
        self.show_plot(fig)

# def compute_theta_of_fischer(self):
