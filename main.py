from src.helper import check_requirements
check_requirements() 

from PyQt5.QtWidgets import QApplication

from src.QT_GUI import Window
from src.NNmanager import MessagePassingModel



if __name__ == "__main__":
    import sys
    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the GUI
    window = Window()
    # window.show()
    # Run the application's event loop
    sys.exit(app.exec_())