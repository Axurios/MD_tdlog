from helper import check_requirements

check_requirements()

from PyQt5.QtWidgets import QApplication
from QtGui import GUI

if __name__ == "__main__":
    import sys

    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the GUI
    window = GUI()
    window.show()
    # Run the application's event loop
    sys.exit(app.exec_())
