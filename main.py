from PyQt5.QtWidgets import QApplication
from MD_tdlog.QtGui import GUI
from MD_tdlog.helper import check_requirements

if __name__ == "__main__":
    import sys
    check_requirements()
    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the GUI
    window = GUI()
    window.show()
    # Run the application's event loop
    sys.exit(app.exec_())
