import os
from tkinter import filedialog
import subprocess
import sys


def check_requirements(requirements_file="requirements.txt"):
    try:
        with open(requirements_file) as f:
            required_packages = [
                pkg.strip() for pkg in f if pkg.strip() and not pkg.startswith("#")
            ]

        missing_packages = []
        for pkg in required_packages:
            package_name = pkg.split(">=")[0]
            try:
                __import__(package_name)
            except ImportError:
                missing_packages.append(pkg)

        if missing_packages:
            print(f"Missing packages: {missing_packages}")
            # Ask the user for permission to install
            user_response = (
                input(
                    "Some packages are missing. Do you want to install them? (yes/no): "
                )
                .strip()
                .lower()
            )

            if user_response in {"yes", "y"}:
                try:
                    subprocess.check_call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "-r",
                            requirements_file,
                        ]
                    )
                    print("All packages installed successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"Error during installation: {e}")
                    os._exit(1)
            else:
                print("Please install the missing packages manually.")
                os._exit(1)
        else:
            print("All required packages are installed.")
    except FileNotFoundError:
        print(f"Requirements file '{requirements_file}' not found.")
        os._exit(1)


# Modified select_file function
def select_file(file_name_var):
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
    )
    if file_path:
        file_name_var.set(file_path)  # Update the displayed filename # noqa:
        return file_path  # Return the file path to the caller
    return None


if __name__ == "__main__":
    # Example usage
    check_requirements("MD_tdlog/requirements.txt")
