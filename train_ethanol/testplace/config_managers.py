import os 
import h5py
import xml.etree.ElementTree as ET

from h5py import File 
from typing import Dict, Any
import numpy as np

class H5Manager : 
    def __init__(self, path_h5 : os.PathLike[str], mode : str = "writing") :
        """H5Manager object for: writing and reading
        
        Parameters
        ----------

        path_h5 : os.PathLike[str]
            Path to .h5 file to write / read

        mode : str 
            Writing or reading mode
        """
        self.path_h5 = path_h5
        self.mode = mode 
        self.h5_file : File = None

        if mode == "writing" :
            self.init_h5()
        elif mode == "reading" : 
            self.read_h5()

    def init_h5(self) -> None : 
        """Init .h5 file for writing"""
        self.h5_file = h5py.File(self.path_h5, 'a')
        return 
    
    def close(self) -> None :
        """Close .h5"""
        self.h5_file.close()
        return 

    def read_h5(self) -> None : 
        """Init .h5 file reading"""
        self.h5_file = h5py.File(self.path_h5, "r")
        return

    def add_or_update_data(self, key_simulation : str,
                            data : Dict[str, Any]) -> None :
        """Update the .h5 file 
        
        Parameters
        ----------

        key_simulation : str 
            Name of the simulation to update in .h5

        data : Dict[str, Any]
            Data associated to ```key_simulation``` to update in .h5

        """

        if key_simulation not in self.h5_file:
            # Create group for configuration
            simulation_group = self.h5_file.create_group(key_simulation)

        else : 
            # Group is already created
            simulation_group = self.h5_file[key_simulation]

        for key, val in data.items() : 
            try : 
                simulation_group.create_dataset(key, data=val, compression="gzip", compression_opts=9)
            except : 
                simulation_group.create_dataset(key, data=val)
 
        return 
    
    def extract_data(self, key_simulation : str) -> Dict[str, Any] : 
        """Extract data from .h5 for a given simulation
        
        Parameters
        ----------

        key_simulation : str 
            Name of the simulation to update in .h5


        Returns
        -------

        Dict[str, Any]
            Simulation data 
        """


        h5_config = self.h5_file[key_simulation]
        data = {}

        for key in h5_config.keys() : 
            data[key] = h5_config[key][()]

        return data
    

    
class XMLManager:
    def __init__(self, path_xml: os.PathLike[str], mode: str = "writing"):
        """XMLManager object for: writing and reading
        
        Parameters
        ----------
        path_xml : os.PathLike[str]
            Path to .xml file to write / read

        mode : str 
            Writing or reading mode
        """
        self.path_xml = path_xml
        self.mode = mode 

        print(f"Initializing XMLManager with path: {self.path_xml} and mode: {self.mode}")

        if mode == "writing":
            self.init_xml()
        elif mode == "reading": 
            self.read_xml()

    def init_xml(self) -> None:
        """Init .xml file for writing"""
        self.tree = None
        self.root = ET.Element('TimeOfFailure')
        print("XML initialized with root element 'TimeOfFailure'")
        return 

    def read_xml(self) -> None:
        """Read .xml file"""
        try:
            print(f"Attempting to read XML file: {self.path_xml}")
            self.tree = ET.parse(self.path_xml)
            self.root = self.tree.getroot()
            print(f"Successfully read XML file: {self.path_xml}")
        except Exception as e:
            print(f"Error reading XML file {self.path_xml}: {e}")
    
    def parse_xml(self) -> Dict[str, Any]:
        """Parse .xml file and return data as a dictionary"""
        data = {}
        for child in self.root:
            data[child.tag] = child.text.strip()
        return data
    
    def generate_xml(self, data: Dict[str, Any]) -> None:
        """Generate .xml file from data
        
        Parameters
        ----------
        data : Dict[str, Any]
            Data to store in .xml file
        """
        print(f"Starting XML generation for data: {data}")
        
        for key, val in data.items():
            ET.SubElement(self.root, key).text = str(val)
            print(f"Added element: {key} with value: {val}")

        self.tree = ET.ElementTree(self.root)

        # Ensure the directory exists
        dir_path = os.path.dirname(self.path_xml)
        if dir_path and not os.path.exists(dir_path):
            print(f"Directory does not exist. Creating: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        
        # Print full file path before writing
        print(f"Writing to file: {os.path.abspath(self.path_xml)}")

        # Write XML to file
        try:
            self.tree.write(self.path_xml, encoding="utf-8", xml_declaration=True)
            print(f"XML file written successfully to: {self.path_xml}")
        except Exception as e:
            print(f"Error writing XML file {self.path_xml}: {e}")

class XMLManager : 
    def __init__(self, path_xml : os.PathLike[str], mode : str = "writing") :
        """H5Manager object for: writing and reading
        
        Parameters
        ----------

        path_h5 : os.PathLike[str]
            Path to .h5 file to write / read

        mode : str 
            Writing or reading mode
        """
        self.path_xml = path_xml
        self.mode = mode 

        if mode == "writing" :
            self.init_xml()
        elif mode == "reading" : 
            self.read_xml()

    def init_xml(self) -> None : 
        """Init .xml file for writing"""
        self.tree = None 
        self.root = ET.Element('TimeOfFailure')
        return 
    
    def read_xml(self) -> None : 
        """Init .xml file reading"""
        self.tree = ET.parse(self.path_xml)
        self.root = self.tree.getroot()
        return
    
    def parse_xml(self) -> Dict[str, Any] : 
        """Parse .xml file 
        
        Returns 
        -------

        Dict[str, Any]
            Data contain in .xml file
        """
        data = {}
        for child in self.root : 
            data[child.tag] = child.text.strip()

        return data
    
    def generate_xml(self, data : Dict[str, Any]) -> None : 
        """Generate .xml file from data 
        
        Parameters
        ----------

        data : Dict[str, Any]
            Data to store in .xml file
        """
        for key, val in data.items() :
            ET.SubElement(self.root, key).text = str(val)

        self.tree = ET.ElementTree(self.root)
        self.tree.write(self.path_xml)
        return 
    
# ###############################
# # INPUTS 
# ###############################
# data = {"Into":1.0,
#         "The":2.0,
#         "Storm":'a'}
# path_xml = './test.xml'

# data_h5 = {"Into":1.0,
#            "The":np.array([2.0]),
#            "Storm":'a'}
# path_h5 = './test.h5'
# ################################

# xml_obj = XMLManager(path_xml, mode='writing')
# xml_obj.generate_xml(data)

# xml_obj2 = XMLManager(path_xml, mode='reading')
# print(xml_obj2.parse_xml()) 

# h5_obj = H5Manager(path_h5, mode='writing')
# h5_obj.add_or_update_data('TimeOfFailure', data_h5)
# h5_obj.close()

# h5_obj2 = H5Manager(path_h5, mode='reading')
# print(h5_obj2.extract_data('TimeOfFailure'))