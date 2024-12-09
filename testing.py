from ase import Atoms
import numpy as np

info = {"m":12, "n": 13}

a = Atoms(symbols='In136Pd12', pbc=True, cell=[25.506480972, 25.506480972, 25.506480972])
a.new_array("google", np.array(range(148)))
print(a)