import numpy as np
from typing import Tuple

class TimeOfFailureAnalysis : 
    def __init__(self, name_simu : str,
                 init_kinetic : np.ndarray,
                 init_pot : np.ndarray,
                 init_tot : np.ndarray,
                 positions_array : np.ndarray,
                 temperature : float) : 
        
        self.name_sim = name_simu
        self.failure = False
        self.kB = 8.6173303e-5

        # buffer for energetic quantities
        self.kinetic = init_kinetic
        self.pot = init_pot
        self.tot = init_tot

        # should be 3D (N,3,N_buffer)
        self.positions = positions_array

        self.temperature = temperature

    def check_kinetic_failure(self, kinetic : float,
                              positions : np.ndarray, 
                              nb_sigma : float = 1.0) -> bool : 
        
        # fluctuations
        sigma_energy = np.sqrt(1.5*self.positions.shape[0])*self.kB*self.temperature
        
        # failure => atoms are moving free
        if (np.abs(self.kinetic-kinetic) < nb_sigma*sigma_energy).all() : 
            self.failure = True

        self.kinetic[:-1] = self.kinetic[1:]
        self.kinetic[-1] = kinetic

        self.positions[:,:,:-1] = self.positions[:,:,1:]
        self.positions[:,:,-1] = positions

        return self.failure
    
    def check_total_energy_failure(self, 
                                   total_energy : float,
                                   nb_sigma : float = 3.0) -> bool :
        # fluctuations
        sigma_energy = np.sqrt(1.5*self.positions.shape[0])*self.kB*self.temperature
        
        # failure => atoms are moving free
        if (np.abs(self.tot - total_energy) > nb_sigma*sigma_energy).all() : 
            self.failure = True

        self.tot[:-1] = self.tot[1:]
        self.kinetic[-1] = total_energy

        return self.failure
    
    def analysis_failure(self, kinetic_energy : float,
                         total_energy : float,
                         positions : np.ndarray) -> Tuple[bool, str] :
        bool_kin = self.check_kinetic_failure(kinetic_energy,
                                              positions)
        bool_tot = self.check_total_energy_failure(total_energy)

        if bool_kin and not bool_tot : 
            return True, 'kinetic_failure'
        
        if not bool_kin and bool_tot : 
            return True, 'total_energy_failure'
        
        if bool_kin and bool_tot : 
            return True, 'kinetic_and_total_energy_failure'

        else : 
            return False, 'no_failure'