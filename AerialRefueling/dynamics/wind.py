import sys
sys.path.append('.')
import numpy as np
from tools.rotations import *
import control.matlab as mat
import parameters.aerosonde_parameters as P

class WindSimulation():

    def __init__(self, Vsw=np.array([[0.],[0.],[0.]])):
        self.Vsw = Vsw # Initialize the wind class with a steady wind velocity Vsw

    def DrydenGust(self, Va, t):
        # Dryden gust model parameters (pg 56 UAV book)
        Lu = 200           # [m]
        Lv = Lu            # [m]
        Lw = 50            # [m]
        sigma_u = 1.06     # [m/s]
        sigma_v = sigma_u  # [m/s]
        sigma_w = 0.7      # [m/s]

        # Dryden transfer functions (section 4.4 UAV book) 
        # Transfer function for longitudinal turbulence, lateral turbulence, vertical turbulence
        au=sigma_u*np.sqrt(2*Va/Lu)
        av=sigma_v*np.sqrt(3*Va/Lv)
        aw=sigma_w*np.sqrt(3*Va/Lw)
        Hu = mat.tf([0, au],[1, Va/Lu])
        Hv = mat.tf([av, av*Va/(np.sqrt(3)*Lv)],[1, 2*Va/Lv, (Va/Lv)**2])
        Hw = mat.tf([aw, aw*Va/(np.sqrt(3)*Lw)],[1, 2*Va/Lw, (Va/Lw)**2])

        # Generate a white noise value for longitudinal turbulence, lateral turbulence, vertical turbulence
        T=[0, t]
        white_noise_u = np.random.normal(0,1,1)
        white_noise_v = np.random.normal(0,1,1)
        white_noise_w = np.random.normal(0,1,1)

        # Solve transfer function
        y_u, _, _ = mat.lsim(Hu, white_noise_u[0], T)
        y_v, _, _ = mat.lsim(Hv, white_noise_v[0], T)
        y_w, _, _ = mat.lsim(Hw, white_noise_w[0], T)

        # Gust components
        wg_u = y_u[1]
        wg_v = y_v[1]
        wg_w = y_w[1]

        return np.array([[wg_u],[wg_v],[wg_w]])

    def getAircraftWindResponse(self, states, Va, sim_time):
        # Extract the aircraft state vector
        pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = states.flatten()
        
        # Calculate the total wind velocity in the body frame
        gusts = self.DrydenGust(Va, sim_time)
        windtot = Rvb(phi, theta, psi)*self.Vsw + gusts
        Var = np.array([[u - windtot[0][0]],
                        [v - windtot[1][0]],
                        [w - windtot[2][0]]])
        
        # Calculate the relative airspeed components
        ur, vr, wr = Var.flatten()
        Va = np.sqrt(ur**2 + vr**2 + wr**2)
        alpha = np.arctan(wr/ur)
        beta  = np.arcsin(vr/Va)

        return Va, alpha, beta