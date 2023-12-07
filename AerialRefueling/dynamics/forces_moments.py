import sys
sys.path.append('.')
import numpy as np
import parameters.aerosonde_parameters as P
from tools.rotations import Quaternion2Euler, Quaternion2Rotation, Euler2Rotation
from math import cos, sin, tan
import control
from control.matlab import *
from dynamics.wind import WindSimulation 

class ForcesMoments:

    def __init__(self):
        self.P=P
        self.Ts=P.ts_simulation
        self.wind = WindSimulation()

    def compute(self, state, delta):
        # Inertial parameters
        jx= self.P.jx
        jy= self.P.jy
        jz= self.P.jz
        jxz= self.P.jxz
        gravity=self.P.gravity
        mass=self.P.mass
        Va0=self.P.Va0

        # Aerodynamic parameters
        S_wing        = self.P.S_wing
        b             = self.P.b
        c             = self.P.c
        S_prop        = self.P.S_prop
        rho           = self.P.rho
        e             = self.P.e
        AR            = self.P.AR
        C_L_0         = self.P.C_L_0
        C_D_0         = self.P.C_D_0
        C_m_0         = self.P.C_m_0
        C_L_alpha     = self.P.C_L_alpha
        C_D_alpha     = self.P.C_D_alpha
        C_m_alpha     = self.P.C_m_alpha
        C_L_q         = self.P.C_L_q
        C_D_q         = self.P.C_D_q
        C_m_q         = self.P.C_m_q
        C_L_delta_e   = self.P.C_L_delta_e
        C_D_delta_e   = self.P.C_D_delta_e
        C_m_delta_e   = self.P.C_m_delta_e
        M             = self.P.M 
        alpha0        = self.P.alpha0
        epsilon       = self.P.epsilon
        C_D_p         = self.P.C_D_p
        C_Y_0         = self.P.C_Y_0
        C_ell_0       = self.P.C_ell_0
        C_n_0         = self.P.C_n_0
        C_Y_beta      = self.P.C_Y_beta
        C_ell_beta    = self.P.C_ell_beta
        C_n_beta      = self.P.C_n_beta 
        C_Y_p         = self.P.C_Y_p
        C_ell_p       = self.P.C_ell_p
        C_n_p         = self.P.C_n_p
        C_Y_r         = self.P.C_Y_r
        C_ell_r       = self.P.C_ell_r
        C_n_r         = self.P.C_n_r
        C_Y_delta_a   = self.P.C_Y_delta_a
        C_ell_delta_a = self.P.C_ell_delta_a
        C_n_delta_a   = self.P.C_n_delta_a
        C_Y_delta_r   = self.P.C_Y_delta_r
        C_ell_delta_r = self.P.C_ell_delta_r
        C_n_delta_r   = self.P.C_n_delta_r
        C_prop        = self.P.C_prop
        k_motor       = self.P.k_motor

        u     = state.item(3)
        v     = state.item(4)
        w     = state.item(5)
        phi   = state.item(6)
        theta = state.item(7)
        psi   = state.item(8)
        p     = state.item(9)
        q     = state.item(10)
        r     = state.item(11)

        delta_a = delta.item(1)
        delta_e = delta.item(0)
        delta_r = delta.item(2)
        delta_t = delta.item(3)

        Va    = np.sqrt(u ** 2 + v ** 2 + w ** 2)
        alpha = np.arctan2(w, u)
        beta  = np.arctan2(v, Va)

        qbar  = 0.5*rho*Va**2
        ca    = np.cos(alpha)
        sa    = np.sin(alpha)
    
        # Compute gravitaional forces
        f_x = -mass*gravity*np.sin(theta)
        f_y =  mass*gravity*np.cos(theta)*np.sin(phi)
        f_z =  mass*gravity*np.cos(theta)*np.cos(phi)
        
        # Compute lift and drag forces
        tmp1  = np.exp(-M*(alpha-alpha0))
        tmp2  = np.exp(M*(alpha+alpha0))
        sigma = (1+tmp1+tmp2)/((1+tmp1)*(1+tmp2))
        CL = (1-sigma)*(C_L_0+C_L_alpha*alpha)
        CD = C_D_0 + 1/(np.pi*e*AR)*(C_L_0+C_L_alpha*alpha)**2
        CL = CL +np.sign(alpha)* sigma*2*sa*sa*ca
        
        # Compute aerodynamic forces
        f_x = f_x + qbar*S_wing*(-CD*ca + CL*sa)
        f_x = f_x + qbar*S_wing*(-C_D_q*ca + C_L_q*sa)*c*q/(2*Va)
        
        f_y = f_y + qbar*S_wing*(C_Y_0 + C_Y_beta*beta)
        f_y = f_y + qbar*S_wing*(C_Y_p*p + C_Y_r*r)*b/(2*Va)
        
        f_z = f_z + qbar*S_wing*(-CD*sa - CL*ca)
        f_z = f_z + qbar*S_wing*(-C_D_q*sa - C_L_q*ca)*c*q/(2*Va)
        
        # Compute aerodynamic torques
        tau_phi   = qbar*S_wing*b*(C_ell_0 + C_ell_beta*beta)
        tau_phi   = tau_phi + qbar*S_wing*b*(C_ell_p*p + C_ell_r*r)*b/(2*Va)

        tau_theta = qbar*S_wing*c*(C_m_0 + C_m_alpha*alpha)
        tau_theta = tau_theta + qbar*S_wing*c*C_m_q*c*q/(2*Va)
        
        tau_psi   = qbar*S_wing*b*(C_n_0 + C_n_beta*beta)
        tau_psi   = tau_psi + qbar*S_wing*b*(C_n_p*p + C_n_r*r)*b/(2*Va)

        # Compute control forces
        f_x = f_x + qbar*S_wing*(-C_D_delta_e*ca+C_L_delta_e*sa)*delta_e
        f_y = f_y + qbar*S_wing*(C_Y_delta_a*delta_a + C_Y_delta_r*delta_r)
        f_z = f_z + qbar*S_wing*(-C_D_delta_e*sa-C_L_delta_e*ca)*delta_e
        
        # Compute control torques
        tau_phi   = tau_phi + qbar*S_wing*b*(C_ell_delta_a*delta_a + C_ell_delta_r*delta_r)
        tau_theta = tau_theta + qbar*S_wing*c*C_m_delta_e*delta_e
        tau_psi   = tau_psi + qbar*S_wing*b*(C_n_delta_a*delta_a + C_n_delta_r*delta_r)
        
        # Compute propulsion forces
        motor_temp = k_motor**2*delta_t**2-Va**2
        f_x        = f_x + 0.5*rho*S_prop*C_prop*motor_temp
        
        f_m = np.array([[f_x],     [f_y],       [f_z], 
                        [tau_phi], [tau_theta], [tau_psi]], dtype = float)
                
        return f_m, Va