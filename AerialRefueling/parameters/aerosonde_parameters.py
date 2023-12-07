import sys
sys.path.append('.')
import numpy as np

######################################################################################
#                       Sample times, etc
######################################################################################
ts_simulation = 0.01  # Smallest time step for simulation
start_time    = 0.0   # Start time for simulation
end_time      = 400.  # End time for simulation
ts_plotting   = 0.1   # Refresh rate for plots
ts_video      = 0.1   # Write rate for video
ts_control    = 0.01  # Sample rate for the controller

######################################################################################
#                        Initial Conditions
######################################################################################
# Initial states for Aerosonde UAV
pn    =  0.         # initial north position
pe    =  0.         # initial east position
pd    = -1.         # initial down position
u     =  0.         # initial velocity along body x-axis
v     =  0.         # initial velocity along body y-axis
w     =  0.         # initial velocity along body z-axis
phi   =  0.         # initial roll angle
theta =  0.         # initial pitch angle
psi   =  0.         # initial yaw angle
p     =  0.         # initial roll rate
q     =  0.         # initial pitch rate
r     =  0.         # initial yaw rate
state0 = np.array([[pn], 
                   [pe], 
                   [pd], 
                   [u],      
                   [v],      
                   [w],      
                   [phi],    
                   [theta],  
                   [psi],    
                   [p],      
                   [q],      
                   [r]])    

# Initial conditions for control surfaces
deltae  = 0.0       # Elevator
deltaa  = 0.0       # Aileron
deltar  = 0.0       # Rudder  
deltat  = 0.0       # Throttle
delta0  = np.array([[deltae],
                    [deltaa],
                    [deltar],
                    [deltat]])

# Trim conditions
Va0 = 0
Va  = 35
Y   = 0
R   = np.inf
Va2 = 35
Y2  = 0
R2  = np.inf
######################################################################################
#                   Physical parameters 
######################################################################################
mass          = 13.5         # [kg]
jx            = 0.824        # [kg m^2]
jy            = 1.135        # [kg m^2]
jz            = 1.759        # [kg m^2]
jxz           = 0.12         # [kg m^2]
gravity       = 9.806650     # [m/s^2]

######################################################################################
#                  Aerodynamic parameters 
######################################################################################
S_wing        =  0.55         # Wing area [m^2]
b             =  2.90         # Wingspan [m]
c             =  0.19         # Wing chord [m]
S_prop        =  0.2027       # Propellor area [m^2]
rho           =  1.2682       # [kg / m^3]
e             =  0.9          # Oswald's Efficiency Factor
AR            =  b**2/S_wing  #
C_L_0         =  0.23         # Zero AOA lift coefficient
C_D_0         =  0.043        # Intercept of linarized drag slope
C_m_0         =  0.0135       # Intercept of pitching moment
C_L_alpha     =  5.61         # Given in book
C_D_alpha     =  0.030        # Drag slope
C_m_alpha     = -2.74         # Pitching moment slope
C_L_q         =  7.95         # Needs to be normalized by c/2*Va
C_D_q         =  0.0          # Drag wrt pitch rate
C_m_q         =-38.21         # Pitching moment wrt q
C_L_delta_e   =  0.13         # Lift due to elevator deflection
C_D_delta_e   =  0.0135       # Drag due to elevator deflection
C_m_delta_e   = -0.99         # Pitching moment from elevator
M             = 50.0          # Barrier function coefficient for AOA
alpha0        =  0.47         # Angle at which stall occurs [deg]
epsilon       =  0.16         #
C_D_p         =  0.0          # Minimum drag
C_Y_0         =  0.0
C_ell_0       =  0.0
C_n_0         =  0.0
C_Y_beta      = -0.98
C_ell_beta    = -0.13
C_n_beta      =  0.073
C_Y_p         =  0.0
C_ell_p       = -0.51         # ell=p
C_n_p         = -0.069
C_Y_r         =  0.0
C_ell_r       =  0.25
C_n_r         = -0.095
C_Y_delta_a   =  0.075
C_ell_delta_a =  0.17
C_n_delta_a   = -0.011
C_Y_delta_r   =  0.19
C_ell_delta_r =  0.0024
C_n_delta_r   = -0.069
C_prop        =  1.
k_motor       = 80.           #80
k_T_p         =  0.
k_omega       =  0.