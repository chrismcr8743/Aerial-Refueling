import sys
sys.path.append('.')  
import numpy as np
import matplotlib.pyplot as plt
import parameters.aerosonde_parameters as P
from dynamics.wind import WindSimulation 
from dynamics.dynamics import MavDynamics 
from dynamics.forces_moments import ForcesMoments 
from dynamics.dynamics2 import MavDynamics2 
from ctrl.autopilot import AutoPilot
from ctrl.transfer_fun import TransferFunctions 
from ctrl.compute_trim import ComputeTrim
from ctrl.path_follow import PathFollower
from viewers.mav_animation import Mav_Animation 
from viewers.data_plotter_state import dataPlotterState
from msg.msg_path import msg_path

# Initialize
anim         = Mav_Animation()
# plot_state   = dataPlotterState()
wind         = WindSimulation()
dynamics     = MavDynamics()
dynamics2    = MavDynamics2()
forcesm      = ForcesMoments()
trim         = ComputeTrim()
TransferFun  = TransferFunctions()
Autopilot    = AutoPilot()
Autopilot2   = AutoPilot()
path_follow  = PathFollower()
path_follow2 = PathFollower()
path         = msg_path()
path2        = msg_path()

delta = P.delta0
Va    = P.Va
Y     = P.Y
R     = P.R

# MAV1
path.line_origin     = np.array([[100, 50.0, -100]]).T # Define origin of inital path for Mav1
path.line_direction  = path.line_direction / np.linalg.norm(path.line_direction)
x_trim, u_trim = trim.compute_trim(Va,Y,R)       # Compute trim states and deflections for Mav1  
xTrim = x_trim[np.newaxis].transpose() 
uTrim = u_trim[np.newaxis].transpose()
xTrim[0] = xTrim[0]+10 # Initial pn 
xTrim[2] = xTrim[2]-55 # Initial pd 
dynamics._state = xTrim                          # Set as inital state for dynamics 
delta = uTrim                                    # Set inital deflections 
TransferFun.compute_tf_models(xTrim, uTrim)      # Compute transfer function models models from trim conditions
gains = TransferFun.state_space(xTrim, uTrim)    # Get gains for the autopilot

# MAV2
Va2   = P.Va
Y2    = P.Y
R2    = P.R2
path2.line_origin    = np.array([[100, 50.0, -90]]).T # Define origin of inital path for Mav1
path2.line_direction = path.line_direction / np.linalg.norm(path.line_direction)
x_trim, u_trim = trim.compute_trim(Va2,Y2,R2)    # Compute trim states and deflections for Mav2
xTrim2 = x_trim[np.newaxis].transpose() 
uTrim2 = u_trim[np.newaxis].transpose()
xTrim2[2] = xTrim2[2]-40 # Initial pd 
dynamics2._state2 = xTrim2                       # Set as inital state for dynamics2 
delta2 = uTrim2                                  # Set inital deflections 
TransferFun.compute_tf_models(xTrim2, uTrim2)    # Compute transfer function models from trim conditions
gains2 = TransferFun.state_space(xTrim2, uTrim2) # Get gains for the autopilot


print("\nPress Q to exit simulation")
sim_time = P.start_time
while sim_time < P.end_time:
    t_next_plot = sim_time+P.ts_plotting
    while sim_time < t_next_plot:

        # MAV1
        Va_c = 40 # Set commanded airspeed
        h_c, chi_c = path_follow.update(path, dynamics._state) # Get commanded altitude, commanded course
        Va = dynamics.getVa()
        pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = dynamics._state.flatten()
        u_state1 = np.array([sim_time, phi, theta, psi, p, q, r, Va, pd, Va_c, h_c, chi_c])
        deltaAP, phi_c, theta_c, chi_c, altitude_state = Autopilot.autopilot(gains, u_state1) 
        delta1 = np.array([[deltaAP.item(0)], [deltaAP.item(1)], [deltaAP.item(2)], [deltaAP.item(3)]])
        fm1, Va1 = forcesm.compute(dynamics._state, delta1)
        dynamics.update(fm1)
        chi = dynamics.get_chi()
        
        # MAV2
        if sim_time < 35: 
            Va_c2 = 40 # Set commanded airspeed
            h_c2, chi_c2 = path_follow2.update(path2, dynamics2._state2) # Get commanded altitude, commanded course
        elif sim_time < 40:
            Va_c2 = 35 # Set commanded airspeed
            h_c2, chi_c2 = path_follow2.update(path2, dynamics2._state2) # Get commanded altitude, commanded course
        else: 
            Va_c2 = 40
            chi_c2 = np.deg2rad(15) # Set commanded course
        Va = dynamics2.getVa()
        pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = dynamics2._state2.flatten()
        u_state1 = np.array([sim_time, phi, theta, psi, p, q, r, Va, pd, Va_c2, h_c2, chi_c2])
        deltaAP, phi_c, theta_c, chi_c, altitude_state = Autopilot2.autopilot(gains, u_state1) 
        delta2 = np.array([[deltaAP.item(0)], [deltaAP.item(1)], [deltaAP.item(2)], [deltaAP.item(3)]])
        fm2, Va2 = forcesm.compute(dynamics2._state2, delta2)
        dynamics2.update(fm2)
        chi2 = dynamics2.get_chi()
        
        sim_time += P.ts_simulation
    plt.pause(0.01)

    # Update animations and plots
    anim.update(dynamics._state, dynamics2._state2)
    # plot_state.update(sim_time, dynamics._state, dynamics2._state2)
