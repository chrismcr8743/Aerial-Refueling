import sys
sys.path.append('.')
import numpy as np
import parameters.aerosonde_parameters as P
import control.matlab as mat
from control.matlab import tf, lsim
from tools.rotations import Euler2Rotation


class MavDynamics2:
    def __init__(self):
        self.ts_simulation = P.ts_simulation
        self.jx = P.jx
        self.jy = P.jy
        self.jz = P.jz
        self.jxz = P.jxz
        self.mass = P.mass
        self.gravity = P.gravity
        self._state2=P.state0

    def update(self, forces_moments):
        '''
            Integrate the differential equations defining dynamics. 
            Inputs are the forces and moments on the aircraft.
            Ts is the time step between function calls.
        '''

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self.ts_simulation

        k1 = self.f(self._state2, forces_moments)
        k2 = self.f(self._state2 + time_step / 2. * k1, forces_moments)
        k3 = self.f(self._state2 + time_step / 2. * k2, forces_moments)
        k4 = self.f(self._state2 + time_step * k3, forces_moments)

        self._state2 += time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def f(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # Extract the states
        u     = state.item(3)
        v     = state.item(4)
        w     = state.item(5)
        phi   = state.item(6)
        theta = state.item(7)
        psi   = state.item(8)
        p     = state.item(9)
        q     = state.item(10)
        r     = state.item(11)
        # Extract the forces and moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l_in = forces_moments.item(3)
        m_in = forces_moments.item(4)
        n_in = forces_moments.item(5)
        body_vel = np.array([u, v, w]).T
        inertial_vel = Euler2Rotation(phi, theta, psi) @ body_vel
  
        m = P.mass
        g = P.gravity
        ForceVecBody = 1 / m * np.array([[fx], [fy], [fz]], dtype=float)
        jx = self.jx
        jy = self.jy
        jz = self.jz
        jxz = self.jxz
        gamma = jx * jz - jxz ** 2
        gamma1 = (jxz * (jx - jy + jz)) / gamma
        gamma2 = (jz * (jz - jy) + jxz ** 2) / gamma
        gamma3 = jz / gamma
        gamma4 = jxz / gamma
        gamma5 = (jz - jx) / jy
        gamma6 = jxz / jy
        gamma7 = ((jx - jy) * jx + jxz ** 2) / gamma
        gamma8 = jx / gamma

        north_dot = inertial_vel[0]
        east_dot = inertial_vel[1]
        down_dot = inertial_vel[2]

        # Position dynamics 
        temp1 = np.array([[r * v - q * w],
                          [p * w - r * u],
                          [q * u - p * v]]) + ForceVecBody

        u_dot = temp1[0][0]
        v_dot = temp1[1][0]
        w_dot = temp1[2][0]

        # Rotational kinematics
        ang_vel = np.array([[p], [q], [r]], dtype=float)
        Rgb = np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]])
        temp2 = Rgb @ ang_vel

        phi_dot = temp2[0][0]
        theta_dot = temp2[1][0]
        psi_dot = temp2[2][0]

        temp3 = np.array([[gamma1 * p * q - gamma2 * q * r], [gamma5 * p * r - gamma6 * (p ** 2 - r ** 2)],
                          [gamma7 * p * q - gamma1 * q * r]]) + np.array(
            [[gamma3 * l_in + gamma4 * n_in], [(1 / jy) * m_in], [gamma4 * l_in + gamma8 * n_in]], dtype=float)
        p_dot = temp3[0][0]
        q_dot = temp3[1][0]
        r_dot = temp3[2][0]

        # Collect the derivative of the states
        x_dot = np.array([[north_dot], [east_dot], [down_dot], [u_dot], [v_dot], [w_dot],
                          [phi_dot], [theta_dot], [psi_dot], [p_dot], [q_dot], [r_dot]], dtype=float)

        return x_dot

    def get_chi(self):
        # Extract the necessary states for computing chi
        u     = self._state2.item(3)
        v     = self._state2.item(4)
        w     = self._state2.item(5)
        phi   = self._state2.item(6)
        theta = self._state2.item(7)
        psi   = self._state2.item(8)
        # Compute the rotation matrix from body to NED frame
        R = Euler2Rotation(phi, theta, psi)
        # Convert body frame velocities to NEDcframe
        V_ned   = np.dot(R, np.array([u, v, 0]).T)  
        V_north = V_ned[0]
        V_east  = V_ned[1]
        # Compute the course angle chi
        chi = np.arctan2(V_east, V_north)
        return chi
    
    def getVa(self):
        return np.sqrt(self._state2.item(3)**2 + self._state2.item(4)**2 + self._state2.item(5)**2)