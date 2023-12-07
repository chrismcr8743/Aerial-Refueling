import sys
sys.path.append('.')  
import numpy as np
import parameters.aerosonde_parameters as P
from math import atan, atan2, sin, cos, tan
from tools.rotations import Euler2Rotation

class PathFollower:
    def __init__(self):
        self.chi_inf = np.radians(40)  # Approach angle for line path following
        self.k_path  = 0.03            # Gain for line path following
        self.k_orbit = .01             # Gain for orbit path following
        self.gravity = P.gravity       # Gravitational constant

    def update(self, path, state):
        if path.flag == 'line':
            return self._follow_straight_line(path, state)
        elif path.flag == 'orbit':
            return self._follow_orbit(path, state)
        else:
            raise ValueError('Path type not supported')

    def _follow_straight_line(self, path, state):

        # States for computing chi
        u     = state.item(3)  
        v     = state.item(4)  
        phi   = state.item(6)  
        theta = state.item(7)  
        psi   = state.item(8) 
        # Compute the rotation matrix from body to NED frame
        R = Euler2Rotation(phi, theta, psi)
        # Convert body frame velocities to NEDcframe
        V_ned = np.dot(R, np.array([u, v, 0]).T)  
        V_north = V_ned[0]
        V_east = V_ned[1]
        # Compute the course angle chi
        chi = np.arctan2(V_east, V_north)
        # Compute the cross-track error
        chi_q = atan2(path.line_direction.item(1), path.line_direction.item(0))
        chi_q = self._wrap(chi_q, chi)
        ep = np.array([state.item(0), state.item(1), -state.item(2)]) - path.line_origin
        path_error = -sin(chi_q) * ep.item(0) + cos(chi_q) * ep.item(1)
        # Compute the course command
        chi_c = chi_q - self.chi_inf * (2 / np.pi) * np.arctan(self.k_path * path_error)
        # Compute the altitude command
        h_c = -path.line_origin.item(2) - (ep.item(0)**2 + ep.item(1)**2) * path.line_direction.item(2) / np.linalg.norm(path.line_direction[0:2])

        return h_c, chi_c

    def _follow_orbit(self, path, state):

        # States for computing chi
        u     = state.item(3)  
        v     = state.item(4)  
        phi   = state.item(6)  
        theta = state.item(7)  
        psi   = state.item(8) 
        # Compute the rotation matrix from body to NED frame
        R = Euler2Rotation(phi, theta, psi)
        # Convert body frame velocities to NEDcframe
        V_ned = np.dot(R, np.array([u, v, 0]).T)  
        V_north = V_ned[0]
        V_east = V_ned[1]
        # Compute the course angle chi
        chi = np.arctan2(V_east, V_north)
        # Determine orbit direction
        direction = 1.0 if path.orbit_direction == 'CW' else -1.0
        # Compute distance from orbit center
        d = np.linalg.norm(np.array([state.item(0), state.item(1)]) - path.orbit_center[0:2])
        # Compute the orbit error
        orbit_error = (d - path.orbit_radius) / path.orbit_radius
        # Compute the angular position on the orbit
        varphi = np.arctan2(state.item(1) - path.orbit_center.item(1), state.item(0) - path.orbit_center.item(0))
        varphi = self._wrap(varphi, chi)
        # Compute the course command
        chi_c = varphi + direction * (np.pi/2.0 + np.arctan(self.k_orbit * orbit_error))
        # Compute the altitude command
        h_c = -path.orbit_center.item(2)

        return h_c, chi_c

    def _wrap(self, angle, ref):
        # Wrap angle to [-pi, pi) relative to ref
        while angle - ref > np.pi:
            angle -= 2.0 * np.pi
        while angle - ref < -np.pi:
            angle += 2.0 * np.pi
        return angle
