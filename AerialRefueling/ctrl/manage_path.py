import sys
sys.path.append('.')  
import numpy as np

class Path:
    def __init__(self, desired_distance, altitude_tolerance, kp_course, kp_speed, ki_speed, kd_speed):

        self.desired_distance   = desired_distance
        self.altitude_tolerance = altitude_tolerance
        self.kp_course = kp_course
        self.kp_speed  = kp_speed
        self.ki_speed  = ki_speed
        self.kd_speed  = kd_speed
        self.integral_error = 0.0
        self.previous_relative_position = np.array([0.0, 0.0])


    def update_path(self, leader_state, follower_state, dt):

        # Extract positions and velocities
        position_leader   = np.array([leader_state.item(0), leader_state.item(1)])
        position_follower = np.array([follower_state.item(0), follower_state.item(1)])
        velocity_leader   = np.array([leader_state.item(3), leader_state.item(4)])
        # Calculate the relative position
        relative_position = position_leader - position_follower
        distance_error = np.linalg.norm(relative_position) - self.desired_distance
        # Update the integral error for the PID controller
        self.integral_error += distance_error * dt
        # Calculate the derivative error for the PID controller
        derivative_error = (relative_position - self.previous_relative_position) / dt
        self.previous_relative_position = relative_position
        # Calculate the PID control for speed to maintain the desired distance
        speed_control = (self.kp_speed * distance_error + 
                         self.ki_speed * self.integral_error + 
                         self.kd_speed * np.dot(derivative_error, relative_position/np.linalg.norm(relative_position))
        )
        # Calculate the desired speed for the follower
        Va_leader = np.linalg.norm(velocity_leader)
        Va_follower_desired = Va_leader - speed_control
        # Calculate the desired course for the follower
        desired_position_behind = position_leader - self.desired_distance * \
                                (velocity_leader / np.linalg.norm(velocity_leader))
        bearing_to_desired_position = np.arctan2(desired_position_behind[1] - position_follower[1], 
                                                 desired_position_behind[0] - position_follower[0])
        course_error = bearing_to_desired_position - np.arctan2(velocity_leader[1], velocity_leader[0])
        course_control = self.kp_course * course_error
        # Calculate the desired course angle
        chi_follower_desired = np.rad2deg(bearing_to_desired_position + course_control)
        # Check altitude tolerance and set desired altitude if necessary
        if abs(follower_state.item(2) - leader_state.item(2)) > self.altitude_tolerance:
            h_follower_desired = -leader_state.item(2)
        else:
            h_follower_desired = follower_state.item(2)

        return Va_follower_desired, chi_follower_desired, h_follower_desired