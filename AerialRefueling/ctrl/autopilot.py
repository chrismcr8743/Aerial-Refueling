import sys
sys.path.append('.')
import numpy as np
import parameters.aerosonde_parameters as P


class AutoPilot:
    def _init_(self):
        self.course_integrator = 0.0
        self.course_differentiator = 0.0
        self.course_error_d1 = 0.0

    def autopilot(self, gains, u):
        global dt
        global altitude_state
        global initialize_integrator
        global altitude_take_off_zone
        global altitude_hold_zone
        t     = u[0]
        phi   = u[1]
        theta = u[2]
        chi   = u[3]
        p     = u[4]
        q     = u[5]
        r     = u[6]
        Va    = u[7]
        h     =-u[8]
        Va_c  = u[9]
        h_c   = u[10]
        chi_c = u[11]
        self.gains = gains
        dt = P.ts_simulation
        altitude_take_off_zone = 60
        altitude_hold_zone = 50
        
        # Lateral Autopilot
        if t == 0:
            delta_r = 0
            phi_c = self.course_hold(chi_c, chi, r, 1, dt)
            delta_a = self.roll_hold(phi_c, phi, p, 1, dt)
        else:
            delta_r = 0
            phi_c = self.course_hold(chi_c, chi, r, 0, dt)
            delta_a = self.roll_hold(phi_c, phi, p, 0, dt)

        # Longitudinal Autopilot
        if t == 0:
            if h <= altitude_take_off_zone:
                altitude_state = 0
            elif h <= h_c - altitude_hold_zone:
                altitude_state = 1
            elif h >= h_c + altitude_hold_zone:
                altitude_state = 2
            else:
                altitude_state = 3
            initialize_integrator = 1
        
        # Takeoff
        if altitude_state == 0:
            delta_t = 1 
            theta_c = 10*(np.pi/180)
            if h >= altitude_take_off_zone:
                altitude_state = 1
                initialize_integrator = 1
            else:
                initialize_integrator = 0
        
        # Climb
        elif altitude_state == 1:
            delta_t = 1
            theta_c = self.airspeed_hold_pitch(Va_c, Va, initialize_integrator, dt)

            if h >= h_c - altitude_take_off_zone:
                altitude_state = 3
                initialize_integrator = 1
            elif h <= altitude_take_off_zone:
                altitude_state = 0
                initialize_integrator = 1
            else:
                initialize_integrator = 0
        
        # Descent
        elif altitude_state == 2:
            delta_t = 0
            theta_c = self.airspeed_hold_pitch(Va_c, Va, initialize_integrator, dt)
            if h <= h_c + altitude_hold_zone:
                altitude_state = 3
                initialize_integrator = 1
            else:
                initialize_integrator = 0
        
        # Altitude hold
        elif altitude_state == 3:
            delta_t = self.airspeed_hold_throttle(Va_c, Va, initialize_integrator, dt)
            theta_c = self.altitude_hold(h_c, h, initialize_integrator, dt)
            if h <= h_c - altitude_hold_zone:
                altitude_state = 1
                initialize_integrator = 1
            elif h >= h_c + altitude_hold_zone:
                altitude_state = 2
                initialize_integrator = 1
            else:
                initialize_integrator = 0

        if t == 0:
            delta_e = self.pitch_hold(theta_c, theta, q, 1, dt)
        else:
            delta_e = self.pitch_hold(theta_c, theta, q, 0, dt)

        delta = np.array([delta_e, delta_a, delta_r, delta_t])

        return delta, phi_c, theta_c, chi_c, altitude_state

    
    def roll_hold(self, phi_c, phi, p, flag, dt):
        global roll_integrator
        global roll_differentiator
        global roll_error_d1

        limit1 =  0.7854
        limit2 = -0.7854

        kp = self.gains.item(4)
        kd = self.gains.item(5)
        ki = 0.0

        if flag == 1.0:
            roll_integrator     = 0.0
            roll_differentiator = 0.0
            roll_error_d1       = 0.0

        error               = phi_c - phi
        roll_integrator     = roll_integrator + (dt / 2) * (error + roll_error_d1)
        roll_differentiator = p
        roll_error_d1       = error

        u     = kp * error + ki * roll_integrator + kd * roll_differentiator
        u_sat = self.sat(u, limit1, limit2)

        if ki != 0.0:
            roll_integrator = roll_integrator + dt / ki * (u_sat - u)

        return u_sat

    def course_hold(self, chi_c, chi, r, flag, dt):
        global course_integrator
        global course_differentiator
        global course_error_d1

        limit1 =  np.deg2rad(30)  
        limit2 = -np.deg2rad(30)

        kp = self.gains.item(2)
        kd = 0.0
        ki = self.gains.item(3)

        if flag == 1.0:
            course_integrator     = 0.0
            course_differentiator = 0.0
            course_error_d1       = 0.0

        error                 = chi_c - chi
        course_integrator     = course_integrator + (dt/2)*(error+course_error_d1)
        course_differentiator = r
        course_error_d1       = error

        u     = kp*error + ki*course_integrator + kd*course_differentiator
        u_sat = self.sat(u, limit1, limit2)

        if ki != 0.0:
            course_integrator = course_integrator*(dt/ki)*(u_sat-u)

        return u_sat

    def pitch_hold(self, theta_c, theta, q, flag, dt):
        global pitch_hold_integrator
        global pitch_hold_differentiator
        global pitch_hold_error_d1

        limit1 =  0.7854
        limit2 = -0.7854

        kp = self.gains.item(0)
        kd = self.gains.item(1)
        ki = 0.0

        if flag == 1.0:
            pitch_hold_integrator     = 0.0
            pitch_hold_differentiator = 0.0
            pitch_hold_error_d1       = 0.0

        error                     = theta_c - theta
        pitch_hold_integrator     = pitch_hold_integrator + (dt / 2) * (error - pitch_hold_error_d1)
        pitch_hold_differentiator = q
        pitch_hold_error_d1       = error

        u     = kp * error + ki * pitch_hold_integrator + kd * pitch_hold_differentiator
        u_sat = self.sat(u, limit1, limit2)

        if ki != 0.0:
            pitch_hold_integrator = pitch_hold_integrator + dt / ki * (u_sat - u)

        return u_sat

    def altitude_hold(self, h_c, h, flag, dt):
        global altitude_integrator
        global altitude_differentiator
        global altitude_error_d1

        tau    =  5.0
        limit1 =  0.7854
        limit2 = -0.7854

        kp = self.gains.item(10)
        kd = 0.0
        ki = self.gains.item(11)

        if flag == 1.0:
            altitude_integrator     = 0.0
            altitude_differentiator = 0.0
            altitude_error_d1       = 0.0

        error                   = h_c - h
        altitude_integrator     = altitude_integrator+(dt/2)*(error-altitude_error_d1)
        altitude_differentiator = (2*tau - dt) / (2*tau + dt) * altitude_differentiator + 2/(2*tau + dt)*(error - altitude_error_d1)
        altitude_error_d1       = error

        u     = kp*error + ki*altitude_integrator + kd*altitude_differentiator
        u_sat = self.sat(u, limit1, limit2)

        if ki != 0.0:
            altitude_integrator = altitude_integrator + dt / ki * (u_sat - u)

        return u_sat

    def airspeed_hold_pitch(self, Va_c, Va, flag, dt):
        global airspeed_pitch_integrator
        global airspeed_pitch_differentiator
        global airspeed_pitch_error_d1

        tau    =  5.0
        limit1 =  np.deg2rad(45)
        limit2 = -np.deg2rad(45)

        kp = self.gains.item(6)
        kd = 0.0
        ki = self.gains.item(7)

        if flag == 1.0:
            airspeed_pitch_integrator     = 0.0
            airspeed_pitch_differentiator = 0.0
            airspeed_pitch_error_d1       = 0.0

        error                         = (Va_c - Va)
        airspeed_pitch_integrator     = airspeed_pitch_integrator + (dt/2)*(error + airspeed_pitch_error_d1)
        airspeed_pitch_differentiator = (2*tau - dt)/(2*tau + dt)*airspeed_pitch_differentiator + 2/(2*tau+dt)*(error-airspeed_pitch_error_d1)

        airspeed_pitch_error_d1       = error

        u = kp*error + ki*airspeed_pitch_integrator + kd*airspeed_pitch_differentiator
        u_sat = self.sat(u, limit1, limit2)

        if ki != 0.0:
            airspeed_pitch_integrator = airspeed_pitch_integrator + dt/ki * (u_sat-u)

        return u_sat*-1.0


    def airspeed_hold_throttle(self, Va_c, Va, flag, dt):
        global airspeed_throttle_integrator
        global airspeed_throttle_differentiator
        global airspeed_throttle_error_d1

        tau    = 5.0
        limit1 = 1.0
        limit2 = 0.0

        kp = self.gains.item(8)
        kd = 0.0
        ki = self.gains.item(9)

        if flag == 1.0:
            airspeed_throttle_integrator     = 0.0
            airspeed_throttle_differentiator = 0.0
            airspeed_throttle_error_d1       = 0.0

        error                            = (Va_c - Va)
        airspeed_throttle_integrator     = airspeed_throttle_integrator + (dt / 2) * (error + airspeed_throttle_error_d1)        
        airspeed_throttle_differentiator = (2*tau - dt)/(2*tau + dt) * airspeed_throttle_differentiator + 2/(2*tau+dt) * (error - airspeed_throttle_error_d1)

        airspeed_throttle_error_d1       = error

        u     = kp * error + ki * airspeed_throttle_integrator + kd * airspeed_throttle_differentiator
        
        u_sat = self.sat(u, limit1, limit2)

        if ki != 0.0:
            airspeed_throttle_integrator = airspeed_throttle_integrator + dt / ki * (u_sat - u)

        return u_sat

    def sat(self,inn,up_limit,low_limit):
        if inn>up_limit:
            out=up_limit
        elif inn<low_limit:
            out=low_limit
        else:
            out=inn
        return out
