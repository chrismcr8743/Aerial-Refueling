import sys
sys.path.append('.')
import numpy as np
import parameters.aerosonde_parameters as P
import control.matlab as mat
from control.matlab import tf, lsim

class TransferFunctions:
    def compute_tf_models(self, x_trim, u_trim):
        Va_trim = np.sqrt(x_trim[3,0]**2 + x_trim[4,0]**2 +x_trim[5,0]**2)
        alpha_trim = np.arctan(x_trim[5,0]/x_trim[3,0])
        theta_trim = x_trim[7,0]
        # Define transfer function constants
        gamma  = P.jx*P.jz-P.jxz**2
        gamma3 = P.jz/gamma
        gamma4 = P.jxz/gamma
        beta   = np.arcsin(x_trim[4,0]/Va_trim)

        C_p_p    = gamma3*P.C_ell_p+gamma4+P.C_n_p
        a_phi1   = -(1/2)*P.rho*Va_trim**2*P.S_wing*P.b*C_p_p*(P.b/(2*Va_trim))
        C_p_d_a  = gamma3*P.C_ell_delta_a+gamma4*P.C_n_delta_a
        a_phi2   =  (1/2)*P.rho*Va_trim**2*P.S_wing*P.b*C_p_d_a
        a_theta1 = -((P.rho*Va_trim**2*P.c*P.S_wing)/(2*P.jy))*P.C_m_q*(P.c/(2*Va_trim))
        a_theta2 = -((P.rho*Va_trim**2*P.c*P.S_wing)/(2*P.jy))*P.C_m_alpha
        a_theta3 =  ((P.rho*Va_trim**2*P.c*P.S_wing)/(2*P.jy))*P.C_m_delta_e
        a_V1     = ((P.rho*Va_trim*P.S_wing)/P.mass)* \
                    (P.C_D_0+P.C_D_alpha+P.C_D_delta_e*u_trim[0,0])+ \
                   ((P.rho*P.S_prop)/P.mass)* \
                     P.C_prop*Va_trim
        a_V2     = ((P.rho*P.S_prop)/P.mass)*P.C_prop*P.k_motor**2*u_trim[1,0]
        a_V3     =   P.gravity
        a_beta1  = -((P.rho*Va_trim*P.S_wing)/(2*P.mass*np.cos(beta)))*P.C_Y_beta
        a_beta2  =  ((P.rho*Va_trim*P.S_wing)/(2*P.mass*np.cos(beta)))*P.C_Y_delta_r
        
        self.a_phi1   = a_phi1
        self.a_phi2   = a_phi2
        self.a_theta1 = a_theta1
        self.a_theta2 = a_theta2
        self.a_theta3 = a_theta3
        self.a_V1     = a_V1
        self.a_V2     = a_V2
        
        # Define transfer functions
        T_phi_delta_a   = tf([a_phi2],[1,a_phi1,0])
        T_chi_phi       = tf([P.gravity/Va_trim],[1,0])
        T_theta_delta_e = tf(a_theta3,[1,a_theta1,a_theta2])
        T_h_theta       = tf([Va_trim],[1,0])
        T_h_Va          = tf([theta_trim],[1,0])
        T_Va_delta_t    = tf([a_V2],[1,a_V1])
        T_Va_theta      = tf([-a_V3],[1,a_V1])
        T_beta_delta_r  = tf([a_beta2],[1,a_beta1])
        print('................Open Loop Transfer Functions.............')
        print('T_phi_delta_a=', T_phi_delta_a)
        print('T_theta_delta_e=', T_theta_delta_e)
        print('T_h_theta=', T_h_theta)
        print('T_beta_delta_r =', T_beta_delta_r)
        print('T_phi_delta_a=', T_phi_delta_a)
        return(T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta,  T_h_Va, T_Va_delta_t, T_Va_theta, T_Va_theta, T_beta_delta_r)
    

    def state_space(self, x_trim, u_trim):

        Va = np.sqrt(x_trim.item(3)**2 + x_trim.item(4)**2 + x_trim.item(5)**2)
        alpha = np.arctan(x_trim.item(5)/x_trim.item(3))
        beta = np.arctan(x_trim.item(4)/Va)

        gamma  =   P.jx * P.jz - P.jxz ** 2
        gamma1 =  (P.jxz * (P.jx - P.jy + P.jz)) / gamma
        gamma2 =  (P.jz * (P.jz - P.jy) + P.jxz ** 2) / gamma
        gamma3 =   P.jz / gamma
        gamma4 =   P.jxz / gamma
        gamma5 =  (P.jz - P.jx) / P.jy
        gamma6 =   P.jxz / P.jy
        gamma7 = ((P.jx - P.jy) * P.jx + P.jxz ** 2) / gamma
        gamma8 =   P.jx / gamma

        C_p_0       = gamma3*P.C_ell_0 + gamma4*P.C_n_0
        C_p_Beta    = gamma3*P.C_ell_beta + gamma4*P.C_n_beta
        C_p_p       = gamma3*P.C_ell_p + gamma4*P.C_n_p
        C_p_r       = gamma3*P.C_ell_r + gamma4*P.C_n_r
        C_p_delta_a = gamma3*P.C_ell_delta_a + gamma4*P.C_n_delta_a
        C_p_delta_r = gamma3*P.C_ell_delta_r + gamma4*P.C_n_delta_r
        C_r_0       = gamma4*P.C_ell_0 + gamma8*P.C_n_0
        C_r_Beta    = gamma4*P.C_ell_beta + gamma8*P.C_n_beta
        C_r_p       = gamma4*P.C_ell_p + gamma8*P.C_n_p
        C_r_r       = gamma4*P.C_ell_r + gamma8*P.C_n_r
        C_r_delta_a = gamma4*P.C_ell_delta_a + gamma8*P.C_n_delta_a
        C_r_delta_r = gamma4*P.C_ell_delta_r + gamma8*P.C_n_delta_r

        Y_v       = ((P.rho*P.S_wing*x_trim.item(4))/(4*P.M*Va))*(P.C_Y_p*x_trim.item(9) + P.C_Y_r*x_trim.item(11)) + ((P.rho*P.S_wing*x_trim.item(4))/P.M)*(P.C_Y_0 + P.C_Y_beta*beta + P.C_Y_delta_a*u_trim[1,0] + P.C_Y_delta_r*u_trim[2,0]) + ((P.rho*P.S_wing*P.C_Y_beta)/(2*P.M))*np.sqrt(x_trim.item(3)**2 + x_trim.item(5)**2)
        Y_p       =  x_trim.item(5) + ((P.rho*Va*P.S_wing*P.b)/(4*P.M))*P.C_Y_p
        Y_r       = -x_trim.item(3) + ((P.rho*Va*P.S_wing*P.b)/(4*P.M))*P.C_Y_r
        Y_delta_a = ((P.rho*Va**2*P.S_wing)/(2*P.M)) * P.C_Y_delta_a
        Y_delta_r = ((P.rho*Va**2*P.S_wing)/(2*P.M)) * P.C_Y_delta_r
        L_v       = ((P.rho*P.S_wing*P.b**2*x_trim.item(4))/(4*Va))*(C_p_p*x_trim.item(9) + C_p_r*x_trim.item(11)) + (P.rho*P.S_wing*P.b*x_trim.item(4))*(C_p_0 + C_p_Beta*beta + C_p_delta_a*u_trim[1,0] + C_p_delta_r*u_trim[2,0]) + (P.rho*P.S_wing*P.b*C_p_Beta/2)*np.sqrt(x_trim.item(3)**2 + x_trim.item(5)**2)
        L_p       =  gamma1*x_trim.item(10) + (P.rho*Va*P.S_wing*P.b**2/4)*C_p_p
        L_r       = -gamma2*x_trim.item(10) + (P.rho*Va*P.S_wing*P.b**2/4)*C_p_r
        L_delta_a = (P.rho*Va**2*P.S_wing*P.b/2)*C_p_delta_a
        L_delta_r = (P.rho*Va**2*P.S_wing*P.b/2)*C_p_delta_r
        N_v       = ((P.rho*P.S_wing*P.b**2*x_trim.item(4))/(4*Va))*(C_r_p*x_trim.item(9) + C_r_r*x_trim.item(11)) + (P.rho*P.S_wing*P.b*x_trim.item(4))*(C_r_0 + C_r_Beta*beta + C_r_delta_a*u_trim[1,0] + C_r_delta_r*u_trim[2,0]) + (P.rho*P.S_wing*P.b*C_r_Beta/2)*np.sqrt(x_trim.item(3)**2 + x_trim.item(5)**2)
        N_p       =  gamma7*x_trim.item(10) + (P.rho*Va*P.S_wing*P.b**2 / 4)*C_r_p
        N_r       = -gamma1*x_trim.item(10) + (P.rho*Va*P.S_wing*P.b**2 / 4)*C_r_r
        N_delta_a = (P.rho*Va**2*P.S_wing*P.b / 2)*C_r_delta_a
        N_delta_r = (P.rho*Va**2*P.S_wing*P.b / 2)*C_r_delta_r

        C_D         = P.C_D_0 + (P.C_D_alpha * alpha)
        C_L         = P.C_L_0 + (P.C_L_alpha * alpha)
        C_x_a       = -P.C_D_alpha * np.cos(alpha) + P.C_L_alpha * np.sin(alpha)
        C_x_0       = -P.C_D_0 * np.cos(alpha) + P.C_L_0 * np.sin(alpha)
        C_x_d_e     = -P.C_D_delta_e * np.cos(alpha) + P.C_L_delta_e * np.sin(alpha)
        C_x_q       = -P.C_D_q * np.cos(alpha) + P.C_L_q * np.sin(alpha)
        C_Z         = -C_D * np.sin(alpha) - C_L * np.cos(alpha)
        C_Z_q       = -P.C_D_q * np.sin(alpha) - P.C_L_q * np.cos(alpha)
        C_Z_delta_e = -P.C_D_delta_e * np.sin(alpha) - P.C_L_delta_e * np.cos(alpha)
        C_Z_0       = -P.C_D_0 * np.sin(alpha) - P.C_L_0 * np.cos(alpha)
        C_Z_alpha   = - P.C_D_alpha * np.sin(alpha) - P.C_L_alpha * np.cos(alpha)

        X_u       = ((x_trim.item(3) * P.rho * P.S_wing) / P.M) * (C_x_0 + (C_x_a * u_trim[1,0]) + (C_x_d_e * u_trim[0,0])) - ((P.rho * P.S_wing * x_trim.item(5) * C_x_a) / (2 * P.M)) + ((P.rho * P.S_wing * P.c * C_x_q * x_trim.item(3) * x_trim.item(10)) / (4 * P.M * Va)) - ((P.rho * P.S_prop * P.C_prop *x_trim.item(3)) / P.M)
        X_w       = -x_trim.item(10) + ((x_trim.item(5) * P.rho * P.S_wing) / P.M) * (C_x_0 + (C_x_a * u_trim[1,0]) + (C_x_d_e * u_trim[0,0])) + ((P.rho * P.S_wing * P.c * C_x_q * x_trim.item(5) * x_trim.item(10)) / (4 * P.M * Va)) + ((P.rho * P.S_wing * x_trim.item(3) * C_x_a) / (2 * P.M)) - ((P.rho * P.S_prop * P.C_prop * x_trim.item(5)) / P.M)
        X_q       = -x_trim.item(5) + ((P.rho * Va * P.S_wing * C_x_q * P.c) / (4 * P.M))
        X_delta_e = (P.rho * (Va ** 2) * P.S_wing * C_x_d_e) / (2 * P.M)
        X_delta_t = (P.rho * P.S_prop * P.C_prop * (P.k_motor ** 2) * u_trim[3,0]) / P.M
        Z_u       = x_trim.item(10) + ((x_trim.item(3) * P.rho * P.S_wing) / (P.M)) * (C_Z_0 + (C_Z_alpha * alpha) + (C_Z_delta_e * u_trim[0,0])) - ((P.rho * P.S_wing * C_Z_alpha *x_trim.item(5)) / (2 * P.M)) + ((x_trim.item(3) * P.rho * P.S_wing * C_Z_q * P.c * x_trim.item(10)) / (4 * P.M * Va))
        Z_w       = ((x_trim.item(5) * P.rho * P.S_wing) / (P.M)) * (C_Z_0 + (C_Z_alpha * alpha) + (C_Z_delta_e * u_trim[0,0])) + ((P.rho * P.S_wing * C_Z_alpha * x_trim.item(3)) / (2 * P.M)) + ((x_trim.item(5) * P.rho * P.S_wing * C_Z_q * P.c * x_trim.item(10)) / (4 * P.M * Va))
        Z_q       = x_trim.item(3) + (P.rho * Va * P.S_wing * C_Z_q * P.c) / (4 * P.M)
        Z_delta_e = (P.rho * (Va ** 2) * P.S_wing * C_Z_delta_e) / (2 * P.M)
        M_u       = ((x_trim.item(3) * P.rho * P.S_wing * P.c) / P.jy) * (P.C_m_0 + (P.C_m_alpha * alpha) + (P.C_m_delta_e * u_trim[0,0])) - ((P.rho * P.S_wing * P.c * P.C_m_alpha * x_trim.item(5)) / (2 * P.jy)) + ((P.rho * P.S_wing * (P.c ** 2) * P.C_m_q * x_trim.item(10) * x_trim.item(3)) / (4 * P.jy * Va))
        M_w       = ((x_trim.item(5) * P.rho * P.S_wing * P.c) / P.jy) * (P.C_m_0 + P.C_m_alpha * alpha + P.C_m_delta_e * u_trim[0,0]) + ((P.rho * P.S_wing * P.c * P.C_m_alpha * x_trim.item(3)) / (2 * P.jy)) + ((P.rho * P.S_wing * P.c ** 2 * P.C_m_q * x_trim.item(10) * x_trim.item(5)) / (4 * P.jy * Va))
        M_q       = (P.rho * Va * P.c ** 2 * P.S_wing * P.C_m_q) / (4 * P.jy)
        M_delta_e = (P.rho * (Va ** 2) * P.S_wing * P.c * P.C_m_delta_e) / (2 * P.jy)

        Alat = np.array([[Y_v, Y_p, Y_r, P.gravity*np.cos(x_trim.item(7))*np.cos(x_trim.item(6)), 0],
                            [L_v, L_p, L_r, 0, 0], 
                            [N_v, N_p, N_r, 0, 0],
                            [0, 1, np.cos(x_trim.item(6))*np.tan(x_trim.item(7)), x_trim.item(10)*np.cos(x_trim.item(6))*np.tan(x_trim.item(7))-x_trim.item(11)*np.sin(x_trim.item(6))*np.tan(x_trim.item(7)), 0],
                            [0, 0, np.cos(x_trim.item(6))*(1/np.cos(x_trim.item(7))), x_trim.item(9)*np.cos(x_trim.item(6))*(1/np.cos(x_trim.item(7))) - x_trim.item(11)*np.sin(x_trim.item(6))*(1/np.cos(x_trim.item(7))), 0]
        ]) 
        Blat = np.array([[Y_delta_a, Y_delta_r], 
                            [L_delta_a, L_delta_r],
                            [N_delta_a, N_delta_r], 
                            [0, 0], 
                            [0, 0]
        ])
        Alon = np.array([[X_u, X_w, X_q, -P.gravity*np.cos(x_trim.item(7)), 0],
                            [Z_u, Z_w, Z_q, -P.gravity*np.sin(x_trim.item(7)), 0],
                            [M_u, M_w, M_q, 0, 0],
                            [0, 0, 1, 0, 0],
                            [np.sin(x_trim.item(7)), -np.cos(x_trim.item(7)), 0, x_trim.item(3)*np.cos(x_trim.item(7)) + x_trim.item(5)*np.sin(x_trim.item(7)), 0]
        ])
        Blong = np.array([[X_delta_e, X_delta_t], 
                            [Z_delta_e, 0], 
                            [M_delta_e, 0], 
                            [0, 0], 
                            [0, 0]
        ])


        # Roll
        zeta    = 0.707
        tr_roll = 0.5
        wn_roll = 2.2/tr_roll
        kp_roll = wn_roll**2/self.a_phi1
        kd_roll = (2*zeta*wn_roll-self.a_phi1)/self.a_phi2
        ki_roll = 0.

        # Course hold
        zeta_course = 0.5
        wn_course   = 1
        kp_course   = (2*zeta_course*wn_course*Va)/P.gravity
        ki_course   = (wn_course**2*Va)/P.gravity
        kd_course   = 0.

        # Pitch attitude hold
        zeta_pitch = 0.1
        tr_pitch   = 0.1
        wn_pitch   = 2.2/tr_pitch
        kp_pitch   = (wn_pitch**2-self.a_theta2)/self.a_theta3
        kd_pitch   = (2*zeta_pitch*wn_pitch-self.a_theta1)/self.a_theta3
        ki_pitch   = 0.
        ktheta_DC  = (kp_pitch*self.a_theta3)/(self.a_theta2+kp_pitch*self.a_theta3)

        # Altitude from pitch gain
        tr_altitude = 1.
        wn_altitude = 2.2/tr_altitude
        kp_altitude = (2*zeta*wn_altitude)/(ktheta_DC*Va)
        kd_altitude = 0.
        ki_altitude = wn_altitude**2/(ktheta_DC*Va)

        # Airspeed from pitch
        tr_airspeed = 0.01
        wn_airspeed = 2.2/tr_airspeed
        kp_airspeed = (self.a_V1-2*zeta*wn_airspeed)/ktheta_DC
        kd_airspeed = 0.
        ki_airspeed = (wn_airspeed**2)/(ktheta_DC*P.gravity)

        # Airspeed from throttle
        tr_throttle   = 15
        wn_throttle   = 2.2 / tr_throttle
        kp_throttle   = (2 * zeta * wn_throttle - self.a_V1)/self.a_V2
        kd_throttle   = 0.
        ki_throttle   = wn_throttle**2 / self.a_V2


        header = "{:<10} {:>10} {:>10} {:>10}".format("Gains", "kp", "ki", "kd")
        divider = "-" * 45
        print(divider)
        print(header)
        print(divider)
        print("{:<10} {:10.4f} {:10.4f} {:10.4f}".format("Roll", kp_roll, ki_roll, kd_roll))
        print("{:<10} {:10.4f} {:10.4f} {:10.4f}".format("Pitch", kp_pitch, ki_pitch, kd_pitch))
        print("{:<10} {:10.4f} {:10.4f} {:10.4f}".format("Course", kp_course, ki_course, kd_course))
        print("{:<10} {:10.4f} {:10.4f} {:10.4f}".format("Airspeed", kp_airspeed, ki_airspeed, kd_airspeed))
        print("{:<10} {:10.4f} {:10.4f} {:10.4f}".format("Throttle", kp_throttle, ki_throttle, kd_throttle))
        print("{:<10} {:10.4f} {:10.4f} {:10.4f}".format("Altitude", kp_altitude, ki_altitude, kd_altitude))
        print(divider)

        return np.array([kp_pitch, kd_pitch, kp_course, ki_course, kp_roll, kd_roll, kp_airspeed, ki_airspeed, kp_throttle, ki_throttle, kp_altitude, ki_altitude])
    
