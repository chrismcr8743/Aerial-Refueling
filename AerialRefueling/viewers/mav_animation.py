"""
Class for plotting a uav

Author: Raj # 
"""
import sys
sys.path.append('.')# one directory up
from math import cos, sin
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
from tools.rotations import Quaternion2Euler, Quaternion2Rotation, Euler2Rotation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Mav_Animation():
    def __init__(self, scale=0.25):

        
        self.scale=scale
        self.flag_init = True
        fig = plt.figure(2,figsize=(8,8))
        # fig = plt.figure(2, figsize=(16, 8))
        # fig.set_size_inches(16, 16, forward=True)

        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-10,10])
        self.ax.set_ylim([-10,10])
        self.ax.set_zlim([-10,10])
  
        self.ax.set_title('3D Animation')
        self.ax.set_xlabel('East(m)')
        self.ax.set_ylabel('North(m)')
        self.ax.set_zlabel('Height(m)')

        self.cube1 = None  
        self.cube2 = None  

        
        #self.update(state0)
    def cube_vertices(self,pn,pe,pd,phi,theta,psi):
        w=self.scale
        V=np.array([[2.8,0,0],
        [  0.1,  0.5,-0.5],
        [  0.1, -0.5,-0.5],
        [  0.1, -0.5, 0.5],
        [  0.1,  0.5, 0.5],
        [   -5,    0,   0],
        [    0,  3.5,   0],
        [ -1.5,  3.5,   0],
        [ -1.5, -3.5,   0],
        [    0, -3.5,   0],
        [-4.25, 1.65,   0],
        [   -5, 1.65,   0],
        [   -5,-1.65,   0],
        [-4.25,-1.65,   0],
        [-4.25,    0,   0],
        [   -5,    0,-1.5]]) 
        pos_ned=np.array([pn, pe, pd])

        # create m by n copies of pos_ned and used for translation
        ned_rep= np.tile(pos_ned, (16,1)) # 8 vertices # 21 vertices for UAV

        R=Euler2Rotation(phi,theta,psi)

        #rotate 
        vr=np.matmul(R,V.T).T
        # translate
        vr=vr+ned_rep
        # rotate for plotting north=y east=x h=-z
        R_plot=np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, -1]])
        
        vr=np.matmul(R_plot,vr.T).T

        Vl=vr.tolist()
        Vl1=Vl[0]
        Vl2=Vl[1]
        Vl3=Vl[2]
        Vl4=Vl[3]
        Vl5=Vl[4]
        Vl6=Vl[5]
        Vl7=Vl[6]
        Vl8=Vl[7]
        Vl9=Vl[8]
        Vl10=Vl[9]
        Vl11=Vl[10]
        Vl12=Vl[11]
        Vl13=Vl[12]
        Vl14=Vl[13]
        Vl15=Vl[14]
        Vl16=Vl[15]




        #f1=. #v1 v2 v3 v4
        #f2  #v5 v6 v7 v8
        #f3= #v3 v4 v8 v7
        #f4 #v2 v1 v5 v6
        #f5 #v1 v4 v8 v5
        #f6 #v3 v7 v6 v2

        verts=[
            [Vl1,Vl2,Vl3],  #face 1
            [Vl1,Vl3,Vl4],  # face 2
            [Vl1,Vl2,Vl3],
            [Vl1,Vl4,Vl5], # face 3
            [Vl2,Vl3,Vl4,Vl5], # face 4
            [Vl3,Vl4,Vl6], # face 5
            [Vl3,Vl2,Vl6],
            [Vl2,Vl6,Vl5],  #face 1
            [Vl5,Vl6,Vl4],  # face 2
            [Vl7,Vl8,Vl9,Vl10], # face 3
            [Vl11,Vl12,Vl13,Vl14], # face 4
            [Vl15,Vl16,Vl6]]  # face 6
        return(verts)    

    

    def update(self, state1, state2):
        # Extract state components for both UAVs
        pn1, pe1, pd1, phi1, theta1, psi1 = self.extract_state_components(state1)
        pn2, pe2, pd2, phi2, theta2, psi2 = self.extract_state_components(state2)
        
        # Update axes limits based on the first UAV's position
        self.update_axes_limits(pn2, pe2, pd2)
        # self.update_axes_limits(pn1, pe1, pd1)


        # Draw both UAVs
        self.draw_cube(pn1, pe1, pd1, phi1, theta1, psi1, 1)
        self.draw_cube(pn2, pe2, pd2, phi2, theta2, psi2, 2)
        # if self.flag_init == True:
        #     self.flag_init = False
    
    def extract_state_components(self, state):
        pn = state[0, 0]
        pe = state[1, 0]
        pd = state[2, 0]
        phi = state[6, 0]
        theta = state[7, 0]
        psi = state[8, 0]
        return pn, pe, pd, phi, theta, psi

    def update_axes_limits(self, pn, pe, pd):
        self.ax.set_xlim([pn - 20, pn + 20])
        self.ax.set_ylim([pe - 20, pe + 20])
        self.ax.set_zlim([-pd - 20, -pd + 20])
    
    def draw_cube(self, pn, pe, pd, phi, theta, psi, uav_id):
        verts = self.cube_vertices(pn, pe, pd, phi, theta, psi)
        if uav_id == 1:
            if self.cube1 is None:
                poly = Poly3DCollection(verts, facecolors=['r'], alpha=.6)
                self.cube1 = self.ax.add_collection3d(poly)
                #plt.pause(0.001)
            else:
                self.cube1.set_verts(verts)
                #plt.pause(0.001)
        elif uav_id == 2:
            if self.cube2 is None:
                poly = Poly3DCollection(verts, facecolors=['b'], alpha=.6)
                self.cube2 = self.ax.add_collection3d(poly)
                plt.pause(0.001)
            else:
                self.cube2.set_verts(verts)
                plt.pause(0.001)
        
          

    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = -self.roll
        pitch = -self.pitch
        yaw = self.yaw
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw), z]
             ])

    def plot(self):  # pragma: no cover
        T = self.transformation_matrix()

        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        #plt.cla() # use handle 
        if self.flag_init is True:
            body, =self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0], p1_t[0], p3_t[0], p4_t[0], p2_t[0]],
                        [p1_t[1], p2_t[1], p3_t[1], p4_t[1], p1_t[1], p3_t[1], p4_t[1], p2_t[1]],
                        [p1_t[2], p2_t[2], p3_t[2], p4_t[2], p1_t[2], p3_t[2], p4_t[2], p2_t[2]], 'k-') # rotor
            self.handle.append(body)

            

            traj, =self.ax.plot(self.x_data, self.y_data, self.z_data, 'b:')# trajectory
            self.handle.append(traj)

            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            self.ax.set_zlim(0, 4)
            plt.xlabel('North')
            plt.ylabel('East')
            self.flag_init = False 
            plt.pause(0.001) # can be put in the main file
        else:
            self.handle[0].set_data([p1_t[0], p2_t[0], p3_t[0], p4_t[0], p1_t[0], p3_t[0], p4_t[0], p2_t[0]],
                        [p1_t[1], p2_t[1], p3_t[1], p4_t[1], p1_t[1], p3_t[1], p4_t[1], p2_t[1]])
            self.handle[0].set_3d_properties([p1_t[2], p2_t[2], p3_t[2], p4_t[2],p1_t[2], p3_t[2], p4_t[2], p2_t[2]])


            self.handle[1].set_data(self.x_data, self.y_data)
            self.handle[1].set_3d_properties(self.z_data)
            print(self.handle)
            plt.pause(0.001)


