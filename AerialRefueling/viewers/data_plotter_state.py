import sys
sys.path.append('.')
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np
from datetime import datetime
import os
plt.ion() 


class dataPlotterState:
    ''' 
        This class plots the time histories for the pendulum data.
    '''

    def __init__(self):
        # Number of subplots = num_of_rows*num_of_cols
        self.num_rows = 12    # Number of subplot rows
        self.num_cols = 1    # Number of subplot columns

        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True, figsize=(6,8))

        # Instantiate lists to hold the time and data histories
        self.time_history = []  
        self.statepn1 = []  
        self.statepn2 = []  
        self.statepe1 = []  
        self.statepe2 = []
        self.statepd1 = [] 
        self.statepd2 = []
        self.stateu1 = []
        self.stateu2 = []
        self.statev1 = []  
        self.statev2 = []
        self.statew1 = [] 
        self.statew2 = []
        self.statephi1 = [] 
        self.statephi2 = []
        self.statetheta1 = []  
        self.statetheta2 = []
        self.statepsi1 = []
        self.statepsi2 = []
        self.statep1 = []
        self.statep2 = []
        self.stateq1 = []
        self.stateq2 = []
        self.stater1 = []
        self.stater2 = []
    
        # Create a handle for every subplot.
        self.handle = []
        self.handle.append(myPlot(self.ax[0], ylabel='$p_n$ $(m)$')) 
        self.handle.append(myPlot(self.ax[1], ylabel='$p_e$ $(m)$'))
        self.handle.append(myPlot(self.ax[2], ylabel='$p_d$ $(m)$'))
        self.handle.append(myPlot(self.ax[3], ylabel='$u$ $(m/s)$'))
        self.handle.append(myPlot(self.ax[4], ylabel='$v$ $(m/s)$'))
        self.handle.append(myPlot(self.ax[5], ylabel='$w$ $(m/s)$'))
        self.handle.append(myPlot(self.ax[6], ylabel='$θ$ $(rad)$'))
        self.handle.append(myPlot(self.ax[7], ylabel='$\phi$ $(rad)$'))
        self.handle.append(myPlot(self.ax[8], ylabel='$\psi$ $(rad)$'))
        self.handle.append(myPlot(self.ax[9], ylabel='$p$ $(rad/s)$'))
        self.handle.append(myPlot(self.ax[10], ylabel='$q$ $(rad/s)$'))
        self.handle.append(myPlot(self.ax[11], xlabel='t(s)', ylabel='$r$ $(rad/s)$'))

    def update(self,t, state1, state2 ):
        '''
            Add to the time and data histories, and update the plots.
        '''
        # Update the time history of all plot variables
        pn1, pe1, pd1, u1, v1, w1, phi1, theta1, psi1, p1, q1, r1 = self.extract_state_components(state1)
        pn2, pe2, pd2, u2, v2, w2, phi2, theta2, psi2, p2, q2, r2 = self.extract_state_components(state2)
        
        self.time_history.append(t)  
        self.statepn1.append(pn1)  
        self.statepn2.append(pn2) 
        self.statepe1.append(pe1)  
        self.statepe2.append(pe2)
        self.statepd1.append(-pd1)  
        self.statepd2.append(-pd2)
        self.stateu1.append(u1)  
        self.stateu2.append(u2)
        self.statev1.append(v1)  
        self.statev2.append(v2)
        self.statew1.append(w1)  
        self.statew2.append(w2)
        self.statephi1.append(phi1)  
        self.statephi2.append(phi2)
        self.statetheta1.append(theta1)  
        self.statetheta2.append(theta2)
        self.statepsi1.append(psi1)  
        self.statepsi2.append(psi2)
        self.statep1.append(p1)  
        self.statep2.append(p2)
        self.stateq1.append(q1)  
        self.stateq2.append(q2)
        self.stater1.append(r1)  
        self.stater2.append(r2)

        # Update the plots with associated histories
        self.handle[0].update(self.time_history, [self.statepn1, self.statepn2])
        self.handle[1].update(self.time_history, [self.statepe1, self.statepe2])
        self.handle[2].update(self.time_history, [self.statepd1, self.statepd2])
        self.handle[3].update(self.time_history, [self.stateu1, self.stateu2])
        self.handle[4].update(self.time_history, [self.statev1, self.statev2])
        self.handle[5].update(self.time_history, [self.statew1, self.statew2])
        self.handle[6].update(self.time_history, [self.statephi1, self.statephi2])
        self.handle[7].update(self.time_history, [self.statetheta1, self.statetheta2])
        self.handle[8].update(self.time_history, [self.statepsi1, self.statepsi2])
        self.handle[9].update(self.time_history, [self.statep1, self.statep2])
        self.handle[10].update(self.time_history, [self.stateq1, self.stateq2])
        self.handle[11].update(self.time_history, [self.stater1, self.stater2])

    def extract_state_components(self, state):
        pn = state[0, 0]
        pe = state[1, 0]
        pd = state[2, 0]
        u = state[3, 0]
        v = state[4, 0]
        w = state[5, 0]
        phi = state[6, 0]
        theta = state[7, 0]
        psi = state[8, 0]
        p = state[9, 0]
        q = state[10, 0]
        r = state[11, 0]

        return pn, pe, pd, u, v, w, phi, theta, psi, p, q, r
    


class myPlot:
    ''' 
        Create each individual subplot.
    '''
    def __init__(self, ax,
                 xlabel='',
                 ylabel='',
                 title='',
                 legend=None):
        ''' 
            ax - This is a handle to the  axes of the figure
            xlable - Label of the x-axis
            ylable - Label of the y-axis
            title - Plot title
            legend - A tuple of strings that identify the data. 
                     EX: ("data1","data2", ... , "dataN")
        '''
        self.legend = legend
        self.ax = ax                  # Axes handle
        self.colors = ['r', 'b', 'g', 'c', 'm', 'y', 'b']
        # A list of colors. The first color in the list corresponds
        # to the first line object, etc.
        # 'b' - blue, 'g' - green, 'r' - red, 'c' - cyan, 'm' - magenta
        # 'y' - yellow, 'k' - black
        self.line_styles = ['-', '-', '--', '-.', ':']
        # A list of line styles.  The first line style in the list
        # corresponds to the first line object.
        # '-' solid, '--' dashed, '-.' dash_dot, ':' dotted

        self.line = []

        # Configure the axes
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_title(title)
        self.ax.grid(True)

        # Keeps track of initialization
        self.init = True   

    def update(self, time, data):
        ''' 
            Adds data to the plot.  
            time is a list, 
            data is a list of lists, each list corresponding to a line on the plot
        '''
        if self.init == True:  # Initialize the plot the first time routine is called
            for i in range(len(data)):
                # Instantiate line object and add it to the axes
                self.line.append(Line2D(time,
                                        data[i],
                                        color=self.colors[np.mod(i, len(self.colors) - 1)],
                                        ls=self.line_styles[np.mod(i, len(self.line_styles) - 1)],
                                        label=self.legend if self.legend != None else None))
                self.ax.add_line(self.line[i])
            self.init = False
            # add legend if one is specified
            if self.legend != None:
                plt.legend(handles=self.line)
        else: # Add new data to the plot
            # Updates the x and y data of each line.
            for i in range(len(self.line)):
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])

        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()