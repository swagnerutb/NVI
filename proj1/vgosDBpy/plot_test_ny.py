import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter as DF

from vgosDBpy.data.PathParser import findCorrespondingTime
from vgosDBpy.data.combineYMDHMS import combineYMDHMwithSec
from vgosDBpy.data.readNetCDF import getDataFromVar, header_info_to_plot
from vgosDBpy.data.getName import get_name_to_print as name, get_unit_to_print as unit
"""
from PathParser import findCorrespondingTime
from combineYMDHMS import combineYMDHMwithSec
from readNetCDF import getDataFromVar
from getRealName import get_name_to_print as name
"""
import os
from datetime import datetime

class Plotfunction():

    def __init__(self):
        self.paths = []
        self.vars = []


    """
    ALWYAS CALL THIS METHOD FROM OUTSIDE THIS FILE
    """
    def plot_generall(self,paths, vars, fig, state): # generall function tahta is always called from other function then calls the other functions.
        self.data_reset()
        if len(paths) != len(vars): # controll
            return
        if default_time(state) is False:
                return ( plot_no_time(paths, vars, fig) )
        else:
            return ( plot_time(paths, vars,fig) )
        save_paths_vars(paths,vars)

    def save_paths_vars(self,paths,vars):
        self.paths= paths
        self.vars= vars

    def plot_append_generall(self, path, var, fig, state):
        self.paths.append(path)
        self.vars.append(var)
        self.plot_generall(self.paths,self.vars, fig, state)

    def data_reset(self):
        self.paths = []
        self.vars= []


    def plot_no_time(self,paths, vars, fig):
        if len(paths) == 2:
            return ( plot_two_vars(paths, vars, fig) )
        elif len(paths) == 3:
            return ( plot_three_vars(paths, vars, fig) )
        elif len(paths) == 1:
            return plot_one_var(paths, vars, fig)

    def plot_time(self,paths, vars, fig):
        if len(paths) == 1:
            return ( plot_var_time(paths[0], vars[0], fig) )
        elif  len(paths) == 2:
            return ( plot_two_var_time(paths, vars, fig) )

    def plot_one_var (self,paths, vars, fig):
        y = getDataFromVar(paths[0],vars[0])
        x = range(1,len(y)+1)
        axis = []
        data = []
        axis.append(fig.add_subplot(1,1,1))
        axis[0].plot(x,y)
        axis[0].set_title(header_info_to_plot(paths[0]) + '\n' + 'Plot' + name(vars[0]) )
        axis[0].set_xlabel('Index')
        axis[0].set_ylabel(name(vars[0])+unit(paths[0],vars[0]))
        data.append( pd.Series(y,index=x) )
        return axis, data

    def plot_two_vars(self,paths, vars, fig):

        # retrive data to plot
        x = getDataFromVar(paths[0], vars[0])
        y = getDataFromVar(paths[1], vars[1])
        axis = []
        data = []
        #create figure
        axis.append(fig.add_subplot(1,1,1))
        axis[0].plot(x,y)
        axis[0].set_title(header_info_to_plot(paths[0]) + '\n' + 'Plot' + name(vars[0]) + 'versus ' + name(vars[1] ) )
        #ax.plot(x,y)
        axis[0].set_xlabel(name(vars[0])+unit(paths[0],vars[0]))
        axis[0].set_ylabel(name(vars[1])+unit(paths[1],vars[1]))
        data.append( pd.Series(y,index=x) )
        return axis, data

    def plot_three_vars (self,paths,vars,fig) :
        #retrive data
        axis = []
        data= []

        #get data
        x = getDataFromVar(paths[0], vars[0])
        y1 = getDataFromVar(paths[1], vars[1])
        y2 = getDataFromVar(paths[2], vars[2])

        axis.append(fig.add_subplot(1,1,1))

        #first y-axis
        color = 'tab:red'
        axis[0].set_xlabel(name(vars[0])+unit(paths[0],vars[0]))
        axis[0].set_ylabel(name(vars[1])+unit(paths[1],vars[1]))
        axis[0].plot(x, y1, color=color)
        axis[0].tick_params(axis=vars[1], labelcolor=color)

        #second y-axis
        axis[1] = axis[0].twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        axis[1].plot(x, y2, color=color)
        axis[1].set_ylabel(name(vars[2])+unit(paths[2],vars[2]))
        axis[1].tick_params(axis=vars[2], labelcolor=color)
        plt.title(header_info_to_plot(path1)+ "\nPlot " +name(vars[1]) + "and " + name(vars[2]) + " against " +name(vars[0]))
        data.append( pd.Series(y1, index = x) )
        data.append( pd.Series(y2, index = x) )
        return axis, data

    def plot_var_time(self, path, var, fig):
        #retrive time data and if not possible just return
        axis = []
        data = []

        timePath = findCorrespondingTime(path)
        if timePath is "":
            return
        time_data = []
        time = combineYMDHMwithSec(timePath)
        for t in time:
            time_data.append(t)
        #retive y-axis data
        y = getDataFromVar(path,var)

        #Create plot
        axis.append( fig.add_subplot(1,1,1) )
        axis[0].set_title(header_info_to_plot(path)+ "\n " + "Plot " + name(var) + " versus Time " )
        if len(time_data) == len(y):
            #axis[0].xticks(rotation= 80)
            axis[0].plot(time_data, y)
            axis[0].set_xlabel('Time')
            axis[0].set_ylabel(name(var)+unit(var))
            #axis[0].set_xticklabels(axis[0].get_xticklabels(), rotation=80)
            data.append(pd.Series(y, index = time_data ))
        else:
            raise ValueError('Time and data do not have same length')
        return axis, data

    def plot_two_var_time (self, paths, vars, fig):

        # Define return arrays'
        axis = []
        data = []

        #try to retrive time data, if can not just return
        timePath = findCorrespondingTime(paths[0])
        time= combineYMDHMwithSec(timePath)
        time_data=[]
        i=0
        for t in time:
            time_data.append(t)
        # retriv y-axis data
        y1 = getDataFromVar(paths[0], vars[0])
        y2 = getDataFromVar(paths[1], vars[1])

        if len(time) == len(y1) and len(time) == len(y2):
            axis.append(fig.add_subplot(1,1,1))
            #ax1 = fig.add_subplot(1,1,1)
            color = 'tab:red'
            axis[0].set_xlabel('Time H:M:S')
            axis[0].set_ylabel(name(vars[0])+ unit(paths[0],vars[0]))
            axis[0].plot(time_data, y1, color=color)
            #plt.xticks( rotation= 80 )
            axis[0].tick_params(axis=vars[0], labelcolor=color)
            axis.append( axis[0].twinx() )  # instantiate a second y-axis that shares the same x-axis
            color = 'tab:blue'
            axis[1].plot(time_data, y2, color=color)
            axis[1].set_ylabel(name(vars[1])+ unit(paths[1],vars[1]))
            axis[1].tick_params(axis=vars[1], labelcolor=color)
            plt.title(header_info_to_plot(paths[0])+ "\nPlot " +name(vars[0]) + "and " + name( vars[1]) + " over time ")
            data.append(pd.Series(y1, index = time_data))
            data.append(pd.Series(y2, index = time_data))
            return axis, data
        else:
            print("Dimensions do not agree")

    def default_time(self,state):
        if state == 1:
            return True
        else:
            return False
