import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter as DF
from netCDF4 import Dataset

from vgosDBpy.data.combineYMDHMS import combineYMDHMwithSec, checkIfTimeAvailable, default_time,findCorrespondingTime
from vgosDBpy.data.readNetCDF import header_plot, get_data_to_plot
from vgosDBpy.data.getName import get_name_to_print as name, get_unit_to_print as unit

import os
from datetime import datetime

"""
ALWYAS CALL THIS METHOD FROM OUTSIDE THIS FILE
"""


"""
Helper-class to Plotfunction_class that stores the data for the axis.
"""
class AxisItem():
    def __init__(self):
        self.path  = ''
        self.var = ''
        self.data = ''
        self.isEmpty = True # to keep track if the axis stores any data or not

    def createAxisItem(self,path,var,data):
        self.path  = path
        self.var = var
        self.data = data
        self.isEmpty = False

    """
    output: [string]
    """
    def get_axis_label(self):
        var = self.var.split("_")[0]
        var_unit = unit(self.path, var)
        var_unit_list = var_unit.split(",")

        try:
            nbr = int(self.var[-1])
        except:
            nbr = -1

        if(len(var_unit_list) > 1 and nbr > -1):
            if(not var_unit_list[nbr].endswith("]")):
                var_unit = var_unit_list[nbr]+"]"
            else:
                var_unit = " ["+var_unit_list[nbr][1:]
        
        # Making sure rate is expressed in units
        if(nbr==1 and var_unit.split("[")[1][0] != "-" and not (var_unit.endswith("/sec]") or var_unit[nbr].endswith("/second]"))):
            var_unit = var_unit.replace(']','/sec]')
        
        return var + var_unit


    def empty(self):
        self.path  = ''
        self.var = ''
        self.data = ''
        self.isEmpty = True

    def getPath(self):
        return self.path

    def getVar(self):
        return self.var

    def getData(self):
        return self.data

"""
Class calls a function to extract the data form netCDF files for the plots, and adds it to axis connected to a figure
and returns the data and axis
"""
class Plotfunction_class():

    def __init__(self):
        self.data = []
        self.axis = []
        self.x = AxisItem()
        self.y1 = AxisItem()
        self.y2 = AxisItem()
        self.y = []
        self.place = 0
        self.timedata = {} #key=path to directory, value = TimeUTC.nc in Timestamps

    """
    Resets all the data to its initial state
    """
    def clear(self, clear_timedata = False):
        self.data = []
        self.axis = []
        self.x = AxisItem()
        self.y1 = AxisItem()
        self.y2 = AxisItem()
        self.y = []
        self.place = 0
        if(clear_timedata==True):
            self._clear_timedata()

    def _clear_axis(self):
        self.axis = []

    def _clear_data(self):
        self.data = []

    def _clear_timedata(self):
        self.timedata = {}

    """
    output:
        self.x [AxisItem]
    """
    def _getX(self):
        return  self.x

    """
    output:
        self.y1 [AxisItem]
    """
    def _getY1(self):
        return self.y1
    """
    output:
        self.y2 [AxisItem]
    """
    def _getY1(self):
        return self.y2

    """
    creates an axis that is connected to the figure figure
    input:
        fig: [Figure]
    """
    def _createAxis(self,fig):
        self.axis.append(fig.add_subplot(1,1,1)) # connects the axis to a figure

    """
    that shares the same x-axis so the same function as the one above
    must be called after _createAxis
    """
    def _appendAxis(self):
        self.axis.append(self.axis[0].twinx()) # this axis in connected to the same figure



    """
    Adds information to the 'x' AxisItem
    input:
        path: [string],
        var: [string], data: [array]
    """
    def add_to_x_axis(self,path,var,data): # adds data to the x-Axis
        data = np.squeeze(np.asarray(data))
        self.x.createAxisItem(path,var,data)



    """
    Adds information to the 'y1' or 'y2' AxisItem
    input:
        path: [string],
        var: [string], data: [array]
    """
    def add_to_y_axis(self,path,var,data): # adds data to a y-axis, y1 if empty else y2, if both full do nothing
        data = np.squeeze(np.asarray(data))
        
        if self.y1.isEmpty == True:
            self.y1.createAxisItem(path,var,data)
        elif self.y2.isEmpty == True:
            self.y2.createAxisItem(path,var,data)

    """
    Adds timeData to the 'x' AxisItem, by finding the data and calling _add_to_x_axis
    input:
        path: [string],
        var: [string] = TIME,
        data: [array]
    """
    def _add_time_to_xAxis(self,path,var='Time',data= []): # help function to add_to_x_axis
        timePath = findCorrespondingTime(path)
        timePath_dir = os.path.dirname(timePath)
        if(timePath_dir in self.timedata):
            time_data = self.timedata[timePath_dir]
        else:
            time_data = combineYMDHMwithSec(timePath)
            self.timedata[timePath_dir] = time_data
        self.add_to_x_axis(path,'Time',time_data)

    """
    Adds index to the 'x' AxisItem, by creating the data and calling _add_to_x_axis
    input:
        path: [string]
        data: [array]
    """
    def _add_index_to_xAxis(self,path,data): # help function to add_to_x_axis
        nbrIdx = len(data) + 1
        idx = range(1,nbrIdx)
        self.add_to_x_axis(path,'Index', idx)


    """
    Adds the data stored int he axis to the directory 'data'
    """
    def _append_data(self,vars=None): # created the data_map to be returned from the dat stored in the axis
        if(vars is None):
            if self.y1.isEmpty == False:
                self.data.append(pd.Series(self.y1.getData(), index = self.x.getData() ) )

            if self.y2.isEmpty == False:
                self.data.append(pd.Series(self.y2.getData(), index = self.x.getData() ) )

        else:
            if(self.y1.isEmpty == False and vars.endswith("0")):
                self.data.append(pd.Series(self.y1.getData(), index = self.x.getData() ) )
            if(self.y2.isEmpty == False and vars.endswith("1")):
                self.data.append(pd.Series(self.y2.getData(), index = self.x.getData() ) )

    def plotFunction(self,paths,vars,fig,state,custom_x_axis=False):
        """
        The function that is called from other files to retrieve the right data and lables for a plot
        input:
            paths: [array[strings]],
            vars: [array[srings]],
            fig: [figure]
            state: [int]
            custom_x_axis: [boolean] (optional argument)
        output:
            axis: [Axis]
            data: [directory '{}']
        """
        #clears all info
        self.clear()
        nbr = len(paths)
        self.place = -1 # keeps track of if any path was used to the x-axis, and if so not to be added to a y-axis


        if(default_time(state) == True and checkIfTimeAvailable(paths, vars) == True): # check if time on x-axis
            plot_to_time = True
        else :
            plot_to_time = False
        """
        So far we are not working with neither data nor axis just creating x, y1, y2
        """
        # First find which data that should be x-axis
        # possible options is time, index or data from path.
        
        Temp =  get_data_to_plot(paths[0],vars[0])

        if(custom_x_axis==False and len(vars) < 3):
            if(nbr == 1 and plot_to_time == False):
                self._add_index_to_xAxis(paths[0],Temp[0])
                self.add_to_y_axis(paths[0],vars[0],Temp)
            else:
                if(plot_to_time == True):
                    self._add_time_to_xAxis(paths[0])
                else:
                    #This plots two curves on y-axis
                    for i in range(len(paths)):
                        Temp = get_data_to_plot(paths[i],vars[i])
                        self._add_index_to_xAxis(paths[i],Temp[0])
                        k=0
                        for itm in Temp:
                            self.add_to_y_axis(paths[i],vars[i],itm)

        if(custom_x_axis==True or len(vars) >= 3):
            if(custom_x_axis == True):
                #Reverse lists to get x and y-axes right
                paths = paths[::-1]
                vars = vars[::-1]
                if(len(vars) >= 3):
                    temp_path = paths[2]
                    paths[2] = paths[1]
                    paths[1] = temp_path
                    temp_var = vars[2]
                    vars[2] = vars[1]
                    vars[1] = temp_var
            elif(len(vars) >= 3):
                temp_path = paths[1]
                paths[1] = paths[0]
                paths[0] = temp_path
                temp_var = vars[1]
                vars[1] = vars[0]
                vars[0] = temp_var


            for i in range(len(paths)):
                Temp = get_data_to_plot(paths[i],vars[i])
                if(len(Temp) == 1 and self.x.isEmpty == True):
                    self.add_to_x_axis(paths[i], vars[i], Temp)
                    self.place = i
            
            if self.x.isEmpty == True:
                self._add_index_to_xAxis(paths[0],Temp)

        if(nbr != 1 or plot_to_time == True):
            if self.x.isEmpty == True:
                    self._add_index_to_xAxis(paths[0],Temp)

            # now we have data stored in x
            # find the data to store in y1 and y2 or just y1

            for i in range(len(paths)):
                if i != self.place:
                    if(vars[i] != vars[i].split("_")[0]):
                        Temp = get_data_to_plot(paths[i],vars[i])
                        self.add_to_y_axis(paths[i],vars[i],Temp)
                    else:
                        Temp = get_data_to_plot(paths[i],vars[i])
                        for temp_data in Temp:
                            self.add_to_y_axis(paths[i],vars[i],temp_data)

        """
        now we move on the defining axis and data using x, y1, y2
        """
        # use x, y1 and y2 to generate the axis and data to return
        
        # first handle y1 meaning working with axis[0]
        if self.y1.isEmpty == False:
            self._createAxis(fig)
            color ='black'
            x_label_ = self.x.get_axis_label()
            y1_label_ = self.y1.get_axis_label()
            self.axis[0].set_xlabel(x_label_)
            self.axis[0].set_ylabel(y1_label_,color=color)
            self.axis[0].plot(self.x.getData(), self.y1.getData(), color=color, label = name(self.y1.getVar()).split("_")[0])
            
            self.axis[0].tick_params(axis='y', labelcolor=color)

        # if two y-axis add the second one
        if self.y2.isEmpty == False:
            self._appendAxis() # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            y2_label_ = self.y2.get_axis_label()
            self.axis[1].plot(self.x.getData(), self.y2.getData(), color=color,label = name(self.y2.getVar()))
            self.axis[1].set_ylabel(y2_label_,color=color)
            self.axis[1].tick_params(axis='y', labelcolor=color)

        # set titel to entire plot handeling all diferent cases
        if self.y2.isEmpty == True:
            self.axis[0].set_title(header_plot(paths[0])+ "\nPlot "+name(self.y1.getVar()).split("_")[0]+" against "+name(self.x.getVar()).split("_")[0])
        elif self.y2.isEmpty == False:
            self.axis[0].set_title(header_plot(paths[0])+ "\nPlot "+name(self.y1.getVar()).split("_")[0]+" and "+name(self.y2.getVar()).split("_")[0]+" against "+name(self.x.getVar()).split("_")[0])
        
        self._append_data()

        return self.axis, self.data


"""
checks if a specific netCDF file and varible has a related timestamp
input : paths: [array[string], vars: [array[string]]
output: boolean
"""
def checkIfTimeAvailable(paths,vars):
    c = 0
    for path in paths:
        timePath = findCorrespondingTime(path)
        if timePath == "":
            return False
        time_data_len = combineYMDHMwithSec(timePath, get_length_only=True)
        y = get_data_to_plot(path,vars[c])
        if time_data_len != len(y[0]):
            return False
        c += 1
    return True

"""
checks if status is to have the time on the x-axis in plot
input:
    state: [int]
output:
    boolean
"""
def default_time(state):
    if state == 1:
        return True
    else:
        return False
