from netCDF4 import Dataset
import pandas as pd
import os


"""
Takes in a path to a TimeUTC.nc file and combines the YMDHM with the related seconds
input:
    timeFilePath [string]
output:
    YMDHMS [array[pd.Timestamp]]
"""
def combineYMDHMwithSec(timeFilePath, get_length_only=False):
    """
    Note:   Executes very slowly when get_length_only=False (i.e. a list of timestamps is returned)
            as the function then needs to produce the list that is returned.
    """
    with Dataset(timeFilePath,"r") as time_file:
        seconds = time_file.variables["Second"]
        YMDHM = time_file.variables["YMDHM"]

        if (len(seconds) != len(YMDHM)):
            return None

        if(get_length_only == True):
            return len(seconds)

        # match each Timestamp with the corresponding second
        if(YMDHM[0][0] < 100):
            # Sometimes the year is only denoted by the last two digit
            # It is assumed that  the year belongs to 2000-2999 which is why we add 2000
            return [pd.Timestamp(int(YMDHM[i][0])+2000,int(YMDHM[i][1]),int(YMDHM[i][2]),int(YMDHM[i][3]),int(YMDHM[i][4]),int(seconds[i])) for i in range(len(YMDHM))]
        
        return [pd.Timestamp(int(YMDHM[i][0]),int(YMDHM[i][1]),int(YMDHM[i][2]),int(YMDHM[i][3]),int(YMDHM[i][4]),int(seconds[i])) for i in range(len(YMDHM))]


"""
checks if a specific netCDF file and varible has a related timestamp
input :
    paths: [array[string], vars: [array[string]]
output:
    boolean
"""
def checkIfTimeAvailable(paths,vars):
    c = 0
    for path in paths:
        timePath = findCorrespondingTime(path)
        if timePath == "":
            return False
        time_data = []
        time = combineYMDHMwithSec(timePath)
        for t in time:
            time_data.append(t)
        y = get_data_to_plot(path,vars[c])
        if len(time_data) != len(y[0]):
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
"""
Returns the timeUTC netCDF that is related to the netCDF-file given as input
input:
    path: [string]
output:
    time_path: [string]
"""
def findCorrespondingTime(pathToNetCDF):
    if os.path.isfile(pathToNetCDF):
        time_path = os.path.join(os.path.dirname(pathToNetCDF),'TimeUTC.nc')

        if(os.path.isfile(time_path)):
            return time_path
        else: # returning an empty string is seen as other functions as a sign to default with index instead
            return "" #default: " "
