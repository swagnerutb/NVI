import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris

def sked_digit_to_string(sked_nbr):
    try:
        if(int(sked_nbr) < 10):
            return "00"+str(sked_nbr)
        elif(int(sked_nbr) < 100):
            return "0"+str(sked_nbr)
        else:
            return str(sked_nbr)
    except: #if sked_nbr is already string
        return sked_nbr

def read_base(path,scale=90):
    """Read base data"""
    deg2rad = np.pi/180

    theta = np.linspace(0,2*np.pi,num=361)

    x1 = scale*np.sin(theta)
    x2 = scale*np.cos(theta)

    df = pd.read_csv(path,sep = '\s+')

    df["hor1"] = pd.DataFrame(x1,columns={"hor1"})
    df["hor2"] = pd.DataFrame(x2,columns={"hor2"})

    df["com_hor1"] = scale*np.sin(deg2rad*df["Kk_Az"].astype(float))*np.cos(deg2rad*df["Kk_El"].astype(float)) #scale*np.sin(H6*deg2rad)*COS($I6*deg2rad)
    df["com_hor2"] = scale*np.cos(deg2rad*df["Kk_Az"].astype(float))*np.cos(deg2rad*df["Kk_El"].astype(float))

    mask = [0, 10, 238, 14, 244, 10, 260, 5, 280, 10, 295, 25, 300, 35, 305,
         40, 310, 45, 325, 40, 330, 35, 335, 30, 340, 10, 350, 5, 360]

    mask1 = [mask[2*i] for i in range(int(len(mask)/2)+1)]
    mask2 = [mask[2*i+1] for i in range(int(len(mask)/2))]
    df["usable1"] = np.nan
    df["usable2"] = np.nan
    for i in range(1,len(mask1)):
        for k in range(mask1[i-1],mask1[i]+1):
            df.at[k,"usable1"] = np.cos(deg2rad*mask2[i-1])*df.loc[k,"hor1"].astype(float)
            df.at[k,"usable2"] = np.cos(deg2rad*mask2[i-1])*df.loc[k,"hor2"].astype(float)
    return df

def get_schedule(path,sked,scale=90):
    """
    Get data from given sked\n
    path: path to directory of schedules\n
    sked: name of .solve file
    """
    deg2rad = np.pi/180

    df = pd.read_csv(path+"/"+sked,sep="\s+",header=None,names=range(22))
    sources_temp = df.iloc[1,1]
    df = df.iloc[30:]

    df.rename(columns={0:"source",
                        1:"epoch1",2:"epoch2",3:"epoch3",4:"epoch_h",5:"epoch_min",6:"epoch_sec",
                        7:"station1",8:"station2",
                        9:"sigma",
                        10:"az_st1",11:"el_st1",
                        12:"az_st2",13:"el_st2"},inplace=True)
    df = df.loc[:,:"el_st2"]

    df["epoch_sec_tot"] = df["epoch_h"]*3600 + df["epoch_min"]*60 + df["epoch_sec"]

    for i in range(30,30+len(df["el_st2"])):
        elem_split = df.at[i,"el_st2"].split("-")
        if(elem_split[0] == ''):
            df.at[i,"el_st2"] = "-"+elem_split[1]
        else:
            df.at[i,"el_st2"] = elem_split[0]

    df["plot_1"] = scale*np.sin(deg2rad*df["az_st1"].astype(float))*np.cos(deg2rad*df["el_st1"].astype(float))
    df["plot_2"] = scale*np.cos(deg2rad*df["az_st1"].astype(float))*np.cos(deg2rad*df["el_st1"].astype(float))

    # df["nbr_sources"] = 0
    # df.at[30,"nbr_sources"] = sources_temp
    
    return df

def observations(sked_nbr,path_sk,scale=90):
    """
    Get observations from station 1
    """
    sked_nbr = sked_digit_to_string(sked_nbr)
    if(sked_nbr == '500'):
        sked_nbr = 'cov'

    sked = f'sk_d10_00h_{sked_nbr}.solve'
    df_sk = get_schedule(path_sk,sked,scale)
    return df_sk

def euclidean_dist(p1_x, p1_y, p2_x, p2_y,):
    return np.sqrt(np.power(p1_x-p2_x,2)+np.power(p1_y-p2_y,2))

def rms_val(data):
    """Calculate RMS"""
    val = 0
    for d in data:
        val += np.power(d,2)
    return np.sqrt(val/len(data))


########## Paths ##########
if(True):
    path_to_skeds = '/Users/sam/Desktop/NVI/proj4/sk_d10_00h'
    path_sked_tot = '/Users/sam/Desktop/NVI/proj4/ut1_d10_00h.txt'
    path_sked_holes = '/Users/sam/Desktop/NVI/proj4/sked_data/holes_data.csv'
    path_output = '/Users/sam/Desktop/NVI/proj4/kk_azel2.txt'

########## Reading csv files ##########
if(True):
    ## Info on all skeds
    df_sked_tot = pd.read_csv(path_sked_tot,sep='\s+',header=None)
    df_sked_tot.rename(columns={8:'RMS',6:'formal_error',2:"num_obs"},inplace=True)
    df_sked_tot.sort_values(by=['RMS'],inplace=True)
    sked_list = df_sked_tot.index.tolist()
    print("\n##################################\n",df_sked_tot,"\n##################################\n")

    ## Base data: Horizon, common horizon and usable horizon
    df_base = read_base(path_output)

    ## Data on holes in skeds
    df_holes = pd.read_csv(path_sked_holes,sep=',',index_col=0)
    df_holes_indexlist = df_holes.index.tolist()
    df_holes_collist = df_holes.columns.tolist()

########## Calculations ##########
if(False):
    nbr_skeds = int(len(df_holes_collist)/3)

    ## Getting all radii
    holes_rad = np.zeros(shape=(nbr_skeds,51)) #holes_rad = nbr_skeds * len(minutes)
    for i in range(nbr_skeds):
        sked_ = sked_digit_to_string(i)
        holes_rad[i,:] = df_holes[f'{sked_}_rad']


    ## RMS
    holes_rms = np.zeros(shape=(nbr_skeds,3)) #holes_rms = [rms_x, rms_y, rms_rad]
    rms_dist_to_p = np.zeros(shape=(nbr_skeds))
    for i in range(nbr_skeds):
        sked_ = sked_digit_to_string(i)
        holes_rms[i,0] = rms_val(df_holes[f'{sked_}_x'])
        holes_rms[i,1] = rms_val(df_holes[f'{sked_}_y'])
        holes_rms[i,2] = rms_val(df_holes[f'{sked_}_rad'])
        point = [20,60]
        rms_dist_to_p[i] = rms_val(euclidean_dist(df_holes[f'{sked_}_x'],df_holes[f'{sked_}_y'],point[0],point[1]))

    ## Variance
    holes_var = np.zeros(shape=(nbr_skeds,3)) #holes_var = [var_x, var_y, var_rad]
    var_dist_to_p = np.zeros(shape=(nbr_skeds))
    for i in range(nbr_skeds):
        sked_ = sked_digit_to_string(i)
        holes_var[i,0] = np.var(df_holes[f'{sked_}_x'])
        holes_var[i,1] = np.var(df_holes[f'{sked_}_y'])
        holes_var[i,2] = np.var(df_holes[f'{sked_}_rad'])
        point = [20,60]
        var_dist_to_p[i] = np.var(euclidean_dist(df_holes[f'{sked_}_x'],df_holes[f'{sked_}_y'],point[0],point[1]))

    ## Mean values
    holes_mean = np.zeros(shape=(nbr_skeds,3)) #holes_mean = [mean_x, mean_y, mean_rad]
    mean_dist_to_p = np.zeros(shape=(nbr_skeds))
    for i in range(nbr_skeds):
        sked_ = sked_digit_to_string(i)
        holes_mean[i,0] = np.mean(df_holes[f'{sked_}_x'])
        holes_mean[i,1] = np.mean(df_holes[f'{sked_}_y'])
        holes_mean[i,2] = np.mean(df_holes[f'{sked_}_rad'])
        point = [20,60]
        mean_dist_to_p[i] = np.mean(euclidean_dist(df_holes[f'{sked_}_x'],df_holes[f'{sked_}_y'],point[0],point[1]))





