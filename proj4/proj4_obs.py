import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, delaunay_plot_2d
from sklearn.linear_model import LinearRegression
import sys #remove
import math

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

def plot_base(df_,plt_=plt):
    """Plot horizon, usable horizon, mutual horizon"""
    #Plot horizon
    #good: start 80, fin 340
    # start = 80
    # fin = 276 #max: 361

    # plt.plot(df.loc[fin:,'hor1'],df.loc[fin:,'hor2'],c='tab:blue')
    # plt.plot(df.loc[:start,'hor1'],df.loc[:start,'hor2'],c='tab:blue')
    plt_.plot(df_['hor1'],df_['hor2'],c='tab:blue')

    #Plot usable horizon
    #good: start 83, fin 287
    # start = 83
    # fin = 320
    # plt.plot(df.loc[:start,"usable1"],df.loc[:start,"usable2"],'r--',linewidth=1,c='red')
    # plt.plot(df.loc[fin:,"usable1"],df.loc[fin:,"usable2"],'r--',linewidth=1,c='red')
    plt_.plot(df_["usable1"],df_["usable2"],'r--',linewidth=1,c='red')

    #Plot common view
    #good: start 80, fin 265
    rm_start = 80
    rm_end = 276
    plt_.plot(df_.loc[:rm_start,"com_hor1"],df_.loc[:rm_start,"com_hor2"],c='grey')
    plt_.plot(df_.loc[rm_end:,"com_hor1"],df_.loc[rm_end:,"com_hor2"],c='grey')

def plot_observations(df_sk,plt_=plt,colour='orange'):
    plt_.scatter(df_sk["plot_1"],df_sk["plot_2"],marker="x",c=colour,s=4)

def euclidean_dist(p1_x, p1_y, p2_x, p2_y,):
    return np.sqrt(np.power(p1_x-p2_x,2)+np.power(p1_y-p2_y,2))

def rms_val(data):
    """Calculate RMS"""
    val = 0
    for d in data:
        val += np.power(d,2)
    return np.sqrt(val/len(data))

def observations(sked_nbr,path_sk,scale=90):
    """
    Plots view from station 1
    """
    sked_nbr = sked_digit_to_string(sked_nbr)
    if(sked_nbr == '500'):
        sked_nbr = 'cov'

    sked = f'sk_d10_00h_{sked_nbr}.solve'
    df_sk = get_schedule(path_sk,sked,scale)
    return df_sk

def obs_10min_window(df_,current_time,min=10):
    """
    Get all observations for time t in [t-min/2,t+min/2]
    """
    return df_[np.abs(np.subtract(df_['epoch_sec_tot'].astype(float),current_time)) < 60*min/2]

########## Paths ##########
if(True):
    path_skeds = '/Users/sam/Desktop/NVI/proj4/sk_d10_00h'
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
    print(sked_list)
    print("\n##################################\n",df_sked_tot,"\n##################################\n")

    ## Base data: Horizon, common horizon and usable horizon
    df_base = read_base(path_output)

########## Plotting ##########
if(False):
    sked_list_best = sked_list[:3]
    sked_list_mid = sked_list[250:250+3]
    sked_list_worst = sked_list[-3:] #[266,329,437,126,401]

    plt_rows = 3
    fig, axs = plt.subplots(plt_rows, len(sked_list_best))

    for idx in range(len(sked_list_best)):
        sked = sked_list_best[idx]
        plot_base(df_base,plt_=axs[0,idx])
        df_sked = observations(sked, path_skeds)
        plot_observations(df_sked,plt_=axs[0,idx])
        axs[0,idx].set_title("sked: " + str(sked), fontsize=10)

    for idx in range(len(sked_list_mid)):
        plt_row = 1
        sked = sked_list_mid[idx]
        plot_base(df_base,plt_=axs[plt_row,idx])
        df_sked = observations(sked, path_skeds)
        plot_observations(df_sked,plt_=axs[plt_row,idx])
        axs[plt_row,idx].set_title("sked: " + str(sked), fontsize=10)

    for idx in range(len(sked_list_worst)):
        plt_row = 2
        sked = sked_list_worst[idx]
        plot_base(df_base,plt_=axs[plt_row,idx])
        df_sked = observations(sked, path_skeds)
        plot_observations(df_sked,plt_=axs[plt_row,idx])
        axs[plt_row,idx].set_title("sked: " + str(sked), fontsize=10)

    plt.show()

########## Saving data ##########

# Making a more accessible csv file
df = pd.DataFrame(columns=['sked','RMS','x','y'])
sked_list = [432]
for sked in range(501):
    if(sked%50==0):
        print(f"sked = {sked}")
        if(sked%100==0):
            print("########## df ##########\n", df)
    df_sk = observations(sked,path_skeds)

    rms = df_sked_tot.at[sked,'RMS']
    ind_list = df_sk['plot_1'].index.tolist()
    for ind in ind_list:
        df = df.append({'sked':sked, 'RMS':rms, 'x':df_sk.at[ind,'plot_1'], 'y':df_sk.at[ind,'plot_2']},ignore_index=True)

# df.to_csv(path_or_buf='/Users/sam/Desktop/NVI/proj4/sked_data/all_obs_data.csv', sep=',')




"""
# Tanken här var att kolla avståndet alla punkter emellan och ha det som input till algoritmen
np.random.seed(123)
inds = range(50*500)
df_dist_between = pd.DataFrame(index=inds)
print(df_dist_between)
for sked in range(501):

    rand_ind = []
    for _ in range(100):
        rand_ind.append(np.random.randint(0,8))
    
    df_sk = observations(sked,path_skeds)
    print(df_sk)
    for time in range(5,55):
        df3 = obs_10min_window(df_sk,time)
        df3_ind = df3.index.tolist()
        #choose 8 points
        p1 = [df3.at[df3_ind[0],'plot_1'],df3.at[df3_ind[0],'plot_2']]
        p2 = [df3.at[df3_ind[1],'plot_1'],df3.at[df3_ind[1],'plot_2']]
        print("p1 = ",p1)
        print("p2 = ",p2)
        sys.exit()
        # euclidean_dist()
        print(df3)
        break
"""

