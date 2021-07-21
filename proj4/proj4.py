import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, delaunay_plot_2d
from sklearn.linear_model import LinearRegression
import sys #remove
import math

def sked_digit_to_string(sked_nbr):
    try:
        if(sked_nbr < 10):
            return "00"+str(sked_nbr)
        elif(sked_nbr < 100):
            return "0"+str(sked_nbr)
        else:
            return str(sked_nbr)
    except: #if sked_nbr is already string
        return sked_nbr

def read_base(path,scale=90):
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

def plot_base(df):
    #Plot horizon
    #good: start 80, fin 340
    # start = 80
    # fin = 276 #max: 361

    # plt.plot(df.loc[fin:,'hor1'],df.loc[fin:,'hor2'],c='tab:blue')
    # plt.plot(df.loc[:start,'hor1'],df.loc[:start,'hor2'],c='tab:blue')
    plt.plot(df['hor1'],df['hor2'],c='tab:blue')

    #Plot visible horizon
    #good: start 83, fin 287
    start = 83
    fin = 320
    plt.plot(df.loc[:start,"usable1"],df.loc[:start,"usable2"],'r--',linewidth=1,c='red')
    plt.plot(df.loc[fin:,"usable1"],df.loc[fin:,"usable2"],'r--',linewidth=1,c='red')
    # plt.plot(df["usable1"],df["usable2"],'r--',linewidth=1,c='red')

    #Plot common view
    #good: start 80, fin 265
    rm_start = 80
    rm_end = 276
    plt.plot(df.loc[:rm_start,"com_hor1"],df.loc[:rm_start,"com_hor2"],c='grey')
    plt.plot(df.loc[rm_end:,"com_hor1"],df.loc[rm_end:,"com_hor2"],c='grey')

def plot_observations(df_sk,show=False):
    plt.scatter(df_sk["plot_1"],df_sk["plot_2"],marker="x",c='orange',s=4)

def plot_3d_obs(df_sk, df1, plot_horizon=True):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if(plot_horizon==True):
        plot_base(df1)

    xdata = df_sk["plot_1"]
    ydata = df_sk["plot_2"]
    zdata = df_sk["epoch_sec_tot"]
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='winter')

def observations(sked_nbr,path_output,path_sk,plot=False,color='orange',scale=90):
    """
    Plots view from station 1
    """
    sked_nbr = sked_digit_to_string(sked_nbr)
    if(sked_nbr == '500'):
        sked_nbr = 'cov'

    df1 = read_base(path_output,scale)

    if(plot==True):
        plot_base(df1)

    #Scatter observations

    if(sked_nbr != 'all'):
        sked = f'sk_d10_00h_{sked_nbr}.solve'
        df_sk = get_schedule(path_sk,sked,scale)
        if(plot==True):
            plt.scatter(df_sk["plot_1"].astype(float),df_sk["plot_2"].astype(float),marker="x",c=color,s=4)
            return df1, df_sk
        else:
            return df1, df_sk
    else:
        for i in range(500):
            nbr = str(i)
            if(len(nbr)==1):
                nbr = "00"+nbr
            elif(len(nbr)==2):
                nbr = "0"+nbr
            sked = f'sk_d10_00h_{nbr}.solve'
            df_sk = get_schedule(path_sk,sked,scale)
            plt.scatter(df_sk["plot_1"],df_sk["plot_2"],marker="x",c='orange',s=4)
        sked = 'sk_d10_00h_cov.solve'
        df_sk = get_schedule(path_sk,sked,scale)
        plt.scatter(df_sk["plot_1"],df_sk["plot_2"],marker="x",c='blue',s=4)
        return df1, df_sk #note, dt2 only contains last 

def obs_10min_window(df,current_time,min=10):
    """
    Get all observations for time t in [t-min/2,t+min/2]
    """
    return df[np.abs(np.subtract(df['epoch_sec_tot'].astype(float),current_time)) < 60*min/2]

def is_inside(coord,df1):
    inside = True

    k1 = 0
    k1nbr = 10
    #Sort list by |com_hor1 - coord| and extract the smallest distance
    df_sort = df1.iloc[(df1['com_hor1']-coord[0]).abs().argsort()[:k1nbr]]

    if(df_sort['com_hor2'].values[0] > 0):
        inside = inside and (df_sort['com_hor2'].values[0] < coord[1])
    else:
        k1 = 1
        # We know that the common horizon is > 0.
        # This does however need to be changed for a general case.
        while(df_sort['com_hor2'].values[k1] < 0 and k1 < k1nbr-1):
            k += 1
        inside = inside and (df_sort['com_hor2'].values[k1] < coord[1])

    #########################
    k2 = 0
    k2nbr = 10
    #Sort list by |usable1 - coord| and extract the smallest distance
    df_sort = df1.iloc[(df1['usable1']-coord[0]).abs().argsort()[:k2nbr]] #varför begränsa sig?

    if(df_sort['usable2'].values[0] > 0):
        inside = inside and (df_sort['usable2'].values[0] > coord[1])
    else:
        while(df_sort['usable2'].values[k2] < 0 and k2 < k2nbr-1): #varför begränsa sig?
            k2 += 1
        inside = inside and (df_sort['usable2'].values[k2] > coord[1])

    return inside

def euclidean_dist(x1,x2,y1,y2):
    return np.sqrt(np.power(np.subtract(x1,x2),2)+np.power(np.subtract(y1,y2),2))

def euclidean_dist_difflen(x1,x2,y1,y2):
    """
    Computing distance between two 10 min interval observations, possibly of different length.
    """
    dist = 0
    if(len(x1) <= len(x2)):
        for i in range(len(x1)): #Only add distance to closest neighbour
            # min_dist = np.sort(np.sqrt(np.power(x1[i]-x2,2) + np.power(y1[i]-y2,2)))[0]
            # dist += min_dist
            dist += np.sum(np.sqrt(np.power(x1[i]-x2,2) + np.power(y1[i]-y2,2)))
    elif(len(x1) > len(x2)):
        for i in range(len(x2)): #Only add distance to closest neighbour
            # min_dist = np.sort(np.sqrt(np.power(x1-x2[i],2) + np.power(y1-y2[i],2)))[0]
            # dist += min_dist
            dist += np.sum(np.sqrt(np.power(x1-x2[i],2) + np.power(y1-y2[i],2)))
    return dist

def rms_val(data):
    val = 0
    for d in data:
        val += np.power(d,2)
    return np.sqrt(val/len(data))

def plot_hole_mvmt(hole,sked,df_sked_tot,plot_horizon=True,show_plot=True,save_fig=False):
    """
    hole is array where col1: mid_x, col2: mid_y, col3: radius
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if(plot_horizon==True):
        plot_base(df1)

    mid_x = hole[:,0]
    mid_y = hole[:,1]
    r = hole[:,2]

    zdata = np.linspace(0,len(mid_x)-1,len(mid_x))

    cmap_name = 'winter'
    cmap = matplotlib.cm.get_cmap(cmap_name)

    for k in range(len(mid_x)):
        ang = np.linspace(0,2*np.pi,360)
        xp = hole[k,2]*np.cos(ang)
        yp = hole[k,2]*np.sin(ang)
        ax.plot3D(mid_x[k]+xp,mid_y[k]+yp,zdata[k],c=cmap(k/(len(mid_x)-1)),linewidth=1)

    ax.scatter3D(mid_x, mid_y, zdata, c=zdata, cmap=cmap_name,s=1.5)
    ax.set_zlabel('minutes')
    rms_ = df_sked_tot.loc[sked,'RMS']
    ax.set_title(f'Hole(t) for sked {sked} where RMS = {rms_}')
    
    if(show_plot==True):
        plt.show()
    if(save_fig==True):
        rms_ = str(rms_).replace(".","-")
        plt.savefig(f"/Users/sam/Desktop/NVI/proj4/plots/holes/hole_rms{rms_}_sked{sked}.eps")


#############################################################

path_output = '/Users/sam/Desktop/NVI/proj4/kk_azel2.txt'
path_sk = '/Users/sam/Desktop/NVI/proj4/sk_d10_00h'

path_sked_tot = '/Users/sam/Desktop/NVI/proj4/ut1_d10_00h.txt'
df_sked_tot = pd.read_csv(path_sked_tot,sep='\s+',header=None)
df_sked_tot.rename(columns={8:'RMS',6:'formal_error',2:"num_obs"},inplace=True)
df_sked_tot.sort_values(by=['RMS'],inplace=True)
print(df_sked_tot)

################## PLOT SPECIFIC SCHEDULE ###################
plot_obs = False

sked_nbr = 401 #in order of lowest RMS: 454,89,271,...,437,126,401
df1, df_sk = observations(sked_nbr,path_output,path_sk,plot=plot_obs) # df_sk contains the .solve files, df1 is base circumstances
if(plot_obs==True):
    plt.grid(True)
    plt.legend()
    plt.show()

############## 3D PLOT OBS POSITION OVER TIME ###############
plot_obs_3d = False

if(plot_obs_3d==True):
    plot_3d_obs(df_sk,df1,True)

    plt.show()

#############################################################
#################### OBSERVATION SPREAD #####################

# Different measures of spread of data points:
# min_dist_approach: gets distance to closest other point,
# and uses this as a measure of the spread of data points

testing = True
"""
Saving csv files
"""
saving_csv = False
"""
Distance between adjacent 10 min interval observations
"""
dist_interval_approach = False
"""
Nbr of sources per 10 min interval
"""
nbr_of_sources_approach = False
"""
Finds biggest circle without observations in the sky
"""
circle_approach = False
"""
Turn sky into grid and check sources in each square
"""
grid_approach = False
"""
Use distance to closest other point
"""
min_dist_approach = False
"""
Use the variance of the distance between data point and some point in the sky
"""
var_approach = False

start_time = 5*60
end_time = 55*60

time_list = np.linspace(start_time,end_time,int(1+(end_time-start_time)/60))


if(saving_csv):
    df_interval_obs = pd.DataFrame(columns=('sked+time','sked_RMS','x','y'))

if(dist_interval_approach):
    df_sked_tot["dist_interval_mean"] = np.nan

if(nbr_of_sources_approach):
    df_sked_tot["nbr_sources_mean"] = np.nan
    df_sked_tot["nbr_sources_max"] = np.nan
    df_sked_tot["nbr_sources_min"] = np.nan

if(circle_approach):
    df_sked_tot["hole_mid_x_var"] = np.nan
    df_sked_tot["hole_mid_y_var"] = np.nan

    df_sked_tot["hole_mid_x_rms"] = np.nan
    df_sked_tot["hole_mid_y_rms"] = np.nan

    df_sked_tot["hole_rad"] = np.nan
    df_sked_tot["hole_rms"] = np.nan
    x_min = -87
    y_min = 7

    # df_all_holes = pd.DataFrame(columns=('sked+time','sked_RMS','x','y','rad'))

if(grid_approach==True):
    df_sked_tot["grid_sum"] = np.nan
    df_sked_tot["V_mean"] = np.nan
    x_grid_res = 40
    y_grid_res = 20
    threshold_per_square = 1
    x_max = 89
    x_min = -87
    y_max = 90
    y_min = 7

if(var_approach==True):
    df_sked_tot["var_dist"] = np.nan #get mean value of observations in 10 min interval

if(min_dist_approach==True):
    df_sked_tot["mean_max_min_dist"] = np.nan #get mean value of observations in 10 min interval

colours = ['orange', 'blue']
temp_sked_list = [454,89,271,496,322,266,329,437,126,401] #[454,89,271,496,322,266,329,437,126,401] #[454,89,271,496,322,266,329,437,126,401]#[499,3,12,51,423] #best: 454, worst: 401

#Sked list in ascending order of RMS
sked_list = df_sked_tot.index.tolist()


########## For saving holes_approach ##########
# hole_sked_list = []
# for t in range(501):
#     t = sked_digit_to_string(t)
#     hole_sked_list.extend([f"{t}_x",f"{t}_y",f"{t}_rad"])

# df_holes = pd.DataFrame(index=range(len(time_list)),columns=hole_sked_list)
########################################
min_count = 100
max_count = 0
for sked in range(501):
    if(sked%20 == 0):
        print("sked nbr:",sked)
    sked_nbr = sked
    df1, df_sk = observations(sked_nbr,path_output,path_sk,plot=plot_obs)

    if(saving_csv):
        sked_rms = df_sked_tot.loc[sked,'RMS']
    if(dist_interval_approach):
        obs_dist = np.zeros(shape=(len(time_list)-1,1))
    if(nbr_of_sources_approach):
        nbr_sources = np.zeros(shape=(len(time_list),1))
    if(circle_approach==True):
        hole = np.zeros(shape=(len(time_list),3)) #rows: [mid x, mid y, radius, sked]
    if(grid_approach==True):
        grid_val = np.zeros(shape=(len(time_list),1))
        V = np.zeros(shape=(len(time_list),1)) #the chi-squared test
    if(var_approach==True):
        var_list = np.zeros(shape=(len(time_list),1))
    if(min_dist_approach==True):
        max_min_dist = np.zeros(shape=(len(time_list),1))

    for i in range(len(time_list)):
        df3 = obs_10min_window(df_sk,time_list[i])
        len_df = len(df3['plot_1'])

        if(testing):
            if(len_df > max_count):
                max_count = len_df
            if(len_df < min_count):
                min_count = len_df

        if(saving_csv):
            ###### df_interval_obs['sked+time','sked_RMS','x','y']
            
            x = df3['plot_1'].to_numpy()
            y = df3['plot_2'].to_numpy()

            try:
                prev_ind = df_interval_obs.index[-1]
            except:
                prev_ind = -1
            
            df_interval_obs = df_interval_obs.append(df3[['plot_1','plot_2']].rename(columns={'plot_1':'x','plot_2':'y'}),ignore_index=True)
            
            curr_ind = df_interval_obs.index[-1]
            df_interval_obs.loc[prev_ind+1:curr_ind,'sked+time'] = sked_digit_to_string(sked)+'+'+str(i)
            df_interval_obs.loc[prev_ind+1:curr_ind,'sked_RMS'] = sked_rms
        
        if(dist_interval_approach):
            if(i > 0): #we need to be able to compare to previous interval
                df3_prev = obs_10min_window(df_sk,time_list[i-1])
                x1 = df3['plot_1'].to_numpy()
                y1 = df3['plot_2'].to_numpy()
                x2 = df3_prev['plot_1'].to_numpy()
                y2 = df3_prev['plot_2'].to_numpy()
                obs_dist[i-1,0] = euclidean_dist_difflen(x1,x2,y1,y2)

        if(nbr_of_sources_approach):
            nbr_sources[i,0] = len_df

        if(circle_approach):
            xydata = np.array([df3["plot_1"],df3["plot_2"]]).T

            vor = Voronoi(xydata)
            # voronoi_plot_2d(vor)

            tri = Delaunay(xydata)
            tri_cc = vor.vertices #points in Voronoi diagram

            distance = [0 for i in range(len(tri_cc))]

            points_temp = np.inf*np.ones(shape=(1,2))
            for ii in range(len(tri_cc[:,0])):
                if is_inside(tri_cc[ii,:],df1):
                    point=xydata[tri.simplices[ii,0]]#tri.simplices[tri.vertices[ii,0],:] #the first one, or any other (they are the same distance)
                    distance[ii] = euclidean_dist(tri_cc[ii,0],point[0],tri_cc[ii,1],point[1])

            r_list = np.sort(distance)
            r_max = r_list[-1]

            ind = distance.index(r_max)
            point = tri_cc[ind,:]

            ### We go on to check if larger holes with mid at horizon can be found
            rm_start1 = 80
            rm_end1 = 276
            df1_temp = df1.loc[:rm_start1,['com_hor1','com_hor2']].append(df1.loc[rm_end1:,['com_hor1','com_hor2']])
            df1_temp = df1_temp[df1_temp['com_hor1'] > -40]
            
            for k in df1_temp.index.tolist():
                d_min = -1
                for row_obs in xydata:
                    d1 = euclidean_dist(df1_temp.at[k,'com_hor1'].astype(float),row_obs[0],df1_temp.at[k,'com_hor2'].astype(float),row_obs[1])
                    if(d1 < d_min or d_min == -1):
                        d_min = d1
                if(d_min > r_max):
                    r_max = d_min
                    point = [df1_temp.at[k,'com_hor1'].astype(float), df1_temp.at[k,'com_hor2'].astype(float)]

            rm_start2 = 83
            rm_end2 = 320 #287 för hela vägen ut till com_hor, men hamnar alltid där då
            df2_temp = df1.loc[:rm_start2,['usable1','usable2']].append(df1.loc[rm_end2:,['usable1','usable2']])

            for k in df2_temp.index.tolist():
                d_min = -1
                for row_obs in xydata:
                    d1 = euclidean_dist(df2_temp.at[k,'usable1'].astype(float),row_obs[0],df2_temp.at[k,'usable2'].astype(float),row_obs[1])
                    if(d1 < d_min or d_min == -1):
                        d_min = d1
                if(d_min > r_max):
                    r_max = d_min
                    point = [df2_temp.at[k,'usable1'].astype(float), df2_temp.at[k,'usable2'].astype(float)]

            hole[i,0] = point[0] #hole x-coord
            hole[i,1] = point[1] #hole y-coord
            hole[i,2] = r_max #hole radius
            

            # ############### SAVE ALL HOLES DATA ###############
            # df_all_holes.at[len(time_list)*sked+i,'sked+time'] = sked_digit_to_string(sked)+'+'+str(i)
            # df_all_holes.at[len(time_list)*sked+i,'sked_RMS'] = df_sked_tot.loc[sked,'RMS']
            # df_all_holes.at[len(time_list)*sked+i,'x'] = point[0]
            # df_all_holes.at[len(time_list)*sked+i,'y'] = point[1]
            # df_all_holes.at[len(time_list)*sked+i,'rad'] = r_max

            # print("\n\ndf_all_holes:\n", df_all_holes)
            
            
            ############### SHOW HOLE PLOT ###############
            # show each time interval
            if(False):
                ang = np.linspace(0,2*np.pi,360)
                xp = hole[i,2]*np.cos(ang)
                yp = hole[i,2]*np.sin(ang)

                plot_base(df1)
                plot_observations(df3)

                plt.scatter(hole[i,0], hole[i,1])
                plt.plot(hole[i,0]+xp, hole[i,1]+yp)
                plt.show()
            ######################################

        if(grid_approach):
            idx_list = df3['plot_1'].index.tolist()
            nbr_obs = len(idx_list)
            
            # points_x = np.linspace(x_min,x_max,x_grid_res)
            # points_y = np.linspace(y_min,y_max,y_grid_res)

            grid = np.zeros(shape=(x_grid_res, y_grid_res))

            for ind in idx_list:
                x = df3.loc[ind,'plot_1']
                y = df3.loc[ind,'plot_2']
                
                # print("-----")
                # print(f"(x,y) = ({x},{y})")
                
                ix = math.floor(x_grid_res*(x - x_min)/(x_max-x_min))
                iy = math.floor(y_grid_res*(y - y_min)/(y_max-y_min))

                # print(f"(ix,iy) = ({ix},{iy})")

                grid[ix,iy] += 1

            if(False):
                v = 0
                v1 = 0
                ps = (x_grid_res*y_grid_res)
                for q1 in range(x_grid_res):
                    for q2 in range(y_grid_res): #there are x_grid_res*y_grid_res categories and nbr_obs indep. obs.
                        v += np.power(grid[q1,q2]-nbr_obs*ps,2)/(nbr_obs*x_grid_res*y_grid_res)

                V[i,0] = v

                # grid = np.zeros(shape=(x_grid_res, y_grid_res))
                for ind1 in range(len(grid[:,0])):
                    for ind2 in range(len(grid[0,:])):
                        g = grid[ind1,ind2]
                        if(g > 3):
                            grid_val[i,0] += g-1
                        # grid_val = np.sum([g-1 if g > 0 else 0 for g in grid])

            # plot_base(df1)
            # plot_observations(df3)
            # plt.show()


            """
            grid_ = np.array([np.linspace(x_min,x_max,x_grid_res),
                              np.linspace(y_min,y_max,y_grid_res)])

            grid_dist = -1*np.ones(shape=(x_grid_res,y_grid_res,3))
            
            for x in range(x_grid_res):
                for y in range(y_grid_res):
                    x_val = grid_[0][x]
                    y_val = grid_[1][y]
                    if(is_inside([x_val,y_val],df1)):
                        for ind in idx_list:
                            dist = np.sqrt(np.power(x_val-df3.at[ind,'plot_1'].astype(float),2)+np.power(y_val-df3.at[ind,'plot_2'].astype(float),2))
                            if(dist < grid_dist[x,y,2] or grid_dist[x,y,2] == -1):
                                grid_dist[x,y,0] = x_val
                                grid_dist[x,y,1] = y_val
                                grid_dist[x,y,2] = dist

            dist_max = grid_dist.max()
            print("dist_max:\n",dist_max)
            grid_val[i,2] = dist_max
            xy_coord = np.where(grid_dist==dist_max)
            print("xy_coord:",xy_coord)
            
            # grid_val[i,0] = grid_[0][xy_coord[0][0]]
            # grid_val[i,1] = grid_[1][xy_coord[1][0]]

            ang = np.linspace(0,2*np.pi,360)
            xp = dist_max*np.cos(ang)
            yp = dist_max*np.sin(ang)
            plt.plot(grid_val[i,0]+xp, grid_val[i,1]+yp)
            plt.scatter(grid_val[i,0], grid_val[i,1])
            plt.show()
            """

        if(var_approach==True):
            x = 20
            y = 60
            #Note that this does not take direction into account
            d1 = np.sqrt(np.power(df3["plot_1"].astype(float)-20,2)+
                         np.power(df3["plot_2"].astype(float)-60,2))

            var_list[i] = np.var(d1)

        if(min_dist_approach==True):
            min_dist = np.inf*np.ones(shape=(len_df,1))

            idx_list = df3['plot_1'].index.tolist()

            for i1 in idx_list:
                for i2 in idx_list:
                    if(i1 != i2):
                        d1 = np.sqrt(np.power(df3["plot_1"].loc[i1].astype(float)-df3["plot_1"].loc[i2].astype(float),2)+
                                    np.power(df3["plot_2"].loc[i1].astype(float)-df3["plot_2"].loc[i2].astype(float),2))
                        
                        # Get smallest distance to next point
                        #if(d1 < df3['min_dist'].loc[i1]):
                        if(d1 < min_dist[i1%len_df]):
                            min_dist[i1%len_df] = d1
                            #df3.at[i1,'min_dist'] = d1

            max_min_dist[i] = np.sum(min_dist) #sum of sqrt to increase sig. of multiple distances (more uniform)
            #max_min_dist[i] = df3['min_dist'].max()

    if(dist_interval_approach):
        df_sked_tot.at[sked,"dist_interval_mean"] = np.mean(obs_dist[:,0])

    if(nbr_of_sources_approach):
        df_sked_tot.at[sked,"nbr_sources_mean"] = np.mean(nbr_sources[:,0])
        df_sked_tot.at[sked,"nbr_sources_max"] = np.max(nbr_sources[:,0])
        df_sked_tot.at[sked,"nbr_sources_min"] = np.min(nbr_sources[:,0])

    if(circle_approach==True):
        # sked = sked_digit_to_string(sked)
        # df_holes[f"{sked}_x"] = hole[:,0]
        # df_holes[f"{sked}_y"] = hole[:,1]
        # df_holes[f"{sked}_rad"] = hole[:,2]

        plot_circles = False
        if(plot_circles==True):
            for k in range(len(hole[:,0])):
                q = k%2
                ang = np.linspace(0,2*np.pi,360)
                xp = hole[k,2]*np.cos(ang)
                yp = hole[k,2]*np.sin(ang)

                plt.plot(hole[k,0]+xp,hole[k,1]+yp,color=colours[q],linewidth=1)
                plt.scatter(hole[k,0],hole[k,1],color=colours[q],s=1)

            # plt.show()
        
        # plt.plot(hole[:,0],hole[:,1],'o', label=f"Hole position over time for sked {sked}",linewidth=1)
        # if(sked != 500):
        #     if(rms_ > 10):
        #         plt.plot(hole[:,2], label=f"sked {sked}, rms {rms_}",c='red',linewidth=1)
        #     elif(rms_ < 6):
        #         plt.plot(hole[:,2], label=f"sked {sked}, rms {rms_}",c='tab:blue',linewidth=1)
        #     else:
        #         plt.plot(hole[:,2], label=f"sked {sked}, rms {rms_}",c='orange',linewidth=1)
        # else:
        #     plt.plot(hole[:,2], label=f"sked cov, rms {rms_}",c='green',linewidth=1)
        # plt.show()

        df_sked_tot.at[sked,"hole_mid_x_var"] = np.var(hole[:,0])
        df_sked_tot.at[sked,"hole_mid_y_var"] = np.var(hole[:,1])

        df_sked_tot["hole_mid_x_rms"] = rms_val(hole[:,0])
        df_sked_tot["hole_mid_y_rms"] = rms_val(hole[:,1])

        df_sked_tot.at[sked,"hole_rad"] = np.mean(hole[:,2])
        df_sked_tot.at[sked,"hole_rms"] = rms_val(hole[:,2])


        # plot_hole_mvmt(hole,sked,df_sked_tot,show_plot=True,save_fig=False)

    if(grid_approach==True):
        df_sked_tot.at[sked,"V_mean"] = np.mean(V[:,0])
        df_sked_tot.at[sked,"grid_sum"] = np.mean(grid_val[:,0])

    if(min_dist_approach==True):
        df_sked_tot.at[sked,"mean_max_min_dist"] = np.mean(max_min_dist)

    if(var_approach==True):
        df_sked_tot.at[sked,"var_dist"] = np.mean(var_list)

if(testing):
    print(f"min_count: {min_count}")
    print(f"max_count: {max_count}")

if(saving_csv):
    df_interval_obs.to_csv(path_or_buf='/Users/sam/Desktop/NVI/proj4/sked_data/interval_obs.csv', sep=',')

if(dist_interval_approach):
    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  df_sked_tot["dist_interval_mean"].to_numpy() #.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    plt.plot(x_lin,y_pred,c='k')
    ###
    plt.scatter(xdata, ydata, c='tab:blue', s=5)
    plt.ylabel('mean distance between observations in 10 min interval')
    plt.xlabel('UT1_RMS')
    plt.title('Cov = '+str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)
    plt.show()

if(nbr_of_sources_approach):
    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  df_sked_tot["nbr_sources_mean"].to_numpy() #.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    plt.plot(x_lin,y_pred,c='k')
    ###
    plt.scatter(xdata, ydata, c='tab:blue', s=5)
    plt.ylabel('nbr_sources_mean')
    plt.xlabel('UT1_RMS')
    plt.title('Cov = '+str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)
    plt.show()

    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  df_sked_tot["nbr_sources_max"].to_numpy() #.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    plt.plot(x_lin,y_pred,c='k')
    ###
    plt.scatter(xdata, ydata, c='tab:blue', s=5)
    plt.ylabel('nbr_sources_max')
    plt.xlabel('UT1_RMS')
    plt.title('Cov = '+str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)
    plt.show()

    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  df_sked_tot["nbr_sources_min"].to_numpy() #.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    plt.plot(x_lin,y_pred,c='k')
    ###
    plt.scatter(xdata, ydata, c='tab:blue', s=5)
    plt.ylabel('nbr_sources_min')
    plt.xlabel('UT1_RMS')
    plt.title('Cov = '+str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)
    plt.show()

if(circle_approach):
    ##Save all df_holes to csv or something
    # df_all_holes.to_csv(path_or_buf='/Users/sam/Desktop/NVI/proj4/sked_data/XXXXXXXXX.csv', sep=',')

    if(False):
        xdata = df_sked_tot['RMS'].to_numpy()
        ydata =  df_sked_tot["hole_rms"].to_numpy() #.reshape((-1, 1))
        model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
        x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
        y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
        plt.plot(x_lin,y_pred,c='k')
        ###
        plt.scatter(xdata, ydata, c='tab:blue', s=5)
        plt.ylabel('RMS of hole radius (sep. 10 min int)')
        plt.xlabel('UT1_RMS')
        plt.title('Cov = '+str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                            fontsize=10)
        plt.show()

        xdata = df_sked_tot['RMS'].to_numpy()
        ydata =  df_sked_tot.at[sked,"hole_rad"].to_numpy() #.reshape((-1, 1))
        model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
        x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
        y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
        plt.plot(x_lin,y_pred,c='k')
        ###
        plt.scatter(xdata, ydata, c='tab:blue', s=5)
        plt.ylabel('mean radius of hole (sep. 10 min int)')
        plt.xlabel('UT1_RMS')
        plt.title('Cov = '+str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                            fontsize=10)
        plt.show()
    
        # plt.savefig(f"/Users/sam/Desktop/NVI/proj4/plots/holes/hole_rms{rms_}_sked{sked}.eps")

if(grid_approach==True):
    # plt.scatter(df_sked_tot["RMS"], df_sked_tot["V_mean"])
    # plt.show()

    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  df_sked_tot["V_mean"].to_numpy() #.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    plt.plot(x_lin,y_pred,c='k')
    ###
    plt.scatter(xdata, ydata, c='tab:blue', s=5)
    plt.ylabel('V_mean')
    plt.xlabel('UT1_RMS')
    plt.title('Cov = '+str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)
    
    plt.show()

    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  df_sked_tot["grid_sum"].to_numpy() #.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    plt.plot(x_lin,y_pred,c='k')
    ###
    plt.scatter(xdata, ydata, c='tab:blue', s=5)
    plt.ylabel('grid_sum')
    plt.xlabel('UT1_RMS')
    plt.title('Cov = '+str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)
    
    plt.show()

if(var_approach==True):
    cov_mat1 = np.cov(df_sked_tot["var_dist"].astype(float),df_sked_tot["RMS"].astype(float))
    print("cov(var_dist, RMS):\n", cov_mat1)
    cov_mat2 = np.cov(df_sked_tot["var_dist"].astype(float),df_sked_tot["formal_error"].astype(float))
    print("cov(var_dist, formal_error):\n", cov_mat2)

    cov_mat3 = np.cov(df_sked_tot["num_obs"].astype(float),df_sked_tot["RMS"].astype(float))
    print("cov(num_obs, RMS):\n", cov_mat3)
    cov_mat4 = np.cov(df_sked_tot["num_obs"].astype(float),df_sked_tot["formal_error"].astype(float))
    print("cov(num_obs, formal_error):\n", cov_mat4)

    plt.scatter(df_sked_tot["RMS"],df_sked_tot["var_dist"])
    plt.xlabel("RMS")
    plt.ylabel("mean var_dist")
    plt.show()

if(min_dist_approach==True):
    cov_mat1 = np.cov(df_sked_tot["mean_max_min_dist"].astype(float),df_sked_tot["RMS"].astype(float))
    print("cov(mean_max_min_dist, RMS):\n", cov_mat1)
    cov_mat2 = np.cov(df_sked_tot["mean_max_min_dist"].astype(float),df_sked_tot["formal_error"].astype(float))
    print("cov(mean_max_min_dist, formal_error):\n", cov_mat2)

    cov_mat3 = np.cov(df_sked_tot["num_obs"].astype(float),df_sked_tot["RMS"].astype(float))
    print("cov(num_obs, RMS):\n", cov_mat3)
    cov_mat4 = np.cov(df_sked_tot["num_obs"].astype(float),df_sked_tot["formal_error"].astype(float))
    print("cov(num_obs, formal_error):\n", cov_mat4)

    plt.scatter(df_sked_tot["RMS"],df_sked_tot["mean_max_min_dist"])
    plt.xlabel("RMS")
    plt.ylabel("mean_max_min_dist")
    plt.show()

















"""
#############################################################
########################### TOTAL ###########################

plot2 = False
path = '/Users/sam/Desktop/NVI/proj4'
file_name = 'ut1_d10_00h.txt'
df = pd.read_csv(path+"/"+file_name,sep='\s+',header=None)
df.rename(columns={8:'RMS',6:'formal_error',2:"num_obs"},inplace=True)
df.sort_values(by=['RMS'],inplace=True)

df['nbr_unique'] = np.nan

ind_list = list(df.index)

# for ind in ind_list: #gives 24 sources for all schedules
#     df1, df_sk = observations(ind,plot=True,color='green')
#     df.at[ind,'nbr_unique'] = df_sk.loc[30,'nbr_sources'].astype(float)

for ind in ind_list:
    df1, df_sk = observations(ind,plot=True,color='green')
    df.at[ind,'nbr_unique'] = df_sk['Source'].nunique() #already says in the .solve file, but gives different answer

column_list = list(df.columns)
column_list = [col for col in column_list if not isinstance(col,int)] #filter for only non-int columns
print(df)
for n in range(len(column_list)):
    for m in range(n+1,len(column_list)):
        print(f"=========== col1: {column_list[n]}, col2: {column_list[m]} ===========")
        cov_mat = np.cov(df[column_list[n]].astype(float),df[column_list[m]].astype(float))
        print(cov_mat)
        print("Pearson's correlation coefficient:", cov_mat[0,1]/np.sqrt(cov_mat[0,0]*cov_mat[1,1])) #most meaningful for normal distr.
        sp_corr, _ = spearmanr(df[column_list[n]].astype(float),df[column_list[m]].astype(float))
        print("Spearman's correlation coefficient:",sp_corr)
        print("===================================================")


if(plot2==True):
    ind_list_ = ind_list
    for ind_ in ind_list_:
        observations(ind_,plot=True)
    plt.show()

if(plot2==True):
    plt.plot(df['RMS'],label='RMS')
    plt.plot(df['formal_error'],label='formal_error')
    plt.legend()
    plt.show()

"""