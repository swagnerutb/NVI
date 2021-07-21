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
    """Get sked in {int} or {string} and return sked in three digit string"""
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

def plot_base(df):
    """Plot horizon, usable horizon, mutual horizon"""
    #Plot horizon
    #good: start 80, fin 340
    # start = 80
    # fin = 276 #max: 361

    # plt.plot(df.loc[fin:,'hor1'],df.loc[fin:,'hor2'],c='tab:blue')
    # plt.plot(df.loc[:start,'hor1'],df.loc[:start,'hor2'],c='tab:blue')
    plt.plot(df['hor1'],df['hor2'],c='tab:blue')

    #Plot usable horizon
    #good: start 83, fin 287
    # start = 83
    # fin = 320
    # plt.plot(df.loc[:start,"usable1"],df.loc[:start,"usable2"],'r--',linewidth=1,c='red')
    # plt.plot(df.loc[fin:,"usable1"],df.loc[fin:,"usable2"],'r--',linewidth=1,c='red')
    plt.plot(df["usable1"],df["usable2"],'r--',linewidth=1,c='red')

    #Plot common view
    #good: start 80, fin 265
    rm_start = 80
    rm_end = 276
    plt.plot(df.loc[:rm_start,"com_hor1"],df.loc[:rm_start,"com_hor2"],c='grey')
    plt.plot(df.loc[rm_end:,"com_hor1"],df.loc[rm_end:,"com_hor2"],c='grey')

def euclidean_dist(p1_x, p1_y, p2_x, p2_y,):
    return np.sqrt(np.power(p1_x-p2_x,2)+np.power(p1_y-p2_y,2))

def rms_val(data):
    """Calculate RMS"""
    val = 0
    for d in data:
        val += np.power(d,2)
    return np.sqrt(val/len(data))

def get_3d_linreg_coeffs(x1data,x2data,ydata):
    """Get coeffs in 3D lin. reg."""
    x1 = x1data - np.mean(x1data)
    x2 = x2data - np.mean(x2data)
    y = ydata - np.mean(ydata)

    beta1 = (np.sum(y*x1)*np.sum(x2*x2) - np.sum(y*x2)*np.sum(x1*x2))/(np.sum(x1*x1)*np.sum(x2*x2) - np.sum(x1*x2)*np.sum(x1*x2))
    beta2 = (np.sum(y*x2)*np.sum(x1*x1) - np.sum(y*x1)*np.sum(x1*x2))/(np.sum(x1*x1)*np.sum(x2*x2) - np.sum(x1*x2)*np.sum(x1*x2))
    beta0 = np.mean(ydata) - beta1*np.mean(x1data) - beta2*np.mean(x2data)

    return beta0, beta1, beta2

def plot_hole_mvmt(sked,df_holes,df_sked_tot,df_base,save_fig=False):
    """Plot hole(t) in given sked"""
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    sked = sked_digit_to_string(sked)
    plot_base(df_base)

    mid_x = df_holes[f"{sked}_x"]
    mid_y = df_holes[f"{sked}_y"]
    r =  df_holes[f"{sked}_rad"]

    zdata = np.linspace(0,len(mid_x)-1,len(mid_x))

    cmap_name = 'winter'
    cmap = matplotlib.cm.get_cmap(cmap_name)

    for k in range(len(mid_x)):
        ang = np.linspace(0,2*np.pi,360)
        xp = r[k]*np.cos(ang)
        yp = r[k]*np.sin(ang)
        #ax.plot3D(mid_x[k]+xp,mid_y[k]+yp,zdata[k],c=cmap(k/(len(mid_x)-1)),linewidth=1)

    ax.scatter3D(mid_x, mid_y, zdata, c=zdata, cmap=cmap_name,s=1.5)
    ax.set_zlabel('minutes')
    rms_ = df_sked_tot.loc[int(sked),'RMS']
    ax.set_title(f'Hole(t) for sked {sked} where RMS = {rms_}')

    if(save_fig==True):
        rms_ = str(rms_).replace(".","-")
        plt.savefig(f"/Users/sam/Desktop/NVI/proj4/plots/holes/hole_rms{rms_}_sked{sked}.eps")


########## Paths ##########
path_sked_tot = '/Users/sam/Desktop/NVI/proj4/ut1_d10_00h.txt'
path_sked_holes = '/Users/sam/Desktop/NVI/proj4/sked_data/holes_data.csv'
path_output = '/Users/sam/Desktop/NVI/proj4/kk_azel2.txt'

########## Reading csv files ##########
## Info on all skeds
df_sked_tot = pd.read_csv(path_sked_tot,sep='\s+',header=None)
df_sked_tot.rename(columns={8:'RMS',6:'formal_error',2:"num_obs"},inplace=True)
df_sked_tot.sort_values(by=['RMS'],inplace=True)
print("\n##################################\n",df_sked_tot,"\n##################################\n")

## Base data: Horizon, common horizon and usable horizon
df_base = read_base(path_output)

## Data on holes in skeds
df_holes = pd.read_csv(path_sked_holes,sep=',',index_col=0)
df_holes_indexlist = df_holes.index.tolist()
df_holes_collist = df_holes.columns.tolist()
print("\n##################################\n",df_holes,"\n##################################\n")


## Data for individual skeds
#sked_nbr = 126
#path_output = '/Users/sam/Desktop/NVI/proj4/kk_azel2.txt'
#path_sk = '/Users/sam/Desktop/NVI/proj4/sk_d10_00h'
#df1, df_sk = observations(sked_nbr,path_output,path_sk,plot=plot_obs)

########## Calculations ##########
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


########## Plotting ##########
## Plot base
if(True):
    plot_base(df_base)
    plt.scatter(20,60,marker='x',c='r')
    plt.show()


## Plotting RMS of x-coord, y-coord, hole radius and euclidean distance to some point vs UT1_RMS
if(True):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('RMS vs UT1_RMS')
    
    ### Lin. reg. - x-coord
    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  holes_rms[:,0]#.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    axs[0, 0].plot(x_lin,y_pred,c='k')
    ###
    axs[0, 0].scatter(xdata, ydata, c='tab:blue', s=5)
    axs[0, 0].set(ylabel = 'RMS_x-coord')
    axs[0, 0].set_title('Cov = '+str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)

    ### Lin. reg. - y-coord
    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  holes_rms[:,1]#.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    axs[0, 1].plot(x_lin,y_pred,c='k')
    ###
    axs[0, 1].scatter(xdata, ydata, c='tab:orange', s=5)
    axs[0, 1].set(ylabel = 'RMS_y-coord')
    axs[0, 1].set_title('Cov = ' + str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)

    ### Lin. reg. - radius
    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  holes_rms[:,2]#.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    axs[1, 0].plot(x_lin,y_pred,c='k')
    ###
    axs[1, 0].scatter(xdata, ydata, c='tab:green', s=5)
    axs[1, 0].set(ylabel = 'RMS_holeSize', xlabel='UT1_RMS')
    axs[1, 0].set_title('Cov = ' + str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)

    ### Lin. reg. - distance to point
    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  rms_dist_to_p#.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    axs[1, 1].plot(x_lin,y_pred,c='k')
    ###
    axs[1, 1].scatter(xdata, ydata, c='tab:red', s=5)
    axs[1, 1].set(ylabel = 'RMS_of_distToPoint[20,60]', xlabel='UT1_RMS')
    axs[1, 1].set_title('Cov = ' + str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)

    plt.show()


## Plotting variance of x-coord, y-coord, hole radius and euclidean distance to some point vs UT1_RMS
if(True):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Variance vs UT1_RMS')

    ### Lin. reg. - x-coord
    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  holes_var[:,0]#.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    axs[0, 0].plot(x_lin,y_pred,c='k')
    ###
    axs[0, 0].scatter(xdata, ydata, c='tab:blue', s=5)
    axs[0, 0].set(ylabel = 'var_x-coord')
    axs[0, 0].set_title('Cov = ' + str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)

    ### Lin. reg. - y-coord
    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  holes_var[:,1]#.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    axs[0, 1].plot(x_lin,y_pred,c='k')
    ###
    axs[0, 1].scatter(xdata, ydata, c='tab:orange', s=5)
    axs[0, 1].set(ylabel = 'var_y-coord')
    axs[0, 1].set_title('Cov = ' + str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)

    ### Lin. reg. - radius
    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  holes_var[:,2]#.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    axs[1, 0].plot(x_lin,y_pred,c='k')
    ###
    axs[1, 0].scatter(xdata, ydata, c='tab:green', s=5)
    axs[1, 0].set(ylabel = 'var_holeSize', xlabel='UT1_RMS')
    axs[1, 0].set_title('Cov = ' + str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)

    ### Lin. reg. - distance to point
    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  var_dist_to_p#.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    axs[1, 1].plot(x_lin,y_pred,c='k')
    ###
    axs[1, 1].scatter(xdata, ydata, c='tab:red', s=5)
    axs[1, 1].set(ylabel = 'var_of_distToPoint[20,60]', xlabel='UT1_RMS')
    axs[1, 1].set_title('Cov = ' + str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)

    plt.show()


## Plotting RMS of hole radius vs. mean of hole radius vs. UT1_RMS
if(True):
    fig, axs = plt.subplots(2, 2)
    # fig.suptitle('RMS vs UT1_RMS')

    ### Lin. reg. - Hole_rms (rms of size) vs. UT1_RMS 
    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  holes_rms[:,2]#.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    axs[0, 0].plot(x_lin,y_pred,c='k')
    ###
    axs[0, 0].scatter(xdata, ydata, c='tab:blue', s=5)
    axs[0, 0].set(ylabel='holes_RMS', xlabel='UT1_RMS')
    axs[0, 0].set_title('Cov = '+str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)

    ## Lin. reg. - Hole_mean (mean of size) vs. UT1_RMS
    xdata = df_sked_tot['RMS'].to_numpy()
    ydata =  holes_mean[:,2]#.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    axs[0, 1].plot(x_lin,y_pred,c='k')
    ###
    axs[0, 1].scatter(xdata, ydata, c='tab:orange', s=5)
    axs[0, 1].set(ylabel='holes_mean', xlabel='UT1_RMS')
    axs[0, 1].set_title('Cov = '+str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)

    ## Lin. reg. - Hole_rms (rms of size) vs. Hole_mean (mean of size)
    xdata = holes_mean[:,2]
    ydata =  holes_rms[:,2]#.reshape((-1, 1))
    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    axs[1, 0].plot(x_lin,y_pred,c='k')
    ###
    axs[1, 0].scatter(xdata, ydata, c='tab:green', s=5)
    axs[1, 0].set(ylabel='holes_rms', xlabel='holes_mean')
    axs[1, 0].set_title('Cov = '+str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)


    #-----------
    ## Lin. reg. - Hole_rms (rms of size) vs. Hole_mean (mean of size)
    xdata = df_sked_tot['RMS'].to_numpy()
    # ydata =  np.multiply(holes_rms[:,2],holes_mean[:,2])#.reshape((-1, 1))
    ydata = np.multiply(np.amax(holes_rad,1)-np.amin(holes_rad,1),holes_var[:,2])

    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)
    axs[1, 1].plot(x_lin,y_pred,c='k')
    ###
    axs[1, 1].scatter(xdata, ydata, c='tab:red', s=5)
    axs[1, 1].set(ylabel='---', xlabel='UT1_RMS')
    axs[1, 1].set_title('Cov = '+str(np.round(np.cov(xdata,ydata)[0,1],5))+' with linreg: y = '+str(np.round(model.coef_[0],3))+'x + '+str(np.round(model.intercept_,3)),
                        fontsize=10)
    
    plt.show()


## 3D plot of holes rms vs. holes mean vs UT1_RMS
if(True):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    cmap_name = 'winter'
    cmap = matplotlib.cm.get_cmap(cmap_name)

    xdata = holes_rms[:,2]
    ydata = holes_mean[:,2]
    zdata = df_sked_tot['RMS']

    model = LinearRegression().fit(xdata.reshape((-1, 1)), ydata)
    x_lin = np.linspace(np.min(xdata),np.max(xdata),20)
    y_pred = model.intercept_ + np.multiply(model.coef_,x_lin)

    beta0, beta1, beta2 = get_3d_linreg_coeffs(xdata,ydata,zdata)
    y_lin = np.linspace(np.min(ydata),np.max(ydata),20)

    ax.plot3D(x_lin, y_pred, beta0 + beta1*x_lin + beta2*y_lin,c='k')

    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap=cmap_name, s=2.5)
    ax.set_xlabel('hole size rms')
    ax.set_ylabel('hole size mean')
    ax.set_zlabel('UT1_RMS')
    # rms_ = df_sked_tot.loc[int(sked),'RMS']
    # ax.set_title(f'Hole(t) for sked {sked} where RMS = {rms_}')

    print("\n##################################")
    print("Coefficients in 3D linreg:")
    print(f"     beta0 = {beta0}")
    print(f"     beta1 = {beta1}")
    print(f"     beta2 = {beta2}")
    print(f"     UT1_RMS = {np.round(beta0,5)} + {np.round(beta1,5)}*holes_rms + {np.round(beta2,5)}*holes_mean")
    print("##################################")

    plt.show()


## Plot holes of given sked
if(True):
    sked = 401
    plot_hole_mvmt(sked,df_holes,df_sked_tot,df_base)
    plt.show()
