import numpy as np

deg2rad = np.pi/180

def AzEl2HaDec(az_deg,el_deg,lat_deg,long_deg,ha_deg,dec_deg):
    az = deg2rad*az_deg
    el = deg2rad*el_deg
    lat = deg2rad*lat_deg

    ha = np.arctan2(-np.sin(az)*np.cos(el), -np.cos(az)*np.sin(lat)*np.cos(el)+np.sin(el)*np.cos(lat))
    dec = np.arcsin(np.sin(lat)*np.sin(el)+np.cos(lat)*np.cos(el)*np.cos(az))

    ha_deg = ha/deg2rad
    dec_deg = dec/deg2rad

    return [az_deg,el_deg,lat_deg,long_deg,ha_deg,dec_deg]

def HaDec2AzEl(ha_deg,dec_deg,lat_deg,long_deg,az_deg,el_deg):
    ha  = ha_deg*deg2rad
    dec = dec_deg*deg2rad
    lat = lat_deg*deg2rad
    long = long_deg*deg2rad

    ch = np.cos(ha)
    sh = np.sin(ha)
    cl = np.cos(lat)
    sl = np.sin(lat)
    cd = np.cos(dec)
    sd = np.sin(dec)

    x = -ch*cd*sl+sd*cl
    y = -sh*cd
    z = ch*cd*cl+sd*sl

    r = np.sqrt(x*x+y*y)
    az = np.arctan2(y,x)
    el = np.arctan2(z,r)

    az_deg = az/deg2rad
    el_deg = el/deg2rad
    return [ha_deg,dec_deg,lat_deg,long_deg,az_deg,el_deg]

def norm(arr):
    norm = 0
    for i in range(len(arr)):
        norm = norm + np.power(arr[i],2)
    return np.sqrt(norm)



# 3d position of Kokee and Wettzell
xyz_kk = [-5543831.7445, -2054585.5895, 2387828.9744]
xyz_wz = [4075658.8067, 931824.8831, 4801516.2728]


#Kokee:
long_kk = np.arctan2(xyz_kk[1],xyz_kk[0])/deg2rad
lat_kk  = np.arcsin(xyz_kk[2]/norm(xyz_kk))/deg2rad
#Wetzell:
long_wz  = np.arctan2(xyz_wz[1],xyz_wz[0])/deg2rad
lat_wz   = np.arcsin(xyz_wz[2]/norm(xyz_wz))/deg2rad

file_name = "output_azel_testpy.txt"
f = open(file_name, "w")
f.write("Kokee,"+ str(long_kk) + "," + str(lat_kk))
f.write("\nWettzell" + "," + str(long_wz) + "," + str(lat_wz))

el_deg_wz = 8.0 #Default Lower el at Wettzell.

f2 = open(file_name,"a")
f2.write("\nWz_Az, Wz_El, Kk_Az, Kk_El")
for iaz in range(360):
    az_deg_wz = iaz
    try:
        [az_deg_wz,el_deg_wz,lat_wz,long_wz,ha_deg,dec_deg] = AzEl2HaDec(az_deg_wz,el_deg_wz,lat_wz,long_wz,ha_deg,dec_deg)
    except:
        ha_deg = 0
        dec_deg = 0
        [az_deg_wz,el_deg_wz,lat_wz,long_wz,ha_deg,dec_deg] = AzEl2HaDec(az_deg_wz,el_deg_wz,lat_wz,long_wz,ha_deg,dec_deg)
    ha_deg = ha_deg + (long_kk-long_wz)

    try:
        [ha_deg,dec_deg,lat_deg,long_deg,az_deg_kk,el_deg_kk] = HaDec2AzEl(ha_deg,dec_deg,lat_kk,long_kk,az_deg_kk,el_deg_kk)
    except:
        az_deg_kk = 0
        el_deg_kk = 0
        [ha_deg,dec_deg,lat_deg,long_deg,az_deg_kk,el_deg_kk] = HaDec2AzEl(ha_deg,dec_deg,lat_kk,long_kk,az_deg_kk,el_deg_kk)


    f2.write("\n" + str(az_deg_wz) + "," + str(el_deg_wz) + "," + str(az_deg_kk) + "," + str(el_deg_kk))

f.close()