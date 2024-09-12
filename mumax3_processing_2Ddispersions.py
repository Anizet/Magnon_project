import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.solid_capstyle'] = "round"
plt.rcParams['lines.dash_capstyle'] = "round"
plt.rcParams['figure.dpi'] = 150
#plt.rcParams['font.family'] = 'Georgia'

color1 = "gold"


#Define the a directory path using GUI, to convert OVF files to numpy files
def get_dir_path():
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory()
    return dir_path

dir_path = get_dir_path()


from subprocess import run, PIPE, STDOUT
from glob import glob
from os import path
from numpy import load

# p : convert all ovf files in the output directory to numpy files
#Regions data
p = run(["mumax3-convert","-numpy",dir_path+"/regions00*.ovf"], stdout=PIPE, stderr=STDOUT)
#Read the numpy files
fields = {}
for npyfile in glob(dir_path+"/regions00*.npy"):
    key = path.splitext(path.basename(npyfile))[0]
    fields[key] = load(npyfile)
regions = np.stack([fields[key] for key in sorted(fields.keys())])
np.shape(regions)
plt.imshow(np.log(regions[0,0,:,168,:]),aspect='auto', origin='lower', cmap="twilight_shifted")
plt.show()

# m YIG data
p = run(["mumax3-convert","-numpy",dir_path+"/m_z_zrange29*.ovf"], stdout=PIPE, stderr=STDOUT)
fields = {} 
for npyfile in glob(dir_path+"/m_x_zrange27*.npy"):
    key = path.splitext(path.basename(npyfile))[0]
    fields[key] = load(npyfile)

# mPY YIG data
p = run(["mumax3-convert","-numpy",dir_path+"/m_x.region1_zrange33*.ovf"], stdout=PIPE, stderr=STDOUT)
fields = {} 
for npyfile in glob(dir_path+"/m_x.region1_zrange33*.npy"):
    key = path.splitext(path.basename(npyfile))[0]
    fields[key] = load(npyfile)



#Stack snapshots of the magnetization on top of each other
m = np.stack([fields[key] for key in sorted(fields.keys())])
mPY = np.stack([fields[key] for key in sorted(fields.keys())])
mz = np.stack([fields[key] for key in sorted(fields.keys())])
m0 = np.stack([fields[key] for key in sorted(fields.keys())])
m_disp = np.stack([fields[key] for key in sorted(fields.keys())])

mx27= np.stack([fields[key] for key in sorted(fields.keys())])
mz27= np.stack([fields[key] for key in sorted(fields.keys())])
mx14= np.stack([fields[key] for key in sorted(fields.keys())])
mz14= np.stack([fields[key] for key in sorted(fields.keys())])
mx0= np.stack([fields[key] for key in sorted(fields.keys())])
mz0= np.stack([fields[key] for key in sorted(fields.keys())])

#m_x x-y plot for time
for i in range(np.shape(m)[0]):
    plt.close()
    mxy = m[i, 0, 0, :, :]
    extent_x_y = [0,7840,0,1344]
    plt.imshow((mxy), origin='lower', cmap="twilight_shifted", extent = extent_x_y, vmin=-0.3, vmax=0.3)
    plt.xlabel("x [nm]")
    plt.ylabel("y [nm]")
    plt.colorbar(label = "m_x [-]")
    plt.title(f"x-y slice at center of YIG slab")
    plt.savefig(f"mx_xy_slice_zrange14_t{i*0.05:.2f}.png", bbox_inches='tight', dpi = 300)
    #plt.show()

np.shape(mx27)
# Mean abs mx vs mean abs mz as a function of z
mx_27 = mx27[:, 0, 0, :, :]
mz_27 = mz27[:, 0, 0, :, :]
mx_14 = mx14[:, 0, 0, :, :]
mz_14 = mz14[:, 0, 0, :, :]
mx_0 = mx0[:, 0, 0, :, :]
mz_0 = mz0[:, 0, 0, :, :]

mx_27_mean = []
mz_27_mean = []

mx_14_mean = []
mz_14_mean = []

mx_00_mean = []
mz_00_mean = []

for i in range(np.shape(mx_27)[0]):
    mx_27_mean.append(np.mean((mx_27[i,:,:])))
    mz_27_mean.append(np.mean((mz_27[i,:,:])))
    mx_14_mean.append(np.mean((mx_14[i,:,:])))
    mz_14_mean.append(np.mean((mz_14[i,:,:])))
    mx_00_mean.append(np.mean((mx_0[i,:,:])))
    mz_00_mean.append(np.mean((mz_0[i,:,:])))

t = np.linspace(0,np.shape(mx_27)[0],np.shape(mx_27)[0] ) * 0.05
plt.plot(t, mx_27_mean, "r*-", label = "m_x, PY-YIG boundary")
plt.plot(t, mz_27_mean, "b*-", label = "m_z, PY-YIG boundary")
plt.plot(t, mx_14_mean, "ro-", label = "m_x, YIG center")
plt.plot(t, mz_14_mean, "bo-", label = "m_z, YIG center")
plt.plot(t, mx_00_mean, "rd-", label = "m_x, YIG bottom")
plt.plot(t, mz_00_mean, "bd-", label = "m_z, YIG bottom")
plt.grid()
plt.xlabel("t [ns]")
plt.ylabel("m_* absolute mean amplitude [-]")
plt.legend()
plt.show()


dt = 0.05*1e-9
#Get the number of data points
N = np.shape(mx_27_mean)[0]
#Get the frequency range
f = np.fft.fftfreq(N, dt)
#Get the fft of the data
#Plot the fft
plt.plot(f, np.abs(np.fft.fft(mx_27_mean)), label='mx_z0')
plt.plot(f, np.abs(np.fft.fft(mx_14_mean)), label='mx_z14')
plt.plot(f, np.abs(np.fft.fft(mx_00_mean)), label='mx_z27')
plt.plot(f, np.abs(np.fft.fft(mz_27_mean)), label='mx_z0')
plt.plot(f, np.abs(np.fft.fft(mz_14_mean)), label='mx_z14')
plt.plot(f, np.abs(np.fft.fft(mz_00_mean)), label='mx_z27')
plt.legend()
plt.xlim(0, 0.5e10)
plt.grid()
plt.xlabel("Frequencz (Hz)")
plt.ylabel("Amplitude [-]")
plt.show()

m_ratio_27 = []
m_ratio_14 = []
m_ratio_00 = []
for i in range(np.shape(mx_27)[0]):
    m_ratio_27.append(mx_27_mean[i]/mz_27_mean[i])
    m_ratio_14.append(mx_14_mean[i]/mz_14_mean[i])
    m_ratio_00.append(mx_00_mean[i]/mz_00_mean[i])

plt.plot(t, m_ratio_27, "r*--", label = "PY-YIG boundary")
plt.plot(t, m_ratio_00, "go", markersize = 10, label = "YIG bottom")
plt.plot(t, m_ratio_14,"b*", label = "YIG center")
plt.xlabel("t [ns]")
plt.ylabel("m_x/m_z ratio [-]")
plt.grid()
plt.legend()
plt.show()

#m_x x slice plot for time
np.shape(m)
mx = m[:, 0, 0, 168, :]
x_length = len(mx[1])*4
x_plot = np.linspace(0,x_length, len(mx[1]))

from matplotlib import cm
colors = cm.rainbow(np.linspace(0, 1, np.shape(mx)[0]))
i = 150
for i in range(np.shape(m)[0]):
    plt.close()
    plt.plot(x_plot, mx[i,:], color = colors[i])
    plt.xlabel("x [nm]")
    plt.ylabel("SW x-magnetization [-]")
    #plt.vlines(5850, -1, 1, color = "red")#NOT RIGHT POSITION
    #plt.vlines(6050, -1, 1, color = "red")#NOT RIGHT POSITION
    plt.grid()
    plt.ylim(-0.15,0.15)
    plt.title(f"x slice at x-z center of YIG slab")
    plt.savefig(f"mx_x_slice_y168_zrange27_t{i*0.05:.2f}.png", bbox_inches='tight', dpi = 300)
    #plt.show()























#m region 3 data
p = run(["mumax3-convert","-numpy",dir_path+"/m_x.region3*236.ovf"], stdout=PIPE, stderr=STDOUT)
fields = {}
for npyfile in glob(dir_path+"/m_x.region3*.npy"):
    key = path.splitext(path.basename(npyfile))[0]
    fields[key] = load(npyfile)

m_py = np.stack([fields[key][0,:,168,:] for key in sorted(fields.keys())])

m_py_z = np.stack([fields[key][0,:,168,:] for key in sorted(fields.keys())])

np.shape(m_py)
np.shape(m_py_z)

mx_mean_z0 = []
mz_mean_z0 = []


mx_mean_z14 = []
mz_mean_z14 = []


mx_mean_z27 = []
mz_mean_z27 = []

for i in range(np.shape(m_py)[0]):
    #mx_mean_z.append([])
    #mx_mean_z[i].append(m_py[i,])
    mx_mean_z0.append(np.mean(m_py[i,0,:])) 
    mz_mean_z0.append(np.mean(m_py_z[i,0,:]))
    
    mx_mean_z14.append(np.mean(m_py[i,14,:])) 
    mz_mean_z14.append(np.mean(m_py_z[i,14,:]))
    
    mx_mean_z27.append(np.mean(m_py[i,27,:])) 
    mz_mean_z27.append(np.mean(m_py_z[i,27,:])) 

t = np.linspace(0,np.shape(mx_mean_z0)[0],np.shape(mx_mean_z0)[0])*0.05
x = np.linspace(0,np.shape(mx_mean_z0)[0],np.shape(mx_mean_z0)[0])*4

np.shape(mx_mean_z0)
np.shape(mz_mean_z0)


plt.plot(t, mx_mean_z0,"rd-", label = "m_x, YIG bottom")
plt.plot(t, mz_mean_z0,"bd", label = "m_z, YIG bottom")
plt.plot(t, mx_mean_z14,"ro-", label = "m_x, YIG center")
plt.plot(t, mz_mean_z14,"bo", label = "m_z, YIG center")
plt.plot(t, mx_mean_z27,"r*-", label = "m_x, PY-YIG boundary")
plt.plot(t, mz_mean_z27,"b*", label = "m_z, PY-YIG boundary")
plt.grid()
plt.legend()
plt.xlabel("t [ns]")
plt.ylabel("m_* mean amplitude [-]")
plt.show()

dt = 0.05*1e-9
#Get the number of data points
N = np.shape(mx_mean_z0)[0]
#Get the frequency range
f = np.fft.fftfreq(N, dt)
#Get the fft of the data
mx_fft = np.fft.fft(sim_data[cnames[1]])
#Plot the fft
plt.plot(f, np.abs(np.fft.fft(mx_mean_z0)), label='mx_z0')
plt.plot(f, np.abs(np.fft.fft(mx_mean_z14)), label='mx_z14')
plt.plot(f, np.abs(np.fft.fft(mx_mean_z27)), label='mx_z27')
plt.plot(f, np.abs(np.fft.fft(mz_mean_z0)), label='mx_z0')
plt.plot(f, np.abs(np.fft.fft(mz_mean_z14)), label='mx_z14')
plt.plot(f, np.abs(np.fft.fft(mz_mean_z27)), label='mx_z27')
plt.legend()
plt.xlim(0, 0.5e10)
plt.grid()
plt.xlabel("Frequencz (Hz)")
plt.ylabel("Amplitude [-]")
plt.show()

m_ratio_27_z = []
m_ratio_14_z = []
m_ratio_00_z = []
for i in range(np.shape(m_py)[0]):
    m_ratio_27_z.append(mx_mean_z27[i]/mz_mean_z27[i])
    m_ratio_14_z.append(mx_mean_z14[i]/mz_mean_z14[i])
    m_ratio_00_z.append(mx_mean_z0[i]/mz_mean_z0[i])

plt.plot(t, np.abs(m_ratio_27_z), "r*--", label = "PY-YIG boundary")
plt.plot(t, np.abs(m_ratio_00_z), "go", markersize = 10, label = "YIG bottom")
plt.plot(t, np.abs(m_ratio_14_z),"b*", label = "YIG center")
plt.xlabel("t [ns]")
plt.ylabel("m_x/m_z ratio [-]")
plt.grid()
plt.legend()
plt.show()





#m_x x-y plot for time
for i in range(np.shape(m)[0]):
    plt.close()
    mxy_py = m_py[i, 0, 27, :, :]
    extent_x_y = [0,7840,0,1344]
    plt.imshow(mxy_py, origin='lower', cmap="PiYG", extent = extent_x_y, vmin=-0.3, vmax=0.3)
    plt.xlabel("x [nm]")
    plt.ylabel("y [nm]")
    plt.colorbar(label = "m_x [-]")
    plt.title(f"x-y slice at center of YIG slab")
    plt.xlim(5600,7200)
    plt.ylim(200,1200)
    plt.savefig(f"mx_Z27_region3_t{i*0.05:.2f}.png", bbox_inches='tight', dpi = 300)
    #plt.show()

#  Select the x component
mx = m[:, 0, 0, 168, :]
#m_x_py = m_py[:, 0, 14, 168, :]
x_length = len(mx[1])*4
x_plot = np.linspace(0,x_length, len(mx[1]))
#plot the x component over x,t
#plt.plot(x_plot, mx[0,:], label = "t=0")
#plt.plot(x_plot, mx[np.uint32(np.ceil(np.shape(mx)[0]/2)),:], label = "t=half")
#plt.plot(x_plot, mx[np.shape(mx)[0]-1,:], label = "t=end")

#plt.plot(x_plot, m_x_py[0,:], label = "t=0")
#plt.plot(x_plot, m_x_py[np.uint32(np.ceil(np.shape(mx)[0]/2)),:], label = "t=half")
#plt.plot(x_plot, m_x_py[np.shape(mx)[0]-1,:], label = "t=end")
plt.plot(x_plot, mx[0,:] + m_x_py[0,:], label = "t=0")
plt.plot(x_plot, mx[np.uint32(np.ceil(np.shape(mx)[0]/2)),:] + m_x_py[np.uint32(np.ceil(np.shape(mx)[0]/2)),:], label = "t=half")
plt.plot(x_plot, mx[np.shape(mx)[0]-1,:]+m_x_py[np.shape(mx)[0]-1,:], label = "t=end")
plt.vlines(6145, -0.15, 0.2, color = "red")
plt.vlines(5893, -0.15, 0.2, color = "red")
plt.vlines(7055, -0.15, 0.2, color = "red")
plt.vlines(7280, -0.15, 0.2, color = "red")
plt.show()




















# k FFT loop
i=0 
k_store = []
mx = m[:, 0, 0, 5, :]
i = 150
plt.figure()
for i in range(np.shape(mx)[0]-1):
    mx = m[150, 0, 0, 5, :]
    dx = 4e-9
    N = np.shape(mx)[0]
    k = np.fft.fftfreq(N, dx)
    mx_fft_k = np.fft.fft(mx)
    #k =  np.fft.fftshift(k)
    #mx_fft_k =np.fft.fftshift(mx)
    #Plot the fft
    #plt.plot(k*2*np.pi, np.abs(mx_fft_k))
    k_store.append(np.abs(2*np.pi*k[np.abs(mx_fft_k).argmax(0)]*1e-6))
plt.grid()
plt.xlabel("k_x (1/m)")
plt.ylabel("Amplitude [-]")
plt.xlim(0,0.6e7)
plt.ylim(bottom = 0)
plt.savefig(f"test_k_per_t.png", bbox_inches='tight', dpi = 300)

for number in k_store:
    if number == float(0.0):
        k_store.remove(number)
np.mean(k_store)





























##################### FFT ############################
import scipy.io

def get_dir_path():
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory()
    return dir_path

dir_path = get_dir_path()

#########     KS MODEL    ###################
#HIGHMS_YIG
mat = scipy.io.loadmat(dir_path+'/KSModel_HIGHMS_YIG_f.mat')
sorted(mat.keys())
mat['freq_KS_A12']
mat['wavevector']
#LOWMS_YIG
mat_low_MS = scipy.io.loadmat(dir_path+'/KSModel_TRUElowMS_YIG_f.mat')
#PY
PY_disp = np.genfromtxt("PY_dispersion_KS.csv", delimiter = ",")

############# BYLAYER MODEL ########################
#CASE BY CASE HERE
BL_pk_disp = np.genfromtxt("BLmodel_-800.0_140.0_0_20.0_4.0_113.0_pK.csv", delimiter = ",")
BL_mk_disp = np.genfromtxt("BLmodel_-800.0_140.0_0_20.0_4.0_113.0_mK.csv", delimiter = ",")


np.shape(m)
import scipy.interpolate as spy
# Apply the two dimensional FFT
mx = m[:, 0, 0, 168, :] #mx
mx = mz[:, 0, 0, 168, :] #mz

mx = m0[:, 0, 0, 168, :]
mx = m_disp[:, 0, 0, 168, :]
#mx before the NM
mx = m[:, 0, 0, 168, 250:1500]
#mx after the NM
mx = m[:, 0, 0, 168, 2000:2750]
#region 3
mx = m_py[:,27,:]

np.shape(m_py)

np.shape(mx)[1]
s = np.shape(mx)[1]
mx_fft = np.fft.fft2(mx)
mx_fft = np.fft.fftshift(mx_fft)
mx_fft_before = np.fft.fft2(mx)
mx_fft_before = np.fft.fftshift(mx_fft_before)
mx_fft_after = np.fft.fft2(mx)
mx_fft_after = np.fft.fftshift(mx_fft_after)
np.shape(mx_fft_after)
from scipy.ndimage import zoom
# Interpolate mx_fft_after to match the size of mx_fft_before
zoom_factors = np.array(mx_fft_before.shape) / np.array(mx_fft_after.shape)
mx_fft_after_interpolated = zoom(mx_fft_after, zoom_factors, order=3)  # Using cubic interpolation
print("Shape of mx_fft_before:", mx_fft_before.shape)
print("Shape of mx_fft_after before interpolation:", mx_fft_after.shape)
print("Shape of mx_fft_after after interpolation:", mx_fft_after_interpolated.shape)
mx_fft_all = mx_fft_before + mx_fft_after_interpolated










mPYx = mPY[:, 0, 0, 168, :] #mx
mx_fft = np.fft.fft2(mPYx)
mx_fft = np.fft.fftshift(mx_fft)
np.shape(mPYx)


mx = m[:, 0, 0, 168, :1500] #mx
mx_fft = np.fft.fft2(mx)
mx_fft = np.fft.fftshift(mx_fft)


plt.figure(figsize=(10, 8))
dt = 5e-11
dx = 4e-9
# Show the intensity plot of the 2D FFT
extent = [ -(2*np.pi)/(2*dx)*1e-6, (2*np.pi)/(2*dx)*1e-6, -1/(2*dt)*1e-9, 1/(2*dt)*1e-9] # extent of k values and frequencies
plt.imshow(np.log((np.abs(mx_fft) ** 2)), extent=extent, aspect='auto', origin='lower', cmap="bone", vmin = -5)
#plt.hlines(1.7, 'r', label='k=0')
plt.ylabel(r"$f$ (GHz)", fontsize = 18)
plt.xlabel(r"k ($\frac{rad}{\mu m}$)", fontsize = 18)
#plt.plot(mat_low_MS['wavevector'][0]*1e-6, mat_low_MS['freq_KS_A12'][0][:50001],'r--',lw=2, label = "Low_MS_YIG")
#plt.plot(-mat_low_MS['wavevector'][0]*1e-6, mat_low_MS['freq_KS_A12'][0][:50001],'r--',lw=2)
plt.plot(-mat['wavevector'][0]*1e-6, mat['freq_KS_A12'][0][:50000],'g-.',lw=1.5)
plt.plot(mat['wavevector'][0]*1e-6, mat['freq_KS_A12'][0][:50000],'g-.',lw=1.5, label = "High-MS YIG KS model")
plt.plot(PY_disp[:,0],PY_disp[:,1],'b--', lw=1.5, label = "PY KS Model")
plt.plot(-PY_disp[:,0],PY_disp[:,1],'b--',lw=1.5)
plt.plot(BL_pk_disp[0,:],BL_pk_disp[1,:], color = color1, linestyle = ":", lw=1.5, label = "YIG BL Model")
plt.plot(BL_mk_disp[0,:],BL_mk_disp[1,:],color = color1, linestyle = ":" ,lw=1.5)
plt.ylim(-10,10)
plt.xlim(-100,100)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(loc="lower right", title=r'Theoretical Models',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.show()