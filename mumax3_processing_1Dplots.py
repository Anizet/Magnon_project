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

# Load the txt file data using GUI as pd data frame
def load_data_txt():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    data = pd.read_csv(file_path, sep='\t', header=0)
    return data

sim_data = load_data_txt()
sim_data

#Get sim_data column names
cnames = sim_data.columns

#Plot the data
fig, ax1 = plt.subplots()
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["mx ()"], label=r'$m_x$', color = "mediumslateblue", linestyle = "--")
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["mz ()"], label=r'$m_z$', color = "salmon")
ax2 = ax1.twinx()
ax2.plot(sim_data[cnames[0]]*1e9, sim_data["my ()"], label=r'$m_y$', color = "crimson")
ax1.legend(loc="lower right", title='Magnetization components',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
#ax1.grid()
ax1.set_xlabel("Time (ns)", fontsize = 18)
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_ylabel(r"$m_y$ Magnetization [-]", fontsize = 18, color = "crimson")
ax1.set_ylabel(r"$m_x$ and $m_z$ Magnetization [-]", fontsize = 18)
ax2.set_xlim(min(sim_data[cnames[0]]*1e9), max(sim_data[cnames[0]]*1e9))
plt.show()

#Realize an fft on sim_data[cnames[1]], sim_data[cnames[2]], sim_data[cnames[3]]
#Get the time step
dt = sim_data[cnames[0]].diff().mean()
#Get the number of data points
N = sim_data[cnames[0]].count()
#Get the frequency range
f = np.fft.fftfreq(N, dt)
#Get the fft of the data
mx_fft = np.fft.fft(sim_data["mx ()"])
#my_fft = np.fft.fft(sim_data[cnames[2]])
mz_fft = np.fft.fft(sim_data["mz ()"])
#Plot the fft
plt.plot(f*1e-9, np.abs(mx_fft), label=r'$m_x$', color = "mediumslateblue", linestyle = "--")
#plt.plot(f, np.abs(my_fft), label='my')
plt.plot(f*1e-9, np.abs(mz_fft), label=r'$m_z$', color = "salmon")
plt.legend(loc="upper right", title='Magnetization components',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.xlim(0, 10)
plt.ylim(0, sorted(np.abs(mx_fft), reverse=True)[1] + 1)
plt.grid()
plt.xlabel("Frequency (GHz)", fontsize = 18)
plt.ylabel("FFT absolute Amplitude [-]", fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()

#Plot the data

fig, ax1 = plt.subplots()
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m.region0x ()"], label=r'$m_x$ YIG', color = "mediumslateblue", linestyle = "--")
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m.region0z ()"], label=r'$m_z$ YIG', color = "salmon")
ax2 = ax1.twinx()
ax2.plot(sim_data[cnames[0]]*1e9, sim_data["m.region0y ()"], label=r'$m_y$ YIG', color = "crimson")
ax1.legend(loc="lower right", title='Magnetization components',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
ax1.set_xlabel("Time (ns)", fontsize = 18)
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_ylabel(r"$m_y$ Magnetization [-]", fontsize = 18, color = "crimson")
ax1.set_ylabel(r"$m_x$ and $m_z$ Magnetization [-]", fontsize = 18)
ax2.set_xlim(min(sim_data[cnames[0]]*1e9), max(sim_data[cnames[0]]*1e9))
plt.show()

#Realize an fft on sim_data[cnames[1]], sim_data[cnames[2]], sim_data[cnames[3]]
#Get the time step
dt = sim_data[cnames[0]].diff().mean()
#Get the number of data points
N = sim_data[cnames[0]].count()
#Get the frequency range
f = np.fft.fftfreq(N, dt)
#Get the fft of the data
mx_fft = np.fft.fft(sim_data["m.region0x ()"])
#my_fft = np.fft.fft(sim_data[cnames[5]])
mz_fft = np.fft.fft(sim_data["m.region0z ()"])
#Plot the fft
plt.plot(f*1e-9, np.abs(mx_fft), label=r'$m_x$', color = "mediumslateblue", linestyle = "--")
#plt.plot(f, np.abs(my_fft), label='my')
plt.plot(f*1e-9, np.abs(mz_fft), label=r'$m_z$', color = "salmon")
plt.legend(loc="upper right", title='Magnetization components',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.xlim(0, 10)
plt.ylim(0, sorted(np.abs(mx_fft), reverse=True)[1] + 1)
plt.grid()
plt.xlabel("Frequency (GHz)", fontsize = 18)
plt.ylabel("FFT absolute Amplitude [-]", fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()


#Plot the data
fig, ax1 = plt.subplots()
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m.region1x ()"], label=r'$m_x$ PY', color = "mediumslateblue", linestyle = "--")
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m.region1z ()"], label=r'$m_z$ PY', color = "salmon")
ax2 = ax1.twinx()
ax2.plot(sim_data[cnames[0]]*1e9, sim_data["m.region1y ()"], label=r'$m_y$ PY', color = "crimson")
ax1.legend(loc="lower right", title='Magnetization components',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
ax1.set_xlabel("Time (ns)", fontsize = 18)
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_ylabel(r"$m_y$ Magnetization [-]", fontsize = 18, color = "crimson")
ax1.set_ylabel(r"$m_x$ and $m_z$ Magnetization [-]", fontsize = 18)
ax2.set_xlim(min(sim_data[cnames[0]]*1e9), max(sim_data[cnames[0]]*1e9))
plt.show()

#Realize an fft on sim_data[cnames[1]], sim_data[cnames[2]], sim_data[cnames[3]]
#Get the time step
dt = sim_data[cnames[0]].diff().mean()
#Get the number of data points
N = sim_data[cnames[0]].count()
#Get the frequency range
f = np.fft.fftfreq(N, dt)
#Get the fft of the data
mx_fft = np.fft.fft(sim_data["m.region1x ()"])
#my_fft = np.fft.fft(sim_data[cnames[5]])
mz_fft = np.fft.fft(sim_data["m.region1z ()"])
#Plot the fft
plt.plot(f*1e-9, np.abs(mx_fft), label=r'$m_x$', color = "mediumslateblue", linestyle = "--")
#plt.plot(f, np.abs(my_fft), label='my')
plt.plot(f*1e-9, np.abs(mz_fft), label=r'$m_z$', color = "salmon")
plt.legend(loc="upper right", title='Magnetization components',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.xlim(0, 10)
plt.ylim(0, sorted(np.abs(mx_fft), reverse=True)[1] + 1)
plt.grid()
plt.xlabel("Frequency (GHz)", fontsize = 18)
plt.ylabel("FFT absolute Amplitude [-]", fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()



#Plot the data
fig, ax1 = plt.subplots()
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m.region3x ()"], label=r'$m_x$ uPY', color = "mediumslateblue", linestyle = "--")
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m.region3z ()"], label=r'$m_z$ uPY', color = "salmon")
ax2 = ax1.twinx()
ax2.plot(sim_data[cnames[0]]*1e9, sim_data["m.region3y ()"], label=r'$m_y$ uPY', color = "crimson")
ax1.legend(loc="lower right", title='Magnetization components',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
ax1.set_xlabel("Time (ns)", fontsize = 18)
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_ylabel(r"$m_y$ Magnetization [-]", fontsize = 18, color = "crimson")
ax1.set_ylabel(r"$m_x$ and $m_z$ Magnetization [-]", fontsize = 18)
ax2.set_xlim(min(sim_data[cnames[0]]*1e9), max(sim_data[cnames[0]]*1e9))
plt.show()


#Realize an fft on sim_data[cnames[1]], sim_data[cnames[2]], sim_data[cnames[3]]
#Get the time step
dt = sim_data[cnames[0]].diff().mean()
#Get the number of data points
N = sim_data[cnames[0]].count()
#Get the frequency range
f = np.fft.fftfreq(N, dt)
#Get the fft of the data
mx_fft = np.fft.fft(sim_data["m.region3x ()"])
#my_fft = np.fft.fft(sim_data[cnames[5]])
mz_fft = np.fft.fft(sim_data["m.region3z ()"])
#Plot the fft
plt.plot(f*1e-9, np.abs(mx_fft), label=r'$m_x$', color = "mediumslateblue", linestyle = "--")
#plt.plot(f, np.abs(my_fft), label='my')
plt.plot(f*1e-9, np.abs(mz_fft), label=r'$m_z$', color = "salmon")
plt.legend(loc="upper right", title='Magnetization components',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.xlim(0, 10)
plt.ylim(0, sorted(np.abs(mx_fft), reverse=True)[1] + 1)
plt.grid()
plt.xlabel("Frequency (GHz)", fontsize = 18)
plt.ylabel("FFT absolute Amplitude [-]", fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()




#Plot the data
fig, ax1 = plt.subplots()
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["B_extx (T)"]*1e3, label=r'$B_x$ uPY', color = "mediumslateblue", linestyle = "--")
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["B_extz (T)"]*1e3, label=r'$B_z$ uPY', color = "salmon")
#ax2 = ax1.twinx()
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["B_exty (T)"]*1e3, label=r'$B_y$ uPY', color = "crimson")
ax1.legend(loc="upper right", title='Field components',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
ax1.set_xlabel("Time (ns)", fontsize = 18)
ax1.tick_params(axis='both', labelsize=14)
#ax2.tick_params(axis='both', labelsize=14)
#ax2.set_ylabel(r"$m_y$ Magnetization [-]", fontsize = 18, color = "crimson")
ax1.set_ylabel(r"Field Strength (mT)", fontsize = 18)
ax1.set_xlim(min(sim_data[cnames[0]]*1e9), max(sim_data[cnames[0]]*1e9))
plt.show()


#Realize an fft on sim_data[cnames[1]], sim_data[cnames[2]], sim_data[cnames[3]]
#Get the time step
dt = sim_data[cnames[0]].diff().mean()
#Get the number of data points
N = sim_data[cnames[0]].count()
#Get the frequency range
f = np.fft.fftfreq(N, dt)
#Get the fft of the data
mx_fft = np.fft.fft(sim_data["m.region3x ()"])
#my_fft = np.fft.fft(sim_data[cnames[5]])
mz_fft = np.fft.fft(sim_data["m.region3z ()"])
#Plot the fft
plt.plot(f*1e-9, np.abs(mx_fft), label=r'$m_x$', color = "mediumslateblue", linestyle = "--")
#plt.plot(f, np.abs(my_fft), label='my')
plt.plot(f*1e-9, np.abs(mz_fft), label=r'$m_z$', color = "salmon")


#FFT of external field excitation
Hx_fft = np.fft.fft(sim_data["B_extx (T)"])
plt.plot(f*1e-9, np.abs(Hx_fft), label=r'$B_x$', color = "mediumslateblue", linestyle = "-")
plt.legend(loc="upper right", title='Field components',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.xlim(-10, 10)
plt.ylim(0, sorted(np.abs(Hx_fft), reverse=True)[1] + 1)
plt.grid()
plt.xlabel("Frequency (GHz)", fontsize = 18)
plt.ylabel("FFT absolute Amplitude [-]", fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()

#Plot the data
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["mx ()"], label=r'$m_x$', color = "mediumslateblue", linestyle = "--")
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["mz ()"], label=r'$m_z$', color = "salmon")
ax2.plot(sim_data[cnames[0]]*1e9, sim_data["my ()"], label=r'$m_y$', color = "crimson")
ax1.legend(loc="lower right", title='Magnetization components',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
#ax1.grid()
ax1.set_xlabel("Time (ns)", fontsize = 18)
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_ylabel(r"$m_y$ Magnetization [-]", fontsize = 18, color = "crimson")
ax1.set_ylabel(r"$m_x$ and $m_z$ Magnetization [-]", fontsize = 18)
ax2.set_xlim(min(sim_data[cnames[0]]*1e9), max(sim_data[cnames[0]]*1e9))
plt.show()


#Plot the response at the edges of the absorbing layers
fig, ax1 = plt.subplots()
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m_xrange0-2_x ()"], label=r'ABL-$m_x$', color = "mediumslateblue", linestyle = "-")
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m_xrange0-2_z ()"], label=r'ABL-$m_z$', color = "salmon", linestyle = "-") 
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m.region2x ()"], label=r'ref-$m_x$', color = "mediumslateblue", linestyle = "--")
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m.region2z ()"], label=r'ref-$m_z$', color = "salmon", linestyle = "--")
ax2 = ax1.twinx()
ax2.plot(sim_data[cnames[0]]*1e9, sim_data["m_xrange0-2_y ()"], label=r'ABL-$m_y$', color = "crimson", linestyle = "-")
ax2.plot(sim_data[cnames[0]]*1e9, sim_data["m.region2y ()"], label=r'ref-$m_y$', color = "crimson", linestyle = "--")
ax1.legend(loc="lower right", title='Magnetization components',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
ax1.set_xlabel("Time (ns)", fontsize = 18)
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_ylabel(r"$m_y$ Magnetization [-]", fontsize = 18, color = "crimson")
ax1.set_ylabel(r"$m_x$ and $m_z$ Magnetization [-]", fontsize = 18)
ax2.set_xlim(min(sim_data[cnames[0]]*1e9), max(sim_data[cnames[0]]*1e9))
plt.show()

#Plot the response at the edges of the absorbing layers
fig, ax1 = plt.subplots()
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m_xrange2998-3000_x ()"], label=r'ABL-$m_x$', color = "mediumslateblue", linestyle = "-")
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m_xrange2998-3000_z ()"], label=r'ABL-$m_z$', color = "salmon", linestyle = "-") 
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m.region1x ()"], label=r'ref-$m_x$', color = "mediumslateblue", linestyle = "--")
ax1.plot(sim_data[cnames[0]]*1e9, sim_data["m.region1z ()"], label=r'ref-$m_z$', color = "salmon", linestyle = "--")
ax2 = ax1.twinx()
ax2.plot(sim_data[cnames[0]]*1e9,sim_data["m_xrange2998-3000_y ()"], label=r'ABL-$m_y$', color = "crimson", linestyle = "-")
ax2.plot(sim_data[cnames[0]]*1e9, sim_data["m.region1y ()"], label=r'ref-$m_y$', color = "crimson", linestyle = "--")
ax1.legend(loc="lower right", title='Magnetization components',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
ax1.set_xlabel("Time (ns)", fontsize = 18)
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_ylabel(r"$m_y$ Magnetization [-]", fontsize = 18, color = "crimson")
ax1.set_ylabel(r"$m_x$ and $m_z$ Magnetization [-]", fontsize = 18)
ax2.set_xlim(min(sim_data[cnames[0]]*1e9), max(sim_data[cnames[0]]*1e9))
plt.show()

#Plot the max angle which should be <0.35 rad to make sure that the cell size is sufficiently small compared to the exchange length
plt.plot(sim_data[cnames[0]], sim_data[cnames[19]], label='max angle')
plt.hlines(0.35, sim_data[cnames[0]].min(), sim_data[cnames[0]].max(), 'r', label='0.35 rad limit')
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Angle [rad]")
plt.show()