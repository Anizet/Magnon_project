import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd

# Load the txt file data using GUI as pd data frame
def load_data_xlsx():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    data = pd.read_excel(file_path, header=0)
    return data

sim_data = load_data_xlsx()
sim_data

plt.imshow(sim_data)
plt.show()

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.solid_capstyle'] = "round"
plt.rcParams['lines.dash_capstyle'] = "round"
plt.rcParams['figure.dpi'] = 150

data = np.array(sim_data)
data[:,0]
H = [2,6,10,14,16]
plt.plot(H, data[0,1:6], "r*-", label = "-13 dBm", markersize = 10)
plt.plot(H, data[1,1:6], "b+-", label = "-10 dBm", markersize = 14)
plt.plot(10, data[2,3], "b+", markersize = 14)
plt.plot(10, data[3,3], "b+", markersize = 14)
plt.plot(H, data[4,1:6],"gv-", label = "-5 dBm", markersize = 8)
plt.plot(H, data[5,1:6], "x-", label = "0 dBm", markersize = 14)
plt.plot(H, data[6,1:6],"mo-", label = "7 dBm", markersize = 8)
plt.axhline(19, linestyle = "--", color = "black", label = "Max")
#plt.title("A-row NM switching")
plt.xlabel("Field (mT)", fontsize = 18)
plt.ylabel("Number of nanomagnets switched [-]", fontsize = 18)
plt.yticks(np.arange(0,20,1), fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(loc="upper left", title='Column A (19 NM)',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.grid()
plt.show()



H = [2,6,10,14,16]
data = np.array(sim_data)
plt.plot(H, data[0,6:11], "r*-", label = "-13 dBm", markersize = 10)
plt.plot(H, data[1,6:11], "b+-", label = "-10 dBm", markersize = 14)
plt.plot(10, data[2,8], "b+", markersize = 14)
plt.plot(10, data[3,8], "b+", markersize = 14)
plt.plot(H, data[4,6:11],"gv-", label = "-5 dBm", markersize = 8)
plt.plot(H, data[5,6:11], "x-", label = "0 dBm", markersize = 14)
plt.plot(H, data[6,6:11],"mo-", label = "7 dBm", markersize = 8)
plt.axhline(14, linestyle = "--", color = "black", label = "Max")
plt.title("B-row NM switching")
plt.xlabel("Field (mT)", fontsize = 18)
plt.ylabel("Number of nanomagnets switched [-]", fontsize = 18)
plt.yticks(np.arange(0,15,1), fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(loc="upper left", title='Column B (14 NM)',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.grid()
plt.show()




H = [2,6,10,14,16]
data = np.array(sim_data)
plt.plot(H, data[0,11:16], "r*-", label = "-13 dBm", markersize = 10)
plt.plot(H, data[1,11:16], "b+-", label = "-10 dBm", markersize = 14)
plt.plot(10, data[2,13], "b+", markersize = 14)
plt.plot(10, data[3,13], "b+", markersize = 14)
plt.plot(H, data[4,11:16],"gv-", label = "-5 dBm", markersize = 8)
plt.plot(H, data[5,11:16], "x-", label = "0 dBm", markersize = 14)
plt.plot(H, data[6,11:16],"mo-", label = "7 dBm", markersize = 8)
plt.axhline(9, linestyle = "--", color = "black", label = "Max")
plt.title("C-row NM switching")
plt.xlabel("Field (mT)", fontsize = 18)
plt.ylabel("Number of nanomagnets switched [-]", fontsize = 18)
plt.yticks(np.arange(0,10,1), fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(loc="upper left", title='Column C (9 NM)',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.grid()
plt.show()




H = [2,6,10,14,16]
data = np.array(sim_data)
plt.plot(H, data[0,16:21], "r*-", label = "-13 dBm", markersize = 10)
plt.plot(H, data[1,16:21], "b+-", label = "-10 dBm", markersize = 14)
plt.plot(10, data[2,18], "b+", markersize = 14)
plt.plot(10, data[3,18], "b+", markersize = 14)
plt.plot(H, data[4,16:21],"gv-", label = "-5 dBm", markersize = 8)
plt.plot(H, data[5,16:21], "x-", label = "0 dBm", markersize = 14)
plt.plot(H, data[6,16:21],"mo-", label = "7 dBm", markersize = 8)
plt.axhline(4, linestyle = "--", color = "black", label = "Max")
plt.title("D-row NM switching")
plt.xlabel("Field (mT)", fontsize = 18)
plt.ylabel("Number of nanomagnets switched [-]", fontsize = 18)
plt.yticks(np.arange(0,5,1), fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(loc="upper left", title='Column D (4 NM)',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.grid()
plt.show()




ma.masked_where(np.isnan(data[:,1]), data[:,0])
# Plot versus power

import numpy.ma as ma
# A rowimport numpy.ma 1 as ma
#A 2mT
plt.plotdata([[0,1,4,5,6],0], data[[0,1,4,5,6],1], "r*-", label = "2mT", markersize = 10)
#A 6mT
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],2], "gv-", label = "6mT", markersize = 8)
#A 10mT
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],3], "b+-", label = "10mT", markersize = 14)
plt.plot(data[2,0], data[2,3], "b+", markersize = 14)
plt.plot(data[3,0], data[3,3], "b+", markersize = 14)

plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],4], "x-", label = "14mT", markersize = 14)
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],5], "mo-", label = "16mT", markersize = 8)
plt.axhline(19, linestyle = "--", color = "black", label = "Max")
plt.title("A-row NM switching")
plt.xlabel("Power [dBm]", fontsize = 18)
plt.ylabel("Number of nanomagnets switched [-]", fontsize = 18)
plt.yticks(np.arange(0,20,1), fontsize = 14)
plt.xticks(np.arange(-13,8,1), fontsize = 14)
plt.legend(loc="lower right", title='Column A (19 NM)',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.grid()
plt.show()

# Plot versus power

# B row
#B 2mT
plt.plotdata([[0,1,4,5,6],0], data[[0,1,4,5,6],6], "r*-", label = "2mT", markersize = 10)
#B 6mT
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],7], "gv-", label = "6mT", markersize = 8)
#B 10mT
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],8], "b+-", label = "10mT", markersize = 14)
plt.plot(data[2,0], data[2,8], "b+", markersize = 14)
plt.plot(data[3,0], data[3,8], "b+", markersize = 14)

plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],9], "x-", label = "14mT", markersize = 14)
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],10], "mo-", label = "16mT", markersize = 8)
plt.axhline(14, linestyle = "--", color = "black", label = "Max")
plt.title("B-row NM switching")
plt.xlabel("Power [dBm]", fontsize = 18)
plt.ylabel("Number of nanomagnets switched [-]", fontsize = 18)
plt.yticks(np.arange(0,15,1), fontsize = 14)
plt.xticks(np.arange(-13,8,1), fontsize = 14)
plt.legend(loc="lower right", title='Column B (14 NM)',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.grid()
plt.show()



# Plot versus power

# C row
#C 2mT
plt.plotdata([[0,1,4,5,6],0], data[[0,1,4,5,6],11], "r*-", label = "2mT", markersize = 10)
#C 6mT
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],12], "gv-", label = "6mT", markersize = 8)
#C 10mT
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],13], "b+-", label = "10mT", markersize = 14)
plt.plot(data[2,0], data[2,13], "b+", markersize = 14)
plt.plot(data[3,0], data[3,13], "b+", markersize = 14)
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],14], "x-", label = "14mT", markersize = 14)
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],15], "mo-", label = "16mT", markersize = 8)
plt.axhline(9, linestyle = "--", color = "black", label = "Max")

plt.title("C-row NM switching")
plt.xlabel("Power [dBm]", fontsize = 18)
plt.ylabel("Number of nanomagnets switched [-]", fontsize = 18)
plt.yticks(np.arange(0,10,1), fontsize = 14)
plt.xticks(np.arange(-13,8,1), fontsize = 14)
plt.legend(loc="lower right", title='Column C (9 NM)',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.grid()
plt.show()



# Plot versus power

# D row
#D 2mT
plt.plotdata([[0,1,4,5,6],0], data[[0,1,4,5,6],16], "r*-", label = "2mT", markersize = 10)
#D 6mT
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],17], "gv-", label = "6mT", markersize = 8)
#D 10mT
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],18], "b+-", label = "10mT", markersize = 14)
plt.plot(data[2,0], data[2,18], "b+", markersize = 14)
plt.plot(data[3,0], data[3,18], "b+", markersize = 14)
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],19], "x-", label = "14mT", markersize = 14)
plt.plot(data[[0,1,4,5,6],0], data[[0,1,4,5,6],20], "mo-", label = "16mT", markersize = 8)
plt.axhline(4, linestyle = "--", color = "black", label = "Max")
plt.title("D-row NM switching")
plt.xlabel("Power [dBm]", fontsize = 18)
plt.ylabel("Number of nanomagnets switched [-]", fontsize = 18)
plt.yticks(np.arange(0,5,1), fontsize = 14)
plt.xticks(np.arange(-13,8,1), fontsize = 14)
plt.legend(loc="lower right", title='Column C (4 NM)',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.grid()
plt.show()
