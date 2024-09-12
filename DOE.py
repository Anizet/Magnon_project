from pyDOE3 import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.solid_capstyle'] = "round"
plt.rcParams['lines.dash_capstyle'] = "round"
plt.rcParams['figure.dpi'] = 150
#plt.rcParams['font.family'] = 'Georgia'

color1 = "gold"

bb = bbdesign(3,1)
cc = ccdesign(3,[2,2]) #Face = ccc or cci or ccf

P = [0,20]
T = [0.05,1]
W = [100,200]

bb[np.where(bb[:,0] == 0),0] = (min(P) + max(P))/2
bb[np.where(bb[:,0] == 1),0] = max(P)
bb[np.where(bb[:,0] == -1),0] = min(P)

bb[np.where(bb[:,1] == 0),1] = (min(T) + max(T))/2
bb[np.where(bb[:,1] == 1),1] = max(T)
bb[np.where(bb[:,1] == -1),1] = min(T)

bb[np.where(bb[:,2] == 0),2] = (min(W) + max(W))/2
bb[np.where(bb[:,2] == 1),2] = max(W)
bb[np.where(bb[:,2] == -1),2] = min(W)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(bb[:,0],bb[:,1],bb[:,2], label = "Box Behnken Design")
ax.set_xlabel('Power [dBm]')
ax.set_ylabel('Period [s]')
ax.set_zlabel('Pulse Width [ns]')
plt.show()







cc[np.where(cc[:,0] > 1),0] =  cc[np.where(cc[:,0] > 1),0]  * (max(P)-min(P))/2 + (max(P)+min(P))/2
cc[np.where(cc[:,0] < -1),0] =  cc[np.where(cc[:,0] < -1),0]  * (max(P)-min(P))/2 + (max(P)+min(P))/2
cc[np.where(cc[:,0] == 0),0] = (min(P) + max(P))/2
cc[np.where(cc[:,0] == 1),0] = max(P)
cc[np.where(cc[:,0] == -1),0] = min(P)

cc[np.where(cc[:,1] > 1),1] =  cc[np.where(cc[:,1] > 1),1]  * (max(T)-min(T))/2 + (max(T)+min(T))/2
cc[np.where(cc[:,1] < -1),1] =  cc[np.where(cc[:,1] < -1),1]  * (max(T)-min(T))/2 + (max(T)+min(T))/2
cc[np.where(cc[:,1] == 0),1] = (min(T) + max(T))/2
cc[np.where(cc[:,1] == 1),1] = max(T)
cc[np.where(cc[:,1] == -1),1] = min(T)


cc[np.where(cc[:,2] > 1),2] =  cc[np.where(cc[:,2] > 1),2]  * (max(W)-min(W))/2 + (max(W)+min(W))/2
cc[np.where(cc[:,2] < -1),2] =  cc[np.where(cc[:,2] < -1),2]  * (max(W)-min(W))/2 + (max(W)+min(W))/2
cc[np.where(cc[:,2] == 0),2] = (min(W) + max(W))/2
cc[np.where(cc[:,2] == 1),2] = max(W)
cc[np.where(cc[:,2] == -1),2] = min(W)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(cc[:,0],cc[:,1],cc[:,2], label = "Box Behnken Design")
ax.set_xlabel('Power [dBm]')
ax.set_ylabel('Period [s]')
ax.set_zlabel('Pulse Width [ns]')
plt.show()