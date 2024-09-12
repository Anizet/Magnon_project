import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import scipy.optimize

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.solid_capstyle'] = "round"
plt.rcParams['lines.dash_capstyle'] = "round"
plt.rcParams['figure.dpi'] = 100
#plt.rcParams['font.family'] = 'Georgia'

color1 = "gold"

#euler constant
gamma = 0.5772156649
# Boltzmann constant
k = 1.38064852e-23 #J/K
T = 300 #K
Np = 50 #number of pulses

t = [0.05,0.1,0.5,1]
NS = [9.5/19, 7.75/19, 4.5/19, 2.5/19]
NS = [9.5/13, 7.75/13, 4.5/13, 2.5/13]
sig = 4.1e-20 #J

def func(t, E, f):
    PS = 1/2 * special.erfc((k*T*(gamma + np.log(f*t)) - E)/(np.sqrt(2)*sig))
    FAC = 1 
    for i in range(Np-1):
        FAC += (1-(1/2 * special.erfc((k*T*(gamma + np.log(f*t)) - E)/(np.sqrt(2)*sig))))**i
    return PS*FAC #*(Np-1)

param_bounds = [[k*T,1e9],[5e-19, 1e13]]

params = scipy.optimize.curve_fit(func, t, NS, bounds = param_bounds, p0=[5e-20,7.8e9])
params[0]

x = np.linspace(0.01, 1, 10000)
y = []

for i in range(len(x)):
    y.append(func(x[i],*params[0]))

plt.plot(t,np.array(NS)*100, "kD-", linewidth = 2, markersize = 10, label = "Experimental data")
plt.plot(x,np.array(y)*100, "k--", label = fr"Fitted curve, $E = {params[0][0]:.2e}$ J, $f = {params[0][1]:.2e}$ Hz")
plt.grid(alpha = 0.2)
plt.xlim(0,1)
plt.xlabel("Pulse Period (s)", fontsize = 24)
plt.ylabel("Estimated Switching Probability (%)", fontsize = 24)
#plt.tight_layout()
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(fontsize = 18)
plt.show()

plt.plot(np.log(t),NS)
plt.scatter(np.log(t[0]),func(t[0],*params[0]))
plt.scatter(np.log(t[1]),func(t[1],*params[0]))
plt.scatter(np.log(t[2]),func(t[2],*params[0]))
plt.scatter(np.log(t[3]),func(t[3],*params[0]))
plt.show()


# Test with fixed f
#f = 6.8e9
#def func(t, E):
#    PS = 1/2 * special.erfc((k*T*(gamma + np.log(f*t)) - E)/(np.sqrt(2)*sig)) 
#    FAC = 1 
#    for i in range(Np-1):
#        FAC += (1-(1/2 * special.erfc((k*T*(gamma + np.log(f*t)) - E)/(np.sqrt(2)*sig))))**i
#    return PS*FAC #*(Np-1)
#
#params = scipy.optimize.curve_fit(func, t, NS, bounds = ([k*T],[5e-19]))
#params[0]
#
#plt.plot(t,NS)
#plt.scatter(t[0],func(t[0],*params[0]))
#plt.scatter(t[1],func(t[1],*params[0]))
#plt.scatter(t[2],func(t[2],*params[0]))
#plt.scatter(t[3],func(t[3],*params[0]))
#plt.show()
