import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

k = np.linspace(0,500e6, 1000)/(2*np.pi)+0.1
theta = 90/360 * 2*np.pi #DE wave
phi = 90/360 * 2*np.pi #DE wave
u0 = 4*np.pi*1e-7
H = 10e-3/u0 # In Amps/m  
g = 1.760859630e11 # In rad/s/T
oh = u0 * g * H


#YIG parameters
t = 112e-9
A = 3.7*1e-12 # J/m
M = 23.8e3 # In A/m
lex = (2*A/(u0*M**2)) #m**-2
om =  u0 * g * M


#PY parameters
t = 20e-9
A = 1.13*1e-12 # J/m
M = 800e3 # In A/m
lex = (2*A/(u0*M**2)) #m**-2
om =  u0 * g * M

#P = k*t/2 # Only in the long wavelength limit; neglecting exchange interaction
Fn = 2/(k*t) * (1-np.exp(-k*t))
P = 1 - Fn*(1/2)
F = P + np.sin(theta)**2 * (1-P*(1+np.cos(phi)**2) + om * (P * (1-P)*np.sin(phi)**2)/(oh + lex*om*k**2))
O = np.sqrt((oh + lex*om*k**2)*(oh + lex*om*k**2 + om*F))

#Make the plot nicer
plt.plot(k*(2*np.pi)*1e-6,O/(2*np.pi)*1e-9, color='black', linestyle='-', linewidth=2, label=r'PY(20nm), M$_S$=800 kA/m')
plt.plot(-k*(2*np.pi)*1e-6,O/(2*np.pi)*1e-9, color='black', linestyle='-', linewidth=2)
plt.hlines(min(O/(2*np.pi)*1e-9),-17,17, color = "black",  linestyle='--')
plt.text(x = 20, y = min(O/(2*np.pi)*1e-9), s = f"{min(O/(2*np.pi)*1e-9):.2f} GHz", horizontalalignment="left",verticalalignment="center", fontsize = 10, fontweight="bold")

plt.xlabel(r'k (rad/$\mu$m)', fontsize = 14)
plt.ylabel(r'$f$ (GHz)', fontsize = 14)
plt.grid()
plt.legend()
#plt.xlim(-150,150)
#plt.ylim(0,3.5)
plt.show()

#Write data file for PY dispersion with k*(2*np.pi)*1e-6 and O/(2*np.pi)*1e-9
data = {'k': k*(2*np.pi)*1e-6, 'f': O/(2*np.pi)*1e-9}
df = pd.DataFrame(data)
df.to_csv('C:/Users/Arnaud/Desktop/Magnons_project/PY_dispersion.csv', index=False)





#P = k*t/2 # Only in the long wavelength limit; neglecting exchange interaction
Fn = 2/(k*t) * (1-np.exp(-k*t))
P = 1 - Fn*(1/2)
F = P + np.sin(theta)**2 * (1-P*(1+np.cos(phi)**2) + om * (P * (1-P)*np.sin(phi)**2)/(oh + lex*om*k**2))
O = np.sqrt((oh + lex*om*k**2)*(oh + lex*om*k**2 + om*F))

#Make the plot nicer
plt.plot(k*(2*np.pi)*1e-6,O/(2*np.pi)*1e-9, color='black', linestyle='-', linewidth=2, label=r'YIG(112nm), M$_S$=23.8 kA/m')
plt.plot(-k*(2*np.pi)*1e-6,O/(2*np.pi)*1e-9, color='black', linestyle='-', linewidth=2)
plt.hlines(min(O/(2*np.pi)*1e-9),-17,17, color = "black",  linestyle='--')
plt.text(x = 20, y = min(O/(2*np.pi)*1e-9), s = f"{min(O/(2*np.pi)*1e-9):.2f} GHz", horizontalalignment="left",verticalalignment="center", fontsize = 10, fontweight="bold")

#Compute the derivative of O with respect to k
dO_1 = np.diff(O)/(np.diff(k))

## Neglect exchange :
#P = k*t/2 # Only in the long wavelength limit; neglecting exchange interaction
#F = P + np.sin(theta)**2 * (1-P*(1+np.cos(phi)**2) + om * (P * (1-P)*np.sin(phi)**2)/(oh + lex*om*k**2))
#O = np.sqrt((oh + lex*om*k**2)*(oh + lex*om*k**2 + om*F))
#plt.plot(k*(2*np.pi)*1e-6,O/(2*np.pi)*1e-9, color='blue', linestyle='--', linewidth=2,label='KS Model, M=23.8e3 A/m, no Exchange')

##Attempt at thickness modes
#n = [1,2,3,4,5,6] 
#for i in range(len(n)):
#    kn = np.sqrt(k**2 + n[i]*np.pi/t) 
#    #First approx :
#    Pnn = (k*t)**2/(n[i]**2*np.pi**2)
#    Fnn = Pnn + np.sin(theta)**2 * (1-Pnn*(1+np.cos(phi)**2) + om * (Pnn * (1-Pnn)*np.sin(phi)**2)/(oh + lex*om*kn**2))
#    Onn = np.sqrt((oh + lex*om*kn**2)*(oh + lex*om*kn**2 + om*Fnn))
#    plt.plot(k*(2*np.pi)*1e-6,Onn/(2*np.pi)*1e-9, color='grey', linestyle='--', linewidth=1.5,label=f'M=23.8e3 A/m, no Exchange, thickness mode n={i}')


#Larger Ms
M = 140e3 # In A/m
lex = (2*A/(u0*M**2)) #m**-2
om =  u0 * g * M
#P = k*t/2 # Only in the long wavelength limit; neglecting exchange interaction
Fn = 2/(k*t) * (1-np.exp(-k*t))
P = 1 - Fn*(1/2)
F = P + np.sin(theta)**2 * (1-P*(1+np.cos(phi)**2) + om * (P * (1-P)*np.sin(phi)**2)/(oh + lex*om*k**2))
O = np.sqrt((oh + lex*om*k**2)*(oh + lex*om*k**2 + om*F))
plt.plot(k*(2*np.pi)*1e-6,O/(2*np.pi)*1e-9, color='red', linestyle='-', linewidth=2,label=r'YIG(112nm), M$_S$=140 kA/m')
plt.plot(-k*(2*np.pi)*1e-6,O/(2*np.pi)*1e-9, color='red', linestyle='-', linewidth=2)
plt.hlines(min(O/(2*np.pi)*1e-9),-12,12, color = "red",  linestyle='--')
plt.text(x = 15, y = min(O/(2*np.pi)*1e-9), s = f"{min(O/(2*np.pi)*1e-9):.2f} GHz", horizontalalignment="left",verticalalignment="center", fontsize = 10, fontweight="bold", color = "red")



#Compute the derivative of O with respect to k
dO_2= np.diff(O)/(np.diff(k))

## Neglect exchange :
#P = k*t/2 # Only in the long wavelength limit; neglecting exchange interaction
#F = P + np.sin(theta)**2 * (1-P*(1+np.cos(phi)**2) + om * (P * (1-P)*np.sin(phi)**2)/(oh + lex*om*k**2))
#O = np.sqrt((oh + lex*om*k**2)*(oh + lex*om*k**2 + om*F))
#plt.plot(k*(2*np.pi)*1e-6,O/(2*np.pi)*1e-9, color='magenta', linestyle='--', linewidth=1.5,label='KS Model, M=140e3 A/m, no Exchange')



plt.xlabel(r'k (rad/$\mu$m)', fontsize = 14)
plt.ylabel(r'$f$ (GHz)', fontsize = 14)
plt.grid()
plt.legend()
plt.xlim(-150,150)
plt.ylim(0,3.5)
plt.show()


f = plt.figure()
ax = f.add_subplot(111)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

plt.plot(k[1:]*(2*np.pi)*1e-6,dO_1*1e-3, color='black', linestyle='-', linewidth=1.5)
plt.plot(k[1:]*(2*np.pi)*1e-6,dO_2*1e-3, color='red', linestyle='-', linewidth=1.5)
plt.grid()
plt.legend()
plt.xlim(0,150)
plt.ylim(0,4)
plt.ylabel(r'$V_g$ (km/s)', fontsize = 14)
plt.xlabel(r'k (rad/$\mu$m)', fontsize = 14)
plt.legend([r'YIG(112nm), M$_S$=23.8 kA/m',r'YIG(112nm), M$_S$=140 kA/m'])
plt.show()
