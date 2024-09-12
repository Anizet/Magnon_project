import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import fsolve


#################### Graph setup ####################################
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.solid_capstyle'] = "round"
plt.rcParams['lines.dash_capstyle'] = "round"
plt.rcParams['figure.dpi'] = 150
#plt.rcParams['font.family'] = 'Georgia'

color1 = "salmon"
color2 = "mediumslateblue"
 


################################## Static and layer parameters ####################################
u0 = 4*np.pi*1e-7
H0 = 10e-3/u0 #A/m ######MUST INCLUDE THE EXCHANGE INTERACTION
gamma = 1.760859630e11 # In rad/s/T 

d1 = 56e-9  #nm #70nmmfor test
d2 = 56e-9 #nm 50 nm for test
d0 = 4e-9 #nm
k = np.linspace(0,50e6,500)+0.001 #rad/m
Wtest = np.linspace(0,10e9,50000)*2*np.pi # frequencies in Hz
A1 = 3.7e-12 #J/m Echange constant
A2 = 3.7e-12 #J/m

J1 = 141e3#*u0 #A/m Magnetization
J2 = 140e3#*u0 #A/m

H2 = H0 + 2*A2*k**2/(u0*J2) # A/m Effective field accounting for intralayer exchange
H1 = H0 + 2*A1*k**2/(u0*J1) # A/m

Omh1 = H1/J1 # Dimensionless parameter (see model)
Omh2 = H2/J2

Om1 = Wtest/(u0*gamma*J1) # Dimensionless parameter (see model)
Om2 = Wtest/(u0*gamma*J2)

############################################# Compute solutions for the PARALLEL Bilayer case ###########################################

########################################### + k ####################################################
W_1 = []
W_2 = []

for i in range(len(k)):
    ktest = k[i]
    K1 = Omh1[i]/(Omh1[i]**2 - Om1**2)
    V1 = Om1/(Omh1[i]**2 - Om1**2)
    K2 = Omh2[i]/(Omh2[i]**2 - Om2**2)
    V2 = Om2/(Omh2[i]**2 - Om2**2)
    Q = (1-np.exp(-2*ktest*d1))*(1-np.exp(-2*ktest*d2))*np.exp(-2*ktest*d0)
    A_1 = (K1-V1)*(K1+V1)/((K1+V1+2)*(V1-K1-2))
    A_2 = (K2-V2)*(K2+V2)/((K2+V2+2)*(V2-K2-2))
    term1 = (1+A_1*np.exp(-2*k[i]*d1))*(1+A_2*np.exp(-2*k[i]*d2))
    term2 = A_1*((2+K1+V1)/(2+K2+V2))*((K2+V2)/(K1+V1))*Q
    sol = term1 + term2
    signs = np.sign(sol)
    diff = np.diff(signs)
    indices = np.where(diff != 0)
    W_2.append(Om2[indices[0][0]])
    W_1.append(Om1[indices[0][1]])
    #plt.plot(Om2[:49999],diff, label = f"{i}")
    print(f"Step {i} of {len(k)}")

#plt.hlines(0,-10,10)
#plt.xlabel("X")
#plt.ylabel("Polynomial Solution")
#plt.show()

freq = np.array(W_2)*gamma*u0/(2*np.pi) * J2
freq = freq/1e9

freqB = np.array(W_1)*gamma*u0/(2*np.pi) * J1
freqB = freqB/1e9

plt.plot(k[:len(freq)]*1e-6,freq, color = color1, linestyle = "-")
plt.plot(k[:len(freqB)]*1e-6,freqB, color = color2, linestyle = "-")
plt.xlabel("k [rad/um]")
plt.ylabel("Frequency (GHz)")
plt.show()

########################################### - k ####################################################
W_1_2 = []
W_2_2 = []

for i in range(len(k)):
    ktest = k[i]
    K1 = Omh1[i]/(Omh1[i]**2 - Om1**2)
    V1 = Om1/(Omh1[i]**2 - Om1**2)
    K2 = Omh2[i]/(Omh2[i]**2 - Om2**2)
    V2 = Om2/(Omh2[i]**2 - Om2**2)
    Q = (1-np.exp(-2*ktest*d1))*(1-np.exp(-2*ktest*d2))*np.exp(-2*ktest*d0)
    A_1 = (K1-V1)*(K1+V1)/((K1+V1+2)*(V1-K1-2))
    A_2 = (K2-V2)*(K2+V2)/((K2+V2+2)*(V2-K2-2))
    term1 = (1+A_1*np.exp(-2*k[i]*d1))*(1+A_2*np.exp(-2*k[i]*d2))
    term2 = A_1*((2+K1-V1)/(2+K2-V2))*((K2-V2)/(K1-V1))*Q
    sol = term1 + term2
    signs = np.sign(sol)
    diff = np.diff(signs)
    indices = np.where(diff != 0)
    W_2_2.append(Om2[indices[0][0]])
    W_1_2.append(Om1[indices[0][1]])
    #plt.plot(Om2,sol)
    print(f"Step {i} of {len(k)}")

#plt.hlines(0,-10,10)
#plt.xlabel("X")
#plt.ylabel("Polynomial Solution")
#plt.show()

freq2 = np.array(W_2_2)*gamma*u0/(2*np.pi) * J2
freq2 = freq2/1e9

freq2b = np.array(W_1_2)*gamma*u0/(2*np.pi) * J1
freq2b = freq2b/1e9

plt.plot(k[:len(freq)]*1e-6,freq, color = color1, linestyle = "-")
plt.plot(k[:len(freqB)]*1e-6,freqB, color = color1, linestyle = "-.")
plt.plot(-k[:len(freq2)]*1e-6,freq2, color = color2, linestyle = "-")
plt.plot(-k[:len(freq2b)]*1e-6,freq2b, color = color2, linestyle = "-.")
plt.xlabel("k [rad/um]")
plt.ylabel("Frequency (GHz)")
plt.xlim(-50,50)
plt.ylim(0.5,4)
plt.vlines(0,0,3, color = "black")
plt.show()





############################################# Compute solutions for the ANTIPARALLEL Bilayer case ###########################################

J1 = -J1#*u0 #A/m
J2 = J2#*u0 #A/m
 
H2 = H0 + 2*A2*k**2/(u0*J2) # A/m
H1 = H0 + 2*A1*k**2/(u0*J1) # A/m

Omh1 = H1/J1
Omh2 = H2/J2

Om1 = Wtest/(u0*gamma*J1)
Om2 = Wtest/(u0*gamma*J2)

#Test graphical solution
W_1_3 = []
W_2_3 = []

for i in range(len(k)):
    ktest = k[i]
    K1 = Omh1[i]/(Omh1[i]**2 - Om1**2)
    V1 = Om1/(Omh1[i]**2 - Om1**2)
    K2 = Omh2[i]/(Omh2[i]**2 - Om2**2)
    V2 = Om2/(Omh2[i]**2 - Om2**2)
    Q = (1-np.exp(-2*ktest*d1))*(1-np.exp(-2*ktest*d2))*np.exp(-2*ktest*d0)
    A_1 = (K1-V1)*(K1+V1)/((K1+V1+2)*(V1-K1-2))
    A_2 = (K2-V2)*(K2+V2)/((K2+V2+2)*(V2-K2-2))
    term1 = (1+A_1*np.exp(-2*k[i]*d1))*(1+A_2*np.exp(-2*k[i]*d2))
    term2 = A_1*((2+K1+V1)/(2+K2+V2))*((K2+V2)/(K1+V1))*Q
    sol = term1 + term2
    signs = np.sign(sol)
    diff = np.diff(signs)
    indices = np.where(diff != 0)
    W_2_3.append(Om2[indices[0][0]])
    W_1_3.append(Om1[indices[0][1]])
    #plt.plot(Om2,sol, label = f"{i}")
    print(f"Step {i} of {len(k)}")
#plt.hlines(0,-10,10)
#plt.xlabel("X")
#plt.ylabel("Polynomial Solution")
#plt.legend()
#plt.show()

freq3 = np.array(W_2_3)*gamma*u0/(2*np.pi) * J2
freq3 = freq3/1e9
for i in range(len(freq3)):
    if (np.abs(freq3[i] - freq3[i+1])> 1):
        kbreak3 = i
        freq3 = freq3[i+1:]
        break

plt.plot(-k[kbreak3+1:len(freq3)+kbreak3+1]*1e-6,freq3, color = color1, linestyle = "-")
plt.plot(-k[:len(freq3)]*1e-6,freq3, color = color1, linestyle = "-")
plt.xlabel("k [rad/um]")
plt.ylabel("Frequency (GHz)")
plt.show()



W_1_4 = []
W_2_4 = []

for i in range(len(k)):
    ktest = k[i]
    K1 = Omh1[i]/(Omh1[i]**2 - Om1**2)
    V1 = Om1/(Omh1[i]**2 - Om1**2)
    K2 = Omh2[i]/(Omh2[i]**2 - Om2**2)
    V2 = Om2/(Omh2[i]**2 - Om2**2)
    Q = (1-np.exp(-2*ktest*d1))*(1-np.exp(-2*ktest*d2))*np.exp(-2*ktest*d0)
    A_1 = (K1-V1)*(K1+V1)/((K1+V1+2)*(V1-K1-2))
    A_2 = (K2-V2)*(K2+V2)/((K2+V2+2)*(V2-K2-2))
    term1 = (1+A_1*np.exp(-2*k[i]*d1))*(1+A_2*np.exp(-2*k[i]*d2))
    term2 = A_1*((2+K1-V1)/(2+K2-V2))*((K2-V2)/(K1-V1))*Q
    sol = term1 + term2
    signs = np.sign(sol)
    diff = np.diff(signs)
    indices = np.where(diff != 0)
    if indices:
        W_2_4.append(Om2[indices[0][0]])
        W_1_4.append(Om1[indices[0][1]])
    #plt.plot(Om2,sol)
    print(f"Step {i} of {len(k)}")

plt.hlines(0,-10,10)
plt.xlabel("X")
plt.ylabel("Polynomial Solution")
plt.show()

freq4 = np.array(W_2_4)*gamma*u0/(2*np.pi) * J2
freq4 = freq4/1e9
kbreak4 = []
for i in range(len(freq4)-1):
    if (np.abs(freq4[i] - freq4[i+1])> 1):
        kbreak4.append(i)

freq4 = freq4[kbreak4[1]+1:]

plt.plot(-k[kbreak3+1:len(freq3)+kbreak3+1]*1e-6,freq3,color = color1, linestyle = "-")
plt.plot(k[kbreak4[1]+1:len(freq4)+kbreak4[1]+1]*1e-6,freq4, color = color1, linestyle = "-")
#plt.plot(k[:len(freq4)]*1e-6,freq4, "r-")
plt.xlabel("k [rad/um]")
plt.ylabel("Frequency (GHz)")
plt.xlim(-50,50)
#plt.ylim(0.5,4)
plt.vlines(0,0,3, color = "black")
plt.show()


############################################## FULL PLOT #############################################
plt.plot(k[:len(freq)]*1e-6,freq, color = color1, linestyle = "-", label = "Parallel")
plt.plot(-k[:len(freq2)]*1e-6,freq2, color = color1, linestyle = "-")
plt.plot(-k[kbreak3+1:len(freq3)+kbreak3+1]*1e-6,freq3, color = color2, linestyle = "-", label = "Antiparallel")
plt.plot(k[kbreak4[1]+1:len(freq4)+kbreak4[1]+1]*1e-6,freq4, color = color2, linestyle = "-")
plt.xlim(-50,50)
plt.ylim(0,4)
plt.axvline(0, color = "black")
plt.title(fr"Bilayer Dispersion Model [$d_0={d0*1e9}, d_1={d1*1e9}, d2={d2*1e9}, \mu_0H = {H0:.1f}, M_1 = {J1/1000}, M_2 = {J2/1000}$]")
plt.xlabel(r"k [$\frac{rad}{\mu m}$]", fontsize = 18)
plt.ylabel("Frequency (GHz)", fontsize = 18)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(loc="lower right", title='Magnetic configuration of bilayer',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.grid()
plt.show()

#np.savetxt(f"BLmodel_{J1/1000}_{J2/1000}_{H0*u0:.0f}_{d1*1e9}_{d0*1e9}_{d2*1e9}_mK.csv",[-k[kbreak3+1:len(freq3)+kbreak3+1]*1e-6,freq3], delimiter = ",")
#np.savetxt(f"BLmodel_{J1/1000}_{J2/1000}_{H0*u0:.0f}_{d1*1e9}_{d0*1e9}_{d2*1e9}_pK.csv",[k[kbreak4[1]+1:len(freq4)+kbreak4[1]+1]*1e-6, freq4], delimiter = ",")

############################################## DELTA F PLOT #############################################
plt.plot(k[:min([len(freq), len(freq2)])]*1e-6, np.abs(freq[:min([len(freq), len(freq2)])] - freq2[:min([len(freq), len(freq2)])]),color = color1, linestyle = "-", label = "Parallel")
plt.plot(k[max([kbreak4[1]+1, kbreak3+1]):min([len(freq3), len(freq4)])]*1e-6, np.abs(freq3[max([kbreak4[1]+1, kbreak3+1]):min([len(freq3), len(freq4)])] - freq4[max([kbreak4[1]+1, kbreak3+1]):min([len(freq3), len(freq4)])]), color = color2, linestyle = "-", label = "Anti-Parallel")
plt.xlabel(r"k ($\frac{rad}{\mu m}$)", fontsize = 18)
plt.ylabel(r"$\Delta$ $f$ (GHz)", fontsize = 18)
plt.xlim(0,max(k[:min([len(freq), len(freq2)])])*1e-6)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(loc="lower right", title='Magnetic configuration of bilayer',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
plt.grid()
plt.legend()
plt.show()


########################################### DATASET FIT FOR RESONANCE CONDITION COMPUTATION #############################################
########################### PARALLEL CASE ########################################
# freq + freq2
dataforfit1 = [np.array(freq),np.array(freq2)]
kf2 = k[:len(freq2)]
kf1 = k[:len(freq)]

z1 =np.polyfit(dataforfit1[0][190:],kf1[190:],200)
f1 = np.poly1d(z1)

z2 = np.polyfit(dataforfit1[1],kf2, 100)
f2 = np.poly1d(z2)
#With this fit, get k for a certain f
fnew = np.linspace(1.34,4.9,500)
k2 = f2(fnew+0.02)
k1 = f1(fnew)
# Check if fit is OK
plt.plot(k1,fnew, color = "red", label = "fit")
plt.plot(kf1, freq, color = "blue", label = "BL-model")
plt.plot(-k2,fnew, color = "red", label = "fit")
plt.plot(-kf2, freq2, color = "blue", label = "BL-model")
plt.show()


width = 0.2 #in microns, width of the F-P cavity

fig, ax1 = plt.subplots()
ax1.plot(fnew,(k1+k2)*1e-6*width + 1*np.pi, color = "crimson", linestyle = "-", label = fr"Resonance condition ({H0*u0*1000:.0f}mT) : $(|k_2|+|k_3|)w + \phi_b(=\pi)$")
ax1.plot(fnew,(k1+k2)*1e-6*width + 1.34*np.pi, color = "crimson", linestyle = "-.", label = fr"Resonance condition ({H0*u0*1000:.0f}mT) : $(|k_2|+|k_3|)w + \phi_b(=1.34\pi)$")
for n in range(3):
    ax1.axhline(y = (2*(n+1) + 1)*np.pi, color = "black", linestyle = "--")
ax2 = ax1.twinx()
ax2.plot(freq, kf1*1e-6, color = color1, linestyle = "-", label = r"+k dispersion ($|k_3|$)")
ax2.plot(freq2, kf2*1e-6, color = color2, linestyle = "-", label = r"-k dispersion ($|k_2|$)")
plt.xlabel("f (GHz)")
ax1.set_ylabel(r"$\Delta \phi$ (rad)", fontsize = 18)
ax2.set_ylabel(r"|k| ($\frac{rad}{\mu m}$)", fontsize = 18)
plt.xlim(0.4,5)
ax1.set_xlabel(r"$f$ (GHz)", fontsize = 18)
ax1.legend(loc="upper left", title='Fabry-Perot Resonator',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
ax2.legend(loc="lower right", title='Parallel configuration',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
ax1.set_ylim(min((k1+k2)*1e-6*0.2 + 1*np.pi), max((k1+k2)*1e-6*0.2 + 1*np.pi))
ax2.set_ylim(min(kf1*1e-6), max([kf1[-1]*1e-6,kf2[-1]*1e-6]))
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)
ax1.spines['bottom'].set_linewidth(1.5)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['top'].set_linewidth(1.5)
ax1.spines['right'].set_linewidth(1.5)
plt.show()


########################### ANTIPARALLEL CASE ########################################
# freq3 + freq4
# freq + freq2
dataforfit2 = [np.array(freq3),np.array(freq4)]
kf4 = k[:len(freq4)]
kf3 = k[:len(freq3)]

z3 = np.polyfit(dataforfit2[0][59:],kf3[59:], 100)
f3 = np.poly1d(z3)

z4 = np.polyfit(dataforfit2[1],kf4, 100)
f4 = np.poly1d(z4)
#With this fit, get k for a certain f
fnew = np.linspace(1.95,4,500)
k4 = f4(fnew+0.02)
k3 = f3(fnew)

plt.plot(-k3,fnew, color = "red", label = "fit")
plt.plot(-kf3, freq3, color = "blue", label = "BL-model")
plt.plot(k4,fnew, color = "red", label = "fit")
plt.plot(kf4, freq4, color = "blue", label = "BL-model")
plt.show()

fig, ax1 = plt.subplots()
ax1.plot(fnew,(k3+k4)*1e-6*width + 1*np.pi, color = "crimson", linestyle = "-", label = fr"Resonance condition ({H0*u0*1000:.0f}mT) : $(|k_2|+|k_3|)w + \phi_b(=\pi)$")
ax1.plot(fnew,(k3+k4)*1e-6*width + 1.34*np.pi, color = "crimson", linestyle = "-.", label = fr"Resonance condition ({H0*u0*1000:.0f}mT) : $(|k_2|+|k_3|)w + \phi_b(=1.34\pi)$")
for n in range(3):
    ax1.axhline(y = (2*(n+1) + 1)*np.pi, color = "black", linestyle = "--")
ax2 = ax1.twinx()
ax2.plot(freq3, kf3[-len(freq3):]*1e-6, color = color1, linestyle = "-", label = r"+k dispersion ($|k_3|$)")
ax2.plot(freq4, kf4*1e-6, color = color2, linestyle = "-", label = r"-k dispersion ($|k_2|$)")
plt.xlabel("f (GHz)")
ax1.set_ylabel(r"$\Delta \phi$ (rad)", fontsize = 18)
ax2.set_ylabel(r"|k| ($\frac{rad}{\mu m}$)", fontsize = 18)
plt.xlim(0.4,5)
ax1.set_xlabel(r"$f$ (GHz)", fontsize = 18)
ax1.legend(loc="upper left", title='Fabry-Perot Resonator',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
ax2.legend(loc="lower right", title='Parallel configuration',facecolor='lightgrey', labelcolor='black',edgecolor='black',shadow=True)
ax1.set_ylim(min((k3+k4)*1e-6*0.2 + 1*np.pi), max((k3+k4)*1e-6*0.2 + 1*np.pi))
ax2.set_ylim(min(kf3*1e-6), max([kf3[-1]*1e-6,kf4[-1]*1e-6]))
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)
ax1.spines['bottom'].set_linewidth(1.5)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['top'].set_linewidth(1.5)
ax1.spines['right'].set_linewidth(1.5)
plt.show()



