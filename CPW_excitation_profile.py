import numpy as np
import matplotlib . pyplot as plt
from scipy import fftpack

plt.rcParams['figure.figsize'] = [10, 5]

Off = 1e-6
m0 = 4*np.pi*1e-7 # permeability of free space

#CPW excitation profile
width = 2.1e-6
gap = 1.4e-6
thickness = 0.11e-6

# F i e l d read out below the CPW with distance d − half film t h i c k n e s s might be taken i f
#CPW i s d i r e c t l y on top of film

d = 0.01e-6 #below CPW

#Field evaluated in x-direction over a distance :
length = 100e-6 # in microns
N = 10000
xnew = np.linspace(-length/2, length/2, num=N, endpoint=True)

#Fielld of current carrying wire
#x component
def fieldZOff(x,y,Offx):
    x = x-Offx #shift the field to the center of the wire
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan(x/y)
    mag = 1/r
    return mag*np.sin(phi)*m0/(2*np.pi)

#z component
def fieldXOff(x,y,Offx):
    x = x-Offx #shift the field to the center of the wire
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan(x/y)
    mag = 1/r
    return mag*np.cos(phi)*m0/(2*np.pi)


GL_off = width + gap #Shift of ground line compared to signal line

points1 = 500 #Iterate wire center position over point1 - points in x direction
points2 = 10 #Iterate wire center position over point2 - points in z direction

OffsetX = np.linspace(-width/2, width/2, num=points1, endpoint=True)
OffsetY = np.linspace(0, thickness, num=points2, endpoint=True)


Hx = 0
Hz = 0

for z in OffsetY:
    for i in OffsetX:
        Hx = Hx + fieldXOff(xnew, d+z,i) - 0.5*fieldXOff(xnew, d+z,i-GL_off) - 0.5*fieldXOff(xnew, d+z,i+GL_off)
        Hz = Hz + fieldZOff(xnew, d+z,i) - 0.5*fieldZOff(xnew, d+z,i-GL_off) - 0.5*fieldZOff(xnew, d+z,i+GL_off)
        Hx = Hx/points1/points2 #average over all points
        Hz = Hz/points1/points2 #average over all points


f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(xnew*1e6, Hz, label='Hz')
ax1.plot(xnew*1e6, Hx, label='Hx')

ax1.set_xlim(-10,10)
ax1.legend()
ax1.set_xlabel('x [um]')
ax1.set_ylabel('Field (T) per 1 A input')

ftX=fftpack.fft(Hx)
ftZ=fftpack.fft(Hz)

k=np.arange(len(ftX))*np.pi*2*1e-6/length
ax2.plot(k, np.abs(ftX), label='Hx')
ax2.plot(k, np.abs(ftZ), label='Hz')
ax2.legend()
ax2.set_xlabel('k_x [rad/um]')
ax2.set_ylabel('FFT amplitude')
ax2.set_xlim(0, 10)
plt.tight_layout()
plt.show()
plt.savefig('CPW_excitation_profile.pdf')