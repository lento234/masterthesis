import numpy as np
import pylab as py
py.ion()
py.rc('text', usetex=True)
py.rc('font', family='sans-serif')

#-----------------------------------------------------------------------
# Gaussian kernel
sigma = 1.0
k = 2.0
x = np.linspace(-5,5,100)
z = (1/(k*np.pi*sigma**2))*np.exp(-(x**2)/(k*sigma**2))

py.plot(x,z,'k')
py.xlabel(r'$\left|\mathbf{x}\right|$')
py.ylabel(r'$\zeta$')
py.axis([-5,5,0,0.2])
py.grid(True)
py.savefig('gaussianKernel.pdf')

#-----------------------------------------------------------------------
