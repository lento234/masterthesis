import numpy as np
import pylab as py
py.ion()

#-----------------------------------------------------------------------

fig_width_pt = 246.0 # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = 2*fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 18,
          'text.fontsize': 18,
          'legend.fontsize': 18,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16,
          'text.usetex': True,
          'figure.figsize': fig_size,
	  'font.family': 'sans-serif'}	
py.rcParams.update(params)

#-----------------------------------------------------------------------
# Gaussian kernel

xi = np.linspace(-3,3,1000)

m1 = 1 - 5*xi**2*0.5 + 3*np.abs(xi)**3*0.5
m2 = 0.5*(2- np.abs(xi))**2*(1-np.abs(xi))
m3 = xi*0.

m1[~ (np.abs(xi)<1)] = np.nan
m2[~ ((np.abs(xi)>=1) & (np.abs(xi)<2))] = np.nan
m3[~ (np.abs(xi)>=2)] = np.nan

py.figure(1)
py.axes([0.125,0.15,0.95-0.175,0.95-0.125])
py.plot(xi,m1,'k-')
py.plot(xi,m2,'k-.')
py.plot(xi,m3,'k--')
py.plot([-2,-1,1,2],[0.,0.,0.,0.],'k.')
py.axis([-3,3,-0.2,1.2])
py.xlabel(r'$\xi$')
#py.ylabel(r'$\zeta_{\sigma}$')
py.grid(True)
#py.savefig('interpolationKernel.pdf')

#-----------------------------------------------------------------------
