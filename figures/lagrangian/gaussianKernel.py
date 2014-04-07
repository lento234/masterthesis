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
sigma = 1.0
k = 2.0
x = np.linspace(-5,5,100)
z = (1/(k*np.pi*sigma**2))*np.exp(-(x**2)/(k*sigma**2))

py.figure(1)
py.axes([0.125,0.15,0.95-0.175,0.95-0.125])
py.plot(x,z,'k')
py.axis([-5,5,0,0.2])
py.xlabel(r'$\left|\mathbf{x}\right|$')
py.ylabel(r'$\zeta_{\sigma}$')
py.grid(True)
#py.savefig('figureTest.pdf')
figName = 'gaussianKernel'
py.savefig('%s.eps' % figName)
import os
os.system('epstopdf %s.eps' % figName)
#os.system('pdfcrop %s.pdf' % figName)
os.system('rm %s.eps' % figName)
#-----------------------------------------------------------------------
