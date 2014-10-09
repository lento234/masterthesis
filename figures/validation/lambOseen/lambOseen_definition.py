
# External modules
#import dolfin
import numpy as np
import pylab as py
import dolfin
#import time
py.ion()
py.close('all')
# pHyFlow
#import pHyFlow
#from pHyFlow.blobs.base import induced

import matplotlib as mpl
import scipy.io as sio
"""
colorMapFile = sio.loadmat('/home/lento/Documents/programs/tools/Colormaps/COLORMAP_HOT_COLD.mat')
colorMapData = colorMapFile['COLORMAP_HOT_COLD']
colorMap = mpl.colors.ListedColormap(zip(colorMapData[:,0],colorMapData[:,1],colorMapData[:,2]))
colorMapCold = mpl.colors.ListedColormap(zip(colorMapData[:64,0],colorMapData[:64,1],colorMapData[:64,2]))
colorMapHot = mpl.colors.ListedColormap(zip(colorMapData[64:,0],colorMapData[64:,1],colorMapData[64:,2]))

colorMap = 'jet'
jetCM = py.cm.ScalarMappable(cmap=colorMap)
jetCM = jetCM.to_rgba(np.linspace(0,1,256))

jetCMCold = mpl.colors.ListedColormap(zip(jetCM[:128,0],jetCM[:128,1],jetCM[:128,2]))
jetCMHot = mpl.colors.ListedColormap(zip(jetCM[128:,0],jetCM[128:,1],jetCM[128:,2]))
jetCM = mpl.colors.ListedColormap(zip(jetCM[:,0],jetCM[:,1],jetCM[:,2]))


fig_width_pt = 246.0 # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = 2.5*fig_width_pt*inches_per_pt  # width in inches
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



"""



def from_levels_and_colors(levels, colors, extend='neither'):
    """
    A helper routine to generate a cmap and a norm instance which
    behave similar to contourf's levels and colors arguments.

    Parameters
    ----------
    levels : sequence of numbers
        The quantization levels used to construct the :class:`BoundaryNorm`.
        Values ``v`` are quantizized to level ``i`` if
        ``lev[i] <= v < lev[i+1]``.
    colors : sequence of colors
        The fill color to use for each level. If `extend` is "neither" there
        must be ``n_level - 1`` colors. For an `extend` of "min" or "max" add
        one extra color, and for an `extend` of "both" add two colors.
    extend : {'neither', 'min', 'max', 'both'}, optional
        The behaviour when a value falls out of range of the given levels.
        See :func:`~matplotlib.pyplot.contourf` for details.

    Returns
    -------
    (cmap, norm) : tuple containing a :class:`Colormap` and a \
                   :class:`Normalize` instance
    """
    colors_i0 = 0
    colors_i1 = None

    if extend == 'both':
        colors_i0 = 1
        colors_i1 = -1
        extra_colors = 2
    elif extend == 'min':
        colors_i0 = 1
        extra_colors = 1
    elif extend == 'max':
        colors_i1 = -1
        extra_colors = 1
    elif extend == 'neither':
        extra_colors = 0
    else:
        raise ValueError('Unexpected value for extend: {0!r}'.format(extend))

    n_data_colors = len(levels) - 1
    n_expected_colors = n_data_colors + extra_colors
    if len(colors) != n_expected_colors:
        raise ValueError('With extend == {0!r} and n_levels == {1!r} expected'
                         ' n_colors == {2!r}. Got {3!r}.'
                         ''.format(extend, len(levels), n_expected_colors,
                                   len(colors)))

    cmap = mpl.colors.ListedColormap(colors[colors_i0:colors_i1], N=n_data_colors)

    if extend in ['min', 'both']:
        cmap.set_under(colors[0])
    else:
        cmap.set_under('none')

    if extend in ['max', 'both']:
        cmap.set_over(colors[-1])
    else:
        cmap.set_over('none')

    cmap.colorbar_extend = extend

    norm = mpl.colors.BoundaryNorm(levels, ncolors=n_data_colors)
    return cmap, norm



fig_width_pt = 424#246.0 # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
	
params = {'backend': 'ps',
          'axes.labelsize':  20,
          'text.fontsize':   20,
          'legend.fontsize': 8,#20,
          'xtick.labelsize': 12,
          'ytick.labelsize': 16,
          'text.usetex': True,
          'figure.figsize': fig_size,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Helvetica'}	

py.rcParams.update(params)

# Daeninck's colormap
colorMapDaen = {}
colorMapDaen['levels'] = np.hstack((np.linspace(-20,0,11)[:-1], -0.2, 0.2, np.linspace(0,20,11)[1:]))/20.
colorMapDaen['colors'] = np.array([[0,1,180],[0,1,180],[0,0,232],[0,27,255],
                                   [1,78,254],[0,129,255],[0,180,255],
                       	       [1,231,255],[26,254,227],[77,255,177],
                                   [204,204,254],[255,255,255],[254,204,203],
                                   [180,255,76],[229,255,24],[255,229,0],
                                   [255,178,0],[255,127,0],[255,76,1],
                                   [255,25,1],[230,0,0],[179,1,1],[179,1,1]])/255.
                                   
# Custom colormap
clim = 10     
#cmap,norm = from_levels_and_colors(colorMapDaen['levels']*clim,colorMapDaen['colors'],extend='both')
cmap = mpl.colors.ListedColormap(zip(colorMapDaen['colors'][:,0],colorMapDaen['colors'][:,1],colorMapDaen['colors'][:,2]))

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke

# To add the annotation to the figure
def add_inner_title(ax, title, loc, size=None, **kwargs):

    if size is None:
        size = dict(size=20)#plt.rcParams['text.fontsize'])

    at = AnchoredText(title, loc=loc, prop=size, pad=0., borderpad=0.5, frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])

    return at
    
    





#-----------------------------------------------------------------------------    
# Global parameters

# Fluid Parameters
vInf = np.array([0.,0.]) # Free stream velocity



# Define the lamb-oseen
Gamma = 1   #2.0*np.pi # Total circulation of the lamb-oseen
nu    = 0.001#5e-4   #Gamma / (2.0*np.pi*Re) # Unit Geometry with Re = 1000.
tau   = 100#2e-3  #0.5 # viscous time
#tStart  4.0 #tau/nu
#t0 = tStart
pVortex = np.array([0.,0.])
deltaT = 0.002

# Exact Function : exact vorticity field
def wLambOseen(x,y,t):
    return (Gamma / (4.0*np.pi*nu*(tau+t))) * np.exp( -  ((x-pVortex[0])*(x-pVortex[0]) +
                                                    (y-pVortex[1])*(y-pVortex[1]))/
                                                    (4.0*nu*(tau+t)))
     

                                           ## Exact Function : exact velocity field
def vLambOseen(x,y,t):
    # Radius
    r = np.sqrt((x-pVortex[0])*(x-pVortex[0]) + (y-pVortex[1])*(y-pVortex[1])) + np.spacing(1)
    # Circumferential velocity
    uTheta = (Gamma / (2.0*np.pi*r)) * (1.0 - np.exp( - (r*r)/(4.0*nu*(tau+t))))
    # Angle
    theta = np.arctan2((y-pVortex[1]),(x-pVortex[0]))
    # Return the cartesian velocity field
    return -np.sin(theta)*uTheta, np.cos(theta)*uTheta  
   
t = 0 


#-----------------------------------------------------------------------------    


#-----------------------------------------------------------------------------    


# Mesh parameters
meshFilePath            = './geometry/mesh.xml.gz'
boundaryDomainsFilePath = './geometry/mesh_facet_region.xml.gz'

mesh = dolfin.Mesh(meshFilePath)

xyFE = mesh.coordinates().T

xBounds = [np.min(xyFE[0]), np.max(xyFE[0])]
yBounds = [np.min(xyFE[1]), np.max(xyFE[1])]


#-----------------------------------------------------------------------------    


#-----------------------------------------------------------------------------    
# Plot Hybrid Domain configuration.

fzTemp = list(np.array(fig_size)*1.5)

py.rcParams['figure.figsize'] = fzTemp


h     = 0.01
dBdry = 10*h + np.spacing(10e10) # ensure no floating-point error occurs, add a small increment

xMin, xMax = xBounds
yMin, yMax = yBounds

xBoundaryPolygon = np.array([xMin+dBdry, xMax-dBdry, xMax-dBdry,xMin+dBdry,xMin+dBdry])
yBoundaryPolygon = np.array([yMin+dBdry, yMin+dBdry, yMax-dBdry,yMax-dBdry,yMin+dBdry])

# Define the domains
xyFEPoly = np.array([[-1,1,1,-1,-1],
                     [-1,-1,1,1,-1]])*0.5

bdryPath = mpl.path.Path(np.vstack((xBoundaryPolygon,yBoundaryPolygon)).T,closed=True)
fePath   = mpl.path.Path(xyFEPoly.T,closed=True)

# Plot the domain
fig = plt.figure()
ax = plt.gca()
fepatch = mpl.patches.PathPatch(fePath, fill=True, hatch='+', fc='w', lw=0, alpha=0.5)
intpatch = mpl.patches.PathPatch(bdryPath, fill=True, lw=0, fc='LightPink',alpha=0.7,ec='r')
ax.add_patch(fepatch)
ax.add_patch(intpatch)
ax.plot(xyFEPoly[0],xyFEPoly[1],'r-',lw=1)
ax.plot(xBoundaryPolygon,yBoundaryPolygon,'r--',lw=0.5)
ax.axis('scaled')
ax.axis([-1.,1.,-1.,1.])
ax.axis('off')

# Add text
ax.annotate(r'$\Omega_{E}$', xy=(-0.3, -0.49),fontsize=15,backgroundcolor='None')

ax.annotate(r'$\Omega_{I}$', xy=(0.1, -0.35),fontsize=15,backgroundcolor='None',color='r')
            #bbox=dict(boxstyle='square', fc="w", ec="none"))#,fontsize=25,zorder=20)
ax.annotate(r'$\Omega_{L} \backslash \Omega_E$', xy=(-0.8, -0.),fontsize=15,backgroundcolor='None')
            #bbox=dict(boxstyle='square', fc="w", ec="none"))#,fontsize=25,zorder=20)
#ax.annotate(r'$\partial \Omega_{dirich}$', xy=(0.6, 0.4),fontsize=10)#,fontsize=25,zorder=20)

ax.annotate(r'$\Sigma_{d}$', xy=(0.47, 0.3), xycoords='data',fontsize=10,xytext=(0.6, 0.4), textcoords='data',
           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,
                           connectionstyle="arc3,rad=-0.2",),)

ax.annotate(r'$\Sigma_{o}$', xy=(0.3, 0.365), xycoords='data',fontsize=10,xytext=(0.4, 0.6), textcoords='data',
           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,
                           connectionstyle="arc3,rad=0.2",),)

#$\d_{bdry}$
ax.annotate(r'$d_{bdry}$', xy=(-0.175, 0.44),fontsize=8,backgroundcolor='w')

ax.annotate(r'', xy=(-0.2, 0.385), xycoords='data',fontsize=5,xytext=(-0.2, 0.52), textcoords='data',
           arrowprops=dict(arrowstyle="<|-|>", color="k",
                           patchA=None, patchB=None,
                           connectionstyle="arc3",),)

# Axis
ax.annotate("",(0.+0.01,0.-0.02),(0.+0.01,0.3-0.02),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})
ax.annotate("",(0.,0.-0.01),(0.3,0.-0.01),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})



py.savefig('hlo_dd.pdf')


#-----------------------------------------------------------------------------    



"""


n = 100
x,y = np.meshgrid(np.linspace(-0.5,0.5,n),np.linspace(-0.5,0.5,n))
x2,y2 = np.meshgrid(np.linspace(-1.,1.,2*n),np.linspace(-1.,1.,2*n))
x = x.flatten()
x2 = x2.flatten()
y = y.flatten()
y2 = y2.flatten()

omega = wLambOseen(x,y,t)
omega2 = wLambOseen(x2,y2,t)

#-------------------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
                


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x.reshape(n,n), y.reshape(n,n), omega.reshape(n,n), rstride=1, cstride=1, 
                       cmap=jetCM, linewidth=0, antialiased=False)
ax.set_zlim(0.2, 0.8)
fig.colorbar(surf)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$\omega$')
py.savefig('lambOseen_initialDistribution.pdf')


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x.reshape(n,n), y.reshape(n,n), omega.reshape(n,n), rstride=1, cstride=1,
                       cmap=jetCM, linewidth=0, antialiased=False)
surf = ax.plot_surface(x2.reshape(2*n,2*n), y2.reshape(2*n,2*n), omega2.reshape(2*n,2*n), rstride=1, cstride=1,
                       cmap=jetCM, linewidth=0, antialiased=False,alpha=0.5)

ax.set_zlim(0.2, 0.8)
fig.colorbar(surf)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$\omega$')
py.savefig('lambOseen_initialDistribution2.pdf')



#= ax.plot_surface(x.reshape(n,n), y.reshape(n,n), omega.reshape(n,n), rstride=1, cstride=1, 
#                       cmap=jetCM, linewidth=0, antialiased=False)
#ax.set_zlim(0, 1)
#fig.colorbar(surf)
#ax.set_xlabel(r'$x$')
#ax.set_ylabel(r'$y$')
#ax.set_zlabel(r'$\omega$')

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x.reshape(n,n), y.reshape(n,n), omega.reshape(n,n), rstride=1, cstride=1, 
#                       cmap=jetCM, linewidth=0, antialiased=False,alpha=0.1)
#cset = ax.contourf(x.reshape(n,n), y.reshape(n,n), omega.reshape(n,n), zdir='x', offset=-0.5)#,levels=np.linspace(0,40,40),cmap=cm.jet)
#cset = ax.contourf(x.reshape(n,n), y.reshape(n,n), omega.reshape(n,n), zdir='y', offset=0.5)#,levels=np.linspace(0,40,40),cmap=cm.jet)
#ax.set_zlim(0, 40)
#fig.colorbar(surf)

#
#fig_width_pt = 246.0 # Get this from LaTeX using \showthe\columnwidth
#inches_per_pt = 1.0/72.27               # Convert pt to inch
#golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
#fig_width = 2*fig_width_pt*inches_per_pt  # width in inches
#fig_height = fig_width*golden_mean      # height in inches
#fig_size =  [fig_width,fig_height]
#params = {'backend': 'ps',
#          'axes.labelsize': 18,
#          'text.fontsize': 18,
#          'legend.fontsize': 18,
#          'xtick.labelsize': 16,
#          'ytick.labelsize': 16,
#          'text.usetex': True,
#          'figure.figsize': fig_size,
#	  'font.family': 'sans-serif'}	
#py.rcParams.update(params)

#
#n = 30
#x,y = np.meshgrid(np.linspace(-0.5,0.5,n),np.linspace(-0.5,0.5,n))
#x = x.flatten()
#y = y.flatten()
#vx,vy = vLambOseen(x,y,t)
#
#py.figure(2)
#py.axes([0.125,0.15,0.95-0.175,0.95-0.125])
#py.quiver(x,y,vx,vy)
#py.axis('scaled')
#py.xlabel(r'$x$')
#py.ylabel(r'$y$')
#py.grid()               
#py.axis([-0.25,0.25,-0.25,0.25])     
"""

