
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

#-----------------------------------------------------------------------------    


# Mesh parameters
meshFilePath            = './geometry/mesh.xml.gz'
boundaryDomainsFilePath = './geometry/mesh_facet_region.xml.gz'

mesh = dolfin.Mesh(meshFilePath)

xyFE = mesh.coordinates().T

xBounds = [np.min(xyFE[0]), np.max(xyFE[0]), -0.6, 0.6]
yBounds = [np.min(xyFE[1]), np.max(xyFE[1]), -0.6, 0.6]


#-----------------------------------------------------------------------------   


#-----------------------------------------------------------------------------    
# Plot Hybrid Domain configuration.

fzTemp = list(np.array(fig_size)*1.5)

py.rcParams['figure.figsize'] = fzTemp


h     = 0.01
dBdry = 5*h #+ np.spacing(10e10) # ensure no floating-point error occurs, add a small increment

xMin, xMax, xMin2, xMax2 = xBounds
yMin, yMax, yMin2, yMax2 = yBounds

xSurfPolygon = np.array([xMin+dBdry, xMax-dBdry, xMax-dBdry,xMin+dBdry,xMin+dBdry])
ySurfPolygon = np.array([yMin+dBdry, yMin+dBdry, yMax-dBdry,yMax-dBdry,yMin+dBdry])

xBdryPolygon = np.array([xMin2-dBdry, xMax2+dBdry, xMax2+dBdry,xMin2-dBdry,xMin2-dBdry])
yBdryPolygon = np.array([yMin2-dBdry, yMin2-dBdry, yMax2+dBdry,yMax2+dBdry,yMin2-dBdry])

# Define the domains
xyFESurfPoly = np.array([[xBounds[0],xBounds[1],xBounds[1],xBounds[0],xBounds[0]],
                         [yBounds[0],yBounds[0],yBounds[1],yBounds[1],yBounds[0]]])   
                     
xyFEBdryPoly = np.array([[xBounds[2],xBounds[3],xBounds[3],xBounds[2],xBounds[2]],
                         [yBounds[2],yBounds[2],yBounds[3],yBounds[3],yBounds[2]]])   
                     

xyFar = np.array([[-1,1,1,-1,-1],[-1,-1,1,1,-1]])*2.
                     
bdryPath = mpl.path.Path(np.vstack((xBdryPolygon,yBdryPolygon)).T,closed=True)
surfPath = mpl.path.Path(np.vstack((xSurfPolygon,ySurfPolygon)).T,closed=True)
FESurfPath   = mpl.path.Path(xyFESurfPoly.T,closed=True)
FEBdryPath   = mpl.path.Path(xyFEBdryPoly.T,closed=True)
farPath   = mpl.path.Path(xyFar.T,closed=True)


FERegion = mpl.path.Path(vertices=np.concatenate([FESurfPath.vertices[::-1], FEBdryPath.vertices]),
  		              codes=np.concatenate([FESurfPath.codes, FEBdryPath.codes]))

interpRegion = mpl.path.Path(vertices=np.concatenate([surfPath.vertices[::-1], bdryPath.vertices]),
                             codes=np.concatenate([surfPath.codes, bdryPath.codes]))

wallRegion = mpl.path.Path(vertices=np.concatenate([FESurfPath.vertices[::-1], farPath.vertices]),
                           codes=np.concatenate([FESurfPath.codes, farPath.codes]))


# Plot the domain
fig = plt.figure()
ax = plt.gca()
FEplot = mpl.collections.PathCollection([FERegion], hatch='+',color='None',edgecolor='gray',facecolor='w')#, fill=True, hatch='+',lw=0,alpha=0.5)
ax.add_collection(FEplot)
interPlot = mpl.collections.PathCollection([interpRegion], lw=0, color='None',facecolor='LightPink', alpha=0.7)
ax.add_collection(interPlot)
#wallPlot = mpl.collections.PathCollection([wallRegion], lw=0, ='lightgray')
#wallPlot = mpl.collections.PathCollection([wallRegion], hatch='...', facecolor='w')
#ax.add_collection(wallPlot)


ax.plot(xyFESurfPoly[0],xyFESurfPoly[1],'b-',lw=1)
ax.plot(xyFEBdryPoly[0],xyFEBdryPoly[1],'r-',lw=1)
ax.plot(xBdryPolygon,yBdryPolygon,'r--',lw=0.5)
ax.plot(xSurfPolygon,ySurfPolygon,'r--',lw=0.5)

ax.axis('scaled')
#ax.axis([-1.5,1.5,-1.1,1.1])
ax.axis([-1.1,1.1,-1.1,1.1])
ax.axis('off')

# Add text
#ax.annotate(r'wall', xy=(1.2, -1.),fontsize=15,backgroundcolor='w')

ax.annotate(r'$\Omega_{I}$', xy=(0.4, -0.85),fontsize=20,backgroundcolor='None',color='r')

ax.annotate(r'$\Omega_{L} \backslash \Omega_E$', xy=(-0.5, 0.4),fontsize=15,backgroundcolor='w')

ax.annotate(r'$\Sigma_{w}$', xy=(-0.7, -1.03), xycoords='data',fontsize=15,xytext=(-0.675, -0.85), textcoords='data',
           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,
                           connectionstyle="arc3,rad=0.2",),)

ax.annotate(r'$\Sigma_{d}$', xy=(-0.55, -0.62), xycoords='data',fontsize=15,xytext=(-0.475, -0.5), textcoords='data',
           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,
                           connectionstyle="arc3,rad=0.2",),)

# Axis
py.scatter([-0.175,0.175],[0.,0.],s=2000,c=['CornflowerBlue','Salmon'])
py.plot([0.175],[0.],'k+')
py.plot([-0.175],[0.],'k_')

ax.annotate(r'', xy=(0.35, -0.-0.15), xycoords='data',fontsize=8,xytext=(0.35,-0.+.15), textcoords='data',
           arrowprops=dict(arrowstyle="<|-", color="r", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,
                           connectionstyle="arc3,rad=-0.4",),)

ax.annotate(r'', xy=(-0.35, -0.-0.15), xycoords='data',fontsize=8,xytext=(-0.35, -0.+0.15), textcoords='data',
           arrowprops=dict(arrowstyle="<|-", color="r", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,
                           connectionstyle="arc3,rad=0.4",),)
                           
ax.annotate("",(-0.,-0.1),(-0.,-0.5),arrowprops={'arrowstyle':'<|-','fc':'r','ec':'r'})


ax.annotate("",(xBounds[3]-0.1,0.5),(xBounds[3]+dBdry+0.1,0.5),arrowprops={'arrowstyle':'-','fc':'k','ec':'k'})
ax.annotate("",(xBounds[3]-0.1,0.5),(xBounds[3]+0.02,0.5),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})
ax.annotate("",(xBounds[3]+dBdry+0.1,0.5),(xBounds[3]+dBdry-0.02,0.5),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})

ax.annotate("",(xBounds[1]+0.1,0.2),       (xBounds[1]-dBdry-0.1,0.2),  arrowprops={'arrowstyle':'-','fc':'k','ec':'k'})
ax.annotate("",(xBounds[1]+0.1,0.2),       (xBounds[1]-0.02,0.2),       arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})
ax.annotate("",(xBounds[1]-dBdry-0.1,0.2), (xBounds[1]-dBdry+0.02,0.2), arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})


#ax.annotate(r'$d_{bdry}\cdot{h}$', xy=(0.61, 0.475), xycoords='data',fontsize=10,xytext=(0.65, 0.675), textcoords='data',
#           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                           patchA=None, patchB=None,
#                           connectionstyle="arc3,rad=0.2",),)

ax.annotate(r'', xy=(0.62, 0.475), xycoords='data',fontsize=10,xytext=(0.7, 0.65), textcoords='data',
           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,connectionstyle="arc3,rad=0.3",),)
                           
ax.text(0.7,0.625,'$d_{bdry}$',fontsize=12)                           

ax.annotate(r'', xy=(0.97, 0.17), xycoords='data',fontsize=10,xytext=(0.9, 0.35), textcoords='data',
           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,connectionstyle="arc3,rad=-0.3",),)
                           
ax.text(-0.83,0.05,'$H$',fontsize=15)                                                      
ax.annotate("",(xBounds[0]-0.02,0.), (xBounds[2]+0.02,0.), arrowprops={'arrowstyle':'<|-|>','fc':'k','ec':'k'})                           

ax.text(0.75,0.33,'$d_{surf}$',fontsize=12)                           


"""
#py.savefig('hlo_dd.pdf')
#py.savefig('test.pdf')
"""
py.savefig('hcbdc_dd.pdf')
#-----------------------------------------------------------------------------    

