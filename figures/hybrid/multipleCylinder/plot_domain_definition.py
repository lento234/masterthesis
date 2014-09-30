
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
    
fzTemp = list(np.array(fig_size)*1.75)
py.rcParams['figure.figsize'] = fzTemp

    
#-----------------------------------------------------------------------------    

#-----------------------------------------------------------------------------    

# Custom functions
def rotate(xy,theta=0.):
    xy = np.array(xy).reshape(2,-1)
    theta = np.deg2rad(theta)
    rotMax = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
    return np.dot(rotMax,xy)                       

#-----------------------------------------------------------------------------   


#-----------------------------------------------------------------------------    
# Plot Hybrid Domain configuration.

# Cylinder coordinates
beta = np.linspace(-np.pi,np.pi,1000)

# Polygon coordinates
xyCylinderA = np.array([np.cos(beta), np.sin(beta)]) + np.array([1.5,1.0]).reshape(2,-1)
xyBoundaryA = np.array([1.5*np.cos(beta), 1.5*np.sin(beta)]) + np.array([1.5,1.0]).reshape(2,-1)

xyCylinderB = np.array([np.cos(beta), np.sin(beta)]) + np.array([-1.5,0.]).reshape(2,-1)
xyBoundaryB = np.array([1.5*np.cos(beta), 1.5*np.sin(beta)]) + np.array([-1.5,0.]).reshape(2,-1)


# Paths
cA = mpl.path.Path(xyCylinderA.T,closed=True)
bA = mpl.path.Path(xyBoundaryA.T,closed=True)
cB = mpl.path.Path(xyCylinderB.T,closed=True)
bB = mpl.path.Path(xyBoundaryB.T,closed=True)

feA = mpl.path.Path(vertices=np.concatenate([cA.vertices[::-1], bA.vertices]),
                    codes=np.concatenate([cA.codes, bA.codes]))
                    
feB = mpl.path.Path(vertices=np.concatenate([cB.vertices[::-1], bB.vertices]),
                    codes=np.concatenate([cB.codes, bB.codes]))

#-----------------------------------------------------------------------------    

#-----------------------------------------------------------------------------    

# Plot the domain
fig = plt.figure()
ax = plt.gca()

# Body
ax.add_patch(mpl.patches.PathPatch(cA,fill=True,lw=1,facecolor='lightgrey',zorder=2))
ax.add_patch(mpl.patches.PathPatch(cB,fill=True,lw=1,facecolor='lightgrey',zorder=2))
# Mesh
FEplot = mpl.collections.PathCollection([feA,feB], hatch='+',color='None',edgecolor='0.7',facecolor='w')#, fill=True, hatch='+',lw=0,alpha=0.5)
ax.add_collection(FEplot)

# Plot lines
ax.plot(xyCylinderA[0],xyCylinderA[1],'k-',lw=2)
ax.plot(xyBoundaryA[0],xyBoundaryA[1],'k-',lw=1)
ax.plot(xyCylinderB[0],xyCylinderB[1],'k-',lw=2)
ax.plot(xyBoundaryB[0],xyBoundaryB[1],'k-',lw=1)

# Plot markers
ax.plot([-1.5,1.5],[0.,1.],'k+',markersize=12,markeredgewidth=2)

# Plot lengths
#ax.annotate(r'$L_1$', xy=(0., -1.95),fontsize=20,color='r')#,backgroundcolor='w')
#ax.annotate(r'$L_2$', xy=(3.55, 0.45),fontsize=20,color='r')#,backgroundcolor='w')          
#ax.annotate("",(-1.5,-2),(1.5,-2),fontsize=5,arrowprops={'arrowstyle':'|-|','fc':'r','ec':'r'})
#ax.annotate("",(3.5,-0.05),(3.5,1),fontsize=5,arrowprops={'arrowstyle':'|-|','fc':'r','ec':'r'})


# Plot Radius


# Axis properties
ax.axis('scaled')
ax.axis([-4,4,-2,3])


# Plot radius
ax.annotate(r'$R$', xy=(-0.45-1.5, 0.2-0.),fontsize=15)#,backgroundcolor='w')
ax.annotate("",(-1.5,0.),rotate((0.,1.),30).flatten()+np.array([-1.5,0.]),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})
ax.annotate(r'$R_{ext}$', xy=(-0.4-1.5, -0.65-0.),fontsize=15)#,backgroundcolor='w')
ax.annotate("",(-1.5,0.),rotate((0.,1.5),130).flatten()+np.array([-1.5,0.]),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})

ax.annotate(r'$R$', xy=(-0.45+1.5, 0.2+1.),fontsize=15)#,backgroundcolor='w')
ax.annotate("",(1.5,1.),rotate((0.,1.),30).flatten()+np.array([1.5,1.]),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})
ax.annotate(r'$R_{ext}$', xy=(-0.4+1.5, -0.65+1.),fontsize=15)#,backgroundcolor='w')
ax.annotate("",(1.5,1.),rotate((0.,1.5),130).flatten()+np.array([1.5,1.]),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})


ax.annotate(r'$\left(-R_{ext},0\right)$', xy=(-1.5, 0.1),fontsize=10,color='b')#,backgroundcolor='w')
ax.annotate(r'$\left(R_{ext},R\right)$', xy=(1.5, 1.1),fontsize=10,color='b')#,backgroundcolor='w')

#py.plot([-4,4],[0.,0.],'k-')
#py.plot([0,0],[-3.,3.],'k-')

# Axis
ax.annotate("",(-4.,0.),(4,0.),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})
ax.annotate("",(-0.,-2.),(0,3.),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})
ax.annotate(r'$x$', xy=(3.6, -0.2),fontsize=15)#,backgroundcolor='w')
ax.annotate(r'$y$', xy=(-0.2, 2.6),fontsize=15)#,backgroundcolor='w')

ax.annotate(r'$\mathbf{u}_{\infty}$', xy=(-3.3, 2.6),fontsize=20)#,backgroundcolor='w')
ax.annotate("",(-3.4,2.5),(-2.8,2.5),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})

py.axis('off')
py.savefig('hmisc_dd.pdf')


# Radius R_Ext
#ax.annotate(r'$R_{ext}$', xy=rotate(np.array([[0.6*Rext],[0.075]]),-30.),fontsize=15)#,backgroundcolor='w')
#ax.annotate("",(0.-0.02,0.),rotate(np.array([[Rext],[0.]]),-30.),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})
#ax.annotate(r'$R_{ext}$', xy=rotate(np.array([[0.3*Rext],[0.]]),-60.),fontsize=15)#,backgroundcolor='w')




'''
ax.plot(xyFESurfPoly[0],xyFESurfPoly[1],'k-',lw=2)
ax.plot(xyFEBdryPoly[0],xyFEBdryPoly[1],'k-',lw=1)
#ax.plot(xyBdryPolygon[0],xyBdryPolygon[1],'r--',lw=0.5)
#ax.plot(xySurfPolygon[0],xySurfPolygon[1],'r--',lw=0.5)

ax.axis(np.array([-0.6,1.1,-0.7,0.5]))

#ax.axis('off')

# Add text
#ax.annotate(r'wall', xy=(1.2, -1.),fontsize=15,backgroundcolor='w')

ax.annotate(r'$\Omega_{L} \backslash \Omega_E$', xy=(0.9, -0.62),fontsize=20,backgroundcolor='w',color='b')

#ax.annotate(r'', xy=(-0., -0.4), xycoords='data',fontsize=8,xytext=(-0.15,-0.5), textcoords='data',
#           arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=5, shrinkB=5,
#                           patchA=None, patchB=None,
#                           connectionstyle="arc3,rad=-0.3",),)
ax.annotate(r'$\Omega_{E}$', xy=(0.2, -0.5),fontsize=25,backgroundcolor='w',color='b')

#
#ax.annotate(r'', xy=(-0.1, -0.25), xycoords='data',fontsize=8,xytext=(-0.3,-0.4), textcoords='data',
#           arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=5, shrinkB=5,
#                           patchA=None, patchB=None,
#                           connectionstyle="arc3,rad=-0.2",),)

#ax.annotate(r'$\Omega_{int}$', xy=(-0.35, -0.4),fontsize=20,backgroundcolor='w')

py.plot(0,0,'k+',markersize=12,markeredgewidth=2)
py.plot([-0.4,0.3],[0,0],'k--',lw=0.5)
xy = rotate(np.array([[0.8,-0.45],[0.,0.]]),-20.)
py.plot(xy[0],xy[1],'k-')


ax.annotate(r'', xy=(-0.32,-0.015),xycoords='data',fontsize=12,xytext=(-0.32, 0.12), textcoords='data',
           arrowprops=dict(arrowstyle="<|-|>", color="r", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,
                           connectionstyle="arc3,rad=0.4",),)
          
ax.annotate(r'$\theta$', xy=(-0.42, 0.035),fontsize=20,color='r')#,backgroundcolor='w')
          
ax.annotate(r'$L$', xy=(0.25, 0.1),fontsize=20,color='r')#,backgroundcolor='w')
          
ax.annotate("",rotate((-0.25,0.15),-20),rotate((0.75,0.15),-20),fontsize=5,arrowprops={'arrowstyle':'|-|','fc':'r','ec':'r'})

ax.annotate("",rotate((0.8,-0.1*0.5),-20),rotate((0.8,0.1*0.5),-20),fontsize=5,arrowprops={'arrowstyle':'|-|','fc':'r','ec':'r'})

ax.annotate(r'$W$', xy=(0.78, -0.32),fontsize=20,color='r')#,backgroundcolor='w')

#py.savefig('hellipticAirfoil_dd.pdf')
'''
'''
# Sigma_wall
ax.annotate(r'$\Sigma_{wall}$', xy=(-1.4, -0.6),fontsize=15,backgroundcolor='w')

ax.annotate(r'', xy=(-0.85,-0.33),xycoords='data',fontsize=8,xytext=(-1.2, -0.5), textcoords='data',
           arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,
                           connectionstyle="arc3,rad=-0.1",),)
          
# Sigma_p
ax.annotate(r'$\Sigma_{p}$', xy=(-1.2, -1.1),fontsize=15,backgroundcolor='w')
ax.annotate(r'', xy=(-0.75,-.7),xycoords='data',fontsize=8,xytext=(-1.1, -1.), textcoords='data',
           arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,
                           connectionstyle="arc3,rad=-0.1",),)
               
# Sigma_int
ax.annotate(r'$\Sigma_{int;ext}$', xy=(-1.6, -1.6),fontsize=15,backgroundcolor='w')
ax.annotate(r'', xy=(-0.95,-1.2),xycoords='data',fontsize=8,xytext=(-1.3, -1.5), textcoords='data',
           arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,
                           connectionstyle="arc3,rad=-0.1",),)               
               
# Sigma_d
ax.annotate(r'$\Sigma_{d}$', xy=(-1.4, -2.),fontsize=15,backgroundcolor='w')
ax.annotate(r'', xy=(-1.05,-1.65),xycoords='data',fontsize=8,xytext=(-1.3, -1.9), textcoords='data',
           arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,
                           connectionstyle="arc3,rad=-0.1",),)





# Radius R
py.plot([0.],[0.],'k+')
ax.annotate(r'$R$', xy=(0.3*R, 0.1),fontsize=15)#,backgroundcolor='w')
ax.annotate("",(0.-0.03,0.),(R,0.),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})

# Radius R_Ext
#ax.annotate(r'$R_{ext}$', xy=rotate(np.array([[0.6*Rext],[0.075]]),-30.),fontsize=15)#,backgroundcolor='w')
ax.annotate("",(0.-0.02,0.),rotate(np.array([[Rext],[0.]]),-30.),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})
ax.annotate(r'$R_{ext}$', xy=rotate(np.array([[0.3*Rext],[0.]]),-60.),fontsize=15)#,backgroundcolor='w')

# Arrow dSurf
ax.annotate("",rotate((R-0.05,0.),70),rotate((R+dSurf+0.05,0.),70),arrowprops={'arrowstyle':'-','fc':'k','ec':'k'})
ax.annotate("",rotate((R-0.2,0.),70),rotate((R+0.02,0.),70),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})
ax.annotate("",rotate((R+dSurf+0.2,0.),70),rotate((R+dSurf-0.02,0.),70),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})

ax.text(-0.1,1.2,'$d_{surf}$',fontsize=12,backgroundcolor='w')                           
ax.annotate(r'', xy=(0.43, 0.99), xycoords='data',fontsize=10,xytext=(0.1, 1.2), textcoords='data',
           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,connectionstyle="arc3,rad=0.3",),)                          

# Arrow dbdry
ax.annotate("",rotate((Rext-dBdry-0.03,0.),50),rotate((Rext+0.03,0.),50),arrowprops={'arrowstyle':'<|-|>','fc':'k','ec':'k'})

ax.text(0.5,1.5,'$d_{bdry}$',fontsize=12,backgroundcolor='w')                                       
ax.annotate(r'', xy=(1.2, 1.35), xycoords='data',fontsize=10,xytext=(0.8, 1.5), textcoords='data',
           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
                           patchA=None, patchB=None,connectionstyle="arc3,rad=-0.3",),)

# Free-Stream
ax.text(-3.+0.1,2.05,'$\mathbf{u}_{\infty}$',fontsize=12,backgroundcolor='w')                                       
ax.annotate("",(-3.,2.),(-3.+0.5,2.),arrowprops={'arrowstyle':'<|-','fc':'k','ec':'k'})


py.savefig('hisc_dd.pdf')
#-----------------------------------------------------------------------------    
'''
