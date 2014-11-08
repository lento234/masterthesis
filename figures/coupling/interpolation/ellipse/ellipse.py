"""

Plot the interpolation algorithm
"""


# Import packages
import numpy as np
#import pylab as py
import matplotlib as mpl
mpl.use('PDF')

from matplotlib import pyplot as plt

#from matplotlib.patches import Polygon
import scipy.interpolate as spinterp
import scipy.io as sio
#from mpl.patches import Polygon

backends = ['pdf', 'pgf', 'Qt4Agg', 'GTK', 'GTKAgg', 'ps', 'agg', 'cairo', 
            'MacOSX', 'GTKCairo', 'WXAgg', 'template', 'TkAgg', 'GTK3Cairo', 
            'GTK3Agg', 'svg', 'WebAgg', 'CocoaAgg', 'emf', 'gdk', 'WX']

plt.ion()
#py.switch_backend(backends[0])    

#py.ion()
#py.close('all')


# Rotation function
def rotate(xyPolygon,theta=30):
    theta= np.deg2rad(theta)
    return np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),xyPolygon) 


# ----------------------------------------------------------

# Colormap (from carlos)
colorMapFile = sio.loadmat('./COLORMAP_HOT_COLD.mat')
colorMapData = colorMapFile['COLORMAP_HOT_COLD']
colorMapHC = mpl.colors.ListedColormap(zip(colorMapData[:,0],colorMapData[:,1],colorMapData[:,2]))
colorMapCold = mpl.colors.ListedColormap(zip(colorMapData[:64,0],colorMapData[:64,1],colorMapData[:64,2]))
colorMapHot = mpl.colors.ListedColormap(zip(colorMapData[64:,0],colorMapData[64:,1],colorMapData[64:,2]))

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
clim = 5     
cmap,norm = mpl.colors.from_levels_and_colors(colorMapDaen['levels']*clim,colorMapDaen['colors'],extend='both')
cmap.set_under('k',1.)

# jet Colormap
colorMap = 'jet'
jetCM = plt.cm.ScalarMappable(cmap=colorMap)
jetCM = jetCM.to_rgba(np.linspace(0,1,256))

# Latex Text
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# ----------------------------------------------------------


# ----------------------------------------------------------
# Load data

# Load the FE data
dataFE = np.load('dataFE_t4.npz')

# Make references
xFE = dataFE['x']
yFE = dataFE['y']
cellsFE = dataFE['cells']
wFE = dataFE['vorticity']
contourLevelsFE = dataFE['contourLevels']

# ----------------------------------------------------------

# ----------------------------------------------------------
# Define the geometries

# Square
#theta= np.deg2rad(30) # Angle
# Coordinates
xySquare  = np.array([[-1, 1, 1, -1, -1],
                      [-1, -1, 1, 1, -1]])


# Initial set of blobs
xB = np.array([0,0.35,1,1,1.3,1.3,1.3,1.3,1.3,1.6,1.65,1.7,
	       1.8,1.86,1.8,2.1,1.9,2.2,2.2,2.5,2.4,2.31,0.6,
	       0.9,0.8,1,0.6,0.5*1.2,0.5*2.84,0.5*2.09,
	       0.5*0.5,0.5*1.315,-0.4,0.5*3.86,0.5*3.96,0.])*1.25
        
yB = (np.array([1.5,1.2,1.3, -1.2,2,-0.5,-1.8,0.4,1,1.2,-0.2,
		.5,-2,2,-1.2,2.3,-2.3,0,0.5,2,0.8,-1.2,-1.3,
		0.5,-0.3,-0.5,1.6,1.02,-1.13,0.015,0.8,
		0.397,-1*2*-0.65,1.14,-0.59,-1*2.*-0.5])*0.5)*-1.25

# Lagrangian grid
hLagr = 0.3
xLagr = np.arange(-5-hLagr*0.5,5+hLagr+hLagr*0.5,hLagr)
XLagr,YLagr = np.meshgrid(xLagr,xLagr)

# Integrator
beta = np.linspace(-np.pi,np.pi,1000)

# Ellipse
widthSurf = 2.
heightSurf = 2.*0.25
widthFE   = 4.
heigthFE   = 4.*0.75
# Body polygon
xyEllipse = rotate(np.array([0.5*widthSurf*np.cos(beta),
                             0.5*heightSurf*np.sin(beta)]))


# FE mesh boundary polygon
xyBoundsFE = [-2,2,-1.5,1.5]
xyFEBoundary = rotate(np.array([(0.5+np.spacing(1e12))*widthFE*np.cos(beta),
                                (0.5+np.spacing(1e12))*heigthFE*np.sin(beta)]))

# Surface polygon
dSurf = 0.1
xySurfacePoly = rotate(np.array([(0.5*widthSurf+dSurf)*np.cos(beta),
                                 (0.5*heightSurf+dSurf)*np.sin(beta)]))



# Boundary polygon
dBdry = 0.1
xyBoundaryPoly = rotate(np.array([(0.5*widthFE-dBdry)*np.cos(beta),
                                  (0.5*heigthFE-dBdry)*np.sin(beta)]))


xyPanel = rotate(np.array([(0.5*widthSurf+0.3*dSurf)*np.cos(beta),
                           (0.5*heightSurf+0.3*dSurf)*np.sin(beta)]))


# ----------------------------------------------------------

# ----------------------------------------------------------
# Make patches

# Body
#ellipseBody = mpl.patches.Ellipse(xy=[0.,0.], width=wSurf,height=hSurf,angle=30.,
#                                  fill=True,lw=1,facecolor='LightGrey',zorder=100)

# FE mesh boundary
#ellipseFE = mpl.patches.Ellipse(xy=[0.,0.], width=wFE,height=hFE,angle=30.,
                                  #fill=True,lw=0.5,facecolor='None',zorder=3)



# Ellipse
ellipse = mpl.path.Path(xyEllipse.T,closed=True)

ellipseL = mpl.path.Path(xyEllipse.T + np.array([-2.5,0]),closed=True)
ellipseR = mpl.path.Path(xyEllipse.T + np.array([2.5,0]),closed=True)
#squareN = mpl.path.Path(np.array([[-1, 1, 1, -1, -1],
#                                  [-0.5, -0.5, 0.5, 0.5, -0.5]]).T,closed=True)

# Square Patch
#squarePatch = mpl.patches.PathPatch(square,facecolor='orange',lw=2)
#squarePatch = mpl.patches.PathPatch(square,fill=True,hatch='/',lw=2,rasterized=False,facecolor='w')
#
## Surface Polygon
surface = mpl.path.Path(xySurfacePoly.T,closed=True)
##surfacePatch = mpl.patches.PathPatch(surface,fill=False,lw=2,rasterized=False)
#
## Boundary Polygon
boundary = mpl.path.Path(xyBoundaryPoly.T,closed=True)
##squarePatch = mpl.patches.PathPatch(square,fill=False,hatch='//',lw=2,rasterized=False)
#
## FE Boundary
FEboundary = mpl.path.Path(xyFEBoundary.T,closed=True)
##squarePatch = mpl.patches.PathPatch(square,fill=False,hatch='//',lw=2,rasterized=False)
#
## Interpolation region
interpRegion = mpl.path.Path(vertices=np.concatenate([surface.vertices[::-1], boundary.vertices]),
                             codes=np.concatenate([surface.codes, boundary.codes]))

# ----------------------------------------------------------


# ----------------------------------------------------------
# Calculations

clim = 5
cmap,norm = mpl.colors.from_levels_and_colors(colorMapDaen['levels']*clim,colorMapDaen['colors'],extend='both')


## Determine the cells inside
inside = FEboundary.contains_points(np.vstack((xFE,yFE)).T)
xFEI   = xFE[inside]
yFEI   = yFE[inside]
wFEI   = wFE[inside]

# Determine the cells inside
inside = boundary.contains_points(np.vstack((xB,yB)).T)
xBO    = xB[~inside]
yBO    = yB[~inside]

# Determine the lagrangian nodes inside the interp region
insideS = surface.contains_points(np.vstack((XLagr.flatten(),YLagr.flatten())).T)
insideB = boundary.contains_points(np.vstack((XLagr.flatten(),YLagr.flatten())).T)
insideBoth = np.where(~insideS & insideB)[0]
XL = XLagr.flatten()[insideBoth]
YL = YLagr.flatten()[insideBoth]

# Generate structured grid
#xStructured = np.arange(xyBoundsFE[0]-hLagr*0.5,xyBoundsFE[1]+hLagr*0.5*1.5,hLagr*0.5)
#yStructured = np.arange(xyBoundsFE[2]-hLagr*0.5,xyBoundsFE[3]+hLagr*0.5*1.5,hLagr*0.5)
xStructured = np.arange(xyBoundsFE[0]-hLagr*0.5,xyBoundsFE[1]+hLagr*0.5*1.5,hLagr)
yStructured = np.arange(xyBoundsFE[2]-hLagr*0.5,xyBoundsFE[3]+hLagr*0.5*1.5,hLagr)
xStructured, yStructured = np.meshgrid(xStructured,yStructured)
nStructured = xStructured.shape
xStructured, yStructured = rotate(np.vstack((xStructured.flatten(),yStructured.flatten()))).reshape(2,nStructured[0],nStructured[1])

# Interpolate daa
wStructured = spinterp.griddata(np.vstack((xFEI,yFEI)).T, wFEI, 
                                np.vstack((xStructured.flatten(),
                                           yStructured.flatten())).T).reshape(nStructured[0],nStructured[1])#method='nearest'                                

#masked_array = np.ma.masked_where(np.isnan(wStructured),wStructured)
#wTemp = wStructured.copy()
wStructured[np.isnan(wStructured)] = 0.
#wTemp[np.isnan(wStructured)] = 0.

# Interpolate from Structured grid to blobs
WL = spinterp.griddata(np.vstack((xStructured.flatten(),yStructured.flatten())).T, wStructured.flatten(),
                       np.vstack((XL,YL)).T,method='linear')
                       
selected = np.abs(WL)>0.1

# ----------------------------------------------------------

'''
# ----------------------------------------------------------

# Plot the Lagrangian Domain

#fig = py.figure(1)
#ax  = fig.add_subplot(111)
#ax.set_rasterized(False)
#
## Plot Ellipse
#ax.add_patch(mpl.patches.PathPatch(ellipse,fill=True,lw=1,facecolor='LightGrey',zorder=1))
##py.plot(xySurfacePoly[0],xySurfacePoly[1],'k-')
##ax.add_patch(ellipseFE)
#                          
## Plot blobs
#py.scatter(xB,yB,s=100,c='w',lw=1)
#
#py.axis('scaled')
#py.axis([-3,3,-3,3])
#py.axis('off')

#py.savefig('./lagrangian.pdf')

# ----------------------------------------------------------


# ----------------------------------------------------------
# Plot the full Eulerian domin

#fig = py.figure(2)
#ax  = fig.add_subplot(111)
#
## Plot vorticity
#py.tripcolor(xFE,yFE,wFE,norm=norm,cmap=cmap,zorder=1,rasterized=True)
## Plot mesh
#py.triplot(xFE,yFE,cellsFE,lw=0.5,color='0.75',zorder=2)
## Plot Square
#ax.add_patch(mpl.patches.PathPatch(ellipse,fill=True,lw=1,facecolor='LightGrey',zorder=3))
##ax.add_patch(ellipseFE)
#
#
#py.clim(-clim,clim)
#py.axis('scaled')
#py.axis([-3,3,-3,3])
#py.axis('off')

#py.savefig('./fullEulerian.pdf')

# ----------------------------------------------------------

# ----------------------------------------------------------
# Plot Hybrid Eulerian

#fig = py.figure(3)
#ax  = fig.add_subplot(111)
#
#clim = 5
#cmap,norm = mpl.colors.from_levels_and_colors(colorMapDaen['levels']*clim,colorMapDaen['colors'],extend='both')
#
## Plot Vorticity
#py.tripcolor(xFEI,yFEI,wFEI,norm=norm,cmap=cmap,rasterized=True,zorder=1)
## Plot mesh
#py.triplot(xFEI,yFEI,color='0.75',lw=0.5,zorder=2)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(ellipse,fill=True,lw=1,facecolor='LightGrey',zorder=3))
#
#py.clim(-clim,clim)
#py.axis('scaled')
#py.axis([-3,3,-3,3])
#py.axis('off')

#py.savefig('./eulerian.pdf')

# ----------------------------------------------------------

# ----------------------------------------------------------
# Plot hybrid formation

#fig = py.figure(4)
#ax  = fig.add_subplot(111)
#
## Plot Eulerian grid
#py.triplot(xFEI,yFEI,color='0.75',lw=0.5,zorder=1)
#
## Plot Eulerian boundary
#py.plot(xyFEBoundary[0],xyFEBoundary[1],'k-',lw=0.5,zorder=2)
#
## Plot ellipse
#ax.add_patch(mpl.patches.PathPatch(ellipse,fill=True,lw=1,facecolor='LightGrey',zorder=3))
#
## Plot blobs
#py.scatter(xB+0.075,yB,s=100,c='w',lw=1,zorder=4)
#
#py.axis('scaled')
#py.axis([-3,3,-3,3])
#py.axis('off')
#
##py.plot([-4.5,5.2,5.2,-4.5,-4.5],[-4.5,-4.5,4.5,4.5,-4.5],'k--')
#py.plot(xySurfacePoly[0],xySurfacePoly[1],'k--',zorder=6)
#py.plot(xyPanel[0],xyPanel[1],'0.5',lw=2,zorder=4)
#
#
### Add text
##ax.annotate(r'$\partial \Omega_{E}$', xy=(-0.60, 3.75),fontsize=20,zorder=20)
#ax.annotate(r'$\partial \Omega_{E}$', xy=(0.3, 1.6), xycoords='data',fontsize=20,xytext=(1., 2.2), textcoords='data',
#           arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=5, shrinkB=5,
#                           patchA=None, patchB=None,
#                           connectionstyle="arc3,rad=0.2",),)
##ax.annotate(r'$\partial \Omega_{\mathrm{wall}}$', xy=(-2.5, -0.75),fontsize=20,zorder=20)
#ax.annotate(r'$\partial \Omega_{\mathrm{P}}$', xy=(-0.54, 0.04), xycoords='data',fontsize=20,xytext=(-0.234, 0.84), textcoords='data',
#           arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=5, shrinkB=5,
#                           patchA=None, patchB=None,
#                           connectionstyle="arc3,rad=0.2",),)
#                           
#ax.annotate(r'$\partial \Omega_{body}$', xy=(-0.4,-0.4), xycoords='data',fontsize=20,xytext=(-1.328, -1.20), textcoords='data',
#           arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=5, shrinkB=5,
#                           patchA=None, patchB=None,
#                           connectionstyle="arc3,rad=0.2",),zorder=100)                           
##ax.annotate(r'$\partial \Omega_{P}$', xy=(-1., 1.5),fontsize=20,zorder=20)
##ax.annotate(r'$\Omega_{B}$', xy=(-0.7, 1.25),fontsize=25,zorder=20)
##ax.annotate(r'$\gamma(s)$', xy=(0, 0.575),fontsize=20,zorder=20)
##ax.annotate(r'$\omega_i$', xy=(0.2, 1.7),fontsize=20,zorder=20)
##ax.annotate(r'\textsf{body}$', xy=(-0.4, 0.4),fontsize=15,zorder=20,backgroundcolor='w')
#
#
#py.savefig('./hybrid.pdf')

# ----------------------------------------------------------

# ----------------------------------------------------------
# Plot hybrid formation

#fig = py.figure(41)
#ax  = fig.add_subplot(111)
#
## Plot Eulerian grid
#py.triplot(rotate(np.vstack((xFEI,yFEI)),-30)[0],
#           rotate(np.vstack((xFEI,yFEI)),-30)[1]*0.5,color='0.75',lw=0.5,zorder=1)
## Plot Eulerian boundary
#py.plot(xyFEBoundary[0],xyFEBoundary[1]*0.5,'k-',lw=0.5,zorder=2)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(squareN,fill=True,hatch='//',lw=1,facecolor='w',zorder=3))
## Plot blobs
#xyBlobsTemp = rotate(np.vstack((xB,yB)),-60-30)*0.5
#xyBlobsTemp[1] *= 0.5
#xyBlobsTemp[1] -= 0.5
#xyBlobsTemp = xyBlobsTemp[:,np.abs(xyBlobsTemp[1])>0.7]
#xyBlobsTemp = xyBlobsTemp[:,xyBlobsTemp[0]>-0.475]
##xyBlobsTemp[1] -= 0.75
#py.scatter(xyBlobsTemp[0],
#           -xyBlobsTemp[1],s=600,c='w',lw=1,zorder=4)
#
## Plot lines
#py.plot([-1,1],[0.7,0.7],'k--',lw=1,zorder=5)
#
#py.plot([-1,1],[0.52,0.52],'k-',lw=5,zorder=5)
#
## Draw arrow
#ax.arrow(-0.8,0.7,0,1.0,head_width=0.025,head_length=0.05,fc='k',ec='k',zorder=10)
#ax.arrow(-0.7,0.5,0,0.15,head_width=0.025,head_length=0.05,fc='k',ec='k',zorder=10)
#ax.arrow(0.7,1.5,0,-1+0.05,head_width=0.025,head_length=0.05,fc='k',ec='k',zorder=3)
#
## Add text
#ax.annotate(r'$\Omega_{E}$', xy=(0.5, 0.55),fontsize=25,zorder=20)
#ax.annotate(r'$\Omega_{P}$', xy=(-0.6, 0.55),fontsize=25,zorder=20)
#ax.annotate(r'$\Omega_{B}$', xy=(-0.7, 1.25),fontsize=25,zorder=20)
#ax.annotate(r'$\gamma(s)$', xy=(0, 0.575),fontsize=20,zorder=20)
#ax.annotate(r'$\omega_i$', xy=(0.2, 1.7),fontsize=20,zorder=20)
#ax.annotate(r'\textsf{body}$', xy=(-0.4, 0.4),fontsize=15,zorder=20,backgroundcolor='w')
#
#py.axis('scaled')
#py.axis([-0.9,0.9,0.35,2])
#py.axis('off')
##py.plot([-5,5,5,-5,-5],[-5,-5,5,5,-5],'k--')
#
#py.savefig('./hybrid_domains.pdf')

# ----------------------------------------------------------


# ----------------------------------------------------------
# Plot hybrid formation

#fig = py.figure(41)
#ax  = fig.add_subplot(111)
#
## Plot Eulerian grid
#py.triplot(rotate(np.vstack((xFEI,yFEI)),-30)[0],
#           rotate(np.vstack((xFEI,yFEI)),-30)[1]*0.5,color='0.75',lw=0.5,zorder=1)
## Plot Eulerian boundary
#py.plot(xyFEBoundary[0],xyFEBoundary[1]*0.5,'k-',lw=0.5,zorder=2)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(squareN,fill=True,hatch='//',lw=1,facecolor='w',zorder=3))
## Plot blobs
#xyBlobsTemp = rotate(np.vstack((xB,yB)),-60-30)*0.5
#xyBlobsTemp[1] *= 0.5
#xyBlobsTemp[1] -= 0.5
#xyBlobsTemp = xyBlobsTemp[:,np.abs(xyBlobsTemp[1])>0.7]
#xyBlobsTemp = xyBlobsTemp[:,xyBlobsTemp[0]>-0.475]
##xyBlobsTemp[1] -= 0.75
#py.scatter(xyBlobsTemp[0],
#           -xyBlobsTemp[1],s=600,c='w',lw=1,zorder=4)
#
## Plot lines
#interpRegTemp = mpl.path.Path(np.array([[-1, 1, 1, -1, -1],
#                                        [0.7,0.7,1.3,1.3,0.7]]).T,closed=True)
#
#
##py.plot([-1,1,1,-1,-1],[0.7,0.7,1.3,1.3,0.7],'k--',lw=1,zorder=5)
#ax.add_patch(mpl.patches.PathPatch(interpRegTemp,fill=True,lw=0,facecolor='LightPink', alpha=0.7,zorder=5))
#py.plot([-1, 1, 1, -1, -1],[0.7,0.7,1.3,1.3,0.7],'r--',zorder=6)
#
#py.plot([-1,1],[0.52,0.52],'k-',lw=5,zorder=5)
#
## Draw arrow
#ax.annotate("",(-0.7,0.5),(-0.7,0.7),arrowprops={'arrowstyle':'<|-|>','fc':'r','ec':'r'},zorder=20)
#ax.annotate("",(-0.7,1.3),(-0.7,1.5),arrowprops={'arrowstyle':'<|-|>','fc':'r','ec':'r'},zorder=20)
#
## Add text
#ax.annotate(r'$\Omega_{P}$', xy=(0.3, 0.55),fontsize=25,zorder=20)
#ax.annotate(r'$d_{\mathrm{surf}}\cdot{h}$', xy=(-0.6, 0.57),fontsize=15,zorder=20)
#ax.annotate(r'$d_{\mathrm{bdry}}\cdot{h}$', xy=(-0.6, 1.37),fontsize=15,zorder=20)
#ax.annotate(r'\textsf{body}$', xy=(-0.4, 0.4),fontsize=15,zorder=20,backgroundcolor='w')
#
#py.axis('scaled')
#py.axis([-0.9,0.9,0.35,2])
#py.axis('off')

#py.savefig('./hybrid_domains_withInterpReg.pdf')

# ----------------------------------------------------------



# ----------------------------------------------------------
# Plot hybrid formation with Eulerina data

#fig = py.figure(5)
#ax  = fig.add_subplot(111)
#
#
## Plot Vorticity
##py.tripcolor(xFEI,yFEI,wFEI,contourLevelsFE,rasterized=True,zorder=1)
#py.tripcolor(xFEI,yFEI,wFEI,norm=norm,cmap=cmap,rasterized=True,zorder=1)
#py.clim(-clim,clim)
## Plot Eulerian grid
#py.triplot(xFEI,yFEI,color='0.75',lw=0.5,zorder=2)
## Plot Eulerian boundary
#py.plot(rotate(xyFEBoundary)[0],rotate(xyFEBoundary)[1],'k-',lw=0.5,zorder=3)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(square,fill=True,lw=1,facecolor='LightGrey',zorder=4))
## Plot blobs
#py.scatter(xB,yB,s=100,c='w',lw=1,zorder=5)
#
#py.axis('scaled')
#py.axis([-5,5,-5,5])
#py.axis('off')
#
## Add text
#ax.annotate(r'$\partial \Omega_{E}$', xy=(-0.4, 3.75),fontsize=15,zorder=20)
#ax.annotate(r'$\partial \Omega_{\mathrm{wall}}$', xy=(-0.8, -0.6),fontsize=15,zorder=20)
#ax.annotate(r'$\Omega_{E}$', xy=(-3, 0.5),fontsize=25,zorder=20)
#                           
#
##py.plot([-5,5,5,-5,-5],[-5,-5,5,5,-5],'k--')
#py.savefig('./hybridwithData.pdf')

# ----------------------------------------------------------

# ----------------------------------------------------------
# Plot definition of the interpolation region according to stock
#
#fig = py.figure(6)
#ax  = fig.add_subplot(111)
#
## Plot Eulerian grid
#py.triplot(xFEI,yFEI,color='0.5',lw=0.5,zorder=1)
## Plot Eulerian boundary
#py.plot(xyFEBoundary[0],xyFEBoundary[1],'k-',lw=0.5,zorder=2)
#
#
## Plot the bounding box
#xyBBOX = np.hstack((np.min(xyBoundaryPoly,axis=1),np.max(xyBoundaryPoly,axis=1)))
#py.plot([xyBBOX[0],xyBBOX[2],xyBBOX[2],xyBBOX[0],xyBBOX[0]],
#        [xyBBOX[1],xyBBOX[1],xyBBOX[3],xyBBOX[3],xyBBOX[1]],'g:',lw=2)
#
## Plot square
#ax.add_patch(mpl.patches.PathPatch(ellipse,fill=True,lw=1,facecolor='LightGrey',zorder=3))
## Plot blobs
#py.scatter(xB,yB,s=100,c='w',lw=1,zorder=4)
## Plot interp-region
##col = mpl.collections.PathCollection([interpRegion], lw=0, color='None',facecolor='LightPink', alpha=0.7, zorder=5)
##ax.add_collection(col)
#ax.add_patch(mpl.patches.PathPatch(boundary, lw=0, facecolor='LightPink', alpha=0.7, zorder=5))
## Plot interp region boundary lines
##py.plot(xySurfacePoly[0],xySurfacePoly[1],'r--',zorder=6)
#py.plot(xyBoundaryPoly[0],xyBoundaryPoly[1],'r--',zorder=6)
#
#py.axis('scaled')
#py.axis([-3,3,-3,3])
#py.axis('off')
#
## Draw arrow
### Add text
##ax.annotate(r'$\partial \Omega_{E}$', xy=(-0.60, 3.75),fontsize=20,zorder=20)
##ax.annotate(r'$\partial \Omega_{p}$', xy=(-0.266, 0.2), xycoords='data',fontsize=20,xytext=(-0.2, 0.6), textcoords='data',
##           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
##                           patchA=None, patchB=None,
##                           connectionstyle="arc3,rad=0.2",),zorder=20)
##ax.annotate(r'$\partial \Omega_{\mathrm{wall}}$', xy=(-2.5, -0.75),fontsize=20,zorder=20)
##ax.annotate(r'$\partial \Omega_{int}$', xy=(-1.2, -1.3), xycoords='data',fontsize=20,xytext=(-0.4, -0.7), textcoords='data',
##           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
##                           patchA=None, patchB=None,
##                           connectionstyle="arc3,rad=0.2",),zorder=20)
#ax.annotate('', xy=(-1.2, -1.3), xycoords='data',fontsize=20,xytext=(-0.4, -0.7), textcoords='data',
#           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                           patchA=None, patchB=None,
#                           connectionstyle="arc3,rad=0.2",),zorder=20)
#ax.text(-0.45,-0.75,r'$\partial \Omega_{int}$',fontsize=20,zorder=20)
#
#
##          
##ax.annotate(r'$\{\partial \Omega_{int}\}_{\mathrm{BBOX}}$', xy=(-0.71, 1.57), xycoords='data',fontsize=20,xytext=(-0.25, 2.5), textcoords='data',
##           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
##                           patchA=None, patchB=None,
##                           connectionstyle="arc3,rad=0.2",),zorder=20)
#            
#ax.annotate('', xy=(-0.71, 1.57), xycoords='data',fontsize=20,xytext=(-0.1, 2.), textcoords='data',
#           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                           patchA=None, patchB=None,
#                           connectionstyle="arc3,rad=0.2",),zorder=20)
#ax.text(-0.1,2.,r'$\{\partial \Omega_{int}\}_{\mathrm{BBOX}}$'   ,fontsize=20,zorder=20)                           
#ax.text(-1,1.8,r'$1$',fontsize=15,zorder=20)                           
#ax.add_patch(mpl.patches.Circle((-0.93,1.88),0.15,fc='0.8'))
#ax.text(-1.4,-0.9,r'$2$',fontsize=15,zorder=20)                           
#ax.add_patch(mpl.patches.Circle((-1.34,-0.82),0.15,fc='0.8',zorder=19))
#
##r     
##ax.annotate(r'$\Omega_{int}$', xy=(-0.6, -1.125),fontsize=25,zorder=20)
#
#
##py.plot([-5,5,5,-5,-5],[-5,-5,5,5,-5],'k--')
##
#py.savefig('./interpRegion.pdf')

# ----------------------------------------------------------


# ----------------------------------------------------------
# Plot removing particles in the outer polygon 
#
#fig = py.figure(7)
#ax  = fig.add_subplot(111)
#
## Plot Eulerian grid
#py.triplot(xFEI,yFEI,color='0.5',lw=0.5,zorder=1)
## Plot Eulerian boundary
#py.plot(xyFEBoundary[0],xyFEBoundary[1],'k-',lw=0.5,zorder=2)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(ellipse,fill=True,lw=1,facecolor='LightGrey',zorder=3))
## Plot blobs
#py.scatter(xBO,yBO,s=100,c='w',lw=1,zorder=4)
## Plot interp-region
##col = mpl.collections.PathCollection([interpRegion], facecolor='LightPink', alpha=0.8, zorder=10, lw=0)
##ax.add_patch(mpl.patches.PathPatch(interpRegion,fill=True,lw=0,facecolor='Yellow',alpha=0.8,zorder=10)) #Lime, LawnGreen
##ax.add_collection(col)
##ax.set_clip_path(mpl.patches.Polygon(xySurfacePoly.T))
#
##py.plot(rotate(xySurfacePoly)[0],rotate(xySurfacePoly)[1],'r--',zorder=11)
#py.plot(xyBoundaryPoly[0],xyBoundaryPoly[1],'r--',zorder=4)
#
#py.axis('scaled')
#py.axis([-3,3,-3,3])
#py.axis('off')
#py.plot([-5,5,5,-5,-5],[-5,-5,5,5,-5],'k--')
#
#ax.text(-1.4+2.75,-0.9+2.75,r'$3$',fontsize=15,zorder=20)                           
#ax.add_patch(mpl.patches.Circle((-1.34+2.75,-0.82+2.75),0.15,fc='0.8',zorder=19))
#
#py.savefig('./particleRemoved.pdf')

# ----------------------------------------------------------


# ----------------------------------------------------------
# Generating particles inside the interpolation region, onto the Lagrangian grid

#fig = py.figure(8)
#ax  = fig.add_subplot(111)
#
## Plot the lagrangian grid
#py.plot(XLagr,YLagr,alpha=0.5,color='Lime',zorder=1,lw=0.5)
#py.plot(XLagr.T,YLagr.T,alpha=0.5,color='Lime',zorder=1,lw=0.5)
## Plot Eulerian boundary
#py.plot(xyFEBoundary[0],xyFEBoundary[1],'k-',lw=0.5,zorder=2)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(ellipse,fill=True,lw=1,facecolor='LightGrey',zorder=3))
## Plot blobs (outside)
#py.scatter(xBO,yBO,s=80,c='w',lw=1,zorder=4,edgecolor='0.75')
## Plot the blobs inside
#py.scatter(XL,YL,s=80,c='w',lw=1,zorder=4)
## Plot interpolation boundary lines
#py.plot(xySurfacePoly[0],xySurfacePoly[1],'r--',zorder=5)
#py.plot(xyBoundaryPoly[0],xyBoundaryPoly[1],'r--',zorder=5)
#
#py.axis('scaled')
#py.axis([-3,3,-3,3])
#py.axis('off')
#
#py.savefig('./generatedParticles.pdf')

# ----------------------------------------------------------

# ----------------------------------------------------------
# Interpolate the mesh data from FE to structured grid

#fig = py.figure(9)
#ax  = fig.add_subplot(111)
#
## Plot mesh
#py.triplot(xFEI-2.5,yFEI,'-',color='0.5',lw=0.5,zorder=1)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(ellipseL,fill=True,lw=0,facecolor='LightGrey',edgecolor='None',zorder=2))
#ax.add_patch(mpl.patches.PathPatch(ellipseR,fill=True,lw=0,facecolor='LightGrey',edgecolor='None',zorder=2))
## Plot the structured grid
#py.plot(xStructured+2.5,yStructured,'-',color='0.5',zorder=1,lw=0.5)
#py.plot(xStructured.T+2.5,yStructured.T,'-',color='0.5',zorder=1,lw=0.5)
#
## Add annotation
#ax.annotate("", xy=(2, 1.75), xycoords='data',xytext=(-1.5, 1.75), textcoords='data',
#            arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                            patchA=None, patchB=None,
#                            connectionstyle="arc3,rad=-0.5",),)
#
#py.axis('scaled')
#py.axis([-5,5,-4,4])
#py.axis('off')
##py.plot([-10,10,10,-10,-10],[-5,-5,6,6,-5],'k--')
#
#py.savefig('./interpolation_FE2StructuredGrid.pdf')

# ----------------------------------------------------------


# ----------------------------------------------------------
# Interpolate the mesh data from FE to structured grid

#fig = py.figure(91)
#ax  = fig.add_subplot(111)
#
## Plot mesh
#py.triplot(xFEI,yFEI,'-',color='silver',lw=0.5,zorder=1)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(ellipse,fill=True,lw=0,facecolor='LightGrey',edgecolor='None',zorder=2))
## Plot the structured grid
#py.plot(xStructured,yStructured,'-',color='LightCoral',zorder=1,lw=0.5)
#py.plot(xStructured.T,yStructured.T,'-',color='LightCoral',zorder=1,lw=0.5)
#
## Add annotation
##ax.annotate("", xy=(2, 1.75), xycoords='data',xytext=(-1.5, 1.75), textcoords='data',
##            arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                            #patchA=None, patchB=None,
#                            #connectionstyle="arc3,rad=-0.5",),)
#
## Draw xis
##ax.arrow(xStructured[0,0],yStructured[0,0],rotate([0.,0.5])[0],rotate([0.,0.5])[1],head_width=0.1,head_length=0.1,fc='k',ec='k',zorder=10,lw='0.5')
##ax.arrow(xStructured[0,0],yStructured[0,0],rotate([0.5,0.])[0],rotate([0.5,0.])[1],head_width=0.1,head_length=0.1,fc='k',ec='k',zorder=10,lw='0.5')
#
#ax.arrow(0.,0.,rotate([0.,0.5])[0],rotate([0.,0.5])[1],head_width=0.1,head_length=0.1,fc='k',ec='k',zorder=10,lw='0.5')
#ax.arrow(0.,0.,rotate([0.5,0.])[0],rotate([0.5,0.])[1],head_width=0.1,head_length=0.1,fc='k',ec='k',zorder=10,lw='0.5')
#
#
#py.axis('scaled')
#py.axis([-3,3,-3,3])
#py.axis('off')
#
#py.plot(xStructured[0,0],yStructured[0,0],'ko')
#py.plot([xStructured[0,0], xStructured[0,0]+2.],[yStructured[0,0],yStructured[0,0]],'k--',lw=0.5)
#
#
#ax.annotate(r'$(x_o,y_o)_{str}$', xy=(xStructured[0,0]-0.4, yStructured[0,0]-0.3),fontsize=12,zorder=20)
#ax.annotate(r'$\theta_{\mathrm{loc}}$', xy=(0.55, -2.2),fontsize=12,zorder=20)
##ax.annotate(r"$y'$", xy=rotate([-0.25, 0.9])+2.,fontsize=13,zorder=20)
##ax.annotate(r"$x'$", xy=rotate([0.65, -0.15])+([xStructured[0,0], yStructured[0,0]]),fontsize=13,zorder=20)
##ax.annotate(r"$y'$", xy=rotate([-0.3, 0.5])+([xStructured[0,0], yStructured[0,0]]),fontsize=13,zorder=20)
#ax.annotate(r"$x'$", xy=rotate([0.5, -0.15]),fontsize=13,zorder=20)
#ax.annotate(r"$y'$", xy=rotate([-0.3, 0.5]),fontsize=13,zorder=20)
#
#ax.annotate("", xy=(xStructured[0,0]+1.5, yStructured[0,0]), xycoords='data',xytext=(xStructured[0,0]+1.5-0.25, yStructured[0,0]+0.75), textcoords='data',
#            arrowprops=dict(arrowstyle="<|-|>", color="k", shrinkA=5, shrinkB=5,
#                            patchA=None, patchB=None,
#                            connectionstyle="arc3,rad=-0.2",),)
#
#
#py.plot([-3,3,3,-3,-3],[-3,-3,3,3,-3],'k--')

#py.savefig('./interpolation_FE2andStructuredGrid.pdf')

# ----------------------------------------------------------



# ----------------------------------------------------------
# Interpolate the mesh data from FE to structured grid

fig = plt.figure(10)
ax  = fig.add_subplot(111)

# Plot Vorticity on FE
##py.tripcolor(xFEI-5,yFEI,wFEI,contourLevelsFE,rasterized=True,zorder=1,cmap=colorMapHC)
plt.tripcolor(xFEI-2.5,yFEI,wFEI,norm=norm,cmap=cmap,rasterized=True,zorder=1,shading='gouraud')
plt.clim(-5,5)

# Plot vorticity on structureed
#py.pcolormesh(xStructured+5,yStructured,wStructured,rasterized=True,zorder=1,cmap=colorMapHC)
#wStructuredPlot = np.ma.masked_where(np.isnan(wStructured),wStructured)
plt.pcolormesh(xStructured+2.5,yStructured,wStructured,rasterized=True,zorder=1,cmap=cmap,norm=norm,shading='gouraud')
#py.pcolor(xStructured+2.5,yStructured,wStructured,rasterized=True,zorder=1,cmap=cmap,norm=norm,edgecolor=wStructured.flatten(),shading='gouraud')
plt.clim(-5,5)  

## Plot mesh
plt.triplot(xFEI-2.5,yFEI,'-',color='0.75',lw=0.5,zorder=2)
#
## Plot the structured grid
plt.plot(xStructured+2.5,yStructured,'-',color='0.75',zorder=2,lw=0.5)
plt.plot(xStructured.T+2.5,yStructured.T,'-',color='0.75',zorder=2,lw=0.5)
#
## Plot square
ax.add_patch(mpl.patches.PathPatch(ellipseL,fill=True,facecolor='LightGrey',edgecolor='None',zorder=3))
ax.add_patch(mpl.patches.PathPatch(ellipseR,fill=True,facecolor='LightGrey',edgecolor='None',zorder=3))
#
## Add annotation
ax.annotate("", xy=(2, 1.75), xycoords='data',xytext=(-1.5, 1.75), textcoords='data',
            arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
                            patchA=None, patchB=None,
                            connectionstyle="arc3,rad=-0.5",),)
#
plt.axis('scaled')
plt.axis([-5.5,5.5,-4,4])
plt.axis('off')
#
## Plot Eulerian boundary
#py.plot(xyFEBoundary[0]+2.5,xyFEBoundary[1],'k-',lw=0.5,zorder=2)
#py.plot(xyFEBoundary[0]-2.5,xyFEBoundary[1],'k-',lw=0.5,zorder=2)
## Plot interp region boundary lines
#py.plot(xySurfacePoly[0]+2.5,xySurfacePoly[1],'r--',zorder=6)
#py.plot(xySurfacePoly[0]-2.5,xySurfacePoly[1],'r--',zorder=6)
#
#py.plot(xyBoundaryPoly[0]+2.5,xyBoundaryPoly[1],'r--',zorder=6)
#py.plot(xyBoundaryPoly[0]-2.5,xyBoundaryPoly[1],'r--',zorder=6)
#
#
#
#ax.annotate(r'$\omega$', xy=(-2.2,2),fontsize=20,zorder=20)
#ax.annotate(r'$\hat{\omega}$', xy=(2.2,2),fontsize=20,zorder=20)
#ax.annotate(r'$W$', xy=(0.,3),fontsize=20,zorder=20)


#py.savefig('./interpolation_FE2StructuredGrid_withData.pdf')

# ----------------------------------------------------------


# ----------------------------------------------------------
# Interpolate the mesh data from structured grid to blobs


# Plot hybrid eulerian
#fig = py.figure(11)
#ax  = fig.add_subplot(111)
#
## Plot vorticity on structured
#py.pcolormesh(xStructured-2.5,yStructured,wStructured,rasterized=True,zorder=1,cmap=cmap,norm=norm)
#py.clim(-5,5)         
#
## Plot the structured grid
#py.plot(xStructured-2.5,yStructured,'-',color='0.75',zorder=2,lw=0.5)
#py.plot(xStructured.T-2.5,yStructured.T,'-',color='0.75',zorder=2,lw=0.5)
#
## Plot interp region boundary lines
#py.plot(xySurfacePoly[0]-2.5,xySurfacePoly[1],'r--',zorder=6)
#py.plot(xyBoundaryPoly[0]-2.5,xyBoundaryPoly[1],'r--',zorder=6)
#py.plot(xyFEBoundary[0]-2.5,xyFEBoundary[1],'k-',zorder=6)
#
#
## Plot square
#ax.add_patch(mpl.patches.PathPatch(ellipseL,fill=True,facecolor='LightGrey',edgecolor='None',zorder=3))
#ax.add_patch(mpl.patches.PathPatch(ellipseR,fill=True,facecolor='LightGrey',lw=1,zorder=3))
#
## Plot blobs (outside)
#py.scatter(xBO+2.5,yBO,s=80*0.5,c='w',lw=1,edgecolor='0.75',zorder=4)
#
## Plot the blobs inside
#py.scatter((XL+2.5)[selected],YL[selected],s=80*0.5,c=WL[selected],lw=1,zorder=4,cmap=cmap,norm=norm)
#py.clim(-5,5)
#
## Add annotation
#ax.annotate("", xy=(2, 1.75), xycoords='data',xytext=(-1.5, 1.75), textcoords='data',
#            arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                            patchA=None, patchB=None,
#                            connectionstyle="arc3,rad=-0.5",),)
#
## Plot Eulerian boundary
#py.plot(xyFEBoundary[0]+2.5,xyFEBoundary[1],'k-',zorder=2)
#
#py.axis('scaled')
#py.axis([-5.5,5.5,-4,4])
#py.axis('off')
#
## Plot interp region boundary lines
#py.plot(xySurfacePoly[0]+2.5,xySurfacePoly[1],'r--',zorder=10)
#py.plot(xyBoundaryPoly[0]+2.5,xyBoundaryPoly[1],'r--',zorder=6)
#
#ax.annotate(r'$\hat{\omega}$', xy=(-3.,1.9),fontsize=22,zorder=20)
#ax.annotate(r'$\hat{\omega}h^2$', xy=(0.,3),fontsize=20,zorder=20)
#ax.annotate(r'$\alpha_i$', xy=(2.2,1.9),fontsize=22,zorder=20)
#

#py.savefig('./interpolation_StructuredGrid2Blobs.pdf')

# ----------------------------------------------------------

# ----------------------------------------------------------
# Plot mesh
#
#fig = py.figure(991)
#ax  = fig.add_subplot(111)
#
## Determine the cells inside
#outside = boundary.contains_points(np.vstack((xFEI,yFEI)).T)
#xFEO   = xFEI[~outside]
#yFEO   = yFEI[~outside]
#wFEO   = wFEI[~outside]
#
#
#py.plot(xFEO,yFEO,'r.',zorder=3,label=r'FE boundary nodes')
##py.legend(loc=2,numpoints=1)#r'$\mathbf{x} \in \partial \Omega_E$'
#
## Plot mesh
#py.triplot(xFEI,yFEI,'-',color='silver',lw=0.5,zorder=1)
#
## Plot square
#ax.add_patch(mpl.patches.PathPatch(ellipse,fill=True,lw=0,facecolor='LightGrey',edgecolor='None',zorder=2))
#
#
#
#vy = np.random.randn(1,xFEO.shape[0])
#vx = np.ones(xFEO.shape)
#vy *= 0.25
##py.quiver(xFEO,yFEO,vx-np.abs(wFEO),wFEO*0.25,lw=(0.01,),headaxislength=2)
#
#
## Plot interp region boundary lines
##py.plot(xySurfacePoly[0],xySurfacePoly[1],'r--',zorder=6)
##py.plot(xyBoundaryPoly[0],xyBoundaryPoly[1],'r--',zorder=6)
#py.plot(xyFEBoundary[0],xyFEBoundary[1],'k-',lw=0.5,zorder=2)
## Plot blobs (outside)
#py.scatter(xBO,yBO,s=80*0.5,c='w',lw=1,zorder=4,label=r'')
#
## Plot the blobs inside
##py.scatter((XL)[selected],YL[selected],s=80*0.5,c=WL[selected],lw=1,zorder=4,cmap=cmap,norm=norm)
#py.scatter((XL)[selected],YL[selected],s=80*0.5,c='w',lw=1,zorder=4)
#py.clim(-5,5)
#
#
#
#py.axis('scaled')
#py.axis([-3,3,-3,3])
#py.axis('off')
#
##py.savefig('eulerianDirichletBC.pdf')

# ----------------------------------------------------------
'''
# ----------------------------------------------------------
# Plot the local orientation
import pylab as py
fig = py.figure()
ax  = fig.add_subplot(111)

py.plot(rotate(xyEllipse,theta=-30.)[0],rotate(xyEllipse,theta=-30.)[1],color='0.5',lw=1.5)
# Draw arrow
ax.arrow(0.,0.,0,1.2,head_width=0.05,head_length=0.05,fc='k',ec='k',zorder=10,lw='0.5')
ax.arrow(0.,0.,1.2,0.,head_width=0.05,head_length=0.05,fc='k',ec='k',zorder=10,lw='0.5')
py.plot(0.,0.,'ko')
ax.annotate(r"$(x'_o,y'_o)$", xy=(0.1, 0.075),fontsize=12,zorder=25)
ax.annotate(r"$x'$", xy=(-0.2, 1.1),fontsize=15,zorder=20)
ax.annotate(r"$y'$", xy=(1.1, -0.2),fontsize=15,zorder=20)

py.axis('scaled')
#py.axis([-1.5,1.5,-1.5,1.5])
py.axis([-2,2,-2,2])
py.axis('off')

#py.plot([-2,2,2,-2,-2],[-2,-2,2,2,-2],'k--')

py.savefig('./localOrientation2.pdf')

ax.arrow(-0.7,0.5,0,0.15,head_width=0.025,head_length=0.05,fc='k',ec='k',zorder=10)
ax.arrow(0.7,1.5,0,-1+0.05,head_width=0.025,head_length=0.05,fc='k',ec='k',zorder=3)

'''
# ----------------------------------------------------------


# ----------------------------------------------------------
# Plot the local orientation

#fig = py.figure()
#ax  = fig.add_subplot(111)
#
#py.plot(xyEllipse[0]+2,xyEllipse[1]+2,color='0.5',lw=1.5)
#py.plot([2., 3.],[2.,2.],'k--',lw=0.5)
## Draw arrow
#ax.arrow(2.,2.,rotate([0.,1.])[0],rotate([0.,1.])[1],head_width=0.05,head_length=0.05,fc='k',ec='k',zorder=10,lw='0.5')
#ax.arrow(2.,2.,rotate([1.2,0.])[0],rotate([1.2,0.])[1],head_width=0.05,head_length=0.05,fc='k',ec='k',zorder=10,lw='0.5')
#
#ax.arrow(0.,-0.5,0.,3.5,head_width=0.1,head_length=0.1,fc='k',ec='k',zorder=10,lw='0.5')
#ax.arrow(-0.5,0.,3.5,0.,head_width=0.1,head_length=0.1,fc='k',ec='k',zorder=10,lw='0.5')
#
#ax.arrow(0.,0.,1.9,1.9,head_width=0.075,head_length=0.075,fc='k',ec='k',zorder=10)
#
#py.plot(2.,2.,'ko')
#ax.annotate(r'$(x_o,y_o)$', xy=(2.1, 1.6),fontsize=12,zorder=20)
#ax.annotate(r'$\theta_{\mathrm{loc}}$', xy=(3.1, 2.2),fontsize=12,zorder=20)
#ax.annotate(r"$y'$", xy=rotate([-0.25, 0.9])+2.,fontsize=13,zorder=20)
#ax.annotate(r"$x'$", xy=rotate([1., 0.15])+2.,fontsize=13,zorder=20)
#ax.annotate(r"$x$", xy=(2.8, -0.2),fontsize=17,zorder=20)
#ax.annotate(r"$y$", xy=(-0.2, 2.8),fontsize=17,zorder=20)
#
#py.axis('scaled')
#py.axis([-1,4,-1,4])
#py.axis('off')
#
#
#ax.annotate("", xy=(3., 2.), xycoords='data',xytext=(2.9, 2.5), textcoords='data',
#            arrowprops=dict(arrowstyle="<|-|>", color="k", shrinkA=5, shrinkB=5,
#                            patchA=None, patchB=None,
#                            connectionstyle="arc3,rad=-0.2",),)
#
##py.plot([-.5,3.5,3.5,-.5,-.5],[-.5,-.5,3.5,3.5,-.5],'k--')
#py.savefig('./globalOrientation.pdf')


# ----------------------------------------------------------


# ----------------------------------------------------------
# Interpolation
#
#
#fig = py.figure(figsize=(16,8))
#ax  = fig.add_subplot(111)
#
#
#xGrid = np.arange(0,1+0.2,0.2)
#xGrid, yGrid = np.meshgrid(xGrid,xGrid)
#nGrid = xGrid.shape
#xGrid, yGrid = rotate(np.vstack((xGrid.flatten(),yGrid.flatten()))).reshape(2,nGrid[0],nGrid[1])
#
#
## Plot the structured grid
#py.plot(xGrid,yGrid,'-',c='k',zorder=1,lw=0.5)
#py.plot(xGrid.T,yGrid.T,'-',c='k',zorder=1,lw=0.5)
#
## Points
#py.plot(rotate([0.7,0.7])[0],rotate([0.7,0.7])[1],'o',c='royalblue',zorder=1,lw=1,ms=7,)
##py.plot(rotate([0.6,0.6])[0],rotate([0.6,0.6])[1],'ro',zorder=1,lw=1,mec='r')
#py.plot(rotate([0.6,0.6])[0],rotate([0.6,0.6])[1],'r.',zorder=1,lw=1)
#py.plot(rotate([0.,0.6])[0],rotate([0.,0.6])[1],'r.',zorder=1,lw=1)
#py.plot(0.,0.,'ko',zorder=1,lw=1,ms=5)
#py.plot(rotate([0.6,0.])[0],rotate([0.6,0.])[1],'r.',zorder=1,lw=1)
#py.plot(rotate(np.array([[0.6,0.6],[0.,0.6]]))[0],rotate(np.array([[0.6,0.6],[0.,0.6]]))[1],'r--',zorder=1,lw=2)
#py.plot(rotate(np.array([[0.,0.6],[0.6,0.6]]))[0],rotate(np.array([[0.,0.6],[0.6,0.6]]))[1],'r--',zorder=1,lw=2)
#py.plot([0., 0.5],[0.,0.],'k--',lw=0.5)
#
## Draw axis
#ax.arrow(0.,0.,rotate([0.,0.4])[0],rotate([0.,0.4])[1],head_width=0.03,head_length=0.03,fc='k',ec='k',zorder=10,lw='0.5')
#ax.arrow(0.,0.,rotate([0.4,0.])[0],rotate([0.4,0.])[1],head_width=0.03,head_length=0.03,fc='k',ec='k',zorder=10,lw='0.5')
#
#
#ax.add_patch(mpl.patches.Rectangle(rotate([0.6,0.6]),0.2,0.2,ec='None',fc='LightPink',angle=30.,zorder=1))
#
#
## Texts
#ax.annotate(r'${\mathbf{x}_o}$', xy=(-0.05, -0.1),fontsize=18,zorder=20)
#ax.annotate(r'${\mathbf{x}_i}$', xy=rotate([0.72, 0.65]),fontsize=18,zorder=20)
#ax.annotate(r"$\hat{\jmath}$", xy=rotate([-0.075, 0.6]),fontsize=15,zorder=20)
#ax.annotate(r"$\hat{\jmath}+1$", xy=rotate([-0.15, 0.8]),fontsize=15,zorder=20)
#ax.annotate(r"$\hat{\imath}$", xy=rotate([0.6, -0.05]),fontsize=15,zorder=20)
#ax.annotate(r"$\hat{\imath}+1$", xy=rotate([0.8, -0.05]),fontsize=15,zorder=20)
#ax.annotate(r'$\theta_{\mathrm{loc}}$', xy=(0.32, 0.025),fontsize=18,zorder=20)
#
#
## Arrow
#ax.annotate("", xy=(0.3, 0.), xycoords='data',xytext=(0.25, 0.16), textcoords='data',
#            arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=5, shrinkB=5,
#                            patchA=None, patchB=None,
#                            connectionstyle="arc3,rad=-0.2",),)
#
#
## Arrow
#ax.annotate("", xy=(1.5, 1.15), xycoords='data',xytext=rotate([0.8, 0.8]), textcoords='data',
#            arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=5, shrinkB=5,
#                            patchA=None, patchB=None,ec='r',fc='r',
#                            connectionstyle="arc3,rad=-0.5",),)
#
## Plot number 2
#
## Box
#py.plot([1.5,2.5,2.5,1.5,1.5],[0.25,0.25,1.25,1.25,0.25],'k')
#py.plot(2.2,0.8,'.',c='royalblue',zorder=10,ms=20)
#
## Points
#py.plot([1.5,2.5,2.5,1.5,1.5],[0.25,0.25,1.25,1.25,0.25],'k.',ms=10)
#py.plot([1.5,2.2],[0.8,0.8],'k--',lw=1)
#py.plot([2.2,2.2],[0.25,0.8],'k--',lw=1)
#
## Texts
#ax.annotate(r'$\hat{x}$', xy=(0.5*(1.5+2.2), 0.85),fontsize=25,zorder=20)
#ax.annotate(r'$\hat{y}$', xy=(2.25, 0.5*(0.25+0.8)),fontsize=25,zorder=20)
#ax.annotate(r'$\omega_1$', xy=(1.5-0.1, 0.25-0.1),fontsize=30,zorder=20)
#ax.annotate(r'$\omega_2$', xy=(2.5+0.025, 0.25-0.1),fontsize=30,zorder=20)
#ax.annotate(r'$\omega_3$', xy=(2.5+0.025, 1.25+0.05),fontsize=30,zorder=20)
#ax.annotate(r'$\omega_4$', xy=(1.5-0.1, 1.25+0.05),fontsize=30,zorder=20)
#ax.annotate(r'$\Delta x$', xy=(2, -0.0),fontsize=20,zorder=25)
#ax.annotate(r'$\Delta y$', xy=(2.75, 0.5*(0.25+1.25)),fontsize=25,zorder=20)
#
## Delta arrows
#ax.annotate("", xy=(2.5+0.2, 0.25), xycoords='data',
#                xytext=(2.5+0.2, 1.25), textcoords='data',
#            arrowprops=dict(arrowstyle="<|-|>",lw=2,
#                            connectionstyle="arc3",ec='k',fc='k'),
#            )
#            
#ax.annotate("", xy=(1.5, 0.1), xycoords='data',
#                xytext=(2.5, 0.1), textcoords='data',
#            arrowprops=dict(arrowstyle="<|-|>",lw=2,
#                            connectionstyle="arc3",ec='k',fc='k'),
#            )
#            
#
#py.axis('scaled')
#py.axis([-0.75,3,-0.25,1.5])
#py.axis('off')
#
#
##py.plot([-.5,3.5,3.5,-.5,-.5],[-.5,-.5,3.5,3.5,-.5],'k--')
#py.savefig('./interpolationManual.pdf')


# ----------------------------------------------------------
'''
