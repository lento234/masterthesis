"""

Plot the interpolation algorithm
"""


# Import packages
import numpy as np
import pylab as py
import matplotlib as mpl
#from matplotlib.patches import Polygon
import scipy.interpolate as spinterp
import scipy.io as sio
#from mpl.patches import Polygon

py.ion()
py.close('all')


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
                              

# jet Colormap
colorMap = 'jet'
jetCM = py.cm.ScalarMappable(cmap=colorMap)
jetCM = jetCM.to_rgba(np.linspace(0,1,256))

# Latex Text
py.rc('text', usetex=True)
py.rc('font', family='serif')
# ----------------------------------------------------------


# ----------------------------------------------------------
# Load data

# Load the FE data
dataFE = np.load('dataFE.npz')

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
	       0.5*0.5,0.5*1.315,0.5*1.37,0.5*3.86,0.5*3.96])*2
yB = np.array([-1.5,1.2,1.3, -1.2,2,-0.5,-1.8,0.4,1,1.2,-0.2,
		.5,-2,2,-1.2,2.3,-2.3,0,0.5,2,0.8,-1.2,-1.3,
		0.5,-0.3,-0.5,1.6,1.02,-1.13,0.015,-1.19,
		0.397,-0.79,1.14,-0.59])

# Lagrangian grid
hLagr = 0.5
xLagr = np.arange(-5-hLagr*0.5,5+hLagr+hLagr*0.5,hLagr)
XLagr,YLagr = np.meshgrid(xLagr,xLagr)


# Surface polygon
xySurfacePoly = np.array([[-1,1,1,-1,-1],
		      	  [-1,-1,1,1,-1]])*1.2

xyBoundaryPoly = np.array([[-1,1,1,-1,-1],
		           [-1,-1,1,1,-1]])*2.8

# Define the FE mesh boundary
xyFEBoundary = np.array([[-1,1,1,-1,-1],
	                 [-1,-1,1,1,-1]])*3


# ----------------------------------------------------------

# ----------------------------------------------------------
# Make patches

# Square
square = mpl.path.Path(rotate(xySquare).T,closed=True)

squareL = mpl.path.Path(rotate(xySquare).T + np.array([-5,0]),closed=True)
squareR = mpl.path.Path(rotate(xySquare).T + np.array([5,0]),closed=True)
squareN = mpl.path.Path(np.array([[-1, 1, 1, -1, -1],
		      [-0.5, -0.5, 0.5, 0.5, -0.5]]).T,closed=True)


# Square Patch
#squarePatch = mpl.patches.PathPatch(square,facecolor='orange',lw=2)
#squarePatch = mpl.patches.PathPatch(square,fill=True,hatch='/',lw=2,rasterized=False,facecolor='w')

# Surface Polygon
surface = mpl.path.Path(rotate(xySurfacePoly).T,closed=True)
#surfacePatch = mpl.patches.PathPatch(surface,fill=False,lw=2,rasterized=False)

# Boundary Polygon
boundary = mpl.path.Path(rotate(xyBoundaryPoly).T,closed=True)
#squarePatch = mpl.patches.PathPatch(square,fill=False,hatch='//',lw=2,rasterized=False)

# FE Boundary
FEboundary = mpl.path.Path(rotate(xyFEBoundary).T,closed=True)
#squarePatch = mpl.patches.PathPatch(square,fill=False,hatch='//',lw=2,rasterized=False)

# Interpolation region
interpRegion = mpl.path.Path(vertices=np.concatenate([surface.vertices[::-1], boundary.vertices]),
		             codes=np.concatenate([surface.codes, boundary.codes]))

# ----------------------------------------------------------


# ----------------------------------------------------------
# Calculations

# Determine the cells inside
inside = FEboundary.contains_points(np.vstack((xFE,yFE)).T)
xFEI = xFE[inside]
yFEI = yFE[inside]
wFEI = wFE[inside]

# Determine the cells inside
inside = boundary.contains_points(np.vstack((xB,yB)).T)
xBO = xB[~inside]
yBO = yB[~inside]

# Determine the lagrangian nodes inside the interp region
insideS = surface.contains_points(np.vstack((XLagr.flatten(),YLagr.flatten())).T)
insideB = boundary.contains_points(np.vstack((XLagr.flatten(),YLagr.flatten())).T)
insideBoth = np.where(~insideS & insideB)[0]
XL = XLagr.flatten()[insideBoth]
YL = YLagr.flatten()[insideBoth]

# Generate structured grid
xStructured = np.arange(xyFEBoundary[0,0],xyFEBoundary[0,1]+hLagr*0.5,hLagr*0.5)
yStructured = np.arange(xyFEBoundary[1,1],xyFEBoundary[1,2]+hLagr*0.5,hLagr*0.5)
nStructured = xStructured.shape[0]
xStructured, yStructured = np.meshgrid(xStructured,yStructured)
xStructured, yStructured = rotate(np.vstack((xStructured.flatten(),yStructured.flatten()))).reshape(2,nStructured,nStructured)

# Interpolate data
wStructured = spinterp.griddata(np.vstack((xFEI,yFEI)).T, wFEI, 
                                np.vstack((xStructured.flatten(),
                                           yStructured.flatten())).T,
                                method='nearest').reshape(nStructured,nStructured)


# Interpolate from Structured grid to blobs
WL = spinterp.griddata(np.vstack((xStructured.flatten(),yStructured.flatten())).T, wStructured.flatten(),
                       np.vstack((XL,YL)).T,method='linear')
                       
selected = np.abs(WL)>0.5

# ----------------------------------------------------------


# ----------------------------------------------------------

# Plot the Lagrangian Domain

#fig = py.figure(1)
#ax  = fig.add_subplot(111)
#ax.set_rasterized(False)
#
## Plot Square
#ax.add_patch(mpl.patches.PathPatch(square,fill=True,hatch='///',lw=2,facecolor='w'))
#
## Plot blobs
#py.scatter(xB,yB,s=100,c='w',lw=1)
#
#py.axis('scaled')
#py.axis([-5,5,-5,5])
#py.axis('off')
#py.show()
#
#py.savefig('./lagrangian.pdf')

# ----------------------------------------------------------


# ----------------------------------------------------------
# Plot the full Eulerian domin

#fig = py.figure(2)
#ax  = fig.add_subplot(111)
#
#clim = 5
#cmap,norm = mpl.colors.from_levels_and_colors(colorMapDaen['levels']*clim,colorMapDaen['colors'],extend='both')
#
## Plot vorticity
##py.tripcolor(xFE,yFE,wFE,contourLevelsFE,rasterized=True,zorder=1)
#py.tripcolor(xFE,yFE,wFE,norm=norm,cmap=cmap,zorder=1,rasterized=True)
## Plot mesh
#py.triplot(xFE,yFE,cellsFE,lw=0.5,color='0.75',zorder=2)
## Plot Square
##ax.add_patch(mpl.patches.PathPatch(square,fill=True,hatch='///',lw=2,facecolor='w',zorder=3,rasterized=False))
#ax.add_patch(mpl.patches.PathPatch(square,fill=True,lw=2,facecolor='LightGrey',zorder=3))
#
#
#py.clim(-clim,clim)
#py.axis('scaled')
#py.axis([-5,5,-5,5])
#py.axis('off')
#py.show()
#
#py.savefig('./fullEulerian.pdf')

# ----------------------------------------------------------

# ----------------------------------------------------------
# Plot Hybrid Eulerian
#
#fig = py.figure(3)
#ax  = fig.add_subplot(111)
#
#clim = 5
#cmap,norm = mpl.colors.from_levels_and_colors(colorMapDaen['levels']*clim,colorMapDaen['colors'],extend='both')
#
## Plot Vorticity
##py.tripcolor(xFEI,yFEI,wFEI,contourLevelsFE,rasterized=True,zorder=1)
#py.tripcolor(xFEI,yFEI,wFEI,norm=norm,cmap=cmap,rasterized=True,zorder=1)
## Plot mesh
#py.triplot(xFEI,yFEI,color='0.75',lw=0.5,zorder=2)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(square,fill=True,lw=1,facecolor='LightGrey',zorder=3))
#
#py.clim(-clim,clim)
#py.axis('scaled')
#py.axis([-5,5,-5,5])
#py.axis('off')
#
#py.savefig('./eulerian.pdf')

# ----------------------------------------------------------

# ----------------------------------------------------------
# Plot hybrid formation
#
#fig = py.figure(4)
#ax  = fig.add_subplot(111)
#
## Plot Eulerian grid
#py.triplot(xFEI,yFEI,color='0.75',lw=0.5,zorder=1)
## Plot Eulerian boundary
#py.plot(rotate(xyFEBoundary)[0],rotate(xyFEBoundary)[1],'k-',lw=0.5,zorder=2)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(square,fill=True,lw=1,facecolor='LightGrey',zorder=3))
## Plot blobs
#py.scatter(xB+0.075,yB,s=100,c='w',lw=1,zorder=4)
#
#py.axis('scaled')
##py.axis([-5,5,-5,5])
#py.axis([-4.5,5.2,-4.5,4.5])
#py.axis('off')
##py.plot([-4.5,5.2,5.2,-4.5,-4.5],[-4.5,-4.5,4.5,4.5,-4.5],'k--')
#py.plot(rotate(xySurfacePoly)[0],rotate(xySurfacePoly)[1],'k--',zorder=6)
#py.plot(rotate(xySquare*1.075)[0],rotate(xySquare*1.075)[1],'0.5',lw=2,zorder=4)
#
#
### Add text
##ax.annotate(r'$\partial \Omega_{E}$', xy=(-0.60, 3.75),fontsize=20,zorder=20)
#ax.annotate(r'$\partial \Omega_{E}$', xy=(-1.0, 2.8), xycoords='data',fontsize=20,xytext=(-0.7, 3.75), textcoords='data',
#           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                           patchA=None, patchB=None,
#                           connectionstyle="arc3,rad=0.2",),)
##ax.annotate(r'$\partial \Omega_{\mathrm{wall}}$', xy=(-2.5, -0.75),fontsize=20,zorder=20)
#ax.annotate(r'$\partial \Omega_{\mathrm{P}}$', xy=(-1.11, 0.7), xycoords='data',fontsize=20,xytext=(-1, 1.5), textcoords='data',
#           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                           patchA=None, patchB=None,
#                           connectionstyle="arc3,rad=0.2",),)
#                           
#ax.annotate(r'$\partial \Omega_{\mathrm{wall}}$', xy=(-1.11, 0.05), xycoords='data',fontsize=20,xytext=(-2.5, -1.25), textcoords='data',
#           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                           patchA=None, patchB=None,
#                           connectionstyle="arc3,rad=-0.2",),)                           
##ax.annotate(r'$\partial \Omega_{P}$', xy=(-1., 1.5),fontsize=20,zorder=20)
##ax.annotate(r'$\Omega_{B}$', xy=(-0.7, 1.25),fontsize=25,zorder=20)
##ax.annotate(r'$\gamma(s)$', xy=(0, 0.575),fontsize=20,zorder=20)
##ax.annotate(r'$\omega_i$', xy=(0.2, 1.7),fontsize=20,zorder=20)
##ax.annotate(r'\textsf{body}$', xy=(-0.4, 0.4),fontsize=15,zorder=20,backgroundcolor='w')


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

fig = py.figure(41)
ax  = fig.add_subplot(111)

# Plot Eulerian grid
py.triplot(rotate(np.vstack((xFEI,yFEI)),-30)[0],
           rotate(np.vstack((xFEI,yFEI)),-30)[1]*0.5,color='0.75',lw=0.5,zorder=1)
# Plot Eulerian boundary
py.plot(xyFEBoundary[0],xyFEBoundary[1]*0.5,'k-',lw=0.5,zorder=2)
# Plot square
ax.add_patch(mpl.patches.PathPatch(squareN,fill=True,hatch='//',lw=1,facecolor='w',zorder=3))
# Plot blobs
xyBlobsTemp = rotate(np.vstack((xB,yB)),-60-30)*0.5
xyBlobsTemp[1] *= 0.5
xyBlobsTemp[1] -= 0.5
xyBlobsTemp = xyBlobsTemp[:,np.abs(xyBlobsTemp[1])>0.7]
xyBlobsTemp = xyBlobsTemp[:,xyBlobsTemp[0]>-0.475]
#xyBlobsTemp[1] -= 0.75
py.scatter(xyBlobsTemp[0],
           -xyBlobsTemp[1],s=600,c='w',lw=1,zorder=4)

# Plot lines
interpRegTemp = mpl.path.Path(np.array([[-1, 1, 1, -1, -1],
                                        [0.7,0.7,1.3,1.3,0.7]]).T,closed=True)


#py.plot([-1,1,1,-1,-1],[0.7,0.7,1.3,1.3,0.7],'k--',lw=1,zorder=5)
ax.add_patch(mpl.patches.PathPatch(interpRegTemp,fill=True,lw=0,facecolor='LightPink', alpha=0.7,zorder=5))
py.plot([-1, 1, 1, -1, -1],[0.7,0.7,1.3,1.3,0.7],'r--',zorder=6)

py.plot([-1,1],[0.52,0.52],'k-',lw=5,zorder=5)

# Draw arrow
ax.annotate("",(-0.7,0.5),(-0.7,0.7),arrowprops={'arrowstyle':'<|-|>','fc':'r','ec':'r'},zorder=20)
ax.annotate("",(-0.7,1.3),(-0.7,1.5),arrowprops={'arrowstyle':'<|-|>','fc':'r','ec':'r'},zorder=20)

# Add text
ax.annotate(r'$\Omega_{P}$', xy=(0.3, 0.55),fontsize=25,zorder=20)
ax.annotate(r'$d_{\mathrm{surf}}\cdot{h}$', xy=(-0.6, 0.57),fontsize=15,zorder=20)
ax.annotate(r'$d_{\mathrm{bdry}}\cdot{h}$', xy=(-0.6, 1.37),fontsize=15,zorder=20)
ax.annotate(r'\textsf{body}$', xy=(-0.4, 0.4),fontsize=15,zorder=20,backgroundcolor='w')

py.axis('scaled')
py.axis([-0.9,0.9,0.35,2])
py.axis('off')

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

#fig = py.figure(6)
#ax  = fig.add_subplot(111)
#
## Plot Eulerian grid
#py.triplot(xFEI,yFEI,color='0.5',lw=0.5,zorder=1)
## Plot Eulerian boundary
#py.plot(rotate(xyFEBoundary)[0],rotate(xyFEBoundary)[1],'k-',lw=0.5,zorder=2)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(square,fill=True,lw=1,facecolor='LightGrey',zorder=3))
## Plot blobs
#py.scatter(xB,yB,s=100,c='w',lw=1,zorder=4)
## Plot interp-region
#col = mpl.collections.PathCollection([interpRegion], lw=0, color='None',facecolor='LightPink', alpha=0.7, zorder=5)
#ax.add_collection(col)
## Plot interp region boundary lines
#py.plot(rotate(xySurfacePoly)[0],rotate(xySurfacePoly)[1],'r--',zorder=6)
#py.plot(rotate(xyBoundaryPoly)[0],rotate(xyBoundaryPoly)[1],'r--',zorder=6)
#
#py.axis('scaled')
#py.axis([-5,5,-5,5])
#py.axis('off')
#
## Draw arrow
### Add text
##ax.annotate(r'$\partial \Omega_{E}$', xy=(-0.60, 3.75),fontsize=20,zorder=20)
#ax.annotate(r'$\partial \Omega_{int}$', xy=(-0.93, 2.5), xycoords='data',fontsize=20,xytext=(-0.8, 3.75), textcoords='data',
#           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                           patchA=None, patchB=None,
#                           connectionstyle="arc3,rad=0.2",),zorder=20)
##ax.annotate(r'$\partial \Omega_{\mathrm{wall}}$', xy=(-2.5, -0.75),fontsize=20,zorder=20)
#ax.annotate(r'$\partial \Omega_{\mathrm{P}}$', xy=(-1.11, 0.7), xycoords='data',fontsize=20,xytext=(-1, 1.5), textcoords='data',
#           arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                           patchA=None, patchB=None,
#                           connectionstyle="arc3,rad=0.2",),zorder=20)
#          
#
#
#
##py.plot([-5,5,5,-5,-5],[-5,-5,5,5,-5],'k--')
#
#py.savefig('./interpRegion.pdf')

# ----------------------------------------------------------


# ----------------------------------------------------------
# Plot removing particles in the outer polygon 

#fig = py.figure(7)
#ax  = fig.add_subplot(111)
#
## Plot Eulerian grid
#py.triplot(xFEI,yFEI,color='0.5',lw=0.5,zorder=1)
## Plot Eulerian boundary
#py.plot(rotate(xyFEBoundary)[0],rotate(xyFEBoundary)[1],'k-',lw=0.5,zorder=2)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(square,fill=True,lw=1,facecolor='LightGrey',zorder=3))
## Plot blobs
#py.scatter(xBO,yBO,s=100,c='w',lw=1,zorder=4)
## Plot interp-region
##col = mpl.collections.PathCollection([interpRegion], facecolor='LightPink', alpha=0.8, zorder=10, lw=0)
##ax.add_patch(mpl.patches.PathPatch(interpRegion,fill=True,lw=0,facecolor='Yellow',alpha=0.8,zorder=10)) #Lime, LawnGreen
##ax.add_collection(col)
##ax.set_clip_path(mpl.patches.Polygon(xySurfacePoly.T))
#
##py.plot(rotate(xySurfacePoly)[0],rotate(xySurfacePoly)[1],'r--',zorder=11)
#py.plot(rotate(xyBoundaryPoly)[0],rotate(xyBoundaryPoly)[1],'r--',zorder=4)
#
#py.axis('scaled')
#py.axis([-5,5,-5,5])
#py.axis('off')
##py.plot([-5,5,5,-5,-5],[-5,-5,5,5,-5],'k--')
#
#py.savefig('./particleRemoved.pdf')

# ----------------------------------------------------------


# ----------------------------------------------------------
# Generating particles inside the interpolation region, onto the Lagrangian grid


#fig = py.figure(8)
#ax  = fig.add_subplot(111)
#
## Plot the lagrangian grid
#py.plot(XLagr,YLagr,alpha=0.5,color='Lime',zorder=1)
#py.plot(XLagr.T,YLagr.T,alpha=0.5,color='Lime',zorder=1)
## Plot Eulerian boundary
#py.plot(rotate(xyFEBoundary)[0],rotate(xyFEBoundary)[1],'k-',lw=0.5,zorder=2)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(square,fill=True,lw=1,facecolor='LightGrey',zorder=3))
## Plot blobs (outside)
#py.scatter(xBO,yBO,s=100,c='w',lw=1,zorder=4,edgecolor='0.75')
## Plot the blobs inside
#py.scatter(XL,YL,s=100,c='0.9',lw=1,zorder=4)
## Plot interpolation boundary lines
#py.plot(rotate(xySurfacePoly)[0],rotate(xySurfacePoly)[1],'r--',zorder=5)
#py.plot(rotate(xyBoundaryPoly)[0],rotate(xyBoundaryPoly)[1],'r--',zorder=5)
#
#py.axis('scaled')
#py.axis([-5,5,-5,5])
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
#py.triplot(xFEI-5,yFEI,'-',color='0.5',lw=0.5,zorder=1)
## Plot square
#ax.add_patch(mpl.patches.PathPatch(squareL,fill=True,lw=0,facecolor='LightGrey',edgecolor='None',zorder=2))
#ax.add_patch(mpl.patches.PathPatch(squareR,fill=True,lw=0,facecolor='LightGrey',edgecolor='None',zorder=2))
## Plot the structured grid
#py.plot(xStructured+5,yStructured,'-',color='0.5',zorder=1,lw=0.5)
#py.plot(xStructured.T+5,yStructured.T,'-',color='0.5',zorder=1,lw=0.5)
#
## Add annotation
#ax.annotate("", xy=(4, 3.5), xycoords='data',xytext=(-3, 3.5), textcoords='data',
#            arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                            patchA=None, patchB=None,
#                            connectionstyle="arc3,rad=-0.5",),)
#
#py.axis('scaled')
#py.axis([-10,10,-5,6])
#py.axis('off')
##py.plot([-10,10,10,-10,-10],[-5,-5,6,6,-5],'k--')
#
#py.savefig('./interpolation_FE2StructuredGrid.pdf')

# ----------------------------------------------------------


# ----------------------------------------------------------
# Interpolate the mesh data from FE to structured grid
#
#fig = py.figure(10)
#ax  = fig.add_subplot(111)
#
## Plot Vorticity on FE
##py.tripcolor(xFEI-5,yFEI,wFEI,contourLevelsFE,rasterized=True,zorder=1,cmap=colorMapHC)
#py.tripcolor(xFEI-5,yFEI,wFEI,norm=norm,cmap=cmap,rasterized=True,zorder=1)
#py.clim(-5,5)
#
## Plot vorticity on structureed
##py.pcolormesh(xStructured+5,yStructured,wStructured,rasterized=True,zorder=1,cmap=colorMapHC)
#py.pcolormesh(xStructured+5,yStructured,wStructured,rasterized=True,zorder=1,cmap=cmap,norm=norm)
#py.clim(-5,5)  
#
## Plot mesh
#py.triplot(xFEI-5,yFEI,'-',color='0.75',lw=0.5,zorder=2)
#
## Plot the structured grid
#py.plot(xStructured+5,yStructured,'-',color='0.75',zorder=2,lw=0.5)
#py.plot(xStructured.T+5,yStructured.T,'-',color='0.75',zorder=2,lw=0.5)
#
## Plot square
#ax.add_patch(mpl.patches.PathPatch(squareL,fill=True,facecolor='LightGrey',edgecolor='None',zorder=3))
#ax.add_patch(mpl.patches.PathPatch(squareR,fill=True,facecolor='LightGrey',edgecolor='None',zorder=3))
#
## Add annotation
#ax.annotate('', xy=(4, 3.5), xycoords='data',xytext=(-3, 3.5), textcoords='data',
#            arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                            patchA=None, patchB=None,
#                            connectionstyle="arc3,rad=-0.5",),)
#
#py.axis('scaled')
#py.axis([-10,10,-5,5])
#py.axis('off')
#
#
#py.savefig('./interpolation_FE2StructuredGrid_withData.pdf')

# ----------------------------------------------------------


# ----------------------------------------------------------
# Interpolate the mesh data from structured grid to blobs

#
## Plot hybrid eulerian
#fig = py.figure(11)
#ax  = fig.add_subplot(111)
#
## Plot vorticity on structured
##py.pcolormesh(xStructured-5,yStructured,wStructured,rasterized=True,zorder=1,cmap=colorMapHC)
#py.pcolormesh(xStructured-5,yStructured,wStructured,rasterized=True,zorder=1,cmap=cmap,norm=norm)
#py.clim(-5,5)         
#
## Plot the structured grid
#py.plot(xStructured-5,yStructured,'-',color='0.75',zorder=2,lw=0.5)
#py.plot(xStructured.T-5,yStructured.T,'-',color='0.75',zorder=2,lw=0.5)
#
## Plot square
#ax.add_patch(mpl.patches.PathPatch(squareL,fill=True,facecolor='LightGrey',edgecolor='None',zorder=3))
##ax.add_patch(mpl.patches.PathPatch(squareR,fill=True,facecolor='LightGrey',edgecolor='None',zorder=3))
#ax.add_patch(mpl.patches.PathPatch(squareR,fill=True,facecolor='LightGrey',lw=1,zorder=3))
#
## Plot blobs (outside)
#py.scatter(xBO+5,yBO,s=100*0.5,c='w',lw=1,edgecolor='0.75',zorder=4)
#
## Plot the blobs inside
##py.scatter((XL+5)[selected],YL[selected],s=100*0.5,c=WL[selected],lw=1,zorder=4,cmap=colorMapHC)
#py.scatter((XL+5)[selected],YL[selected],s=100*0.5,c=WL[selected],lw=1,zorder=4,cmap=cmap,norm=norm)
#py.clim(-5,5)
#
## Add annotation
#ax.annotate("", xy=(4, 3.5), xycoords='data',xytext=(-3, 3.5), textcoords='data',
#            arrowprops=dict(arrowstyle="->", color="k", shrinkA=5, shrinkB=5,
#                            patchA=None, patchB=None,
#                            connectionstyle="arc3,rad=-0.5",),)
#
## Plot Eulerian boundary
#py.plot(rotate(xyFEBoundary)[0]+5,rotate(xyFEBoundary)[1],'k-',zorder=2)
#
#py.axis('scaled')
#py.axis([-10,10,-5,5])
#py.axis('off')
#
#py.savefig('./interpolation_StructuredGrid2Blobs.pdf')

# ----------------------------------------------------------
