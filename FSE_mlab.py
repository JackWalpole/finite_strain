#!/usr/bin/env python

"""Visualise finite strian ellipsoids"""

import numpy as np
from copy import copy
from mayavi import mlab
import finite_strain as fs
# needed for plotting the continents
from mayavi.sources.builtin_surface import BuiltinSurface

mlab.figure(1, size=(400, 400), bgcolor=(0, 0, 0))
mlab.clf()

def elliptic_mesh(F):
    """Generate a mesh describing the finite strain ellipse"""
    # draw sphere
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    # put into array
    XYZ = np.vstack([x.flatten(),y.flatten(),z.flatten()])
    # deform according to F
    XYZ = np.dot(F.Fij,XYZ)
    # put back into meshgrid
    a, b, c = np.vsplit(XYZ,3)
    x = np.reshape(a,x.shape)
    y = np.reshape(b,y.shape)
    z = np.reshape(c,z.shape)
    return x, y, z

#### INITIAL FRAME ####

# Data
# initial undeformed finite strain tensor
F = fs.Fij()
# L_file = 'files/_L_3555_25_25'
# P_file = 'files/_P_3555_25_25'

L_file = '/Users/glyjw/Comitac/files/TX2008_cc/_L_files/' + '_L_3555_0_0'
P_file = '/Users/glyjw/Comitac/files/TX2008_cc/_P_files/' + '_P_3555_0_0'

P = fs.Path(L_file,P_file)
listF = [ copy(fs.update_strain(F,L)) for L in P.listLij ]

# Earth's core
Er = 6371.
r2891 = (Er - 2891) / Er
continents_src = BuiltinSurface(source='earth', name='Continents')
# The on_ratio of the Earth source controls the level of detail of the 
# continents outline.
continents_src.data_source.on_ratio = 2
continents_src.data_source.radius = r2891
continents = mlab.pipeline.surface(continents_src, color=(0, 0, 0))

core = mlab.points3d(0, 0, 0, scale_mode='none',
                    scale_factor=r2891*2,
                    color=(0.5, 0.5, 0.5),
                    resolution=50,
                    opacity=.9)

core.actor.property.backface_culling = True

# Position
pos = np.asarray([ (pos.x, pos.y, pos.z) for pos in P.listTrajectory ]) / Er
posx = pos[:,0]
posy = pos[:,1]
posz = pos[:,2]

# Finite Strain Ellipse
scale = 0.05
x, y, z = tuple([scale * x for x in elliptic_mesh(F)])
s = mlab.mesh( x + posx[0], y + posy[0], z + posz[0])
actor = s.actor
actor.property.opacity = 0.5
actor.property.color = (0,1,0)
actor.mapper.scalar_visibility = False
actor.property.backface_culling = True

# Principal Axes
a, b = F.Principal()
b = scale * b
v = mlab.quiver3d( np.ones(3)*pos[0],
                   np.ones(3)*posy[0],
                   np.ones(3)*posz[0], 
                   b[:,0], b[:,1], b[:,2], scale_factor = scale*10 )

# Plot Path Line 
# -- do once else need to reset source which is very inefficient:
# e.g. path.mlab_source.reset(x = posx[0:ii], y=posy[0:ii], z=posz[0:ii])
pathline = mlab.plot3d( posx, posy, posz, tube_radius = None )

mlab.view(azimuth=-30, distance=2.)

#### SUBSEQUENT FRAMES ####

for ii in np.arange(0,len(listF),50):
    F = listF[ii]
    T = P.listTrajectory[ii]
    x, y, z = tuple([scale * x for x in elliptic_mesh(F)])
    s.mlab_source.x = x + T.x / Er
    s.mlab_source.y = y + T.y / Er
    s.mlab_source.z = z + T.z / Er
    a, b = F.Principal()
    b = scale * b
    v.mlab_source.x = T.x / Er * np.ones(3)
    v.mlab_source.y = T.y / Er * np.ones(3)
    v.mlab_source.z = T.z / Er * np.ones(3)
    v.mlab_source.u = b[0,:]
    v.mlab_source.v = b[1,:]
    v.mlab_source.w = b[2,:]

