#!/usr/bin/env python

from __future__ import print_function

"""Visualise finite strian ellipsoids"""

import numpy as np
from mayavi import mlab
import finite_strain as fs
import glob
# needed for plotting the continents
from mayavi.sources.builtin_surface import BuiltinSurface

# mlab.figure(1, size=(400, 400), bgcolor=(0, 0, 0))
# mlab.clf()

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

def sph2cart(lat,lon,r):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    z = r * np.sin(lat)
    y = r * np.cos(lat) * np.sin(lon)
    x = r * np.cos(lat) * np.cos(lon)
    return x,y,z

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
                    opacity=.99)

core.actor.property.backface_culling = True


# where are all the L files
filehome = '/Users/glyjw/Comitac/files/TX2008_cc/_L_files/*'
for Lfile in glob.glob(filehome):
    lat, lon = tuple([float(x) for x in Lfile.split('_')[-2:]])
    
    # only show if on 15 by 15 grid
    print(lat,lon)
    if (lat%15==0 and lon%15==0):
        print('process')
        pass
    else:
        print('skip')
        continue

    P = fs.Path(Lfile)
    F = P.accumulate_strain()

    px, py, pz  = sph2cart(lat, lon, .7)
    
    # Finite Strain Ellipse
    scale = 0.05
    x, y, z = tuple([scale * x for x in elliptic_mesh(F)])
    s = mlab.mesh( x + px, y + py, z +pz)

    actor = s.actor
    actor.property.opacity = 0.5
    actor.property.color = (0,1,0)
    actor.mapper.scalar_visibility = False
    actor.property.backface_culling = True
#
#
# # Position
# pos = np.asarray([ (pos.x, pos.y, pos.z) for pos in P.listTrajectory ]) / Er
# posx = pos[:,0]
# posy = pos[:,1]
# posz = pos[:,2]
#
#
#
#
# # Principal Axes
# a, b = F.Principal()
# b = scale * b
# v = mlab.quiver3d( np.ones(3)*pos[0],
#                    np.ones(3)*posy[0],
#                    np.ones(3)*posz[0],
#                    b[:,0], b[:,1], b[:,2], scale_factor = scale*10 )





