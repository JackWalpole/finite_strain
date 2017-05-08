#!/usr/bin/env python

"""Visualise finite strian ellipsoids"""

import numpy as np
from copy import copy
from mayavi import mlab
import finite_strain as fs

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

# initial undeformed finite strain tensor
F = fs.Fij()


# mesh it up
x, y, z = elliptic_mesh(F)

s = mlab.mesh( x, y, z)
actor = s.actor
actor.property.opacity = 0.5
actor.property.color = tuple(np.random.rand(3))
actor.mapper.scalar_visibility = False
actor.property.backface_culling = True

L_file = 'files/_L_3555_25_25'
P_file = 'files/_P_3555_25_25'

P = fs.Path(L_file,P_file)
listF = [ copy(fs.update_strain(F,L)) for L in P.listLij ]    
# data  = [ copy(deform_sphere(Fij)) for Fij in listFij ]

for F in listF:
    x, y, z = elliptic_mesh(F)
    s.mlab_source.x = x
    s.mlab_source.y = y
    s.mlab_source.z = z
