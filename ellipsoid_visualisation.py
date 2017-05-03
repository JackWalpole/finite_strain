#!/usr/bin/env python

"""Visualise finite strian ellipsoids"""

import numpy as np

from mayavi.api import Engine
from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Surface
from mayavi import mlab


def gen_ellipsoid(engine,position,shape,orientation):
    """given the existence of a scene generate ellipsoid"""
    
    source = ParametricSurface()
    source.function = 'ellipsoid'
    engine.add_source(source)
    
    surface = Surface()
    source.add_module(surface)
    
    actor = surface.actor
    actor.property.opacity = 0.5
    actor.property.color = tuple(np.random.rand(3))
    actor.mapper.scalar_visibility = False
    actor.property.backface_culling = True
    actor.actor.orientation = orientation
    actor.actor.origin = np.zeros(3)
    actor.actor.position = position
    actor.actor.scale = shape
    
    return surface
    

# if __main__ == name:
engine = Engine()
engine.start()
scene = engine.new_scene()
scene.scene.disable_render = True

surfaces = []

for ii in range(10):
    surfaces.append(gen_ellipsoid(engine,np.random.rand(3),np.random.rand(3),np.random.rand(3)*360))
    
scene.scene.disable_render = False

mlab.show()