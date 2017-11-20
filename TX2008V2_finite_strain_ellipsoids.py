#!/usr/bin/env python

"""calculate the finite strain ellipsoid at each location"""

import numpy as np
import finite_strain as fs
import glob


def rotate_to_local(lat,lon,V):
    """rotate column vectors V in the global frame 
    (axes 1,2,3 pointing to lat,lon (0,0); (0,90); and (90,0) respectively)
    to a local frame with 1 pointing Up, 2 pointing E, and 3 pointing N"""
    # specifiy Up, N, and E directions at point (lat,lon) in the global frame
    def vunit(a):
       # Return a unit vector in the direction of a
       return a/np.linalg.norm(a)
    def sph2cart(lat,lon):
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        z = np.sin(lat)
        y = np.cos(lat) * np.sin(lon)
        x = np.cos(lat) * np.cos(lon)
        return np.array([x,y,z])
    def vangle(a,b):
       # Return the angle between two vectors
       return np.arccos(round(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)),15))
    def vrejection(a,b):
       # The vector rejection of a on b (the bit of a perpendicular to b)
       # Equivalent to projection onto the plane normal to b
       return a - (np.linalg.norm(a)*np.cos(vangle(a,b))) * vunit(b)
    # radial vector pointing outwards at surface
    Up = vunit( sph2cart(lat,lon))
    # vector pointing towards north and orthogonal to Up
    N = vunit( vrejection(np.array([0,0,1]),Up))
    # vector orthogonal to both Up and N in a right-handed co-ordinate system.
    E = vunit( np.cross(N,Up))
    # put together in columns of matrix
    local = np.vstack([Up,E,N]).T
    # change base
    return np.dot( np.linalg.inv(local), V)

# where are all the L files
filehome = '/Users/glyjw/Comitac/files/TX2008_cc/_L_files/*'

print '{:>7s}{:>7s}{:>9s}{:>9s}{:>9s}{:>9s}{:>9s}{:>9s}{:>9s}{:>9s}{:>9s}{:>9s}{:>9s}{:>9s}'\
       .format('lat','lon','lam1','lam2','lam3','V1_Up','V1_E','V1_N','V2_Up',\
       'V2_E','V2_N','V3_Up','V3_E','V3_N')

for Lfile in glob.glob(filehome):
    lat, lon = tuple([float(x) for x in Lfile.split('_')[-2:]])
    P = fs.Path(Lfile)
    F = P.accumulate_strain()
    # get eigenvalues and eigenvectors
    w,v = F.Principal()
    # rotate eigenvectors into local frame
    v = rotate_to_local(lat,lon,v)
    # output result
    print '{:7.1f}{:7.1f}{:9.5g}{:9.5f}{:9.5f}{:9.5f}{:9.5f}{:9.5f}{:9.5f}{:9.5f}{:9.5f}{:9.5f}{:9.5f}{:9.5f}'\
    .format(lat, lon, w[0], w[1], w[2], v[0,0], v[1,0], v[2,0], v[0,1], v[1,1], \
    v[2,1], v[0,2], v[1,2], v[q2,2])


    
