#!/usr/bin/env python

import numpy as np
# import matplotlib
# matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

class Lij:
    """the Velocity Gradient Tensor Class"""

    def __init__(self, step, L11, L12, L13, L21, L22, L23, L31, L32, L33, tincr):
        self.step = int(step)
        self.Lij = np.array([[L11, L12, L13],[L21, L22, L23],[L31, L32, L33]],dtype='float_')
        self.tincr = float(tincr)
        
    def strain_rate(self):
        return (self.Lij + np.transpose(self.Lij))/2
    
    def rotation_rate(self):
        return (self.Lij - np.transpose(self.Lij))/2
 
class Trajectory:
    """Forward motion of particle: position and velocity""" 
    def __init__(self, r, lat, lon, x, y, z, vx, vy, vz):
        self.r = float(r)
        self.lat = float(lat)
        self.lon = float(lon)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.vx = float(vx)
        self.vy = float(vy)
        self.vz = float(vz)

    
class Path:
    """list of Lij types and Trajectory along a path"""

    def __init__(self,Lfilename,Pfilename):
        self.Lfilename = Lfilename
        self.Pfilename = Pfilename
        try:
            self._read_L_file()
        except IOError:
            print 'cannot open ' + Lfilename
        try:
            self._read_P_file()
        except IOError:
            print 'cannot open ' + Pfilename
            


    def _read_L_file(self):
        with open(self.Lfilename, 'r') as f:
            # get info from top line
            fl = f.readline().split()
            self.nsteps = int(fl[0])
            self.ictrl  = int(fl[1])
            self.eqincr = float(fl[2])
            self.temp   = float(fl[3])
            # skip the nextline
            _ = f.readline()
            # populate a list of Lij
            self.listLij = []
            for line in f.readlines():
                self.listLij.append( Lij( *tuple( line.split())))
            # if time step is negative, flip
            if self.listLij[0].tincr < 0:
                self.listLij.reverse()
                for l in self.listLij:
                    l.tincr = -l.tincr
                    
    def _read_P_file(self):
        with open(self.Pfilename, 'r') as f:
            # skip the first 2 lines
            _ = f.readline()
            _ = f.readline()
            # populate a list of Trajectory objects
            self.listTrajectory = []
            for line in f.readlines():
                self.listTrajectory.append( Trajectory( *tuple( line.split())))

    
    def accumulate_strain(self):
        """get the total finite deformation along path"""
        # instantiate a finite deformation tensor
        F = Fij()
        # work along the path
        for L in self.listLij:
            F = update_strain(F,L)
        return F
    
    
    # def Flinn(self):
    #     """Get Flinn info along path"""
    
    
                    
class Fij:
    """The Deformation Gradient Tensor Class
      with some methods to calculate attributes"""

    def __init__(self):
        self.Fij = np.identity(3)  
        
    def rightCauchyGreen(self):
        return np.dot(np.transpose(self.Fij),self.Fij)

    def Finger(self):
        return np.linalg.inv(self.rightCauchyGreen())

    def leftCauchyGreen(self):
        return np.dot(self.Fij,np.transpose(self.Fij))

    def Cauchy(self):
        return np.linalg.inv(self.leftCauchyGreen())
    
    def Jacobian(self):
        return np.linalg.det(self.Fij)

    def rightStretch(self):
        U, s, Vh = np.linalg.svd(self.Fij)
        return np.dot( Vh.conj().T, np.dot( np.diag(s), Vh))
    
    def leftStretch(self):
        U, s, Vh = np.linalg.svd(self.Fij)
        return np.dot( U, np.dot( np.diag(s), U.conj().T))
    
    def Rotation(self):
        U, s, Vh = np.linalg.svd(self.Fij)
        return np.dot(U,Vh)
    
    def Principal(self):
        """get principal axis lengths and axes directions of finite strain ellipsoid"""
        w, v = np.linalg.eig(self.rightStretch())
        return w, np.dot(self.Rotation(), v)
    
    # def plot3d(self,axlim=3):
    #     """plot finite strain ellipsoid"""
    #     from mpl_toolkits.mplot3d import Axes3D
    #     import matplotlib.pyplot as plt
    #     from matplotlib.patches import FancyArrowPatch
    #     from mpl_toolkits.mplot3d import proj3d
    #     class Arrow3D(FancyArrowPatch):
    #
    #         def __init__(self, xs, ys, zs, *args, **kwargs):
    #             FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
    #             self._verts3d = xs, ys, zs
    #
    #         def draw(self, renderer):
    #             xs3d, ys3d, zs3d = self._verts3d
    #             xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
    #             self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
    #             FancyArrowPatch.draw(self, renderer)
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     lim = [-axlim,axlim]
    #     ax.set_xlim(lim)
    #     ax.set_ylim(lim)
    #     ax.set_zlim(lim)
    #     ax.set_aspect("equal")
    #
    #     # draw sphere
    #     u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    #     x = np.cos(u)*np.sin(v)
    #     y = np.sin(u)*np.sin(v)
    #     z = np.cos(v)
    #
    #     # put into array
    #     XYZ = np.vstack([x.flatten(),y.flatten(),z.flatten()])
    #     # deform according to F
    #     XYZ = np.dot(self.Fij,XYZ)
    #
    #     # put back into meshgrid
    #     a, b, c = np.vsplit(XYZ,3)
    #     x = np.reshape(a,x.shape)
    #     y = np.reshape(b,y.shape)
    #     z = np.reshape(c,z.shape)
    #
    #     # ax.plot_wireframe(x, y, z, color="r")
    #     ax.plot_surface(x,y,z,rstride=1,cstride=1,shade=True)
    #
    #     # plot principal vectors
    #     w, v = self.Principal()
    #     for jj in np.arange(3):
    #         a = Arrow3D([ 0, 1.2 * w[jj] * v[0, jj] ], [0, 1.2 * w[jj] * v[1, jj] ], [0, 1.2 * w[jj] * v[2, jj] ], mutation_scale=20,
    #                 lw=1, arrowstyle="-|>", color="k")
    #         ax.add_artist(a)
    #
    #     plt.show()
    
def update_strain(F,L):
    """update strain on finite deformation tensor F according to velocity gradients tensor L"""
    half_dtL = ( L.tincr / 2) * L.Lij
    A = np.identity(3) - half_dtL 
    B = np.identity(3) + half_dtL
    F.Fij = np.dot(np.linalg.inv(A), np.dot( B, F.Fij))
    return F
    
def sph2cart(lat,lon,r):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    z = r * np.sin(lat)
    y = r * np.cos(lat) * np.sin(lon)
    x = r * np.cos(lat) * np.cos(lon)
    return x,y,z