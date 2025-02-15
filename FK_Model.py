"""\
------------------------------------------------------------
ForwardKinematics class implementing all the funcions needed
to establish a Forward Kinematics Model.
------------------------------------------------------------\
"""
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.pyplot as plt
import math
import numpy as np

class ForwardKinematics:

    def __init__(self,axis_list,line_ends,coordsys):

        self.axis_list = axis_list
        self.line_ends = line_ends
        self.coordsys = coordsys

    def rotation_matrix(self,rotation_angle, axis_of_rotation): ##MODIFY IN THE FUTURE SO IT WORKS FOR OTHER COORDINATE SYSTEMS
        ax = axis_of_rotation.astype(float)  # Explicitly cast to float
        ax /= np.linalg.norm(ax)
        cosA = math.cos(rotation_angle)
        sinA = math.sin(rotation_angle)
        minusCosA = 1.0 - cosA
        im = np.identity(3)

        cp = np.array([[0,-ax[2],ax[1]],
                        [ax[2],0,-ax[0]],
                        [-ax[1],ax[0],0]])
        op = np.outer(ax,ax)
        
        rm = np.dot(cosA,im) + np.dot(sinA,cp) + np.dot(minusCosA,op)

        return rm


