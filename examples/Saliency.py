# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 22:42:52 2021

@author: Melih Can Yesilli (yesillim@msu.edu)
    
This Code is written to understand the output of TTK functions on a simple example

"""

from ttk_saliency import geo_dist_par,perDiagTTK,triangulate,saliency_weight,Topological_Saliency,Topological_Saliency_r,Saliency_Simplification
from pyevtk.hl import pointsToVTK
from pygeodesic.examples.vtk_helpers import getPointsAndCellsFromPolydata
import numpy as np
import scipy.spatial.distance as dist
import time
import cripser         
import pygeodesic.geodesic as geodesic
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk
from ttk_saliency import saliency_based_clustering, feature_similarity
from itertools import combinations
import sys,os
#%% input parameters
diagtype = 'superlevel'
vtu_available = True
save_output = True
saving_path = 'ENTER THE SAVING PATH HERE' # IF VTU IS NOT AVAILABLE

# check if the vtu file is available, if not conver the surface into a vtu file 
# using pointsToVTK function
if vtu_available:
    inputFilePath = os.path.join(os.path.dirname(__file__),'synthetic_surface_64by64.vtu')
else:
    # PLEASE PROVIDE THE X, Y AND Z COORDINATES OF YOUR SURFACE IF YOU DO NOT HAVE THE VTU FILE FOR YOUR DATA
       
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    x = np.ravel(X)
    y = np.ravel(Y)
    z = np.ravel(Z)
    pointsToVTK(saving_path, x, y, z, data={"data": z})   

#%% TTK

# triangulate the input
reader,polydata = triangulate(inputFilePath)

# persistence diagram computation---------------------------------------------
pd,output,Pairs,criticalPairs = perDiagTTK(reader,diagtype)
vertexID = output['vertexID_des']
pair_index = output['cp_ind_des']

# extract the persistence diagram pairs for only the selected type of diagram
perDiag= np.asarray(pd)
if diagtype == 'sublevel':
    n_features = len(perDiag[perDiag[:,2]==0])
    perDiag=perDiag[perDiag[:,2]==0]
elif diagtype == 'superlevel':
    n_features = len(perDiag[perDiag[:,2]==1])
    perDiag= perDiag[perDiag[:,2]==1]
elif diagtype == 'infinite':
    n_features = len(perDiag[perDiag[:,2]==-1])
    perDiag = perDiag[perDiag[:,2]==-1]


# compute distances between critical points-----------------------------------
DM_app,distances = geo_dist_par(polydata,True,'approximate',pair_index) # recommended for faster computation
# D_M = geo_dist_par(polydata,True,'exact',pair_index)

# flatten the distance matrix that only includes 
distances = DM_app.flatten()

# compute saliency for a range of r-------------------------------------------
r = np.linspace(0.01, max(distances),100)
output_sal = Topological_Saliency_r(pd,r,distances,pair_index,DM_app,diagtype)
saliency = output_sal["saliency"]
chosen_pd_ind =  output_sal["chosen_pd_ind"]
r_threshold_ind = 5 
n_feature_remain = 5


# save initial saliency, neighborhood sizes and persistence diagrams
out = {}
out['saliency'] = saliency
out['r'] = r
if save_output:
    np.save("Saliency_r.npy",out)
    np.save("simple_surface_PD",perDiag)

# apply simplification--------------------------------------------------------
out_simp = Saliency_Simplification(reader,saliency,criticalPairs,chosen_pd_ind,r,r_threshold_ind,vertexID,n_features,n_feature_remain,diagtype)


