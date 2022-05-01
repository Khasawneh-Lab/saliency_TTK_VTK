# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 19:37:11 2021

@author: Melih Can Yesilli (yesillim@msu.edu)

This function loads the saliency and cluster the surface based on saliency.
Depending on the user input it also provides plot of the dendrogram.

NOTE: RUN Saliency.py before running this example.

"""
import os,sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__)))
from ttk_saliency import Surface_Patch_Clustering,feature_similarity
import scipy.spatial.distance as dist
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from vtk import vtkXMLPolyDataReader
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

save_plot = False # set to True if you want to save the plot
#%% parameters

# iteration number for the simplification and the persistence diagram type
iter_num = 3 # the total number of iterations executed by the simplification algorithm 
diagtype = 'superlevel'

# cluster the critical points only 
cluster_n = [5,10,15,25] # numbers of the clusters user would like to see
iteration=1 # iteration number for the simplified surface 
# this enables user to import the corresponding simplified surface 
# and perform clustering on it

# impor the surface data for the corresponding simplified surface
surf_name = 'Simplified_Surface_Iteration_{}.npy'.format(iteration)
surf = np.load(surf_name,allow_pickle=True).item()

input_reader = vtkXMLPolyDataReader()
input_reader.SetFileName('Simplified_Surface_Iter_PolyData_{}'.format(iteration))
input_reader.Update()
Bounds = input_reader.GetOutputDataObject(0).GetBounds()
X_min = Bounds[0]
X_max = Bounds[1]
Y_min = Bounds[2]
Y_max = Bounds[3]
X = np.linspace(X_min,X_max,len(surf['surface']))
Y = np.linspace(Y_min,Y_max,len(surf['surface']))
X, Y = np.meshgrid(X, Y)
Z = surf['surface']

# import saliency information and the range for the neighborhood size
saliency = surf['saliency']
r = surf['r']
pair_index = surf['cp_ind_des']        
    
# eliminate inf values in saliency matrix
saliency = saliency[saliency!=float("inf")].reshape(-1,len(r))
Dist_M = feature_similarity(saliency,r)
Dist_M = dist.squareform(Dist_M)

# MS complex
if diagtype == 'sublevel':
    manifold = surf['Asc_Man']
elif diagtype == 'superlevel':
    manifold = surf['Des_Man']

# obtain the surface clustering 
clustered_surf = Surface_Patch_Clustering(cluster_n,diagtype, Dist_M,'approximate', manifold, pair_index, input_reader.GetOutputDataObject(0))
    
    
    
    
#%% plot the surface patches obtained with given cluster numbers

fig = plt.figure(figsize=(15,7.5))
for i in range(len(cluster_n)):
    saliency_clustering =  clustered_surf[0][i,0]

    color_dimension = saliency_clustering # change to desired fourth dimension
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='Paired')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)
    
    textsize=18
    ax = fig.add_subplot(1, len(cluster_n), i+1, projection='3d')
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
    ax.set_xlabel(('x'),fontsize=textsize)
    ax.set_ylabel(('y'),fontsize=textsize)
    ax.set_zlabel(('z'),fontsize=textsize)
    ax.set_title(("Number of Clusters:{}".format(cluster_n[i])),fontsize=textsize)
    fig.canvas.show()
    ax=plt.gca()
    ax.tick_params(labelsize = textsize-2)
    
    if save_plot:
        plt.savefig(os.path.join(os.path.dirname(__file__),'Updated_Saliency_Clustering_Simplification_Step_{}_Approximate_distance_Segmentation_Use_False.png'.format(iteration)), 
                bbox_inches='tight',dpi = 200)     
    




















