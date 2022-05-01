"""
Author: Melih Can Yesilli
Date: 5/1/2022
Contact: yesillim@msu.edu

Description: This library includes functions that can be used to compute topological saliency for a given surface.
In addition, users can perform clustering and topological simplification based on topological saliency.

Required softwares and packages:
    - TTK (https://topology-tool-kit.github.io/installation.html)
    - VTK Python API
    - pygeodesic (https://pypi.org/project/pygeodesic/)
    - multiprocessing
    - potpourri3d (https://github.com/nmwsharp/potpourri3d)


"""
from vtk import (
    vtkDataObject,
    vtkTableWriter,
    vtkThreshold,
    vtkXMLPolyDataWriter,
    vtkXMLPolyDataReader,
    vtkUnstructuredGridReader,
    vtkXMLUnstructuredGridReader,
    vtkXMLUnstructuredGridWriter,
    vtkXMLStructuredGridWriter,
    vtkDataSetReader,
    vtkUnstructuredGridWriter,
    vtkXMLStructuredGridReader,
    vtkDelaunay2D,
)
import vtk
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk
from topologytoolkit import (
    ttkMorseSmaleComplex,
    ttkPersistenceCurve,
    ttkPersistenceDiagram,
    ttkTopologicalSimplification,
    ttkArrayPreconditioning,
)
from itertools import combinations
from pygeodesic.examples.vtk_helpers import getPointsAndCellsFromPolydata
import multiprocessing
from multiprocessing import Pool
import time
import pygeodesic.geodesic as geodesic
import numpy as np
import math
import scipy.spatial.distance as dist
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import potpourri3d as pp3d
from scipy.spatial.distance import squareform


def triangulate(inputFilePath):
    """
    This function triangulates given vtu file using vtkDelaunay2D function of vtk.
    
    Parameters
    ----------
    inputFilePath : str
        Path to the vtu file

    Returns
    -------
    reader : vtk object 
        triangulated surface/image
    polydata : vtk object
        Output of the vtk object of triangulated surface/image

    """
    # loading the input data 
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(inputFilePath)
    reader.Update()
    
    # triagulating it 
    triangulation = vtkDelaunay2D()
    triangulation.SetInputConnection(reader.GetOutputPort())
    triangulation.Update()
    reader = triangulation
    polydata = reader.GetOutput() 
    
    return reader,polydata



def MS_Complex(reader,saving):
    """
    

    Parameters
    ----------
    reader : vtk object 
        triangulated surface/image
    saving : boolean
        Set to True if user wants to save the MS complex output

    Returns
    -------
    out : dict
        Output of the MS Complex function in TTK. This includes the ascending and descending manifolds.

    """
    # compute MS Complex
    morseSmaleComplex = ttkMorseSmaleComplex()
    morseSmaleComplex.SetInputConnection(reader.GetOutputPort())
    morseSmaleComplex.SetInputArrayToProcess(0, 0, 0, 0, "data")
    morseSmaleComplex.SetDebugLevel(3)
    morseSmaleComplex.Update()

    n_output = morseSmaleComplex.GetOutputDataObject(3).GetPointData().GetNumberOfArrays()
    out = {}
    for i in range(n_output):
        out[morseSmaleComplex.GetOutputDataObject(3).GetPointData().GetArrayName(i)]=vtk_to_numpy(morseSmaleComplex.GetOutputDataObject(3).GetPointData().GetArray(i))
    
    if saving:
        np.save("MS_Complex_Output",out)
    
    return out



def geo_dist_exact(k,points, faces,comb):
    """
    Computes the exact geodesic distance using paralell computing. 

    Parameters
    ----------
    k : int
        index for the list of combinations of selected points in triangulated surfaces
    points : 2D array
        the coordinate points of the points on the surface
    faces : 2D array
        the vertex number of the points that generates faces after triangulation
    comb : list
        the list of combinations between the points 

    Returns
    -------
    d : array
        the distances between the selected points
        

    """
    # select indexes of two critical points
    source_Index = comb[k][0]
    target_Index = comb[k][1]
    geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)
    # compute the distance
    d,path = geoalg.geodesicDistance(source_Index, target_Index)
  
    return d

def geo_dist_approx(solver,source_Index):
    """
    Computes the approximated geodesic distance  

    Parameters
    ----------
    solver : potpourri3d object
        solver for distance computation, check the documentation of potpourri3d for more details
    source_Index : float
        the vertex number in triangulated surface

    Returns
    -------
    dist : array
        the distances between selected vertex number and all surface points

    """
    dist = solver.compute_distance(source_Index)
    return dist


def geo_dist_par(polydata,compute_comb,dist_type,*args):
    """
    Computes the pairwise distance matrix between the given critical points. 
    The exact distance or approaximated distance is computed based on user choice.

    Parameters
    ----------
    polydata : vtkPolyData
        vtkPolyData object obtained after generatin triangulation
    compute_comb : boolean
        Set it to True if you want algorithm to generate combinations whose 
        pairwise distance will be computed. If False, user needs to provide 
        the list of indicies whose pairwise distances needed
    dist_type : string
        Geodesic distance type. If 'exact' is selected, exact distance will be 
        computed. For faster computation, approximated distance is recommended using 
        'approximate' flag.
    *args : 
        vertexID : np.array
            The list of indicies of the points whose pairwise distance needed.
            It is only required if user set compute_comb parameter to True.
        comb : list
            The list of combinations provided by user when compute_comb is set 
            to False
        
    Returns
    -------
    DM_cp: np.array
        2D array that only includes the pairwise distances between the points whose
        indices given in vertexID or whose combinations are provided by the user
    distances: np.array
        2D array that only includes the pairwise distances between selected critical points
        of the surface and the all surface points of the surface. All other distances is shown
        as zero. Its dimenison equals to the dimension of the surface. 

    """
    points, faces = getPointsAndCellsFromPolydata(polydata) # obtain points and faces of the surface
    
    # user choses if exact or approximate geodesic distance will be computed
    # exact distance uses pygeodesic package, while approximate distance uses
    # potpourri3d package. The latter is much faster than the former.
    
    if dist_type =='exact':
        if compute_comb:
            vertexID = args[0]
            comb = combinations(vertexID, 2)
            comb = list(comb)   
        else:
            comb = args[0]
        
        # put all inputs of the distance computation function together
        inputs = []
        for i in range(len(comb)):
            inputs.append((i,points, faces, comb))
        
        # time the paralell computation of distances
        start = time.time()
        n_proc = multiprocessing.cpu_count()
        with Pool(processes=n_proc//2) as p:
            DM_cp  = p.starmap(geo_dist_exact, inputs)
        finish = time.time()
        print('Elapsed time (paralell execution-exact): {}'.format(finish-start))

        
        return DM_cp
    
    elif dist_type=='approximate':
        if compute_comb:
            vertexID = args[0]
            comb = combinations(vertexID, 2)
            comb = list(comb)   
        else:
            comb = args[0]
            
        # generate the solver 
        solver = pp3d.MeshHeatMethodDistanceSolver(points,faces)
        
        # time the computation of distances between surface points and the selected critical points
        start = time.time()
        distances = np.zeros((len(points),len(points)))
        for i in range(len(vertexID)):
            distances[vertexID[i],:] = geo_dist_approx(solver,vertexID[i]) 
            distances[:,vertexID[i]] = distances[vertexID[i],:] 
        finish = time.time()
        print('Elapsed time (serial execution-approximation): {}'.format(finish-start))               
        
        # the distances between selected critical points
        DM_cp = np.zeros((len(comb)))
        for i in range(len(comb)):
            DM_cp[i] = distances[comb[i][0], comb[i][1]]
        #convert it to square matrix
        DM_cp = squareform(DM_cp)         
                
        return DM_cp,distances


def perDiagTTK(reader,diagtype):
    """
    Computes selected type of persistence diagram and provides user with details.

    Parameters
    ----------
    reader : vtk object 
        triangulated surface/image
    diagtype : str
        the type of persistence diagram need to be computed (superlevel, sublevel or infinite)

    Returns
    -------
    pd : list of tuples
        all persistence diagrams including superlevel, sublevel and infinite
    output : dictionary
        the dictionary that includes the all necessary output such as coordinates of critical points,
        desired persistence diagrams, vertex number of all or desired critical points in triangulated 
        surface
    Pairs : vtk object
        vtkThreshold object that includes information related to all persistence diagram types. This information
        contains coordinates of the critical points, persistence values and vertex number of critical points.
    criticalPairs : vtk object
        vtkThreshold object that includes information related to selected persistence diagram type. This information
        contains coordinates of the critical points, persistence values and vertex number of critical points.

    """
    
    diagram = ttkPersistenceDiagram()
    diagram.SetInputConnection(reader.GetOutputPort())
    diagram.SetInputArrayToProcess(0, 0, 0, 0, "data")
    diagram.SetDebugLevel(3)
    diagram.Update()
    
    # obtain critical points from PD
    criticalPairs = vtkThreshold()
    criticalPairs.SetInputConnection(diagram.GetOutputPort())
    criticalPairs.SetInputArrayToProcess(
        0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, "PairIdentifier")
    criticalPairs.ThresholdBetween(-0.1, 999999)
    criticalPairs.Update()
    
    if diagtype == "infinite":
        # obtain Global Min - Global Max pairs
        Pairs = vtkThreshold()
        Pairs.SetInputConnection(criticalPairs.GetOutputPort())
        Pairs.SetInputArrayToProcess(
            0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, "PairType")
        Pairs.ThresholdBetween(-1, -1)
        Pairs.Update()
    elif diagtype == "sublevel":
        # obtain sub-level set persistent
        Pairs = vtkThreshold()
        Pairs.SetInputConnection(criticalPairs.GetOutputPort())
        Pairs.SetInputArrayToProcess(
            0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, "PairType")
        Pairs.ThresholdBetween(0, 0)
        Pairs.Update()   
    elif diagtype=="superlevel":
        Pairs = vtkThreshold()
        Pairs.SetInputConnection(criticalPairs.GetOutputPort())
        Pairs.SetInputArrayToProcess(
            0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, "PairType")
        Pairs.ThresholdBetween(1, 1)
        Pairs.Update()
        
    # vertexID, coordinates and persistence of all critical points including superlevels, sublevels and infinite
    
    vertexID_all = criticalPairs.GetOutput().GetPointData().GetArray("ttkVertexScalarField")
    coord_all = criticalPairs.GetOutput().GetPointData().GetArray("Coordinates")
    persistence_all = criticalPairs.GetOutput().GetCellData().GetArray("Persistence")
    
    # vertexID, coordinates and persistence of critical points belong to diagram chosen above
    
    vertexID_des = Pairs.GetOutput().GetPointData().GetArray("ttkVertexScalarField")
    coord_des = Pairs.GetOutput().GetPointData().GetArray("Coordinates")
    persistence_des = Pairs.GetOutput().GetCellData().GetArray("Persistence")   
    
    pd = []
    pd_pairs_coord = []
    # obtain persistence diagram and coordinates of all points
    for i in range(criticalPairs.GetOutput().GetNumberOfCells()):
        tCell = criticalPairs.GetOutput().GetCell(i)
        pair_type = vtk_to_numpy(criticalPairs.GetOutput().GetCellData().GetArray(1))[i]
        
        birthId = tCell.GetPointId(0)
        bp = coord_all.GetTuple(birthId)[2]
        
        deathId = tCell.GetPointId(1)
        dp = coord_all.GetTuple(deathId)[2]    
        
        pd.append((bp,dp,pair_type))
        pd_pairs_coord.append((coord_all.GetTuple(birthId),coord_all.GetTuple(deathId)))
        
    # obtain persistence diagram and coordinates of desired points only    
    pd_des = []
    pd_pairs_coord_des = []
    # obtain persistence diagram and coordinates of desired (sublevel or superlevel) points
    for i in range(Pairs.GetOutput().GetNumberOfCells()):
        tCell = Pairs.GetOutput().GetCell(i)
        pair_type = vtk_to_numpy(Pairs.GetOutput().GetCellData().GetArray(1))[i]
        
        birthId = tCell.GetPointId(0)
        bp = coord_des.GetTuple(birthId)[2]
        
        deathId = tCell.GetPointId(1)
        dp = coord_des.GetTuple(deathId)[2]    
        
        pd_des.append((bp,dp,pair_type))
        pd_pairs_coord_des.append((coord_des.GetTuple(birthId),coord_des.GetTuple(deathId)))        
        
           
    # convert coordinates of all critical point pairs from list to an array
    pair_coord = np.asarray(pd_pairs_coord)
    # convert the coordinates of the pair of critical points of desired persistence diagram into an array
    pd_pairs_coord_des = np.asarray(pd_pairs_coord_des)
    # convert the coordinates of desired points or all points into numpy array
    coord_all = vtk_to_numpy(coord_all)
    coord_des = vtk_to_numpy(coord_des) 

    # find the index of matching birth and death critical points in vertexID_des
    # For example, if the first matching indices are 0 and 1, it will show that 
    # first and second vertex_ID des are matching and they are a pair in persistence
    # diagram
    
    pair_indices = []
    for i in range(len(pd_pairs_coord_des)):
        p1 = pd_pairs_coord_des[i,0,:].reshape((1,3)).astype("float32")
        p2 = pd_pairs_coord_des[i,1,:].reshape((1,3))[0]
    
        ind1 = np.where(np.all(coord_des==p1,axis=1))[0][0]
        ind2 = np.where(np.all(coord_des==p2,axis=1))[0][0]
        
        pair_indices.append((ind1,ind2))
    pair_index = np.asarray(pair_indices)
    
    # Depending on chosen persistence diagram type provide users with the index of the
    # minima or maxima
    
    vertexID_des = vtk_to_numpy(vertexID_des)
    if diagtype=='sublevel':
        pair_index_des = vertexID_des[pair_index[:,0]] # vertexID of birth times 
    elif diagtype == 'superlevel':
        pair_index_des = vertexID_des[pair_index[:,1]] # vertexID of death times  


    # parse the outputs into a dictionary
    output = {}
    output["vertexID_all"] = vtk_to_numpy(vertexID_all)
    output["coord_all"] = coord_all  
    output["persistence_all"] = vtk_to_numpy(persistence_all) 
    output["pairs_coord_all"] = pd_pairs_coord
    
    output['pair_ind_des'] = pair_index
    output["cp_ind_des"]=pair_index_des
    output["vertexID_des"] = vertexID_des
    output["coord_des"] = coord_des
    output["persistence_des"] = vtk_to_numpy(persistence_des)    
    
    return pd,output,Pairs,criticalPairs


def saliency_weight(vertexID,r,D_M):
    """
    Computes the gaussian weights for each critical point at given neighborhood
    size.

    Parameters
    ----------
    vertexID : np.array
        The vertex number of the critical points for the 
    r : np.array
        The array that includes the neighborhood sizes used to compute the saliency
    D_M : 2D np.array
        The matrix that includes the distances between the critical points

    Returns
    -------
    W : 2D np.array
        The gaussian weights used to compute the saliency for correponding r

    """
    comb = combinations(np.arange(0,len(vertexID)), 2)
    comb = list(comb)
    
    weights = []
    # upper diagonal weights
    for i in range(len(comb)):
        ind1 = comb[i][0]
        ind2 = comb[i][1]
        weights.append(math.exp(-D_M[ind1,ind2]**2/r**2))

    # convert the list of weights into array
    W = dist.squareform(weights) 
    
    # add diagonal weigths
    for i in range(len(W)):
        W[i,i]=1
        
    return W


def Topological_Saliency(perdiag,weight):
    """
    Computes topological saliency for given weights

    Parameters
    ----------
    perdiag : 2D np.array
        The selected persistence diagram ('sublevel', 'superlevel' or 'infinite')
    weight : 2D np.array
        The gaussian weights used to compute the saliency for correponding r

    Returns
    -------
    T : np.array
        Saliency values computed at a certain neighborhood size for the critical points

    """
    T = []
    for i in range(len(perdiag)):
        denominator = 0
        for j in range(len(perdiag)):
            denominator = denominator + weight[i,j]*(perdiag[j,1]-perdiag[j,0])
        T.append(weight[i,i]*(perdiag[j,1]-perdiag[j,0])/denominator)
    return T



def Topological_Saliency_r(pd,r_range,distances,vertexID,D_M,diagtype):
    """
    Provides user with salency of critical points computed at given neighborhood range

    Parameters
    ----------
    pd : list of tuples
        Persistence diagrams including all types
    r_range : np.array
        The array that includes all r values which will be used to compute saliency
    distances : np.array
        Geodesic distance that includes only the distances between critical points
    vertexID : np.array
        The vertex number of critical points for selected peristence diagram.
    D_M : np.array (2D)
        Pairwise distance matrix between the critical points of selected persistence diagram
    diagtype : str
        The type of selected persistence diagram ('superlevel', 'sublevel', or 'infinite')

    Returns
    -------
    output : dict
        The dictionary that includes the saliency values of selected critical points, and the index of selected persistence diagrams
        The saliency matrix will only include the values for selected critical points, for all other points, the saliency value is set to 'inf'
    """
    # conver the list of tuples into an array and generate the saliency matrix
    perDiag= np.asarray(pd)
    saliency = np.ones((len(perDiag),len(r_range)))*float("inf")
    

    # set the saliencies of persistence diagram points that excludes the points 
    # of selected persistence diagram
    if diagtype == 'sublevel':
        pd_index = np.where(perDiag[:,2]==0)[0]
        chosen_pd = perDiag[perDiag[:,2]==0]
    elif diagtype == 'superlevel':
        pd_index = np.where(perDiag[:,2]==1)[0]
        chosen_pd = perDiag[perDiag[:,2]==1] 
    elif diagtype == 'infinite':
        pd_index = np.where(perDiag[:,2]==-1)[0]
   
        chosen_pd = perDiag[perDiag[:,2]==-1]
        
    
    i=0
    for r in r_range:
        weight = saliency_weight(vertexID,r,D_M)
        saliency[pd_index,i] = np.asarray(Topological_Saliency(chosen_pd,weight))
        i +=1
    

    # output object to return
    output = {}
    output["chosen_pd_ind"]= pd_index
    output["saliency"]= saliency
    
    
    return output
    
def save_surf_info(reader, saliency, r, diagtype, inc):
    """
    This function saves the surface information during simplification process. For each iteration,
    saliency and triangulated surface are saved.

    Parameters
    ----------
    reader : vtk object 
        triangulated surface/image
    saliency : 2D np.array
        Saliency values of all critical points for varying critical points 
    r : np.array
        the range of values for neighborhood size
    diagtype : str
        The type of persistence diagrams selected for the analysis. It could be 'superlevel', 'sublevel' or 'infinite'
    inc : int
        The iteration number for simplification process

    Returns
    -------
    None.

    """
    # persistence diagram
    pd,output,Pairs,criticalPairs = perDiagTTK(reader,diagtype)
    pair_index = output['cp_ind_des']
    # MS-Complex
    MS_Cplx = MS_Complex(reader,False)
    AM = MS_Cplx['AscendingManifold']
    DM = MS_Cplx['DescendingManifold']
    # surface data 
    surf = vtk_to_numpy(reader.GetOutputDataObject(0).GetPointData().GetArray(0))
    surf = surf.reshape(-1,int(np.sqrt(len(surf))))
    # saving
    output = {}
    output['surface'] = surf
    output['saliency'] = saliency
    output['r'] = r
    output['cp_ind_des'] = pair_index
    output['Des_Man']= DM
    output['Asc_Man']= AM
    
    save_name = save_name = 'Simplified_Surface_Iteration_{}'.format(inc)
    np.save(save_name,output)
    
    save_name_polydata = 'Simplified_Surface_Iter_PolyData_{}'.format(inc)
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(save_name_polydata)
    writer.SetInputData(reader.GetOutputDataObject(0))
    writer.Write()    
    

    
def Saliency_Simplification(reader,saliency,criticalPairs,chosen_pd_ind,r,r_threshold_ind,vertexID,n_features,n_feature_remain,diagtype):
    """
    This function applies the topological simplification based on topological saliency.
    

    Parameters
    ----------
    reader : vtk object 
        triangulated surface/image
    saliency : 2D np.array
        The saliency values computed at given neighborhood range for all critical points 
    criticalPairs : vtk object
        The vtk object that only includes the selected critical points (local minimas or maximas)
    chosen_pd_ind : np.array
        the indicies of the selected persistence diagram points in the array that includes all types of persistence diagrams
    r : np.array
        The range for the neighbohood size  
    r_threshold_ind : int
        The index of neighborhood size which will be used to threshold saliency values in simplification process
    vertexID : np.array
        The vertex number of the all critical points of the triangulated surface
    n_features : int
        The number of features (critical points). This number depends on the selected type of persistence diagram
    n_feature_remain : int
        The maximum number of features (critical points) that the user wants to have at the end of simplification process
    diagtype : str
        The type of persistence diagram user wants to work on ('sublevel','superlevel','infinite')

    Returns
    -------
    SalientPairs : vtkThreshold object
        The object that includes the information about the most salient critical points after the simplification process
    topologicalSimplification : ttk Object
        The TTK object that includes information about the simplified surface 

    """
    topologicalSimplification = reader
    inc= 0
    # the loop that simplifies the surface until the desired number of features remain
    while n_features>n_feature_remain:
        # save persistence diagram, saliency, critical point information for the original surface
        if inc==0:
            save_surf_info(reader, saliency, r, diagtype, inc)
            
        # define the thereshold for saliency for chosen r value 
        # it will ignore the the infinite saliency values 
        init_threshold_s = max(saliency[chosen_pd_ind,r_threshold_ind])*0.15
        max_saliecny = max(saliency[:,r_threshold_ind])
        

        # create a vtk array for saliency matrix and fill it with saliency values
        # of selected r parameter
        array = vtk.vtkDoubleArray()
        array.SetName("Saliency")
        array.SetNumberOfComponents(1)
        array.SetNumberOfTuples(len(saliency))
        for x in zip(range(len(saliency)), np.reshape(saliency[:,r_threshold_ind],(len(saliency),1))):
            array.SetTuple(*x)
    
        # add saliency matrix into vtk object of critical pairs
        criticalPairs.GetOutput().GetCellData().AddArray(array)
        criticalPairs.Update()
     
        # find the most salient points for the given range
        SalientPairs = vtkThreshold()
        SalientPairs.SetInputConnection(criticalPairs.GetOutputPort())
        SalientPairs.SetInputArrayToProcess(
            0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, "Saliency")
        SalientPairs.ThresholdBetween(init_threshold_s,max_saliecny)
        SalientPairs.Update()
        
        # remove the non-salient points from the surface
        t_Simplification = ttkTopologicalSimplification()
        t_Simplification.SetInputConnection(0, topologicalSimplification.GetOutputPort())
        t_Simplification.SetInputArrayToProcess(0, 0, 0, 0, "data")
        t_Simplification.SetInputConnection(1, SalientPairs.GetOutputPort())
        t_Simplification.SetDebugLevel(3)
        t_Simplification.Update()
        
        # update the topological simplification object since this is an iterative process
        topologicalSimplification = t_Simplification
        
        # triangulate the new simplified surface
        triangulation = vtkDelaunay2D()
        triangulation.SetInputConnection(topologicalSimplification.GetOutputPort())
        triangulation.Update()
        reader = triangulation
        polydata = reader.GetOutput() 
        
        # compute the persistence diagram of the simplified surface
        pd,output,Pairs,criticalPairs = perDiagTTK(reader,diagtype)
        pair_index = output['cp_ind_des']
        perDiag= np.asarray(pd)


        # compute the number of feature remained after each simplification
        prev_n_feat = n_features
        if diagtype == 'sublevel':
            n_features = len(perDiag[perDiag[:,2]==0])
        elif diagtype == 'superlevel':
            n_features = len(perDiag[perDiag[:,2]==1])
        elif diagtype == 'infinite':
            n_features = len(perDiag[perDiag[:,2]==-1])
        
        # compute the distances between critical points of the simplified surface
        # exact distance
        # distances = geo_dist_par(polydata,True,'exact',pair_index)
        # D_M = dist.squareform(distances)
        # approximate distance
        D_M,distances = geo_dist_par(polydata,True,'approximate',pair_index)
        
        # compute the saliency
        output_sal = Topological_Saliency_r(pd,r,distances,pair_index,D_M,diagtype)
        saliency = output_sal["saliency"]
        chosen_pd_ind =  output_sal["chosen_pd_ind"]
        
        print(n_features)
        inc= inc+1 
        # save the surface information
        save_surf_info(reader, saliency, r, diagtype, inc)
        
        # # extract the simplified surface information
        simp = topologicalSimplification
        simp = vtk_to_numpy(simp.GetOutputDataObject(0).GetPointData().GetArray(0))
        simp = simp.reshape(-1,int(np.sqrt(len(simp))))
        
        # # save simplified surface and corresponding saliency information
        save_name = 'Simplified_Surface_Iteration_{}'.format(inc+1)
        output = {}
        output['surface'] = simp
        output['saliency'] = saliency
        np.save(save_name,output)
        
        # if the number of feature does not change with respect to previous
        # iteration break the loop
        if prev_n_feat == n_features:
            break
         
    return SalientPairs,topologicalSimplification


def area_between_features(feat_1, feat_2, saliency,r):
    """
    

    Parameters
    ----------
    feat_1 : int
        index for critical point 1
    feat_2 : int
        index for critical point 2
    saliency : 2D np.array
        Saliency matrix
    r : np.array
        The array that includes the range for neighborhood size

    Returns
    -------
    float
        area between given two saliency curves

    """
    # extract the saliency array for feature 1 and feature 2
    sal_1 = saliency[feat_1,:]
    sal_2 = saliency[feat_2,:]
    
    # compute the areas under these arrays
    area_1 = np.trapz(sal_1,r)
    area_2 = np.trapz(sal_2,r)
    
    return abs(area_1-area_2)    

    

def feature_similarity(saliency,r):
    """
    

    Parameters
    ----------
    saliency : 2D np.array
        The saliency values of the critical points
    r : np.array
        The range for the neighborhood size

    Returns
    -------
    np.array
        The array that includes the areas between saliency curves

    """
    # distance = []
    # for i in range(len(saliency)):
    #     for j in range(i,len(saliency)):
    #         distance.append(area_between_features(i, j, saliency,r))   
    
    
    comb = combinations(np.arange(0,len(saliency)), 2)
    comb = list(comb)    
    
    distance = []
    for i in range(len(comb)):
        ind1 = comb[i][0]
        ind2 = comb[i][1]
        distance.append(area_between_features(ind1, ind2, saliency,r))
    
    return np.asarray(distance)

def plot_dendrogram(model, **kwargs): 
    """
    

    Parameters
    ----------
    model : cluster model
        Selected clustering function
    **kwargs : 
        Additional plot parameters. Please check the documentation for the dendogram function of SciPy. 

    Returns
    -------
    Figure that inlcudes the dendogram plot of the clustering

    """
    # this function directly copied and pasted from 
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
       
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
def saliency_based_clustering(Dist_M, dendrogram_plot,cluster_n):
    """
    

    Parameters
    ----------
    Dist_M : 2D np.array
        The distance matrix that contains the pairwise distances between the critical points of the simplified surface.
    dendrogram_plot : boolean
        Set to True if user wants to obtain dendogram plot for the clustering of the critical points
    cluster_n : int
        Number of clusters user wants to have

    Returns
    -------
    cluster_model : cluster model
        Selected clustering function
    clustering : np.array
        Cluster numbers for each critical point

    """
    if dendrogram_plot:
        cluster_model =  AgglomerativeClustering(n_clusters=cluster_n, affinity='precomputed',
                                  linkage='single',compute_distances=True)
        clustering = cluster_model.fit_predict(Dist_M)         
    else:
        cluster_model =  AgglomerativeClustering(n_clusters=cluster_n, affinity='precomputed',
                                  linkage='single')
        clustering = cluster_model.fit_predict(Dist_M)  
    return cluster_model,clustering


def Surface_Patch_Clustering(cluster_n,diagtype, Dist_M, dist_type, manifold, pair_index, polydata):
    """
    

    Parameters
    ----------
    cluster_n : list
        The list that includes the number of clusters. User try multiple cluster number to compare the outputs
    diagtype : str
        the type of persistence diagrams that saliency is computed for. ('sublevel', 'superlevel', 'infinite')
    Dist_M : 2D np.array
        The distance matrix that contains the pairwise distances between the critical points of the simplified surface
    dist_type : str
        'approximate' or 'exact' geodesic distance. 'approximate' is recommended for faster computation.
    manifold : np.array
        The array that includes the clusters obtained using descending or ascending manifolds from Morse-Smale Complex
    pair_index : np.array
        The vertex numbers of the critical points of trinagulated surface
    polydata : vtkPolyData object
        The VTK object that includes the simplified surface information

    Returns
    -------
    clustered_surf : TYPE
        DESCRIPTION.
    distances: 2D np.array
        The distance matrix that includes the pairwise distances between the critical points only

    """
    labels_clust = np.zeros((len(cluster_n),1),dtype=object)
    clustered_surf = np.zeros((len(cluster_n),1),dtype=object)
    
    for j in range(len(cluster_n)):
        cluster_model,labels = saliency_based_clustering(Dist_M, False,cluster_n[j])
     
        # rearrange the label matrix for each element of the pairs
        new_labels = np.zeros((len(labels),2))
        new_labels[:,0] = pair_index
        new_labels[:,1] = labels
        new_labels = new_labels.astype(int)
        
        labels_clust[j,0] = new_labels   
        # generate final cluster matrices
        clustered_surf[j,0] =  np.full((len(manifold),1),-1)
        
        
    if dist_type=='exact': 
        for i in np.unique(manifold):
            print(i)
            # find the index of cluster i
            ind = np.where(manifold==i)[0]
            
            # find how many critical points are in this cluster
            common_cp, comm1, comm2 = np.intersect1d(ind, new_labels[:,0],return_indices=True)
            common_cp = common_cp.astype(int)
            
            # proceed with computation reduction if there is intersection 
            if len(common_cp)!=0:
                # remove these critical points from the index matrix since we dont need to compute the pairwise distance between them
                ind = np.delete(ind, comm1)
                
                comb=[]
                # generate the combinations of indices 
                for j in range(len(common_cp)):
                    for k in range(len(ind)):
                        comb.append((common_cp[j],ind[k]))
                
                # compute the distance 
                distances = geo_dist_par(polydata,False,'exact',comb)
                distances = np.asarray(distances)
                distances = np.reshape(distances,(-1,len(ind)))
       
                # find the nearest neighbor for each surface point in the cluster and assign them a label
                for j in range(len(ind)):
                    closest_cp_loc = np.where(new_labels[:,0] == common_cp[np.where(distances[:,j]==distances[:,j].min())[0][0]])[0][0]
                    # assing labels for different number of clusters chosen in the beginning
                    for cl_num in range(len(cluster_n)):
                        clustered_surf[cl_num,0][ind[j]] =  labels_clust[cl_num,0][closest_cp_loc,1]
                    
            # proceed with brute force approach if there is no intersection
            else:
                comb=[]
                # generate the combinations of indices 
                for j in range(len(new_labels[:,0])):
                    for k in range(len(ind)):
                        comb.append((new_labels[j,0],ind[k]))  
                        
                # compute the distance 
                distances = geo_dist_par(polydata,False,'exact',comb)
                distances = np.asarray(distances)
                
                # reshape the distances
                distances = distances.reshape(-1,len(ind)) 
                
                # find the nearest neighbor for each surface point in the cluster and assign them a label
                for j in range(len(ind)):
                    closest_cp_loc = np.where(distances[:,j]==distances[:,j].min())[0][0]
                    # assing labels for different number of clusters chosen in the beginning
                    for cl_num in range(len(cluster_n)):
                        clustered_surf[cl_num,0][ind[j]] =  labels_clust[cl_num,0][closest_cp_loc,1]       
                                                                
                print("i:{} ---> no intersection".format(i))
        for cl_num in range(len(cluster_n)):
            # add the labels of critical points into the matrix  
            clustered_surf[cl_num,0][new_labels[:,0].astype(int)] =  labels_clust[cl_num,0][:,1].reshape((len(new_labels[:,0]),1))     
            # make the labels square matrix
            clustered_surf[cl_num,0] = clustered_surf[cl_num,0].reshape(-1,int(np.sqrt(len(manifold))))  
    
    elif dist_type=='approximate':
        DM_cp,distances = geo_dist_par(polydata,True,dist_type,new_labels[:,0])
        segmentation_use = False
        
        if not segmentation_use: 
            # this portion does not use manifold segmentation to cluster
            
            # find the clusters of each surface point
            for i in range(len(manifold)):
                if i not in new_labels[:,0]:
                    # distances to critical points from a surface point
                    dist_to_cps = distances[new_labels[:,0],i]
                    # minimum distance
                    min_dist = min(dist_to_cps)
                    #find the location of the minimum distace
                    closest_cp_loc = np.where(dist_to_cps==min_dist)[0][0]
                    for cl_num in range(len(cluster_n)):
                        clustered_surf[cl_num,0][i] = labels_clust[cl_num,0][closest_cp_loc,1]
            # add the labels of the critical points
            for cl_num in range(len(cluster_n)):
                clustered_surf[cl_num,0][new_labels[:,0]]=labels_clust[cl_num,0][:,1].reshape((len(new_labels[:,0]),1))
                # make the labels square matrix
                clustered_surf[cl_num,0] = clustered_surf[cl_num,0].reshape(-1,int(np.sqrt(len(manifold))))         
        else:
            for i in np.unique(manifold):
                # find the index of cluster i
                ind = np.where(manifold==i)[0]
                
                # find how many critical points are in this cluster
                common_cp, comm1, comm2 = np.intersect1d(ind, new_labels[:,0],return_indices=True)
                common_cp = common_cp.astype(int)
    
                # find the nearest neighbor for each surface point in the cluster and assign them a label
                for j in range(len(ind)):
                    min_dist_to_cp = min(distances[new_labels[:,0],ind[j]])
                    closest_cp_loc = np.where(distances[new_labels[:,0],ind[j]]==min_dist_to_cp)[0][0]
                    # assing labels for different number of clusters chosen in the beginning
                    for cl_num in range(len(cluster_n)):
                        clustered_surf[cl_num,0][ind[j]] =  labels_clust[cl_num,0][closest_cp_loc,1]
      
                                                                    
            for cl_num in range(len(cluster_n)):
                # add the labels of critical points into the matrix  
                clustered_surf[cl_num,0][new_labels[:,0].astype(int)] =  labels_clust[cl_num,0][:,1].reshape((len(new_labels[:,0]),1))     
                # make the labels square matrix
                clustered_surf[cl_num,0] = clustered_surf[cl_num,0].reshape(-1,int(np.sqrt(len(manifold))))            
            
            
    return  clustered_surf,distances




# if __name__ == '__main__':
    
    
#     # input parameters
#     X = np.arange(-5, 5, 0.25)
#     Y = np.arange(-5, 5, 0.25)
#     X, Y = np.meshgrid(X, Y)
#     R = np.sqrt(X**2 + Y**2)
#     Z = np.sin(R)
    
#     x = np.ravel(X)
#     y = np.ravel(Y)
#     z = np.ravel(Z)
    
#     inputFilePath='D:\\Research Stuff\\Surface_Texture_Classification\\TDA_Based_Approachs\\simple_surface.vtu'
    
#     # obtain points and faces of the triangulation
#     points, faces, comb, critical_points_output, critical_points = critical_points(inputFilePath, x,y,z)
     
#     # put all inputs of the distance computation function together
#     inputs = []
#     for i in range(len(comb)):
#         inputs.append((i,points, faces, comb))
    
#     # time the paralell computation of distances
#     start = time.time()
#     with Pool(processes=8) as p:
#         results  = p.starmap(geo_dist, inputs)
#     finish = time.time()
#     print(finish-start)
         



