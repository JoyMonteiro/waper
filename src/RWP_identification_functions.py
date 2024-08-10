import vtk
import numpy as np
from sklearn import cluster
from scipy import spatial
import math
import vtk
from collections import defaultdict
from scipy.spatial import distance
import networkx as nx
import itertools
import xarray as xr
import pyvista as pv

import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from myCmap import joyNLDivCmaptest
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean
from matplotlib.gridspec import GridSpec

import shapely
from shapely.geometry import Polygon
from shapely.geometry import LineString
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from scipy.signal import hilbert
from scipy.fftpack import fft, ifft
from cartopy.util import add_cyclic_point


def addMaxData(lat,lon,scalar_values):
    
    r, c = scalar_values.shape
    check = np.zeros((r, c))
    is_max = np.zeros((r, c))
    vertex_identifiers = np.zeros(r*c)

    rect = pv.RectilinearGrid(lon, lat)
    scalar = scalar_values[::-1, :].ravel()
    rect.point_arrays['v'] = scalar

    count = 0
    k = 0

    for i in range(r):
        for j in range(c):

            vertex_identifiers[k] = k+1
            k += 1
            max_flag = 1

            if(check[i][j] == 1):
                continue

            else:
                if(j == 0):
                    for x in [i-1, i, i+1]:
                        for y in [c-1, j, j+1]:
                            if((0 <= x < r) and (0 <= y < c)):
                                if(scalar_values[x][y] > scalar_values[i][j]):
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

                if(j == c-1):
                    for x in [i-1, i, i+1]:
                        for y in [j-1, j, 0]:
                            if((0 <= x < r) and (0 <= y < c)):
                                if(scalar_values[x][y] > scalar_values[i][j]):
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

                else:
                    for x in [i-1, i, i+1]:
                        for y in [j-1, j, j+1]:
                            if((0 <= x < r) and (0 <= y < c)):
                                if(scalar_values[x][y] > scalar_values[i][j]):
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

            if(max_flag == 1):
                is_max[i][j] = 1
                check[i][j] = 1
                count+=1


    cell_number = rect.GetNumberOfCells()
    cell_id = np.zeros(cell_number)
    for i in range(cell_number):
        cell_id[i] = i

    rect.point_arrays['is max'] = is_max[::-1, :].ravel()
    rect.point_arrays['Vertex_id'] = vertex_identifiers
    rect.cell_arrays["Cell_V"] = cell_id
    #print("max_count", count) 
    return (rect,is_max)
    
    
def interpolateCellVals(inputs):
    numCells = inputs.GetNumberOfCells()
    scalar_v = inputs.GetPointData().GetArray("v")
    cell_scalars = vtk.vtkFloatArray()
    cell_scalars.SetNumberOfComponents(1)
    cell_scalars.SetNumberOfTuples(numCells)
    cell_scalars.SetName("Cell V")
    for i in range(numCells):
        cell = inputs.GetCell(i)
        num_points = cell.GetNumberOfPoints()
        func_value = 0
        for j in range(num_points):
            pid = cell.GetPointId(j)
            func_value += (scalar_v.GetTuple1(pid))
        func_value /= num_points
        cell_scalars.SetTuple1(i, func_value)
    inputs.GetCellData().AddArray(cell_scalars)
    return (inputs)


def clipDataset(dataset,scalar_name,scalar_val):
    clip_dataset=vtk.vtkClipDataSet()
    dataset.GetPointData().SetScalars(dataset.GetPointData().GetArray(scalar_name))
    clip_dataset.SetValue(scalar_val)
    clip_dataset.SetInputData(dataset)
    clip_dataset.Update()
    return (clip_dataset.GetOutput())

def addConnectivityData(dataset):
    connectivity_filter = vtk.vtkConnectivityFilter()
    connectivity_filter.SetInputData(dataset)
    connectivity_filter.SetExtractionModeToAllRegions()
    connectivity_filter.ColorRegionsOn()
    connectivity_filter.Update()
    return (connectivity_filter.GetOutput())

def extractSelectionIds(scalar_field, id_list):
    selectionNode = vtk.vtkSelectionNode()
    selectionNode.SetFieldType(1)
    selectionNode.SetContentType(4)
    selectionNode.SetSelectionList(id_list)
    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)
    extractSelection = vtk.vtkExtractSelection()
    extractSelection.SetInputData(0, scalar_field)
    extractSelection.SetInputData(1, selection)
    extractSelection.Update()
    return extractSelection.GetOutput()




def extractPosMaxIds(scalar_field,thresh):
    pos_max_ids = vtk.vtkIdTypeArray()
    num_pts = scalar_field.GetNumberOfPoints()
    is_max_arr = scalar_field.GetPointData().GetArray("is max")
    scalar_arr = scalar_field.GetPointData().GetArray("v")
    for i in range(num_pts):
        if(is_max_arr.GetTuple1(i) == 1 and scalar_arr.GetTuple1(i) >= thresh):
            pos_max_ids.InsertNextValue(i)
    return pos_max_ids


def clusterMax(scalar_field, connectivity_clipped_scalar_field, max_points):
    # import scalar field and critical point data objects
    scalar_field = connectivity_clipped_scalar_field
    maxima_points = max_points
    base_field = scalar_field

    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(scalar_field)
    geometryFilter.Update()
    scalar_field = geometryFilter.GetOutput()

    triangleFilter = vtk.vtkTriangleFilter()
    triangleFilter.SetInputData(scalar_field)
    triangleFilter.Update()
    scalar_field = triangleFilter.GetOutput()

    maxima_point_id = maxima_points.GetPointData().GetArray("vtkOriginalPointIds")
    num_points = maxima_points.GetNumberOfPoints()
   
    maxima_regions = maxima_points.GetPointData().GetArray("RegionId")

    point_region_id = scalar_field.GetPointData().GetArray("RegionId")
    num_regions = int(np.max(point_region_id)+1)

    dist_matrix = np.full((num_points, num_points), 400)

    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(scalar_field)

    #region_distance_array=[[[0 for col in range(0)]for row in range(0)]for clusters in range(num_regions)]

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(base_field)
    locator.BuildLocator()
    cellIds = vtk.vtkIdList()

    cell_v = base_field.GetCellData().GetArray("Cell V")

    co_ords = np.empty((0, 3))
    for i in range(num_points):
        co_ords = np.append(co_ords, [maxima_points.GetPoint(i)], axis=0)

    for i in range(num_points):
        for j in range(i+1, num_points):
            min_v = 1000
            max_v = 0
            av_v = 0
            p0 = [0, 0, 0]
            p1 = [0, 0, 0]
            dist = 0.0
            region_1 = maxima_regions.GetTuple1(i)
            region_2 = maxima_regions.GetTuple1(j)
            if(region_1 != region_2):
                continue
            dijkstra.SetStartVertex(int(maxima_point_id.GetTuple1(i)))
            dijkstra.SetEndVertex(int(maxima_point_id.GetTuple1(j)))
            dijkstra.Update()
            pts = dijkstra.GetOutput().GetPoints()
            for ptId in range(pts.GetNumberOfPoints()-1):
                pts.GetPoint(ptId, p0)
                pts.GetPoint(ptId+1, p1)
                dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
            locator.FindCellsAlongLine(co_ords[i], co_ords[j], 0.001, cellIds)
            for k in range(cellIds.GetNumberOfIds()):
                if(cell_v.GetTuple1(cellIds.GetId(k)) < min_v):
                    min_v = cell_v.GetTuple1(cellIds.GetId(k))
                    min_cell_id = cellIds.GetId(k)
            dist_matrix[i][j] = dist_matrix[i][j]-min_v
            dist_matrix[j][i] = dist_matrix[i][j]
           

    region_array = [[0 for col in range(0)]for row in range(num_regions)]
    cluster_assign = np.full(num_points, 0)

    median_dist = -np.median(dist_matrix)

    for i in range(num_points):
        region_array[int(point_region_id.GetTuple1(
            int(maxima_point_id.GetTuple1(i))))].append(i)

    prev_max = 0

    for k in range(num_regions):
        if(len(region_array[k]) == 1):
            cluster_assign[region_array[k][0]] = prev_max
            prev_max += 1
            continue
        if(len(region_array[k]) == 2):
            cluster_assign[region_array[k][0]] = prev_max
            cluster_assign[region_array[k][1]] = prev_max
            prev_max += 1
            continue

        num_cluster = int(len(region_array[k]))
        new_dist = np.full((num_cluster, num_cluster), 0)
        

        for i in range(num_cluster):
            for j in range(i+1, num_cluster):
                new_dist[i][j] = dist_matrix[region_array[k]
                                             [i]][region_array[k][j]]
                new_dist[j][i] = new_dist[i][j]

        if(num_cluster == 0):
            continue

        sim_matrix = np.negative(new_dist)

        af_clustering = cluster.AffinityPropagation(preference=np.full(
            num_cluster, median_dist/5.0), affinity='precomputed')
        af_clustering.fit(sim_matrix)
        clusters = af_clustering.labels_ + prev_max
        prev_max = np.max(clusters)+1

        for i in range(num_cluster):
            cluster_assign[region_array[k][i]] = clusters[i]

    cluster_id = vtk.vtkIntArray()
    cluster_id.SetNumberOfComponents(1)
    cluster_id.SetNumberOfTuples(num_points)
    cluster_id.SetName("Cluster ID")

    for i in range(num_points):
        cluster_id.SetTuple1(i, cluster_assign[i])

    maxima_points.GetPointData().AddArray(cluster_id)
    return maxima_points


def addMinData(lat,lon,scalar_values):
    
    scalar_negative = np.negative(scalar_values)
    
    r, c = scalar_values.shape
    check = np.zeros((r, c))
    is_max = np.zeros((r, c))
    vertex_identifiers = np.zeros(r*c)

    rect = pv.RectilinearGrid(lon, lat)
    scalar = scalar_values[::-1, :].ravel()
    rect.point_arrays['v'] = scalar

    count = 0
    k = 0

    for i in range(r):
        for j in range(c):

            k += 1
            max_flag = 1

            if(check[i][j] == 1):
                continue

            else:
                if(j == 0):
                        for x in [i-1,i, i+1]:
                            for y in [c-1, j, j+1]:
                                if((0 <= x < r) and (0 <= y < c)):
                                    if(scalar_values[x][y] < scalar_values[i][j]):
                                        max_flag = 0
                                    else:
                                        check[x][y] = 1   
            
                if(j == c-1):
                        for x in [i-1,i, i+1]:
                            for y in [j-1, j, 0]:
                                if((0 <= x < r) and (0 <= y < c)):
                                    if(scalar_values[x][y] < scalar_values[i][j]):
                                        max_flag = 0
                                    else:
                                        check[x][y] = 1
                                        
                    
                else:
                        for x in [i-1,i, i+1]:
                            for y in [j-1, j, j+1]:
                                if((0 <= x < r) and (0 <= y < c)):
                                    if(scalar_values[x][y] < scalar_values[i][j]):
                                        max_flag = 0
                                    else:
                                        check[x][y] = 1
                    
                if(max_flag == 1 and i!=0):
                    is_max[i][j] = 1
                    check[i][j] = 1
                    count+=1
                    
    cell_number = rect.GetNumberOfCells()
    cell_id = np.zeros(cell_number)
    for i in range(cell_number):
        cell_id[i] = i

    rect.point_arrays['is min'] = is_max[::-1, :].ravel()
    rect.point_arrays['Vertex_id'] = vertex_identifiers
    rect.cell_arrays["Cell_V"] = cell_id
    #print("min points", count)
    return (rect,is_max)


def interpolateCellVals_min(inputs):
	numCells=inputs.GetNumberOfCells()
	scalar_v=inputs.GetPointData().GetArray("v")
	cell_scalars=vtk.vtkFloatArray()
	cell_scalars.SetNumberOfComponents(1)
	cell_scalars.SetNumberOfTuples(numCells)
	cell_scalars.SetName("Cell V")
	for i in range(numCells):
	    cell=inputs.GetCell(i)
	    num_points=cell.GetNumberOfPoints()
	    func_value=0
	    for j in range(num_points):
	        pid=cell.GetPointId(j)
	        func_value+=(scalar_v.GetTuple1(pid))
	    func_value/=num_points
	    cell_scalars.SetTuple1(i,func_value)    
	    inputs.GetCellData().AddArray(cell_scalars)
	return (inputs)

def clipDataset_min(dataset,scalar_name,scalar_val):
	clip_dataset=vtk.vtkClipDataSet()
	dataset.GetPointData().SetScalars(dataset.GetPointData().GetArray("v"))
	clip_dataset.SetValue(scalar_val)
	clip_dataset.SetInputData(dataset)
	clip_dataset.InsideOutOn()
	clip_dataset.Update()
	return (clip_dataset.GetOutput())	

def addConnectivityData_min(dataset):
	connectivity_filter=vtk.vtkConnectivityFilter()
	connectivity_filter.SetInputData(dataset)
	connectivity_filter.SetExtractionModeToAllRegions()
	connectivity_filter.ColorRegionsOn()
	connectivity_filter.Update()
	return (connectivity_filter.GetOutput())

def extractPosMinIds(scalar_field,thresh):
    pos_max_ids = vtk.vtkIdTypeArray()
    num_pts = scalar_field.GetNumberOfPoints()
    is_max_arr = scalar_field.GetPointData().GetArray("is min")
    scalar_arr = scalar_field.GetPointData().GetArray("v")
    for i in range(num_pts):
        if(is_max_arr.GetTuple1(i) == 1 and scalar_arr.GetTuple1(i) <= thresh):
            pos_max_ids.InsertNextValue(i)
    return pos_max_ids

def extractSelectionIds_min(scalar_field,id_list):
	selectionNode=vtk.vtkSelectionNode()
	selectionNode.SetFieldType(1)
	selectionNode.SetContentType(4)
	selectionNode.SetSelectionList(id_list)
	selection=vtk.vtkSelection()
	selection.AddNode(selectionNode)
	extractSelection=vtk.vtkExtractSelection()
	extractSelection.SetInputData(0,scalar_field)
	extractSelection.SetInputData(1,selection)
	extractSelection.Update()
	return extractSelection.GetOutput()	


def clusterMin(scalar_field,connectivity_clipped_scalar_field,max_points):
	#import scalar field and critical point data objects
	scalar_field=connectivity_clipped_scalar_field
	maxima_points=max_points
	base_field=scalar_field

	geometryFilter=vtk.vtkGeometryFilter()
	geometryFilter.SetInputData(scalar_field)
	geometryFilter.Update()
	scalar_field=geometryFilter.GetOutput()

	triangleFilter=vtk.vtkTriangleFilter()
	triangleFilter.SetInputData(scalar_field)
	triangleFilter.Update()
	scalar_field=triangleFilter.GetOutput()

	maxima_point_id=maxima_points.GetPointData().GetArray("vtkOriginalPointIds")
	num_points=maxima_points.GetNumberOfPoints()

	maxima_regions=maxima_points.GetPointData().GetArray("RegionId")
	point_region_id=scalar_field.GetPointData().GetArray("RegionId")
	num_regions = int(np.max(point_region_id)+1)

	dist_matrix=np.full((num_points,num_points),400)

	dijkstra=vtk.vtkDijkstraGraphGeodesicPath()
	dijkstra.SetInputData(scalar_field)

	#region_distance_array=[[[0 for col in range(0)]for row in range(0)]for clusters in range(num_regions)]

	locator=vtk.vtkCellLocator()
	locator.SetDataSet(base_field)
	locator.BuildLocator()
	cellIds=vtk.vtkIdList()

	cell_v=base_field.GetCellData().GetArray("Cell V")

	co_ords=np.empty((0,3))
	for i in range(num_points):
		co_ords=np.append(co_ords,[maxima_points.GetPoint(i)],axis=0)

	for i in range(num_points):
	    for j in range(i+1,num_points):
	        min_v=1000
	        max_v=-1000
	        av_v=0
	        p0 = [0,0,0]
	        p1 = [0,0,0]
	        dist = 0.0
	        region_1=maxima_regions.GetTuple1(i)
	        region_2=maxima_regions.GetTuple1(j)
	        if(region_1!=region_2):
	            continue
	        dijkstra.SetStartVertex(int(maxima_point_id.GetTuple1(i)))
	        dijkstra.SetEndVertex(int(maxima_point_id.GetTuple1(j)))
	        dijkstra.Update()
	        pts = dijkstra.GetOutput().GetPoints()
	        for ptId in range(pts.GetNumberOfPoints()-1):
	            pts.GetPoint(ptId, p0)  
	            pts.GetPoint(ptId+1, p1)
	            dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
	        dist_matrix[i][j]=dist
	        dist_matrix[j][i]=dist
	        locator.FindCellsAlongLine(co_ords[i],co_ords[j],0.001,cellIds)
	        for k in range(cellIds.GetNumberOfIds()):
	            if(cell_v.GetTuple1(cellIds.GetId(k))>max_v):
	                max_v=cell_v.GetTuple1(cellIds.GetId(k))
	        dist_matrix[i][j]=dist_matrix[i][j]+max_v
	        dist_matrix[j][i]=dist_matrix[i][j]

	
	region_array=[[0 for col in range(0)]for row in range(num_regions)]
	cluster_assign=np.full(num_points,0)
	
	median_dist=-np.median(dist_matrix)
	
	for i in range(num_points):
	    region_array[int(point_region_id.GetTuple1(int(maxima_point_id.GetTuple1(i))))].append(i)
	    
	
	prev_max=0
	
	for k in range(num_regions):
	    if(len(region_array[k])==1):
	        cluster_assign[region_array[k][0]]=prev_max
	        prev_max+=1 
	        continue
	    if(len(region_array[k])==2):
	        cluster_assign[region_array[k][0]]=prev_max
	        cluster_assign[region_array[k][1]]=prev_max
	        prev_max+=1 
	        continue
	       
	    num_cluster=int(len(region_array[k]))
	    new_dist=np.full((num_cluster,num_cluster),0)
	
	    for i in range(num_cluster):
	        for j in range(i+1,num_cluster):
	            new_dist[i][j]=dist_matrix[region_array[k][i]][region_array[k][j]]
	            new_dist[j][i]=new_dist[i][j]
	    
	        
	    if(num_cluster==0):
	        continue
	
	    sim_matrix=np.negative(new_dist)
	
	    af_clustering = cluster.AffinityPropagation(preference=np.full(num_cluster,median_dist/5.0),affinity='precomputed')
	    af_clustering.fit(sim_matrix)
	    clusters=af_clustering.labels_ + prev_max
	    prev_max=np.max(clusters)+1
	
	    for i in range(num_cluster):
	        cluster_assign[region_array[k][i]]=clusters[i]
	
	
	cluster_id=vtk.vtkIntArray()
	cluster_id.SetNumberOfComponents(1)
	cluster_id.SetNumberOfTuples(num_points)
	cluster_id.SetName("Cluster ID")
	
	
	for i in range(num_points):
	    cluster_id.SetTuple1(i,cluster_assign[i])
	
	
	#clustered_output=self.GetOutput()
	maxima_points.GetPointData().AddArray(cluster_id)
	#clustered_output.ShallowCopy(maxima_points)
	return maxima_points

def hav_distance(lat1, lon1, lat2, lon2):

    r_earth = 6371.0

    circum = 2*np.pi*r_earth*np.cos(np.radians(30))

    dlat = math.radians(lat1 - lat2)

    dlon = math.radians(lon1 - lon2)

    a = (math.sin(dlat/2))**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * (math.sin(dlon/2))**2
    c = 2 * np.arctan2(math.sqrt(a), math.sqrt(1-a))
    distance = r_earth * c

    return distance


def getIsoContour(scalar_field, val):
    contourFilter = vtk.vtkContourFilter()
    scalar_field.GetPointData().SetScalars(scalar_field.GetPointData().GetArray("v"))
    contourFilter.SetValue(0, val)
    contourFilter.SetInputData(scalar_field)
    contourFilter.Update()
    return contourFilter.GetOutput()


def computeGradients(scalar_field):
    gradientFilter = vtk.vtkGradientFilter()
    scalar_field.GetPointData().SetScalars(scalar_field.GetPointData().GetArray("v"))
    gradientFilter.SetInputData(scalar_field)
    gradientFilter.Update()
    return gradientFilter.GetOutput()


def computeAssocGraph(max_points, min_points, iso_contour):
    # creating an empty graph
    G = nx.Graph()

    num_contour_pts = iso_contour.GetNumberOfPoints()
    point_grad = iso_contour.GetPointData().GetArray("Gradients")
    max_cluster_ids = max_points.GetPointData().GetArray("Cluster ID")
    min_cluster_ids = min_points.GetPointData().GetArray("Cluster ID")
    num_max_pts = max_points.GetNumberOfPoints()
    num_min_pts = min_points.GetNumberOfPoints()
    num_max_clusters = int(np.max(max_cluster_ids)+1)
    num_min_clusters = int(np.max(min_cluster_ids)+1)
    cluster_max_arr = np.full(num_max_clusters, 0.0)
    cluster_min_arr = np.full(num_min_clusters, 100.0)
    cluster_max_point = np.full((num_max_clusters, 2), 0.0)
    cluster_min_point = np.full((num_min_clusters, 2), 0.0)
    assoc_index_array = np.full((num_max_clusters, num_min_clusters), 0.0)

    line_dir_array = np.full((num_max_clusters, num_min_clusters), 0.0)

    assoc_set = set()

    max_scalars = max_points.GetPointData().GetArray("v")
    min_scalars = min_points.GetPointData().GetArray("v")

    cluster_max_dict = defaultdict(list)
    cluster_min_dict = defaultdict(list)

    for i in range(num_max_pts):
        point_coords = max_points.GetPoint(i)
        cluster_id = max_cluster_ids.GetTuple1(i)
        scalar = max_scalars.GetTuple1(i)
        point_tuple = (point_coords, cluster_id, scalar)
        cluster_max_dict[cluster_id].append(point_tuple)
        if(cluster_max_arr[int(max_cluster_ids.GetTuple1(i))] < max_scalars.GetTuple1(i)):
            cluster_max_arr[int(max_cluster_ids.GetTuple1(i))] = max_scalars.GetTuple1(i)
            cluster_max_point[int(max_cluster_ids.GetTuple1(i))][0] = max_points.GetPoint(i)[0]
            cluster_max_point[int(max_cluster_ids.GetTuple1(i))][1] = max_points.GetPoint(i)[1]
   
    for i in range(num_min_pts):
        point_coords = min_points.GetPoint(i)
        cluster_id = min_cluster_ids.GetTuple1(i)
        scalar = min_scalars.GetTuple1(i)
        point_tuple = (point_coords, cluster_id, scalar)
        cluster_min_dict[cluster_id].append(point_tuple)
        if(cluster_min_arr[int(min_cluster_ids.GetTuple1(i))] > min_scalars.GetTuple1(i)):
            cluster_min_arr[int(min_cluster_ids.GetTuple1(i))] = min_scalars.GetTuple1(i)
            cluster_min_point[int(min_cluster_ids.GetTuple1(i))][0] = min_points.GetPoint(i)[0]
            cluster_min_point[int(min_cluster_ids.GetTuple1(i))][1] = min_points.GetPoint(i)[1]
    
    
    assoc_dict = {(-1, -1): 0}

    max_pt_dict = defaultdict(list)
    min_pt_dict = defaultdict(list)

    i = 0

    for i in range(num_contour_pts):
        contour_point = iso_contour.GetPoint(i)
        max_dist = 1000
        min_dist = 1000
        max_id = -1
        min_id = -1
        curr_max_dir_deriv = 0
        curr_min_dir_deriv = 0
        grad_vector = [point_grad.GetTuple3(i)[0], point_grad.GetTuple3(i)[1]]
        curr_max_scalar = 0
        curr_min_scalar = 0

        for j in range(num_max_pts):
            max_point = max_points.GetPoint(j)
            curr_max_id = max_cluster_ids.GetTuple1(j)
            max_dir_vector = [max_point[0]-contour_point[0],max_point[1]-contour_point[1]]
            max_dir_deriv = max_dir_vector[0] * grad_vector[0]+max_dir_vector[1]*grad_vector[1]
            curr_max_dist = (max_dir_vector[0]**2+max_dir_vector[1]**2)**0.5
            #if(max_dir_deriv>0):
            if(curr_max_dist < max_dist):
                max_dist = curr_max_dist
                max_id = curr_max_id
                curr_max_dir_deriv = max_dir_deriv
                curr_max_scalar = max_scalars.GetTuple1(j)
                curr_max_x = max_point[0]

        max_id = int(max_id)
#         point_cords_max = cluster_max_point[max_id]
#         point_tuple_max = (point_cords_max, max_id, cluster_max_arr[max_id])

        for j in range(num_min_pts):
            min_point = min_points.GetPoint(j)
            curr_min_id = min_cluster_ids.GetTuple1(j)
            min_dir_vector = [min_point[0]-contour_point[0],min_point[1]-contour_point[1]]
            min_dir_deriv = min_dir_vector[0]*grad_vector[0]+min_dir_vector[1]*grad_vector[1]
            curr_min_dist = (min_dir_vector[0]**2+min_dir_vector[1]**2)**0.5
            #if(min_dir_deriv > 0):
            if(curr_min_dist < min_dist):
                min_dist = curr_min_dist
                min_id = curr_min_id
                curr_min_dir_deriv = min_dir_deriv
                curr_min_scalar = min_scalars.GetTuple1(j)
                curr_min_x = min_point[0]

        min_id = int(min_id)
#         point_cords_min = cluster_min_point[min_id]
#         point_tuple_min = (point_cords_min, min_id, cluster_min_arr[min_id])
        if(max_id != -1 and min_id!= -1):
            assoc_set.add((int(max_id), int(min_id)))
    count = 0
    
    for elem in assoc_set:
        count+=1
        max_id = elem[0]
        min_id = elem[1]
        max_centre = cluster_max_point[max_id]
        min_centre = cluster_min_point[min_id]
        max_scalar = cluster_max_arr[max_id]
        min_scalar = cluster_min_arr[min_id]
        if(min_id == 0):
            min_id = 100
        G.add_node(max_id, coords=max_centre, cluster_id=max_id,scalar=max_scalar, dictionary=cluster_max_dict[max_id])
        if(min_id ==  100):
        	G.add_node(-min_id, coords=min_centre,
                       cluster_id=min_id,scalar=min_scalar,
                       dictionary=cluster_min_dict[0])
        else:
        	G.add_node(-min_id, coords=min_centre,
                       cluster_id=min_id,scalar=min_scalar,
                       dictionary=cluster_min_dict[min_id])
        G.add_edge(max_id, -min_id, weight=0)
        #print("no. of associations", count)
    return G


def edgeWeight(G, max_id, min_id):

    scalar_tol = 30

    max_scalar = G.nodes[max_id]["scalar"]
    min_scalar = -G.nodes[min_id]["scalar"]

    cluster_max_pts = G.nodes[max_id]["dictionary"]
    cluster_min_pts = G.nodes[min_id]["dictionary"]

    curr_dist = 0.0

    edge_wt = 0.0
    high_val_flag = 0

    if(max_scalar > 50 and min_scalar > 50):
        high_val_flag = 1
    
    for max_pt in cluster_max_pts:
        if(max_pt[2] < 30):
            continue
        if(max_pt[2] < max_scalar-scalar_tol and high_val_flag == 0):
            continue
    
        for min_pt in cluster_min_pts:
            if(min_pt[2] > -30):
                continue
            if(min_pt[2] > -min_scalar+scalar_tol and high_val_flag == 0):
                continue
            curr_dist = hav_distance(max_pt[0][0], max_pt[0][1], min_pt[0][0], min_pt[0][1])
            curr_weight = (max_pt[2]-min_pt[2])/curr_dist
    
            if(curr_weight > edge_wt):
                edge_wt = curr_weight
    
    return(edge_wt)


def scalarPrune(G, scalar_thresh):
    H = nx.Graph()
    edges = [e for e in G.edges()]
    for e in edges:
        start_node = e[0]
        end_node = e[1]
        min_node = 0
        min_scalar = 0
        if(start_node >= 0):
            if(G.nodes[start_node]["scalar"] < -G.nodes[end_node]["scalar"]):
                min_node = start_node
                min_scalar = G.nodes[start_node]["scalar"]

            else:
                min_node = end_node
                min_scalar = -G.nodes[end_node]["scalar"]
        else:
            if(-G.nodes[start_node]["scalar"] < G.nodes[end_node]["scalar"]):
                min_node = start_node
                min_scalar = -G.nodes[start_node]["scalar"]
            else:
                min_node = end_node
                min_scalar = G.nodes[end_node]["scalar"]
                
        if(min_scalar >= scalar_thresh and min_scalar <= 100):
            H.add_node(start_node, coords= G.nodes[start_node]["coords"], cluster_id= G.nodes[start_node]["cluster_id"],scalar=G.nodes[start_node]["scalar"], dictionary=G.nodes[start_node]["dictionary"])
            H.add_node(end_node, coords= G.nodes[end_node]["coords"], cluster_id= G.nodes[end_node]["cluster_id"],scalar=G.nodes[end_node]["scalar"], dictionary=G.nodes[end_node]["dictionary"])
            H.add_edge(start_node, end_node)
    return H


def edgePrune(G, thresh):
    H = nx.Graph()
    edges = [e for e in G.edges()]
    for e in edges:
        start_node = e[0]
        end_node = e[1]
        if(start_node >= 0):
            weight = edgeWeight(G, start_node, end_node)
        else:
            weight = edgeWeight(G, end_node, start_node)
        G[start_node][end_node]["weight"] = weight
        if(weight >= thresh and weight <= 1000):
            H.add_node(start_node, coords= G.nodes[start_node]["coords"], cluster_id= G.nodes[start_node]["cluster_id"],scalar=G.nodes[start_node]["scalar"], dictionary=G.nodes[start_node]["dictionary"])
            H.add_node(end_node, coords= G.nodes[end_node]["coords"], cluster_id= G.nodes[end_node]["cluster_id"],scalar=G.nodes[end_node]["scalar"], dictionary=G.nodes[end_node]["dictionary"])
            H.add_edge(start_node, end_node,weight = weight)
    return H

def GetRankedPaths(G):        

    H = nx.Graph()
    H = G
    path_list = []
    
    start_leaves=[x for x in H.nodes()]
    end_leaves=[x for x in H.nodes()]
    
    print(len(start_leaves), "number of nodes in graph for rankedPaths")
    
    for source in start_leaves:
        print(source)
        for sink in end_leaves:
            if(nx.has_path(G,source=source,target=sink)):
                for path in nx.all_simple_paths(G,source=source,target=sink):
                    path_list.append(path)


    path_wt_dict={}
    
    print(len(path_list), "number of paths found")
    
    for path in path_list:
        curr_wt = 0      
        for i in range(len(path)-1):        
            curr_wt += H[path[i]][path[i+1]]["weight"] 
        path_wt_dict[tuple(path)]=curr_wt

    top_paths=list(filter(lambda f: not any([(path_wt_dict[tuple(f)]<path_wt_dict[tuple(g)] and len(set(f)&set(g))!=0) for g in path_list]),path_list)) 

    return top_paths

def plot_points(lat,lon,extrema):

    fig = plt.figure(figsize=(14, 14))
    m6 = plt.axes(projection=ccrs.PlateCarree())
    # (x0, x1, y0, y1)        
    m6.coastlines()
    
    Y = lat
    X = lon-180
    Z = extrema
    cp = m6.pcolormesh(X,Y,Z,transform=ccrs.PlateCarree())

    count = 0
    r,c = extrema.shape
    for i in range(r):
        for j in range(c):
            if(extrema[i][j]==1):
                    count+=1

    plt.show()
    
def max_cluster_assign(max_points):

    num_points_max = max_points.GetNumberOfPoints()
    cluster_id_max = max_points.GetPointData().GetArray("Cluster ID")
    num_max_clusters = np.max(cluster_id_max)+1

    max_pt_dict = defaultdict(list)
    cluster_max_arr = np.full(num_max_clusters, 0.0)
    cluster_max_point = np.full((num_max_clusters, 2), 0.0)
    max_scalars = max_points.GetPointData().GetArray("v")


    for i in range(num_points_max):
            x,y,z = max_points.GetPoint(i)
            coords = [x,y]
            max_pt_dict[cluster_id_max.GetTuple1(i)].append(coords)
            if(cluster_max_arr[int(cluster_id_max.GetTuple1(i))] < max_scalars.GetTuple1(i)):
                cluster_max_arr[int(cluster_id_max.GetTuple1(i))] = max_scalars.GetTuple1(i)
                cluster_max_point[int(cluster_id_max.GetTuple1(i))][0] = max_points.GetPoint(i)[0]
                cluster_max_point[int(cluster_id_max.GetTuple1(i))][1] = max_points.GetPoint(i)[1]

    return(cluster_max_arr,cluster_max_point,max_pt_dict,num_max_clusters)

def min_cluster_assign(min_points):

    num_points_min = min_points.GetNumberOfPoints()
    cluster_id_min = min_points.GetPointData().GetArray("Cluster ID")
    num_min_clusters = np.max(cluster_id_min)+1

    min_pt_dict = defaultdict(list)
    cluster_min_arr = np.full(num_min_clusters, 0.0)
    cluster_min_point = np.full((num_min_clusters, 2), 0.0)
    min_scalars = min_points.GetPointData().GetArray("v")


    for i in range(num_points_min):
            x,y,z = min_points.GetPoint(i)
            coords = [x,y]
            min_pt_dict[cluster_id_min.GetTuple1(i)].append(coords)
            if(cluster_min_arr[int(cluster_id_min.GetTuple1(i))] > min_scalars.GetTuple1(i)):
                cluster_min_arr[int(cluster_id_min.GetTuple1(i))] = min_scalars.GetTuple1(i)
                cluster_min_point[int(cluster_id_min.GetTuple1(i))][0] = min_points.GetPoint(i)[0]
                cluster_min_point[int(cluster_id_min.GetTuple1(i))][1] = min_points.GetPoint(i)[1]

    return(cluster_min_arr,cluster_min_point,min_pt_dict,num_min_clusters)

        
def plot_clusters(lat,lon,scalar_values,pt_dict,num_clusters,cluster_point): 
    
    def plot_alphashape(ax, alphashape,color):
        coords = []
        if isinstance(alphashape, Polygon):
            coords = list(alphashape.boundary.coords)
        elif isinstance(alphashape, LineString):
            coords = list(alphashape.coords)

        for index in range(len(coords) - 1):
            x_values = [coords[index][0]-180, coords[index+1][0]-180]
            y_values = [coords[index][1], coords[index+1][1]]

            ax.plot(x_values, y_values, linewidth=2, color=color, transform=ccrs.PlateCarree())

    fig = plt.figure(figsize=(14, 14))
    m6 = plt.axes(projection=ccrs.PlateCarree())
    m6.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    m6.set_yticks([0, 30, 60, 90])
    # (x0, x1, y0, y1)        
    m6.coastlines(color="grey")

    Y = lat
    X = lon-180
    Z = scalar_values

    cp = m6.contourf(X,Y,Z,cmap = joyNLDivCmaptest,levels = 20, transform=ccrs.PlateCarree())
    cmap = cm.tab10
    norm = Normalize(vmin=0, vmax= num_clusters)

    for i in range(num_clusters):
            cluster_shape = alphashape.alphashape(pt_dict[i])
            for point in pt_dict[i]:
                if(cluster_point[i][0]==point[0] and 
                   cluster_point[i][1]==point[1]):
                    m6.scatter(cluster_point[i][0]-180,cluster_point[i][1], s=10,
                           color="k", transform=ccrs.PlateCarree())
                else:
                    m6.scatter(point[0]-180, point[1], s=10, color=cmap(norm(i)), transform=ccrs.PlateCarree())
            plot_alphashape(m6, cluster_shape, cmap(norm(i)))

    cax = fig.add_axes([m6.get_position().x1+0.01,m6.get_position().y0,0.02,m6.get_position().height])
    fig.colorbar(cp,cax=cax)       
    plt.show()
    
def all_clusters(lat,lon,scalar_values,max_pt_dict,min_pt_dict,num_max_clusters,num_min_clusters,
                 cluster_max_point,cluster_min_point):
    
    def plot_alphashape(ax, alphashape,color):
        coords = []
        if isinstance(alphashape, Polygon):
            coords = list(alphashape.boundary.coords)
        elif isinstance(alphashape, LineString):
            coords = list(alphashape.coords)

        for index in range(len(coords) - 1):
            x_values = [coords[index][0]-180, coords[index+1][0]-180]
            y_values = [coords[index][1], coords[index+1][1]]

            ax.plot(x_values, y_values, linewidth=2, color=color, transform=ccrs.PlateCarree())

    fig = plt.figure(figsize=(14, 14))
    m6 = plt.axes(projection=ccrs.PlateCarree())
    m6.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    m6.set_yticks([0, 30, 60, 90])
    # (x0, x1, y0, y1)        
    m6.coastlines(color="grey")
    
    Y = lat
    X = lon-180
    Z = scalar_values

    cp = m6.contourf(X,Y,Z,cmap =joyNLDivCmaptest,levels = 20, transform=ccrs.PlateCarree())
    #m6.contour(X,Y,Z, levels=[0], color='k', linewidth=2, transform=ccrs.PlateCarree())
    

    for i in range(num_min_clusters):
            cluster_shape = alphashape.alphashape(min_pt_dict[i])
            for point in min_pt_dict[i]:
                if(cluster_min_point[i][0]==point[0] and 
                   cluster_min_point[i][1]==point[1]):
                    m6.scatter(cluster_min_point[i][0]-180,cluster_min_point[i][1], s=10,
                           color="k", transform=ccrs.PlateCarree())
                else:
                    m6.scatter(point[0]-180, point[1], s=10, color="r", transform=ccrs.PlateCarree())
            plot_alphashape(m6, cluster_shape, "r")

    for i in range(num_max_clusters):
            cluster_shape = alphashape.alphashape(max_pt_dict[i])
            for point in max_pt_dict[i]:
                if(cluster_max_point[i][0]==point[0] and 
                   cluster_max_point[i][1]==point[1]):
                    m6.scatter(cluster_max_point[i][0]-180,cluster_max_point[i][1], s=10,
                           color="k", transform=ccrs.PlateCarree())
                else:
                    m6.scatter(point[0]-180, point[1], s=10, color="g", transform=ccrs.PlateCarree())
            plot_alphashape(m6, cluster_shape, "g")

    cax = fig.add_axes([m6.get_position().x1+0.01,m6.get_position().y0,0.02,m6.get_position().height])
    fig.colorbar(cp,cax=cax)       
    plt.show()
    
def plot_graph(lat,lon,scalar_values,graph):

    fig = plt.figure(figsize=(14, 14))
    m6 = plt.axes(projection=ccrs.PlateCarree())
    m6.set_xticks([0, 60, 120, 180, -60, -120, -180], crs=ccrs.PlateCarree())
    m6.set_yticks([0, 30, 60, 90], crs=ccrs.PlateCarree())
    # (x0, x1, y0, y1)        
    m6.coastlines(color='grey')

    Y = lat
    X = lon-180

    Z = scalar_values
    m6.contour(X,Y,Z, levels=[0], color='k', linewidth=2, transform=ccrs.PlateCarree())
    cp = m6.contourf(X,Y,Z,cmap=joyNLDivCmaptest,levels = 20,transform=ccrs.PlateCarree())

    count = 0

    display_graph = graph
    edges = [e for e in display_graph.edges()]
    for e in edges:
        count+=1
        if(e[0]>=0):
            m6.scatter(display_graph.nodes[e[0]]["coords"][0]-180,
                       display_graph.nodes[e[0]]["coords"][1],color="r",
                       transform=ccrs.PlateCarree(),zorder = 1)
            m6.scatter(display_graph.nodes[e[1]]["coords"][0]-180,
                       display_graph.nodes[e[1]]["coords"][1],color="b",transform=ccrs.PlateCarree(),zorder = 1)
        else:
            m6.scatter(display_graph.nodes[e[1]]["coords"][0]-180,
                       display_graph.nodes[e[1]]["coords"][1],color="r",transform=ccrs.PlateCarree(),zorder = 1)
            m6.scatter(display_graph.nodes[e[0]]["coords"][0]-180,
                       display_graph.nodes[e[0]]["coords"][1],color="b",
                       transform=ccrs.PlateCarree(),zorder = 1)
        x = [display_graph.nodes[e[0]]["coords"][0]-180,display_graph.nodes[e[1]]["coords"][0]-180]
        y = [display_graph.nodes[e[0]]["coords"][1],display_graph.nodes[e[1]]["coords"][1]]
        # plotting the line 1 points 
        m6.plot(x, y,color = "g",transform=ccrs.PlateCarree(),zorder = 1)


    cax = fig.add_axes([m6.get_position().x1+0.01,m6.get_position().y0,0.02,m6.get_position().height])
    fig.colorbar(cp,cax=cax)
    plt.show()
    
def plot_ranked_graph(lat,lon,scalar_values,ranked_paths_graph,pruned_assoc_graph):

    fig = plt.figure(figsize=(14, 14))
    m6 = plt.axes(projection=ccrs.PlateCarree())
    m6.set_xticks([0, 60, 120, 180, -60, -120, -180], crs=ccrs.PlateCarree())
    m6.set_yticks([0, 30, 60, 90], crs=ccrs.PlateCarree())
    # (x0, x1, y0, y1)        
    m6.coastlines(color='grey')

    Y = lat
    X = lon-180

    Z = scalar_values
    m6.contour(X,Y,Z, levels=[0], color='k', linewidth=2, transform=ccrs.PlateCarree())
    cp = m6.contourf(X,Y,Z,cmap=joyNLDivCmaptest,levels = 20,transform=ccrs.PlateCarree())


    cmap = cm.tab20b
    norm = Normalize(vmin=0, vmax=17)

    display_graph = ranked_paths_graph
    G = pruned_assoc_graph
    for i in range(len(display_graph)):
        for j in range(len(display_graph[i])-1):
            x = [G.nodes[display_graph[i][j]]["coords"][0]-180,G.nodes[display_graph[i][j+1]]["coords"][0]-180]
            y = [G.nodes[display_graph[i][j]]["coords"][1],G.nodes[display_graph[i][j+1]]["coords"][1]]
            # plotting the line 1 points 
            m6.plot(x, y,color = "g",transform=ccrs.PlateCarree(),zorder = 1)        


    cax = fig.add_axes([m6.get_position().x1+0.01,m6.get_position().y0,0.02,m6.get_position().height])
    fig.colorbar(cp,cax=cax)
    plt.show()
