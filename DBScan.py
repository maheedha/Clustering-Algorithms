# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 05:13:06 2017

@author: MaHi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:06:12 2017

@author: MaHi
"""

import numpy as np
import scipy.spatial.distance as dist
import time
from sklearn.decomposition import PCA
import colorsys
import matplotlib.pyplot as plt
import queue

"""-------------------------------------------------------------------------------------------
     Function Name  :  loadData
     In Parameter   :  file_name-- Input file from which data is read
     Out Parameters :  dataset  -- numpy array of all data points
                       labels   -- Ground truth extracted from input file  
     Description    :  This function takes the input file, reads the data from it, 
                       separates the input file content into working data set which 
                       contains the datapoints and the ground truth
   -------------------------------------------------------------------------------------------
"""
def loadData (file_name):
    #Load the data set    
    results = []
    
    with open(file_name) as inputfile:
        for line in inputfile:
            results.append(line.strip().split('\t'))
    
    # Data preprocesing        
    myArray = np.asarray(results)
    
    #Separate the raw data into dataset and labels
    dataset = np.delete(myArray, [0,1], axis=1)
    
    # convert the numpy array to float
    dataset = dataset.astype(np.float)
    
    #First coulmn contains the ground truth
    labels = myArray[:,1]

    return dataset,labels


"""-------------------------------------------------------------------------------------------
     Function Name  :  distanceMatrix
     In Parameter   :  dataset -- numpy array of all data points 
     Description    :  This function computes the eucledian distance between every data point
                       and stores them in the form of a matrix
   -------------------------------------------------------------------------------------------
"""
def distanceMatrix(dataset) :
        #Calculate Distance Matrix
    samples = dataset.shape[0]
    distMat = np.empty([samples,samples])
    for i in range(samples) :
        for j in range(samples) :
                distMat[i][j] = dist.euclidean(dataset[i],dataset[j])
    return distMat

"""-------------------------------------------------------------------------------------------
     Function Name  :  dbscan
     In Parameter   :  dataset   -- numpy array of all data points
                       eps       -- predefined epsilon value
                       Minpts    -- predefined minimum points
     Out Parameters :  result    -- clusters computed by the algorithm                   
     Description    :  This function implements the dbscan algorithm and returns
                       the list of datapoint's corresponding clusters
   -------------------------------------------------------------------------------------------
"""
def dbscan(dataset,eps,Minpts) :
    # Initialisation
    samples = dataset.shape[0]
    result = np.asarray(np.zeros((samples)))
    visited = np.zeros((samples), dtype=bool)   
    NeighborPts = []
    cluster =0
    
    #Distanace matrix call to get distMat
    distMat = distanceMatrix(dataset)
    
    for p in range(samples) :
        if not visited[p] :
            visited[p] = 'True'
            NeighborPts = regionQuery(p,eps,distMat)
            if len(NeighborPts) < Minpts :
                result[p] = -1
            else :
                cluster +=1
                
                result = expandCluster(p, NeighborPts, cluster, eps, Minpts, visited,distMat,result) 
    return result

"""-------------------------------------------------------------------------------------------
     Function Name  :  regionQuery
     In Parameter   :  p         -- input datapoint
                       eps       -- predefined epsilon value
     Out Parameters :  points    -- Neighbouring points of p                   
     Description    :  This function computes the neighbouring datapoints
                       of input point p and returns them as a list
   -------------------------------------------------------------------------------------------
"""
def regionQuery(p,eps,distMat) :
    #Initialisation
    points =[]
    
    #compute neighbors based on epsilon
    for i in range(distMat.shape[0]) :
        if(distMat[p][i] <= eps) :
            points.append(i)
            
    return points

"""-------------------------------------------------------------------------------------------
     Function Name  :  expandCluster
     In Parameter   :  p            -- input datapoint
                       NeighborPts  -- Neighbouring points of p
                       cluster      -- cluster of input datapoint p
                       eps          -- predefined epsilon value
                       Minpts       -- predefined minimum points
                       visited      -- Boolean array which says whether input data point is visited or not
                       distMat      -- Distance Matrix
     Out Parameters :  result       -- clusters computed by the algorithm                    
     Description    :  This function checks the neighboring points in the NeighborPts list, mark them as visited
                       and assign them to a cluster
   -------------------------------------------------------------------------------------------
"""
def expandCluster(p, NeighborPts, cluster, eps, MinPts, visited,distMat,result) :
    result[p] = cluster 
    q= queue.Queue()
    for i in NeighborPts :
        q.put(i)
    while(q.qsize()>0):
        ele = q.get()
        if visited[ele] == False :
            visited[ele] = 'True'
            NeighborPts_ = regionQuery(ele,eps,distMat)
            if len(NeighborPts_) >= MinPts :
                for j in NeighborPts_ :
                    q.put(j)
        if result[ele]==0 :
            result[ele] = cluster
    return result
          
"""-------------------------------------------------------------------------------------------
     Function Name  :  validation
     In Parameter   :  result - clusters identified by the kmeans algorithm
                       labels  -- ground truth
     Description    :  This function computes the jaccard coffecient,randindex and prints
                       them to console
   -------------------------------------------------------------------------------------------
"""
def validation(result,labels) :
    num = len(result)
    my_mat = np.empty([num,num])
    g_truth = np.empty([num,num])
    m00 = 0
    m01 =0 
    m10=0
    m11 =0
    
    #Build the clustered matrix
    for i in range (num) :
        for j in range (num) :
            if(result[i] == result[j]) :
                my_mat[i][j] = 1
            else :
                 my_mat[i][j] = 0
                 
    #Build ground truth matrix             
    for i in range (num) :
        for j in range (num) :
            if(labels[i] == labels[j]) :
                g_truth[i][j] = 1
            else :
                 g_truth[i][j] = 0
              
    #compute jaccard coffecient
    for i in range (num) :
        for j in range (num) :
            if(my_mat[i][j] == 0 and g_truth[i][j]==0) :
                m00+=1
            elif (my_mat[i][j] == 0 and g_truth[i][j]==1):
                m01+=1
            elif (my_mat[i][j] == 1 and g_truth[i][j]==0):
                m10+=1
            else :
                m11+=1
                
    j_coff = float(m11)/(m11+m10+m01)
    randi = float(m11+m00)/(m11+m10+m01+m00)
    
    print("Jaccard-Coffecient ",j_coff)
    print("RandIndex ",randi)
    
"""-------------------------------------------------------------------------------------------
     Function Name  :  pca_visualise
     In Parameter   :  result - clusters identified by the kmeans algorithm
     Out Parameters :  dataset  -- numpy array of all data points
     Description    :  This function implements PCA and computes maximum variance along
                       2 dimensions and plot the data points using scatter plot
   -------------------------------------------------------------------------------------------
"""                
def pca_visualise(result,dataset) :
    
    RGBcolors = []
    
    # Compute PCA
    pca=PCA(n_components =2)
    dim_red = pca.fit_transform(dataset)
    
    
    # Get the list of the clusters
    for i in range(len(result)) :
        cluster = set(result)
        clusters= list(cluster)   
    
    #Compute the colors
    for x in range(len(clusters)) :
        h=(x*1.0/(len(clusters)))-0.1
        s=0.5
        v=0.5
        RGBcolors.append(colorsys.hsv_to_rgb (h, s, v))
    col = zip(clusters,RGBcolors)

    # Scatter plot   
    for name,color in col :
       
        plt.scatter(
        dim_red[result==name,0],
        dim_red[result==name,1],
        label=name,
        c=color,
        )
    plt.title('DBScan')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
#start time of the execution
start_time =  time.time()

#Load Input file
inputfile = "new_dataset_1.txt"

#Load the data to get the dataset and labels
dataset,labels = loadData(inputfile)

#DBScan algorithm implementation
eps = 0.8
Minpts =4
result = dbscan(dataset,eps,Minpts)

#compute jaccard and RandIndex
validation(result,labels)

#visualize using scatter plot
pca_visualise(result,dataset)

# print execution time of the algorithm
print("Executuion Time ", time.time()-start_time)


