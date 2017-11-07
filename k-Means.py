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

"""-------------------------------------------------------------------------------------------
     Function Name  :  loadData
     In Parameter   :  file_name -- Input file from which data is read
     Out Parameters :  dataset   -- numpy array of all data points
                       labels    -- Ground truth extracted from input file  
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
     Function Name  :  initialise
     In Parameter   :  k -- Number of clusters
     Out Parameters :  dataset   -- numpy array of all data points
                       centroids    -- Returns the initial centroids  
     Description    :  This function takes number of clusters as input, implements min-max method
                       of initialising centroids and returns them
   -------------------------------------------------------------------------------------------
"""
def initialise (k,dataset) :
    # Initialisation
    attributes = dataset.shape[1]
    centroids = np.zeros((k,attributes))
    
    #select the random value between the min and max along each column of data set
    for i in range(attributes) :
        range_ =max(dataset[:,i])-min(dataset[:,i])
        centroids[:,i] = min(dataset[:,i]) + (range_ * np.random.rand(k))
        
    return centroids

"""-------------------------------------------------------------------------------------------
     Function Name  :  kmeans
     In Parameter   :  k -- Number of clusters
                       dataset   -- numpy array of all data points
                       centroids  -- list of initial centroids
     Out Parameters :  result  -- list of computed clusters for each data point
                       count   -- No.of iterations before converge
     Description    :  This function implements the k means algorithm
   -------------------------------------------------------------------------------------------
"""
def kmeans(k,datset,centroids) :
    
    #Initialisation
    samples = dataset.shape[0]
    result = np.asarray(np.zeros((samples)))
    count =0
    converged = True
    while(converged) :
        converged = False
        for i in range(samples): 
            minimumdist = np.finfo(float).max
            cluster = -1
            
    # Compute Distance            
            for j in range(k):
                if minimumdist > dist.euclidean(centroids[j,:],dataset[i,:]):
                    minimumdist = dist.euclidean(dataset[i,:],centroids[j,:])
                    cluster = j      
# Convergence check                    
            if result[i] != cluster :
                converged = True
                
    #Assign cluster to the record
            result[i] = cluster
        
    #Recalculation of Centroid
        for index in range(k):
            points = np.where(result[:]==index)
            array = dataset[points]
            centroids[index,:] = np.mean(array,axis=0)
            
        count+=1
        
    return result,count
          

"""-------------------------------------------------------------------------------------------
     Function Name  :  validation
     In Parameter   :  result - clusters identified by the kmeans algorithm
                       labels  -- ground truth
     Description    :  This function computes the jaccard coffecient,randindex and prints
                       them to console
   -------------------------------------------------------------------------------------------
"""
def validation(result,labels) :
    #Initialisation
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
                
    j_coff = m11/(m11+m10+m01)
    rand   = (m11+m00)/(m11+m00+m10+m01)
    
    print("Jaccard-Coffecient ",j_coff)
    print("Rand Index", rand)
    
                
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
    
    # Compute PCA ujsing sci-learn library
    pca=PCA(n_components =2)
    dim_red = pca.fit_transform(dataset)
    
    
    # Get the list of the clusters
    for i in range(len(result)) :
        cluster = set(result)
        clusters= list(cluster)
    
    #Compute the colors
    for x in range(len(clusters)) :
        h=(x*1.0/(len(clusters)))
        s=0.6
        v=0.6
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
    plt.title('kMeans')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    
                
# Start time of the execution of code    
start_time =  time.time()

#Load Input file
inputfile = "new_dataset_1.txt"

# Set value of K
k=4
print("K-Value", k)

#Load the data to get the dataset and labels
dataset,labels = loadData(inputfile)



#Manually enter centroids
dp= [1,2,3,4]
centroids = np.empty([len(dp),dataset.shape[1]])

for i in range(len(dp)):
    centroids[i] = dataset[dp[i]-1]
    

#K Means algorithm implementation

result,count = kmeans(k,dataset,centroids)
print("Iterations before converge",count)

#compute jaccard and RandIndex
validation(result,labels)

#visualize using scatter plot
pca_visualise(result,dataset)

#End time 
print("Executuion Time ", time.time()-start_time)


