# Assignment 3 #

import numpy as np 
import matplotlib.pyplot as plt

#### Ex 1.1 ####

# Centering function
def center(data):
    """
    Center the data.
    """
    mean = np.mean(data, axis=0)
    Cdata = (data - mean)
    return Cdata

# Standardization function
def standardize(data):    
    """
    Center and normalize the dataset.
    """
    # Obtain center and scaler
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    # Centering and normalizing
    Sdata = (data - mean) / std
    return Sdata

## PCA
def pca(data):
    """
    Take as argument data with column 
    as features and rows as observations.
    Return eigenvectors and eigenvalues 
    both sorted from eigenvalues (variance).
    """
    # Obtain covariance matrix
    sigma = np.cov(data.T) # np.cov wants features as rows and observations as columns (so transpose the data)
    # Obtain sorted eigenvalues and eigenvectors
    eigvals, eigvect = np.linalg.eig(sigma)
    isorted = eigvals.argsort()[::-1]
    sorted_eigvals = eigvals[isorted]
    sorted_eigvect = eigvect[:, isorted]
    return sorted_eigvals, sorted_eigvect

#### Ex 1.2 ####
print("\n>> Ex 1.2")

# Load and standardize the data
murders = np.loadtxt("murderdata2d.txt")
Sdata = standardize(murders)      

# PCA
eigvals, eigvect = pca(Sdata)
print("\nSorted eigenvalues =\n", eigvals)
print("\nSorted eigenvectors (columns) =\n", eigvect)

## Plot
import matplotlib.pyplot as plt
std1 = np.sqrt(eigvals[0]) # scaler of the eigenvector
std2 = np.sqrt(eigvals[1])
# Plot data
plt.scatter(Sdata[:,0], Sdata[:,1])
plt.title('PC scatterplot of the standardized murders dataset \nwith principal eigenvectors pointing out of the mean')
plt.xlabel('Standardized unemployment')
plt.ylabel('Standardized murders')
plt.axis('equal')
# Plot the eigenvectors
plt.plot([0, eigvect[0,0] * std1], [0, eigvect[1,0] * std1], "r")
plt.plot([0, eigvect[0,1] * std2], [0, eigvect[1,1] * std2], "r")
plt.show()

#### Ex 1.3 ####
print("\n>> Ex 1.3")

pesticides = np.loadtxt("IDSWeedCropTrain.csv", delimiter = ",")[:,:-1] 
Ctrain = center(pesticides)

# Cumulative sum of normalized variance among the principal components
eigvals, eigvect = pca(Ctrain)
cum_var = np.cumsum(eigvals/np.sum(eigvals)) 
Nvar = eigvals/np.sum(eigvals)

# Plot the variance against the pcs
plt.plot(np.arange(1,len(cum_var)+1), eigvals)
plt.title('Variance versus PC indexes')
plt.xlabel('PC indexes')
plt.ylabel('Variance')
plt.xticks(np.arange(1,14,1))
plt.show()

# Plot the normalized variance against the pcs
plt.plot(np.arange(1,len(cum_var)+1), Nvar)
plt.title('Normalized variance versus PC indexes')
plt.xlabel('PC indexes')
plt.ylabel('Variance')
plt.xticks(np.arange(1,14,1))
plt.yticks(np.arange(0,0.7,0.05))
plt.show()

# Plot the cumulative normalized variance against the pcs
plt.plot(np.arange(1,len(cum_var)+1), cum_var)
plt.title('Cumulative normalized variance versus PC indexes')
plt.xlabel('PC indexes')
plt.ylabel('Cumulative variance')
plt.xticks(np.arange(1,14,1))
plt.yticks(np.arange(0.65,1.02,0.05))
plt.show()

# Check how many dimensions are needed to capute the first 95% and 90% of variance
def get_needed_pcs(var_wanted, cum_var_list):
    """
    Return the number of principal components 
    needed to campute the requested variance.
    """
    pcs_needed = 0
    total_var = 0
    for i, var in enumerate(cum_var_list):
        if total_var >= var_wanted:
            break
        else:
            total_var = var
            pcs_needed += 1
    return pcs_needed

print("\nTo capture 95%% of variance we need %d PCs" % get_needed_pcs(0.95, cum_var))
print("To capture 90%% of variance we need %d PCs" % get_needed_pcs(0.90, cum_var))

#### Ex 2 ####

## Dimensionality reduction
#                            - Q = eigenvectors matrix (13 by 2), data = observation by features (1000 by 13).
#                            - We want Q.T (2 by 13) @ data.T (13 by 1000), so to obtain a 2 by 1000 matrix 
#                              (data projected on the two main eigenbasis, dim reduction).
#                            - Project the data in the new basis defined by the two pcs (eigenvectors), 
#                              linear trasformation of data from matrix Q.
#                            - The two matrices needs to be: r by n X n by c -> will output a matrix: r by c.
def dim_reduction(data, dimensions = 2):
    """
    Take a n by m matrix as input and return
    a n by k matrix as output (reduction to kD), 
    where k are the matrix dimensions after the 
    linear transformation.
    """
    # PCA
    eigvals, eigvect = pca(data)
    # Dimensionality reduction
    Q = eigvect[:,:dimensions] # Take first two pcs 
    data_projected = Q.T @ data.T # Project the data in the new space defined by the eigenvectors (2 first pcs)
    return data_projected.T        

# PCA and dimensionality reduction centered pesticides dataset
eigvals, eigvect = pca(Ctrain)
Ctrain_projected = dim_reduction(Ctrain)             

## Plot the data

# Now the eigenvectors are unit basis of my coordinate system: since Q is ortogonal -> Q.T = inv(Q) -> Q.T x Q = I (the basis vectors))                                                                                          
std1, std2 = np.sqrt(eigvals[0]), np.sqrt(eigvals[1])   # Obtain the scalers
Q = eigvect[:,:2]                                       # Obtain the two pcs matrix for the linear transformation
eigenbasis = Q.T @ Q                                    # Since I projected the data in the new space define by the eigenvectors, 
                                                        # now they are my unit basis (eigenbasis)
# Plot the two principal components (eigenbasis)
plt.plot([0, eigenbasis[0,0] * std1], [0, eigenbasis[1,0] * std1], "r")
plt.plot([0, eigenbasis[0,1] * std2], [0, eigenbasis[1,1] * std2], "r")
# Plot the projected data on the space define by the eigenvectors
plt.scatter(Ctrain_projected[:,0], Ctrain_projected[:,1])
plt.title('PCA plot of pesticides centered training dataset')
plt.xlabel('Pc1')
plt.ylabel('Pc2')
plt.show()

#### Ex 3 ####
print("\n>> Ex 3")

def get_loss(list_clusters):
    """
    Obtains the loss function as the sum of 
    the least squares among the clusters.
    """
    loss = 0
    for _, cluster in enumerate(list_clusters):
        mu = np.mean(cluster, axis = 0)
        loss_cluster = sum(np.linalg.norm((cluster - mu), axis=1)**2)
        loss += loss_cluster
    return loss

def kmeans_clustering(data, centroids_list, indexes_only=False):
    """
    Takes an n by m (observations by features) matrix and 
    assign clusters based on points distance from the centroids.
    Return indexes of the clusters if indexes_only equal True,
    return the clusters otherwise. 
    """
    ## Compute distances ##
    dist_lists = []
    k = len(centroids_list)
    # Obtain a list of distances for each cluster
    for i in range(0,k): 
        dist_to_c = np.linalg.norm(data - centroids_list[i], axis=1)
        dist_lists.append(dist_to_c)
    # Convert the lists of distances into an array                           # Each column rapresent the distance from a certain cluster,
    dist_array = np.array(dist_lists).T                                      # each row rapresent the point of the data       

    ## Obtain indexes (rows) of the points assigned to each cluster ##
    row_clusterindex_list = np.argmin(dist_array, axis=1)                    # Each index of the list is a row of the data (point) and the
                                                                             # value is the index of the closer cluster to that point
    # List of lists of indexes (rows) of points assigned to each cluster
    indexes_lists = []
    for i in range(0,k):                                                     # Initialize the indexes_lists with empty lists
        indexes_lists.append([])
    # Append each point row to the assigned cluster
    for pointrow, cluster in enumerate(row_clusterindex_list): 
        indexes_lists[cluster].append(pointrow)

    ## Obtain clusters ##
    cluster_list = []
    for i in range(0,k):
        cluster_list.append(data[indexes_lists[i],:])

    ## Output ##
    if indexes_only == True:
        return indexes_lists
    else:
        return cluster_list

def kmeans_fit(data, k):
    """
    Given a dataset (observations by features matrix), 
    find the optimal k clusters that minimize a least 
    square loss function.
    """
    ## Initialization ##
    centroids_list = []
    # Initialize centroids (just take the first points of the dataset)
    for row in range(0,k):  
        mu = data[row,:]
        centroids_list.append(mu)
    # Assign clusters (based on distance from the centroids)
    cluster_list = kmeans_clustering(data, centroids_list, k)
    # Initialize loss function
    start_loss = get_loss(cluster_list)
    ## Start iterations
    stop = False
    while stop == False:
        # Update centroids
        for i in range(0,k):
            centroids_list[i] = np.mean(cluster_list[i], axis=0)
        # Assign clusters
        cluster_list = kmeans_clustering(data, centroids_list, k)
        # Update loss
        new_loss = get_loss(cluster_list)
        # Stop if there isn't any improve, update starting loss otherwise
        if new_loss == start_loss:
            stop = True
        else:
            start_loss = new_loss
    return centroids_list

# Obtain centroids for centered pesticides dataset
centroids = kmeans_fit(Ctrain, 2)
print("\nThe first centroid (my implementation, centered data)=\n", centroids[0])
print("\nThe second centroid (my implementation, centered data)=\n", centroids[1])


#### Evaluate the cluster prediction (not requested) ####
print("\n>> Extra exercise")

### Plot classes division of pesticides test dataset
pesticides = np.loadtxt("IDSWeedCropTest.csv", delimiter = ",")
XTest = pesticides[:,:-1]
YTest = pesticides[:,-1]
Ctest = center(XTest)
# Divide dataset in crop and weed
crop_indexes = YTest == 1
weed_indexes = YTest == 0
# Dimensionality reduction
crop_2D = dim_reduction(Ctest)[crop_indexes]
weed_2D = dim_reduction(Ctest)[weed_indexes]
# Plot
plt.scatter(crop_2D[:,0], crop_2D[:,1], color = "r", label = "Crop")
plt.scatter(weed_2D[:,0], weed_2D[:,1], color = "b", label = "Weed")
plt.axis('equal')
plt.title('Crop and weed division in pesticides centered test dataset')
plt.xlabel('Pc1')
plt.ylabel('Pc2')
plt.legend(loc = "lower left")
plt.show()

### Plot test set clustering
# Assign clusters 
indexes = kmeans_clustering(Ctest, centroids, indexes_only=True)
# Dimensionality reduction
cluster1 = dim_reduction(Ctest)[indexes[0],:]
cluster2 = dim_reduction(Ctest)[indexes[1],:]
# Plot
plt.scatter(cluster1[:,0], cluster1[:,1], color = "r", label = "Crop")
plt.scatter(cluster2[:,0], cluster2[:,1], color = "b", label = "Weed")
plt.axis('equal')
plt.title('K-means clustering in pesticides centered test dataset, k = 2')
plt.xlabel('Pc1')
plt.ylabel('Pc2')
plt.legend(loc = "lower left")
plt.show()

### Compute the accuracy of the prediction by comparing the index of the clusters and the relative class
# Classes indexes
crop = list(np.where(crop_indexes == True)[0])
weed = list(np.where(weed_indexes == True)[0])
# Cluster indexes
cluster1 = indexes[0]
cluster2 = indexes[1]
# Comparison and accuracy
def clustering_accuracy(iclass1, iclass2, icluster1, icluster2):
    """
    The function compares the clusters predicted to 
    the real classes division of the data. It returns 
    the accuracy of the prediction.
    """
    right_pred = 0
    for _, crop_point in enumerate(iclass1):
        if crop_point in icluster1:
            right_pred += 1
    for _, weed_point in enumerate(iclass2):
        if weed_point in icluster2:
            right_pred += 1
    return right_pred/(len(iclass1) + len(iclass2))

accuracy = clustering_accuracy(crop, weed, cluster1, cluster2)
print("\nThe accuracy of the model on the pesticides test dataset is %.2f%%" % (accuracy*100))