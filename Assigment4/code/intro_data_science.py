### I D S ### M O D U L E ###

### This python script contains function that I implemented during the Introduction to Data Analysis course.
### I am importing this module in my others python scripts in order to reuse the functions I implemented for
### previous assigments.

import numpy as np
import matplotlib.pyplot as plt

### PREPROCESS DATA ###

# Centering function
def center(data):
    """
    Center the data.
    """
    mean = np.mean(data, axis=0)
    Cdata = (data - mean)
    return Cdata

# Normalization function
def normalize(data):
    """
    Normalize the data.
    """
    std = np.std(data, axis = 0)
    Ndata = data / std
    return Ndata

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

### K-NEAREST NEIGHBOR ###

# K-nearest neighbor error
def error_knearest(ypred, ytest):
    """
    Given the predicted classes and the true classes. 
    Return the error of the k-nearest algorithm.
    """
    return sum(ypred!=ytest) / len(ytest)

# K-nearest neighbor accuracy
def accuracy_knearest(ypred, ytest):
    """
    Given the predicted classes and the true classes. 
    Return the accuracy of the k-nearest algorithm.
    """
    accuracy = sum(ypred==ytest) / len(ytest)

# Find most common class among the k closest points
def get_most_common_class(k_indexes, ytrain):      
    """
    Given a list of k indexes and a vector of classes
    return the class that occurs more often. 
    """
    import random
    list_classes = list(ytrain[k_indexes])           
    most_common = max(set(list_classes), key = list_classes.count)
    return most_common          

# K-nearest-neighbour classifier function
def k_nearest_neigh(xtrain, xtest, ytrain, ytest, k=5):
    """
    Predict the class of each point in the xtest based on 
    the class of the k closest points in xtrain, return the
    loss and the accuracy of the model and the vector of the 
    predicted classes
    """
    ypred = []
    # Repeat the loop for each row in the test dataset
    for j in range(len(xtest)): 
        # Return a list for the Euclidean distances between a specific point (row) in the test data set and all points in the train data set                                                     
        list_dist = list(np.linalg.norm(xtrain - xtest[j,:], axis = 1))     
        # Get the indexes of the k-closest points 
        list_indexes = np.argsort(list_dist)[:k]
        # Predict the class by taking the average class of the k-closest points                     
        predicted_class = get_most_common_class(list_indexes, ytrain)                       
        ypred.append(predicted_class)     
    ypred = np.array(ypred)
    # Compute loss and accuracy 
    error = sum(ypred!=ytest) / len(ytest)
    accuracy = sum(ypred==ytest) / len(ytest)
    return error, accuracy, ypred

# Cross-validation
def cross_validation(xtrain, ytrain, k, kfold=5): # k is used for KNN number of neighbours, kfold is used to set the number of folds in CV
    """
    Test the performance of our model by cross-validation 
    using only the training set. By default perform a 
    5-nearest-neighbour classification using a 5-fold 
    cross-validation.
    """
    from sklearn.model_selection import KFold 
    cv = KFold(n_splits = kfold)
    list_error = []
    list_accuracy = []
    # Loop over CV folds 
    for train, test in cv.split(xtrain):   
        # In each cycle divide a dataset (XTrain) in training data (XTrainCV, YTrainCV) and test data (XTestCV, YTestCV) 
        XTrainCV, XTestCV, YTrainCV, YTestCV = xtrain[train] ,xtrain[test] ,ytrain[train] ,ytrain[test]
        # Compute k-nearest-neighbour and return loss and accuracy
        error, accuracy, _ = k_nearest_neigh(xtrain=XTrainCV, xtest=XTestCV, ytrain=YTrainCV, ytest=YTestCV, k=k)
        list_error.append(error)
        list_accuracy.append(accuracy)
    # Compute the average of loss and accuracy between the k-folds (accuracy was not requested in the exercise)
    error = sum(list_error) / len(list_error)
    accuracy = sum(list_accuracy) / len(list_accuracy)                            
    return error, accuracy

# Find best k-hyperparameter by cross-validation (model selection)
def get_kbest_cv(xtrain, ytrain, kmax=12, odd_only=True):
    """
    Find the best k-hyperparameter for k-nearest neighbour
    classification by performing a 5-fold cross-validation
    on the training dataset.
    """
    topk = []
    # If odd_only equal True, perform k-NN only for odd k
    if odd_only == True:                                                      
        step = 2
    # If odd_only equal False, perform k-NN for all k (not requested in the exercise)
    else:
        step = 1    
    # Perform k-NN for k between 1 and kmax
    for k in range(1,kmax,step):
        error = cross_validation(xtrain, ytrain, k)
        topk.append((error, k)) 
    topk.sort()
    topk = np.array(topk)
    kbest = int(topk[0,1])
    return kbest

### MULTIDIMENSIONAL SCALING ###

# PCA
def pca(data, transpose = True):
    """
    Take as argument data with columns 
    as features and rows as observations.
    Return eigenvectors and eigenvalues 
    both sorted from eigenvalues (variance).
    """
    # Obtain covariance matrix
    if transpose == True:
        sigma = np.cov(data.T) # np.cov wants features as rows and observations as columns (so transpose the data)
    else:
        sigma = np.cov(data)
    # Obtain sorted eigenvalues and eigenvectors
    eigvals, eigvect = np.linalg.eig(sigma)
    isorted = eigvals.argsort()[::-1]
    sorted_eigvals = eigvals[isorted]
    sorted_eigvect = eigvect[:, isorted]
    return sorted_eigvals, sorted_eigvect

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

## Dimensionality reduction
#                            - Q = eigenvectors matrix (13 by 2), data = observation by features (1000 by 13).
#                            - We want Q.T (2 by 13) @ data.T (13 by 1000), so to obtain a 2 by 1000 matrix 
#                              (data projected on the two main eigenbasis, dim reduction).
#                            - Project the data in the new basis defined by the two pcs (eigenvectors), 
#                              linear trasformation of data from matrix Q.
#                            - The two matrices needs to be: r by n X n by c -> will output a matrix: r by c.
def mds(data, dimensions = 2):
    """
    Take a n by m matrix as input and return
    a n by k matrix as output (reduction to kD), 
    where k are the matrix columns (dimensions) 
    after the linear transformation.
    """
    # PCA
    eigvals, eigvect = pca(data)
    # Dimensionality reduction
    Q = eigvect[:,:dimensions] # Take first two pcs 
    data_projected = Q.T @ data.T # Project the data in the new space defined by the eigenvectors (2 first pcs)
    return data_projected.T               

# Plot eigenvectors
def plot_eigvect(data):
    """
    The function plot the 2 principal eigenvectors 
    of a dataset, remember to call plt.show().
    """
    # Perform PCA
    eigvals, eigvect = pca(data)
    # Compute scaler of eigenvectors
    std1 = np.sqrt(eigvals[0])
    std2 = np.sqrt(eigvals[1])
    # Plot eigenvectors
    plt.plot([0, eigvect[0,0] * std1], [0, eigvect[1,0] * std1], "r")
    plt.plot([0, eigvect[0,1] * std2], [0, eigvect[1,1] * std2], "r")

#### K-MEANS CLUSTERING ####

# K-means loss function
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

# K-means clustering algorithm
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
    row_clusterindex_list = np.argmin(dist_array, axis=1)                    # Each index of the list is a row of the data (point) and the value is the index of the closer cluster to that point                                                                     
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

# K-means model training (obtain centroids from training data)
def kmeans_fit(data, k, 
               return_loss = False, 
               set_starting_points = False):
    """
    Given a dataset (observations by features matrix), 
    find the optimal k clusters that minimize a least 
    square loss function. Options: retur loss, set 
    starting points (add starting points as argument
    list or tuple).
    """
    ## Initialization ##
    centroids_list = []
    # Initialize centroids 
    if set_starting_points == False:
        # Just take the first points of the dataset
        for row in range(0,k):  
            mu = data[row,:]
            centroids_list.append(mu)
    else:
        # Use the centroids given in list set_starting_points
        centroids_list = set_starting_points
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
        if (start_loss - new_loss) < 1:
            stop = True
        else:
            start_loss = new_loss
    ## Output
    if return_loss == False:
        return centroids_list
    if return_loss == True:
        return centroids_list, new_loss 

# K-means comparison and accuracy
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
    if right_pred != 0:
        accuracy = right_pred/(len(iclass1) + len(iclass2))
        return accuracy
    else:
        return 0

### PLOTTING ###

# Plot data
def plot_details(title = "",
                 xlabel = "X-axis", 
                 ylabel = "Y-axis",
                 ax_equal = False,
                 legend = False, 
                 leg_loc = "upper right",
                 bg = False,
                 bg_color = "lightgray",
                 save = False,
                 filename = False):
    """
    Add plot details like title, xlabe, ylabel and other options: 
    legend and its location, equal axis, background and its color, 
    save file and filename.
    """
    # Title and axis
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Set equal axis
    if ax_equal == True:
        plt.axis('equal') 
    # Legend
    if legend == True:
        legend = plt.legend(frameon = 1, loc = leg_loc, shadow = True)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
    # Set background and grid
    if bg == True:
        plt.grid(zorder=0, color="white")
        ax = plt.gca()
        ax.set_facecolor(bg_color)
    # Output
    if save == True:
        if filename == False:
            plt.savefig(title + ".png")
        else:
            plt.savefig(filename + ".png")
        plt.clf()
    else:
        plt.show()
