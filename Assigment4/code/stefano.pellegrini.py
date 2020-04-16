import numpy as np
import matplotlib.pyplot as plt
import intro_data_science as ids

##### Data exploration with PCA ######


#### Exercise 1 ####
print("\n>> Exercise 1:\n")

# Load data
diatoms = np.loadtxt("diatoms.txt") 
print("Shape of diatoms dataset =", np.shape(diatoms))

# Each row is a diatom cell: the columns of each row alternate 90 x and 90 y: x1, y1, x2, y2, .., .., x90, y90
# So each diatom cell is described by 90 points and each point is defined on a 2D plane (x and y)
col_numbers = len(diatoms.T) 
x_indexes = np.arange(0,col_numbers,2)
y_indexes = np.arange(1,col_numbers,2)

# Append the first point to the indexes to plot the complete outline of the diatom cells
x_indexes = np.append(x_indexes, 0)
y_indexes = np.append(y_indexes, 1)

# Plot one diatom cell
plt.plot(diatoms[0,x_indexes], diatoms[0,y_indexes], color = "red")
ids.plot_details(title = "One diatom cell outline",
                 ax_equal = True,       
                 bg = True)
                 
# Plot all cells on top of each other
for row in range(len(diatoms)):
    plt.plot(diatoms[row,x_indexes], diatoms[row,y_indexes], color = str(row+999))
ids.plot_details(title = "All diatoms cells outline",
                 ax_equal = True,
                 bg = True,
                 bg_color = "wheat")


#### Exercise 2 ####
print("\n>> Exercise 2:\n")

# Create an array where 0:90 rows are the x coordinate of the 90 points 
# and the 90:181 rows are the y coordinates of the 90 points, columns are features.
data_x = diatoms[:,x_indexes].T
data_y = diatoms[:,y_indexes].T
data_xy_orizontal = np.concatenate((data_x, data_y))
print("Shape of data xy orizontal =", np.shape(data_xy_orizontal))

# Plot the instances of the principal components
def plot_instances(data, pc_index, instances, title = "Instances of principal components"):
    """
    Take a dataset where the first 91 rows are the x coordinates 
    and the last 91 are the y coordinates. The function plot n
    instances of the given principal component.
    """
    # Compute the mean of each feature
    mean = np.mean(data_xy_orizontal, axis=1)
    # Obtain eigenvalues and eigenvectors
    evals, evecs = ids.pca(data, transpose = False)
    # Create a zero matrix with 182 rows (first 91 are x coordinate, last 91 are y coordinate) and 5 columns (for 5 instances)
    data_along_pc = np.zeros((182,instances))
    e = evecs[:,pc_index]
    std = np.sqrt(evals[pc_index])
    ## Plot the the instances of the PCs
    for i in range(instances):
        data_along_pc[:,i] = mean + (i-2) * std * e
        plt.plot(data_along_pc[:91,i], data_along_pc[91:,i], label = i+1, color = (i/instances,0,0))
    # Add plot details
    ids.plot_details(title = title,
                    legend = True,
                    ax_equal = True,
                    bg = True)                   

plot_instances(data_xy_orizontal, pc_index = 0, instances = 5,
               title = "Five instances of the first principal component")
plot_instances(data_xy_orizontal, pc_index = 1, instances = 5,
               title = "Five instances of the second principal component")
plot_instances(data_xy_orizontal, pc_index = 2, instances = 5,
               title = "Five instances of the third principal component")

## Check the variance captured by the principal components ## (Not requested)
evals, evect = ids.pca(data_xy_orizontal, transpose = False)
cum_var = np.cumsum(evals/np.sum(evals)) 
Nvar = evals/np.sum(evals)

print("\nFirst PC captured variance = %d%%" % int(Nvar[0]*100))
print("Second PC captured variance = %d%%" % int(Nvar[1]*100))
print("Third PC captured variance = %d%%" % int(Nvar[2]*100))


#### Exercise 3a ####

# Whitening function and test 
def whitening(data):
    """
    Center, decorrelate and normalize a dataset.
    """
    Cdata = ids.center(data)
    n_features = len(data.T)
    Ddata = ids.mds(Cdata, dimensions = n_features)
    Wdata = ids.normalize(Ddata)
    return Wdata


#### Exercise 3b ####
print("\n>> Exercise 3b:\n")

toydata = np.loadtxt("pca_toydata.txt") 
print("Shape of toydata =", np.shape(toydata))

toydata_2d = ids.mds(toydata)
plt.scatter(toydata_2d[:,0], toydata_2d[:,1], zorder=3, ec = "black")
ids.plot_details(title = "Toydata plot (all datapoints) in 2D",
                 ax_equal = True,
                 bg = True)
                
toydata_2d = ids.mds(toydata)
plt.scatter(toydata_2d[:-2,0], toydata_2d[:-2,1], zorder=3, ec = "black")
ids.plot_details(title = "Toydata (leave out last 2 data points) in 2D",
                 ax_equal = True,
                 bg = True)


##### Clustering II ######

#### Exercise 4 ####                     
print("\n>> Exercise 4:\n")

pest_train = np.loadtxt("IDSWeedCropTrain.csv", delimiter = ",") 
ydata = pest_train[:,-1]
data = pest_train[:,:-1]
data = ids.center(data)
# Obtain centroids
centroids = ids.kmeans_fit(data, 2)
# Get classes indexes
crop_indexes = ydata == 1
weed_indexes = ydata == 0
# Get clusters indexes
cluster_indexes = ids.kmeans_clustering(data, centroids, indexes_only=True)

## 2D Projection
# MDS
data_2d = ids.mds(data)
# Divide the data points in the two classes
crop_2D = data_2d[crop_indexes]
weed_2D = data_2d[weed_indexes]
# Assign clusters and MDS
cluster1 = data_2d[cluster_indexes[0],:]
cluster2 = data_2d[cluster_indexes[1],:]
# Obtain centroids in 2D
centroid1_2D = np.mean(cluster1, axis=0)
centroid2_2D = np.mean(cluster2, axis=0)
print("Centroid 1 projected in 2D =", centroid1_2D)
print("Centroid 2 projected in 2D =", centroid2_2D)
# Plot classes
plt.scatter(crop_2D[:,0], crop_2D[:,1], label = "Crop", ec = "black", c = "red", zorder=3)
plt.scatter(weed_2D[:,0], weed_2D[:,1], label = "Weed", ec = "black", zorder=3)
# Plot centroids
plt.scatter(centroid1_2D[0], centroid1_2D[1], marker = "X", color = "yellow", 
                                              ec = "black", s = 200, zorder=4, label = "Centroids")
plt.scatter(centroid2_2D[0], centroid2_2D[1], marker = "X", color = "yellow", 
                                              ec = "black", s = 200, zorder=4)
# Add plot details
ids.plot_details(title = "2D projection of pesticides training dataset",
                 xlabel = "PC1",
                 ylabel = "PC2",
                 ax_equal = True,
                 bg = True,
                 legend = True,
                 leg_loc = "upper left")

## 3D Projection
from mpl_toolkits.mplot3d import Axes3D
# MDS
data_3d = ids.mds(data, 3)
# Divide the data points in the two classes
crop_3D = data_3d[crop_indexes]
weed_3D = data_3d[weed_indexes]
# Assign clusters and MDS
cluster1 = data_3d[cluster_indexes[0],:]
cluster2 = data_3d[cluster_indexes[1],:]
# Obtain centroids in 3D
centroid1_3D = np.mean(cluster1, axis=0)
centroid2_3D = np.mean(cluster2, axis=0)
print("\nCentroid 1 projected in 3D =", centroid1_3D)
print("Centroid 2 projected in 3D =", centroid2_3D)
# Plot classes into 3D
ax = plt.axes(projection='3d')
ax.scatter3D(crop_3D[:,0], crop_3D[:,1], crop_3D[:,2], label = "crop", ec = "black", c = "red", zorder=3)
ax.scatter3D(weed_3D[:,0], weed_3D[:,1], weed_3D[:,2], label = "weed", ec = "black", zorder=3)
# Plot centroids
ax.scatter3D(centroid1_3D[0], centroid1_3D[1], centroid1_3D[2], marker = "X", color = "yellow", 
                                               ec = "black", s = 200, zorder=5, label = "Centroids")
ax.scatter3D(centroid2_3D[0], centroid2_3D[1], centroid2_3D[2], marker = "X", color = "yellow", 
                                               ec = "black", s = 200, zorder=5)
# Plot details
ax.set_title("3D projection of pesticides training dataset", y = 1.1)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")    
ax.set_zlabel("PC3")
legend = ax.legend(frameon = 1, loc = 'upper left', shadow = True)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
plt.show()
