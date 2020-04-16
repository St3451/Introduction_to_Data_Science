#### Assignment 2

import numpy as np # read in the data 
dataTrain = np.loadtxt("IDSWeedCropTrain.csv", delimiter = ",") 
dataTest = np.loadtxt("IDSWeedCropTest.csv", delimiter = ",") 

# Split input variables and labels 
XTrain = dataTrain[:,:-1]                # take all rows and all column except last one
YTrain = dataTrain[:,-1]                 # take all rows and only last column
XTest = dataTest[:,:-1] 
YTest = dataTest[:,-1]


###### Exercise 1 ######
print("\n>> Exercise 1:\n")

## Nearest neighbour
def nearest_neigh(xtrain, xtest, ytrain, ytest):
    """
    For each point of a given dataset (xtest), predict its 
    class by looking at the class of the nearest neighbour, 
    return the accuracy of the model and a vector containing 
    the predicted classes
    """
    ypred = []
    for i in range(len(xtest)):
        # Compute the Euclidean distances between the point i in XTest and all points in XTrain 
        list_dist = np.linalg.norm(xtrain - xtest[i,:], axis = 1) 
        # Return the index of the closest point 
        y_index = int(np.where(list_dist == min(list_dist))[0])  
        # Append the predicted class to a list (predict the XTest class based on its closest point label) 
        ypred.append(ytrain[y_index]) 
    # Convert the list into a vector of predicted classes                            
    ypred = np.array(ypred)    
    # Compute accuracy                                   
    accuracy = sum(ypred == ytest) / len(ytest)                  
    return accuracy, ypred

self_accuracy_nn = nearest_neigh(XTrain, XTrain, YTrain, YTrain)[0]
accuracy_nn = nearest_neigh(XTrain, XTest, YTrain, YTest)[0]
print("1-Nearest Neighbour Training Accuracy (own implementation) =", self_accuracy_nn, "= %d%%" % (self_accuracy_nn*100))
print("1-Nearest Neighbour Test Accuracy (own implementation) =", accuracy_nn, "\u2248 %.1f%%" % (accuracy_nn*100))

## Check if I obtain the same results with Scikit-learn classifier (not requested in the exercise)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create k-NN classifier
knn = KNeighborsClassifier(n_neighbors = 1)
# Fit the classifier to the data
knn_train = knn.fit(XTrain,YTrain) 
knn_test = knn.fit(XTrain,YTrain) 
# Make prediction on test dataset
accTest = accuracy_score(YTest, knn_test.predict(XTest))
accTrain = accuracy_score(YTrain, knn_train.predict(XTrain))
print("1-Nearest Neighbour Training Accuracy (Scikit-learn) =", accTrain, "= %d%%" % (accTrain*100))
print("1-Nearest Neighbour Test Accuracy (Scikit-learn) =", accTest, "= %.1f%%" % (accTest*100))


###### Exercise 2 ######
print("\n>> Exercise 2:\n")

## Find the most common class among the k closest points
def get_most_common_class(k_indexes, ytrain):      
    """
    Given a list of k indexes and a vector of classes
    return the class that occurs more often. 
    """
    import random
    list_classes = list(ytrain[k_indexes])           
    most_common = max(set(list_classes), key = list_classes.count)
    return most_common                   

## K-nearest-neighbour classifier function
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

## Cross-validation
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

## Find best k-hyperparameter by cross-validation (model selection)
def get_top_k_cv(xtrain, ytrain, kmax=12, odd_only=True):
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
        error, accuracy = cross_validation(xtrain, ytrain, k)
        topk.append((error, accuracy, k))   
    topk.sort()
    topk = np.array(topk)
    kbest = int(topk[0,2])
    return topk, kbest
    
topk_cv, kbest_cv = get_top_k_cv(xtrain=XTrain, ytrain=YTrain)             
print("Top-k by CV Out-of-Sample Error (Loss, Accuracy, k) =\n\n", topk_cv)                               # Not requested in the exercise
print("\nK-best =", kbest_cv) 
print("K-best (%d-NN) CV Out-of-Sample Error =" % kbest_cv, topk_cv[0,0])                                 # Not requested in the exercise  
print("K-best (%d-NN) CV Test Accuracy =" % kbest_cv, topk_cv[0,1], "\u2248 %.1f%%" % (topk_cv[0,1]*100)) # Not requested in the exercise  


###### Exercise 3 ######
print("\n>> Exercise 3:\n")

# Estimate the performance on training and test data
kbest_in_error, kbest_in_accuracy, _ = k_nearest_neigh(XTrain, XTrain, YTrain, YTrain)
kbest_out_error, kbest_out_accuracy, _ = k_nearest_neigh(XTrain, XTest, YTrain, YTest)
print("K-best (3-NN) In-Sample Error =", kbest_in_error)
print("K-best (3-NN) Out-of-Sample Error =", kbest_out_error)
print("K-best (3-NN) Training Accuracy =", kbest_in_accuracy, "= %.1f%%" % (kbest_in_accuracy*100))
print("K-best (3-NN) Test Accuracy =", kbest_out_accuracy, "\u2248 %.1f%%" % (kbest_out_accuracy*100))


###### Exercise 4 ######
print("\n>> Exercise 4:\n")

## Data normalization
def norm_data(traindata, testdata):    
    """
    Center and normalize the two datasets based on the 
    mean and standard deviation of the training set
    """
    # Find the affine linear mapping 
    mean_train = np.mean(traindata, axis = 0)
    std_train = np.std(traindata, axis = 0)
    # Center the data
    norm_train = (traindata - mean_train) / std_train
    # Normalize the data
    norm_test = (testdata - mean_train) / std_train
    return norm_train, norm_test

# Normalize the data
norm_XTrain, norm_XTest = norm_data(XTrain, XTest)    

# Find k-best by cross-validation (model selection)                    
topk_cv_norm, kbest_cv_norm = get_top_k_cv(xtrain=norm_XTrain, ytrain=YTrain)       
print("Top-k by CV Out-of-Sample Error on normalized data (Loss, Accuracy, k) =\n\n", topk_cv_norm)    # Not requested in the exercise
print("\nK-best (normalized data) =", kbest_cv_norm)

# Estimate the performance on training and test data
Nkbest_in_error, Nkbest_in_accuracy, _ = k_nearest_neigh(norm_XTrain, norm_XTrain, YTrain, YTrain, kbest_cv_norm)
Nkbest_our_error, Nkbest_out_accuracy, _ = k_nearest_neigh(norm_XTrain, norm_XTest, YTrain, YTest, kbest_cv_norm)
print("K-best (3-NN) In-Sample Error (normalized data) =", Nkbest_in_error)                            # Not requested in the exercise
print("K-best (3-NN) Out-of-Sample Error (normalized data) =", Nkbest_our_error)                       # Not requested in the exercise
print("K-best (3-NN) Training Accuracy (normalized data) =", Nkbest_in_accuracy, "= %.1f%%" % (Nkbest_in_accuracy*100))
print("K-best (3-NN) Test Accuracy (normalized data) =", Nkbest_out_accuracy, "\u2248 %.1f%%" % (Nkbest_out_accuracy*100))