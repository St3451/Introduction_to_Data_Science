# Assigment 5-6 #

import numpy as np
import math
import matplotlib.pyplot as plt
import intro_data_science5 as ids

################ LINEAR REGRESSION ################

######## Exercise 2
print("\n>> Exercise 2\n")
wine_train = np.loadtxt("redwine_training.txt")
wine_test = np.loadtxt("redwine_testing.txt")

# Divide x and y
xtrain = wine_train[:,:-1]
ytrain = wine_train[:,-1]
xtest = wine_test[:,:-1]
ytest = wine_test[:,-1]

# Standardize data                                                              
xtrain, xtest = ids.stdize_train_test(xtrain, xtest)


#### Exercise 2a

# Define multivarlinreg(x, y)
# input: 1) X: the independent variables (data matrix), an (N x D)-dimensional matrix, as a numpy array
#        2) y: the dependent variable, an N-dimensional vector, as a numpy array
#
# output: 1) the regression coefficients, a (D+1)-dimensional vector, as a numpy array
# note: remember to either expect an initial column of 1's in the input X, or to append this within your code

def addone_to_firstcol(data):
    """
    Create a 1 vector of the lenght of the input data, 
    add the one vector as first column of the dataset. 
    """
    # Generate vector of 1
    one = np.ones((len(data), 1))
    # Reverse data
    rev_data = np.flip(data)
    # If data has more than 1 feature (if data is a matrix)
    if len(np.shape(rev_data)) == 2:
        # Append one vector as last column
        addone_rev = np.append(rev_data, one, axis=1)
    # If data has only one feature (if data is a vector)
    else:
        addone_rev = np.column_stack((rev_data, one))
    # Reverse data again
    addone = np.flip(addone_rev)
    return addone

def multivarlinreg(X, y):
    """
    Given a matrix of independet variables (predictors) X and a
    vector of dependent variable y, find analytically the weights
    (coefficients) of the multivariate linear regression function.
    """
    # Add a vector of 1 as first column of the dataset (offset parameter w0)
    X = addone_to_firstcol(X)
    # Find the w matrix (weights, free parameters) analytically: (x.T @ x)**-1 @ x.T @ y
    # (x.T @ x) is an invertible square matrix 
    w = np.linalg.inv((X.T @ X)) @ X.T @ y
    return w


#### Exercise 2b
fixed_acidity = xtrain[:,0]
w_fixed_acidity = multivarlinreg(fixed_acidity, ytrain)
print("Fixed acidity weights =", w_fixed_acidity)


#### Exercise 2c
w_all = multivarlinreg(xtrain, ytrain)
print("\nAll features weights =\n", w_all)

def print_coefficients(w, features_dict, sort = True):
    """
    Take as inputs a weights vector and a dictionary with the 
    names of the features of a dataset. Print the sorted 
    weights (coefficients) with the feature corresponding name.
    """
    # Sort the features based on the parameter weights
    sorted_features_ind = np.argsort(abs(w))[::-1]
    # Print the features and the corresponding weight sorted in descending order
    if sort == True:
        print("\nSorted coefficients:")
        for i, feature_index in enumerate(sorted_features_ind):
            print("\t%.7f =" % w[sorted_features_ind[i]], features_dict[feature_index])
    else:
        print("\nCoefficients:")
        for i in range(len(w)):
            print("\t%.7f =" % w[i], features_dict[i])
    
features_name_dict = {0 : "intercept", 1 : "ﬁxed acidity", 2 : "volatile acidity", 
                      3 : "citric acid", 4 : "residual sugar", 5 : "chlorides",
                      6 : "free sulfur dioxide", 7 : "total sulfur dioxide", 
                      8 : "density", 9 : "pH", 10 : "sulfates", 11 : "alcohol"}
print_coefficients(w_all, features_name_dict, sort = False)
print_coefficients(w_all, features_name_dict, sort = True)


######## Exercise 3
print("\n>> Exercise 3\n")


#### Exercise 3a

# Linear regression function
def linreg_predict(X, w):
    """
    Given a matrix of independet variables (predictors) X and a
    coefficients (weights) vector w. Return the prediction of the 
    dependent variable y.
    """
    X = addone_to_firstcol(X)
    h = X @ w
    return h

# Root means square error
def rmse(h, y):
    """
    Given a vector of the predicted y (h) and the true y. 
    Return the root means square error. 
    """
    square_error = (h - y)**2
    sum_square_error = sum(square_error)
    means_square_error = sum_square_error / len(h)        
    rmsd = np.sqrt(means_square_error)
    return rmsd


#### Exercise 3b (Model for 1-dimensional input variables: fixed acidity)         

## Output error
# Predict the dependent random variable (quality of the wine) on test data
h_fixed_acidity = linreg_predict(xtest[:,0], w_fixed_acidity)
# Compute the root means square output error
rmse_fixed_acidity = rmse(h_fixed_acidity, ytest)
print("Root means square output error (model with only fixed acidity) = ", rmse_fixed_acidity)


#### Exercise 3c (Full model)

## Output error
# Predict the dependet random variable (quality of the wine) on test data
h_all = linreg_predict(xtest, w_all)
# Compute the root means square output error
rmse_all = rmse(h_all, ytest)
print("Root means square output error (full model) = ", rmse_all)


################ RANDOM FOREST ################

######## Exercise 5
print("\n>> Exercise 5\n")

## Random forest prediction
from sklearn.ensemble import RandomForestClassifier
# Load data
pest_train = np.loadtxt("IDSWeedCropTrain.csv", delimiter = ",") 
pest_test = np.loadtxt("IDSWeedCropTest.csv", delimiter = ",") 
# Divide data
xtrain = pest_train[:,:-1]
ytrain = pest_train[:,-1]
xtest = pest_test[:,:-1]
ytest = pest_test[:,-1]

# Create the model with 50 trees
forest = RandomForestClassifier(n_estimators=50, bootstrap = True)
# Fit the model on training data
forest.fit(xtrain, ytrain)
# Predict class on test data
ypred = forest.predict(xtest)
# Compute accuracy
accuracy = sum(ypred==ytest) / len(ytest)
print(f"Random forest accuracy = %.2f%%" % (accuracy*100))

## Nearest neighbor classiﬁer
# Find the best k (number of closest nearest points)
kbest = ids.get_kbest_cv(xtrain, ytrain)[0]
# Prediction accuracy on test set
accuracy = ids.k_nearest_neigh(xtrain, xtest, ytrain, ytest, k = kbest)[1]
print(f"K-nearest neighbor accuracy = %.2f%%" % (accuracy*100))


################ GRADIENT DESCENT ################

######## Exercise 6
print("\n>> Exercise 6\n")

#### Exercise 6.1

# Define the function f(x)
def f(x):
    """
    Just the default function used for 
    the gradient descent.
    """
    return np.exp(-x/2) + 10*(x**2)

# Derivative of f(x) (m or slope of the tangent)
def f_prime(x):
    """
    Derivative of the function f(x).
    """
    return -1/2 * np.exp(-x/2) + 20*x


#### Exercise 6.2

# Tangent of f(x)
def tan(x, x1):
    """
    Tangent of f(x) in a point x1.
    """ 
    m = f_prime(x1) # Compute the slope of the tangent (f'(x))
    y1 = f(x1)
    tan = y1 + m * (x - x1)
    return tan

# Plot function, tangent and steps
def plot_gradient_main(x, x1, num_iter, steps_to_plot, plot_tan):
    """
    Plot predefined f() and tan() functions, also 
    plot the first n steps if requsted
    """
    # If first iteration: plot f(x), the starting point
    if num_iter == 1:
        plt.plot(x, f(x), color = "black", label = "f(x)", linewidth=1.3, zorder=2)
        plt.scatter(x1, f(x1), zorder=3, s = 60, marker = "X", ec = "black", label = "%d° iteration" % num_iter)
    # Plot the other points
    elif num_iter <= steps_to_plot:
        plt.scatter(x1, f(x1), zorder=3, ec = "black", label = "%d° iteration" % num_iter)
    # Plot the tangent
    if num_iter <= 3 and plot_tan == True:
        plt.plot(x, tan(x, x1), linewidth=1.3, zorder=2)

# Plot f(x), tangent and steps
def plot_gradient_details(x1, learningrate, steps_to_plot, plot_tan):
    """
    Plot the last point (hopefully the minimum of 
    the function) and add plot details.
    """
    # Plot last point if requested
    if x1 >= -2 and x1 <= 2: 
        plt.scatter(x1, f(x1), zorder=3, s = 80, marker = "X", 
                               color = "yellow", ec = "black", label = "Last iteration")
    else:
        print("""\nCan't plot last point, its value is out of the plotting coordinates:
              > Learningrate = %.4f
              > Steps to plot = %d
              > Value = %f
              """ % (learningrate, steps_to_plot, x1))
    # Set legend positions and columns number
    if steps_to_plot <= 5:
        ncol, loc = 1, "lower right"
    else:
        ncol, loc = 3, "lower center"
    if plot_tan == False:
        loc = "upper center"
    # Plot details
    title = "GD, {} iterations plotted, {} learning rate".format(steps_to_plot, learningrate)
    plt.legend(loc = loc, ncol = ncol, shadow = True)
    ids.plot_details(title = title,
                     bg = True,
                     save = False,
                     filename = "{}".format(title))

# Gradient descent function alternative plotting
def gradient_descent(learningrate = 0.01, 
                     plot = True, 
                     steps_to_plot = 3, 
                     plot_tan = True,
                     stopvalue = False,
                     stopifinf = True):
    """
    Gradient descent function. It uses default f(x), 
    f'(x) and tan(x) previously defined. It also relies 
    on two other functions to plot steps and tangent and
    to add plot details. 
    """
    ## Initialization
    x1 = 1
    value = f(x1)
    # Setting parameters for convergence check
    num_iter = 1
    convergence = 0
    max_iter = 10000
    last_iteration = 0
    # Set x coordiantes for plotting the functions
    x = np.linspace(-2,2,10000)
    ## Start iterations
    while convergence == 0:
        # Compute the current value of the function we are minimizing
        cur_value = f(x1)
        ## Plot f(x), tangent and steps
        if plot == True:
            plot_gradient_main(x, x1, num_iter, steps_to_plot, plot_tan)
        ## Compute the gradient and take a step in the opposite direction
        grad = f_prime(x1)
        x1 = x1 - learningrate * grad  
        ## Check convergence:
        num_iter += 1     
        # Stop by reaching max iterations number or gradient become infinite
        if num_iter > max_iter:                                        
            convergence = 1
            last_iteration = 1
        elif stopifinf == True:
            if np.isinf(grad):
                convergence = 1
                last_iteration = 1
        # Stop if gradient became smaller than threeshold (False by default)
        if stopvalue == True:
            if abs(grad) < 10**(-10):
                convergence = 1
                last_iteration = 1
        ## Plot last point, add settings and details
        if plot == True:
            if last_iteration == 1:
                plot_gradient_details(x1, learningrate, steps_to_plot, plot_tan)
        # Update the value
        value = cur_value  
    return value, num_iter -1

## Run the algorithm at different learning rates
learningrates = 0.1, 0.01, 0.001, 0.0001
# Plot the first three steps and the tangent at different learing rates
for _, rate in enumerate(learningrates):
    value, num_iter = gradient_descent(rate)
# Plot the first ten steps at different learing rates
for _, rate in enumerate(learningrates):
    value, num_iter = gradient_descent(rate, 
                                        steps_to_plot = 10, 
                                        plot_tan = False)
# Set min gradient threeshold as stop criteria and output the number of iterations and function value
output = {}
for _, rate in enumerate(learningrates):
    value, num_iter = gradient_descent(rate, 
                                        plot = False, 
                                        stopvalue = True,
                                        stopifinf = False)
    output[rate] = (value, num_iter)
# Output
print("\nFunction value and number of iterations took the algorithm to converge:")
for _, rate in enumerate(learningrates):
    print("\t> Rate: {}, Function value = {}, N. iterations = {}".format(rate, output[rate][0], output[rate][1]))


################ Logistic Regression ################

######## Exercise 7
print("\n>> Exercise 7\n")

## Load data

# First dataset
iris_train1 = np.loadtxt("Iris2D1_train.txt")
iris_test1 = np.loadtxt("Iris2D1_test.txt")

iris_train2 = np.loadtxt("Iris2D2_train.txt")
iris_test2 = np.loadtxt("Iris2D2_test.txt")

xtrain1 = iris_train1[:,:-1]
ytrain1 = iris_train1[:,-1]
xtest1 = iris_test1[:,:-1]
ytest1 = iris_test1[:,-1]

xtrain2 = iris_train2[:,:-1]
ytrain2 = iris_train2[:,-1]
xtest2 = iris_test2[:,:-1]
ytest2 = iris_test2[:,-1]


#### Ex. 7.1 

## Split first dataset
# Train
train1_class0 = iris_train1[iris_train1[:,-1] == 0,:]
train1_class1 = iris_train1[iris_train1[:,-1] == 1,:]
# Test
test1_class0 = iris_test1[iris_test1[:,-1] == 0,:]
test1_class1 = iris_test1[iris_test1[:,-1] == 1,:]
## Split second dataset
# Train
train2_class0 = iris_train2[iris_train2[:,-1] == 0,:]
train2_class1 = iris_train2[iris_train2[:,-1] == 1,:]
# Test
test2_class0 = iris_test2[iris_test2[:,-1] == 0,:]
test2_class1 = iris_test2[iris_test2[:,-1] == 1,:]

## Plot classes
def easy_plot_classes(class0, class1, title = "Classes scatter plot", save = False):
    plt.scatter(class0[:,0], class0[:,1], ec = "black", label = "Class 0", zorder=3)
    plt.scatter(class1[:,0], class1[:,1], color = "red", ec = "black", label = "Class 1", zorder=3)
    ids.plot_details(title = title,
                     ax_equal = True,
                     bg = True, 
                     legend = True,
                     leg_loc = "upper left",
                     save = save,
                     filename = title)

## Plot first dataset
# Train1
easy_plot_classes(train1_class0, train1_class1,                        
                  title = "Scatter plot Iris2D1 train", save = False)
# Test1                  
easy_plot_classes(test1_class0, test1_class1,                           
                  title = "Scatter plot Iris2D1 test", save = False)
## Plot second dataset
# Train2
easy_plot_classes(train2_class0, train2_class1,                         
                  title = "Scatter plot Iris2D2 train", save = False)
# Test2                  
easy_plot_classes(test2_class0, test2_class1,                          
                  title = "Scatter plot Iris2D2 test", save = False)


#### Ex. 7.2 

# Define the logistic function
def logistic(input):
    """
    Logistic function.
    """
    out = 1 /(1+np.exp(-input))
    return out

# Define in-sample error function
def logistic_Ein(x, y ,w):
    """
    Logistic regression loss function.
    """
    N = len(x)
    # Compute wx, return a vector (w.T = w)
    wx = w @ x.T
    # y * wx, return a vector
    ywx = y * wx
    # Compute ln(1/s(ywx)), for all values of yn and xn. Sum them and divide for N.
    ln_vector = np.log(1 / logistic(ywx))
    E = np.sum(ln_vector) / N
    return E

# Define the gradient function
def logistic_gradient(x, y, w):
    """
    Logistic regression gradient function.
    """
    N = len(x)
    # Just to return a 0 vector of the right dimension
    g = 0 * w 
    # Compute wx (w.T = w)
    wx = w @ x.T                         # y_pred = logistic(w @ x.T)
    # Compute scalar multiplication yx
    yx = np.multiply(y, x.T).T
    # Plug in -ywx in the logistic function
    logistic_result = logistic(-y * wx)
    # Obtain the gradient (vector of length 3) by computing: column wise sum(-yx * logistic(-ywx)) / nrows
    g = np.sum(np.multiply(logistic_result, -yx.T).T, axis = 0) / N
    return g

# Define the logistic regression function
def logistic_reg(x, y, 
                 max_iter = 20000, 
                 learningrate = 0.1, 
                 grad_threshold = 0, 
                 print_num_iter = False):
    """
    Logistic regression function.
    """  
    ## Initialization 
    N, num_feat = x.shape
    # Add one to dataset matrix                           
    onevec = np.ones((N, 1))                   
    x = np.concatenate((onevec, x), axis = 1)
    # Transform y to a N by 1 matrix of target values -1 and 1
    y = np.array((y - .5) * 2)
    # Initialize learning rate for gradient descent
    learningrate = learningrate        
    # Initialize weights at time step 0    
    np.random.seed(0)
    w = 0.1 * np.random.randn(num_feat + 1)
    # Compute value of logistic log likelihood
    value = logistic_Ein(x,y,w)
    num_iter = 0  
    convergence = 0
    # Keep track of function values
    E = 0
    # Start iterations
    while convergence == 0:
        num_iter = num_iter + 1                       
        ## Compute the gradient and take a step in the opposite direction
        g = logistic_gradient(x,y,w)     
        w_new = w - g * learningrate
        ## Check for improvement
        # Compute in-sample error for new w
        cur_value = logistic_Ein(x,y,w_new)
        # If there is improvement we update w and increase the learning rate
        if cur_value < value:
            w = w_new
            value = cur_value
            learningrate *= 1.1
        # If we don't have improvement we discard w and try again with smaller learning rate
        else:
            learningrate *= 0.9              
        ## Check if stop-criteria are satisfied 
        # Check if gradient norm is below threshold or max iterations reached
        g_norm = np.linalg.norm(g)
        if g_norm < grad_threshold or num_iter > max_iter:
            if print_num_iter == True:
                print("Reached convergence at %d iterations" % (num_iter - 1))
            convergence = 1   
            E = value
    return w, E

# Define function to get the class from probabily
def get_class_by_prob(yprob, threshold = 0.5):
    """
    Function that given an array of probabily, it 
    return an array of classes. The class is 1 if 
    the probabily of that record is larger or equal 
    than threshold, 0 otherwise.
    """
    # Inizialization
    ypred = np.zeros(len(yprob), dtype = np.int)
    # Get indexes of points classified as 1
    indexes_class1 = np.where(yprob >= threshold)
    # Update the inizialized array
    ypred[indexes_class1] = 1
    return ypred

# Define function for target prediction by logistic regression
def logistic_pred(x, w, threshold = 0.5):
    """
    Logistic regression target prediction function.
    """
    N = len(x)
    # Add a first column with ones
    onevec = np.ones((N,1))                            
    x = np.concatenate((onevec, x), axis = 1) 
    # Compute the probability that the class is 1
    yprob = logistic(w @ x.T) 
    # Get the predicted class
    ypred = get_class_by_prob(yprob, threshold)
    return yprob, ypred

# Define the function that obtain w from train set and use it to predict y of test set
def logistic_train_and_pred(xtrain, ytrain, xtest, 
                            prob = False, 
                            print_w = False, 
                            print_num_iter = False):
    """
    Function that train the model (obtains w) from
    training set and predict the target values of 
    a test set.
    """
    w = logistic_reg(xtrain, ytrain, print_num_iter = print_num_iter)[0]
    if print_w == True:
        print("Parameters of the linear model =", w)
    yprob, ypred = logistic_pred(xtest, w)
    if prob == False:
        return ypred
    elif prob == True:
        return yprob
    else:
        print("\n>> Error: prob argument has the wrong type <<\n")


#### Ex. 7.3

# Function to compute the error
def zero_one_error(ypred, ytrue):
    """
    Evaluate the performance of a classifier
    as a 0-1 loss function. 
    """
    error = np.sum(ypred != ytrue) / len(ypred)
    return error

# Function to plot the decision boundary (not requested)           
def plot_log_decision_boundary(xtrain, ytrain, xtest, ytest,
                               a_xlim = -5, bxlim = 5,
                               title = "Logistic regresison decision boundary"):
    """
    The function uses the train set to obtain the parameters w. 
    Then evaluate the performance of the model obtained by 
    plotting both, the decision boundary and the true class 
    division of the data.
    """
    # Get the logistic regression parameters by gradient descent
    w, _ = logistic_reg(xtrain, ytrain)
    # Get x and y values for the decision boundary
    x_values = [np.min(xtest[:, 1] + a_xlim), np.max(xtest[:, 1] + bxlim)]
    y_values = np.dot((-1. / w[2]), (np.dot(w[1], x_values) + w[0]))
    # Get colors of the classes
    class0 = xtest[ytest == 0]
    class1 = xtest[ytest == 1]
    # Plot points
    plt.scatter(class0[:, 0], class0[:, 1], label = "Class 0", color = "C0", ec = "black", zorder = 2)
    plt.scatter(class1[:, 0], class1[:, 1], label = "Class 1", color = "red", ec = "black", zorder = 2)
    # Plot decision boundary
    plt.plot(x_values, y_values, label = 'Decision Boundary', color = "green", zorder = 3)
    # Add plot details
    ids.plot_details(title = title,
                     ax_equal = True,
                     bg = True, 
                     legend = True,
                     leg_loc = "upper left",
                     save = False,
                     filename = title)

### Report the parameters of the linear model, train and test accuracy, decision boundary
## Iris2D1 data
# Train 1
print("\n> Iris2D1 data:")
ypred_train1 = logistic_train_and_pred(xtrain1, ytrain1, xtrain1, 
                                       print_w = True, print_num_iter = True)
print("Training error =", zero_one_error(ypred_train1, ytrain1))
plot_log_decision_boundary(xtrain1, ytrain1, xtrain1, ytrain1, a_xlim = 0.8, bxlim = 2.2,
                           title = "Iris2D1 train logistic regression decision boundary")
# Test 1
ypred_test1 = logistic_train_and_pred(xtrain1, ytrain1, xtest1)
print("Test error =", zero_one_error(ypred_test1, ytest1))
plot_log_decision_boundary(xtrain1, ytrain1, xtest1, ytest1, a_xlim = 0.8, bxlim = 2.2,
                           title = "Iris2D1 test logistic regression decision boundary")
## Iris2D2 data
# Train 2
print("\n> Iris2D2 data:")
ypred_train2 = logistic_train_and_pred(xtrain2, ytrain2, xtrain2, 
                                       print_w = True, print_num_iter = True)
print("Training error =", zero_one_error(ypred_train2, ytrain2))
plot_log_decision_boundary(xtrain2, ytrain2, xtrain2, ytrain2, a_xlim = 2, bxlim = 2.4,
                           title = "Iris2D2 train logistic regression decision boundary")
# Test 2
ypred_test2 = logistic_train_and_pred(xtrain2, ytrain2, xtest2)
print("Test error =", zero_one_error(ypred_test2, ytest2))
plot_log_decision_boundary(xtrain2, ytrain2, xtest2, ytest2, a_xlim = 2.2, bxlim = 2.4,
                           title = "Iris2D2 test logistic regression decision boundary")


################ Dimensionality Reduction, Classiﬁcation and Clustering ################

######## Exercise 9
print("\n>> Exercise 9\n")

xdigit = np.loadtxt("MNIST_179_digits.txt")
ydigit = np.loadtxt("MNIST_179_labels.txt")

#### Exercise 9a (Perform k-means clustering and compare to true classes division)

## Clustering
print("Clustering, no prior MDS:")
# Obtain centroids and clusters indexes
centroids = ids.kmeans_fit(xdigit, 3)                                          
clusters_indexes = ids.kmeans_clustering(xdigit, centroids, indexes_only=True)  
                                                                               
# Plot 3 cluster centers as images
for n, centroid in enumerate(centroids):
    # Plot the image (rapresented by 28 x 28 pixels)
    centroid = centroid.reshape((28, 28))
    plt.imshow(centroid, cmap = 'viridis')  
    title = "Cluster %d image (no prior MDS)" % (n+1)
    ids.plot_details(title = title,
                     xlabel = "X-axis (pixels)",
                     ylabel = "Y-axis (pixels)",
                     save = False,
                     filename = title)
    
# Count the proportion of 1s, 7s and 9s in each cluster
def get_labels_proportion(clusters_indexes, true_y):
    labels = 1,7,9
    for n, cl_indexes in enumerate(clusters_indexes):
        proportion1 = sum(true_y[cl_indexes] == labels[0]) / len(cl_indexes)
        proportion2 = sum(true_y[cl_indexes] == labels[1]) / len(cl_indexes)
        proportion3 = sum(true_y[cl_indexes] == labels[2]) / len(cl_indexes)
        print("\t> Cluster %d:" % (n+1))
        print("\t\t  Proportion of 1 = %.2f%%" % (proportion1 * 100))
        print("\t\t  Proportion of 7 = %.2f%%" % (proportion2 * 100))
        print("\t\t  Proportion of 9 = %.2f%%" % (proportion3 * 100))

get_labels_proportion(clusters_indexes, ydigit)

print("\n")
# Function to project data divided in clusters (classes/clusters) (not requested)
def plot_projected_clusters(data, 
                            clusters_indexes,
                            centroids = None,
                            title = "Projection of dataset",
                            labels = None,
                            leg_loc = "upper left",
                            save = False,
                            filename = "projected_data"):
    """
    Given the a dataset and a list of indexes, perform 2D 
    projection of data divided by the provided clusters 
    (classes or clusters). Also, plot centroids if provided. 
    """
    # Num of clusters division
    num_clusters = len(clusters_indexes)
    # MDS
    data_2d = ids.mds(data)
    # Assign clusters/classes and MDS
    clusters = []
    for n in range(num_clusters):
        group = data_2d[clusters_indexes[n],:]
        clusters.append(group)
    # Obtain centroids in 2D
    if centroids != None:
        centroids_2D = ids.project_centroids(data, centroids)
    # Plot clusters/classes and centroids
    for n, group in enumerate(clusters):
        # Plot clusters/classes
        if labels != None:
            label = "Class %d" % (labels[n])
        else:
            label = "Cluster %d" % (n+1)
        plt.scatter(group[:,0], group[:,1], label = label, ec = "black", zorder=3)
        if centroids != None:
            # Plot centroids
            label = None
            if n == len(clusters) - 1:
                label = "Centroids"    
            plt.scatter(centroids_2D[n][0], centroids_2D[n][1], marker = "X", color = "yellow", 
                                                                ec = "black", s = 120, 
                                                                zorder=4, label = label)                               
    # Add plot details
    ids.plot_details(title = title,
                     xlabel = "PC1",
                     ylabel = "PC2",
                     ax_equal = True,
                     bg = True,
                     legend = True,
                     leg_loc = leg_loc,
                     save = save,
                     filename = filename)

# Plot k-means predicted clusters division in 2D and centroids (not requested)             
plot_projected_clusters(xdigit, 
                        clusters_indexes = clusters_indexes,
                        centroids = centroids,
                        title = "2D k-means clusters projection of MNIST (no prior MDS)",
                        save = False,
                        filename = "2D k-means clusters projection of MNIST (no prior MDS)")
                
## True classes
# Obtain true classes indexes                    
bol_indexes1 = ydigit == 1
bol_indexes7 = ydigit == 7
bol_indexes9 = ydigit == 9
classes_indexes = bol_indexes1, bol_indexes7, bol_indexes9

# Plot true classes division in 2D and centroids (not requested)                         
plot_projected_clusters(xdigit, 
                        clusters_indexes = classes_indexes,
                        labels = [1,7,9],
                        title = "2D true classes projection of MNIST data",
                        save = False,
                        filename = "2D true classes projection of MNIST data")


#### Exercise 9b (Classification)

# K-NN test accuracy by cross validation (no prior MDS)
print("\nPerforming k-NN, please wait..")
kbest, knn_accuracy = ids.get_kbest_cv(xtrain = xdigit, 
                                       ytrain = ydigit, 
                                       kmax=5, 
                                       kfold=3)        
print("K-best = %d, %d-NN test classification accuracy = %.2f%% (no prior MDS)" % (kbest, kbest, knn_accuracy))


######## Ex. 10 (Clustering and classiﬁcation after dimensionality reduction)

#### Ex. 10a

# Perform PCA and compute the cumulative variance in percentage
eigvals, eigvect = ids.pca(xdigit)
cum_var = np.cumsum(eigvals/np.sum(eigvals)) * 100

# Plot the cumulative variance (in %) against the pcs (plot only the real part of the complex numbers)
plt.plot(np.arange(1, len(cum_var)+1), cum_var.real)
plt.yticks(np.linspace(cum_var[0].real,100,10))
plt.xticks(np.linspace(1,len(cum_var),10))
ids.plot_details(title = 'Cumulative variance versus PC indexes',
                 xlabel = "PC indexes",
                 ylabel = "Cumulative variance in %",
                 bg = True,
                 save = False,
                 filename = "Cumulative variance versus PC indexes")

#### Ex. 10b

# Define the function to plot the image
def plot_image(centroid, n_cluster, dim):
    """
    Plot the image as a n by n pixels image.
    """
    image = centroid.reshape((28, 28))
    plt.imshow(image.real, cmap = 'viridis')
    title = "Cluster %d image (after MDS to %d pcs)" % ((n_cluster+1), dim)
    ids.plot_details(title = title,
                     xlabel = "X-axis (pixels)",
                     ylabel = "Y-axis (pixels)",
                     save = False,
                     filename = title)

def plot_centroids_image(centroids_reduced, eigvect, original_data):
    """
    Plot the centroid image.
    """
    # Compute the mean of the dataset before MDS
    mean = np.mean(original_data, axis = 0)
    # # Get the original dimensions
    dim = len(centroids_reduced[0])
    # Check if the centroids has been projected into 2D or more
    if dim > 2:
        # Merge the centroids into an array
        centroids_array = np.array(centroids_reduced).T
        # Reproject the centroids in 784 dimensions
        centroids784D = (eigvect[:,:dim] @ centroids_array).T
    else:
        # Merge the centroids into an array
        centroids_array = np.array(centroids_reduced)
        # Reproject the centroids in 784 dimensions
        centroids784D = (centroids_array @ eigvect[:,:2].T)
    # Add the mean of the original data
    centroids784D = centroids784D + mean
    # Plot the image of the three centroids (rapresented by 28 x 28 pixels)
    for n_cluster, centroid in enumerate(centroids784D):
        plot_image(centroid, n_cluster, dim)

def kmeans_MNIST(xdata, ydata, dimensions, k = 3):
    """
    The function apply MDS to the xdata (MNIST), then it perform k-means 
    clustering and evaluate its prediction. To evaluate the clustering 
    performance after MDS, the function print the labels proportion 
    for each cluster, plot the centroids and the clusters projection in 
    2D and finally plot each centroid as a 28 by 28 pixels image.
    """
    # Perform mds
    xdata_reduced = ids.mds(xdata, dimensions)   
    # Obtain centroids and cluster indexes
    centroids_reduced = ids.kmeans_fit(xdata_reduced, k)                                        
    clusters_indexes_reduced = ids.kmeans_clustering(xdata_reduced, 
                                                     centroids_reduced, 
                                                     indexes_only=True)
    # Get the proportion of the labels for each cluster
    get_labels_proportion(clusters_indexes_reduced, ydata)
    # Plot the centroids and the predicted clusters division in 2D (not requested)
    plot_projected_clusters(xdata_reduced, 
                            clusters_indexes = clusters_indexes_reduced,
                            centroids = centroids_reduced,
                            title = "2D k-means clusters projection of MNIST (after MDS to %d pcs)" % dimensions,
                            save = False,
                            filename = "plots/2D k-means clusters projection of MNIST (after MDS to %d pcs)" % dimensions)
    # Plot the image of the three centroids
    plot_centroids_image(centroids_reduced, eigvect, xdigit)

## K-means clustering on data projected on 200 pcs
print("\nClustering after MDS to 200 pcs:")
kmeans_MNIST(xdigit, ydigit, 200)

## K-means clustering on data projected on 20 pcs
print("\nClustering after MDS to 20 pcs:")
kmeans_MNIST(xdigit, ydigit, 20)

## K-means clustering on data projected on 2 pcs
print("\nClustering after MDS to 2 pcs:")
kmeans_MNIST(xdigit, ydigit, 2)


#### Ex. 10c

# K-NN test accuracy by cross validation (after MDS to 20 pcs)
xdigit20D = ids.mds(xdigit, 20)
print("\nPerforming k-NN, please wait..")
kbest, knn_accuracy = ids.get_kbest_cv(xtrain = xdigit20D, 
                                       ytrain = ydigit, 
                                       kmax=5, 
                                       kfold=3)        
print("K-best = %d, %d-NN test classification accuracy = %.2f%% (after MDS to 20 pcs)" % (kbest, kbest, knn_accuracy))

# K-NN test accuracy by cross validation (after MDS to 200 pcs)
xdigit200D = ids.mds(xdigit, 200)
print("\nPerforming k-NN, please wait..")
kbest, knn_accuracy = ids.get_kbest_cv(xtrain = xdigit200D, 
                                       ytrain = ydigit, 
                                       kmax=5, 
                                       kfold=3)        
print("K-best = %d, %d-NN test classification accuracy = %.2f%% (after MDS to 200 pcs)" % (kbest, kbest, knn_accuracy))