# Import modules
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

### Exercise 1

# Load data
data = np.loadtxt("smoking.txt")
# Divide the dataset into two groups consisting of smokers and non-smokers
non_smokers = data[data[:,4] == 0]
smokers = data[data[:,4] == 1]
# Computes the average lung function among the smokers and among the non-smokers
avg_fev_nonsmok = np.mean(non_smokers[:,1])
avg_fev_smok = np.mean(smokers[:,1])
print("\nAverage lung function among non-smokers:", avg_fev_nonsmok)
print("Average lung function among smokers:", avg_fev_smok)


### Exercise 2

# Make a box plot of the FEV1 in the two groups
fig = plt.figure()
ax = fig.add_subplot(111)
labels = ["Non-smokers", "Smokers"]
box = ax.boxplot([non_smokers[:,1], smokers[:,1]], 
                 labels = labels, 
                 patch_artist = True, 
                 notch = True, 
                 sym = 'x')
ax.set(yticks = np.arange(0, 8))
colors = ['#1f77b4', 'red']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
ax.set_title("Boxplot over lung functions for non-smokers and smokers", 
              fontsize = 15, 
              y = 1.04)
ax.set_ylabel("Lung function", fontsize = 12)  
plt.grid(axis = "y", alpha = 0.5)
plt.show()

### Exercise 3

## Implemented T-test
def t_test(x, y, alpha = 0.05):
    """
    Determine if there is a significant difference 
    between the means of two groups of data
    """
    N1 = len(x)
    N2 = len(y)
    s1 = np.std(x)
    s2 = np.std(y)
    # Calculate the t-statistics
    t = (avg_fev_nonsmok - avg_fev_smok) / np.sqrt( (s1**2/N1) + (s2**2/N2) ) 
    # Calculate the degree of freedom
    num = ((s1**2)/N1 + (s2**2)/N2)**2
    den1 = (s1**4) / ((N1**2)*(N1-1))
    den2 = (s2**4) / ((N2**2)*(N2-1))
    v = int(num / (den1+den2))
    # Calculoate the p-value
    p_value = 2 * sp.t.cdf(-abs(t), v)
    # Binary response indicating rejection or non rejection of the null hypothesis
    if p_value < alpha:
        reject_null = True
    else:
        reject_null = False
    return{"t_stats": t, "p_value": p_value, "df": v, "alpha": alpha, "reject_null": reject_null}

t = t_test(non_smokers[:,1], smokers[:,1])
print("\nT-statistics =", t["t_stats"],
      "\nDegree of freedom =", t["df"], 
      "\np-value = ", t["p_value"], 
      "\nReject null hypothesis =", t["reject_null"])


### Exercise 4

## Correlation between age and FEV1 for all observations
pearson = sp.stats.pearsonr(data[:,0], data[:,1])
spearman = sp.stats.spearmanr(data[:,0], data[:,1])
print("\nPearson (all data) =", pearson[0])
print( "Spearman (all data) =", spearman[0])
## Correlation between age and FEV1 for the two groups divided
# Pearson correlation
pearson_nosmoke = sp.stats.pearsonr(non_smokers[:,0], non_smokers[:,1])
pearson_smoke = sp.stats.pearsonr(smokers[:,0], smokers[:,1])
print("\nPearson (non-smokers) =", pearson_nosmoke[0])
print("Pearson (smokers ) =", pearson_smoke[0])                     
# Spearman correlation
spearman_nosmoke = sp.stats.spearmanr(non_smokers[:,0], non_smokers[:,1])
spearman_smoke = sp.stats.spearmanr(smokers[:,0], smokers[:,1])
print("\nSpearman (non-smokers) =", spearman_nosmoke[0])
print("Spearman (smokers) =", spearman_smoke[0])

# 2D plot
plt.style.use("ggplot")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(non_smokers[:,0], non_smokers[:,1], 
           label = "Non-smokers",
           color = "#1f77b4",
           ec = "darkblue")
ax.scatter(smokers[:,0], smokers[:,1], 
           label = "Smokers",
           color = "red",
           ec = "darkred")
ax.set(yticks = np.arange(8),
       xticks = np.arange(2, 21))
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')         
ax.set_title("2D plot of lung function over age for smokers and non-smokers", 
             fontsize = 14, 
             y = 1.04)
ax.set_xlabel("Age", fontsize = 12)
ax.set_ylabel("Lung function", fontsize = 12)
legend = ax.legend(frameon = 1, loc = 'upper left', shadow = True)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
plt.show()


### Exercise 5

# Histograms
plt.style.use("default")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(non_smokers[:,0], 
        label = "Non-smokers", 
        bins = np.arange(3,21) - 0.5,
        edgecolor='black')
ax.hist(smokers[:,0], 
        label = "Smokers", 
        bins = np.arange(3,21) - 0.5,
        color = "red",
        edgecolor = 'black',
        rwidth = 0.75)
ax.set(yticks = np.arange(0,11) * 10,
       xticks = np.arange(3, 20))
ax.set_title("Histogram over age for non-smokers and smokers", 
             fontsize = 15, 
             y = 1.04)
ax.set_xlabel('Age', fontsize = 12)
ax.set_ylabel('Count', fontsize = 12)
ax.grid(axis = "y", alpha = 0.5)
legend = ax.legend(frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
plt.show()