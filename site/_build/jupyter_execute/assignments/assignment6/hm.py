## Homework - Instructions

![](https://github.com/rpi-techfundamentals/hm-01-starter/blob/master/notsaved.png?raw=1)

**WARNING!!!  If you see this icon on the top of your COLAB sesssion, your work is not saved automatically.**


**When you are working on homeworks, make sure that you save often. You may find it easier to save intermident copies in Google drive. If you save your working file in Google drive all changes will be saved as you work. MAKE SURE that your final version is saved to GitHub.** 

Before you turn this problem in, make sure everything runs as expected. First, restart the kernel (in the menubar, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All).  You can speak with others regarding the assignment but all work must be your own. 


### This is a 30 point assignment graded from answers to questions and automated tests that should be run at the bottom. Be sure to clearly label all of your answers and commit final tests at the end.  

files = "https://github.com/rpi-techfundamentals/introml_website_fall_2020/raw/master/files/assignment6.zip" 
!pip install otter-grader && wget $files && unzip -o assignment6.zip

### Load Data
Here we have 2 files. One we will use for PCA and the other for cluster analysis. 


# Load the data here
import pandas as pd
df_cluster  = pd.read_csv("cluster.csv")
df_pca  = pd.read_csv("pca.csv")

## EDA

Do some simple exploritory analysis on the data so that you understand the structure of the data. 



### 1. PCA Data Baseline Regression.
  On the PCA data, perform a 50/50 train test split with `random state` equal to 99. Predict `y` with all variables using regression analysis. 
  
  1a. Calculate the r2 for train and assign to `pca1_r2_train` 
  
  1b. calculate the r2 for test and assign to `pca1_r2_test`. 






### PCA Analysis

So you should find that the overall r2 is quite high, but we have a really complex model with 150 predictors. Run PCA with 4, 5, and 6 components. For example, running with 4 components means setting `n_components=4`. 

* Check out the variance explained from each of the numbers of principal components. When you find taht increasing the number of components only increases the variance explained by a small amount.  * 







After experimenting with 4, 5, and 6 components, explain why 5 is the correct number of components.

man1="""
Answer here
"""

2. Usign just the 5 PCA components as X, perform a 50/50 train test split with `random state` equal to 99. Predict `y` with all variables using regression analysis. 

2a. Calculate the r2 for train and assign to `pca2_r2_train`.

2b. Calculate the r2 for test and assign to `pca2_r2_test`. 




## Challenge Problem: Feature Selection.  
While we obtained a decent R2 with PCA, it wasn't as good as had with the origional data.  Rather than dimensionality reduction using principal components, try to use feature selection to get 4 components that explain >99% of the variance. 

List those features here. 

man2="""
Answer here
"""

### 3. Cluster Data Baseline Regression
 On the Cluster data, perform a 50/50 train test split with `random state` equal to 99. Predict `y` with all variables using regression analysis. 
 
3a.  Calculate the r2 for train and assign to `cluster1_r2_train`. 

3b. calculate the r2 for test and assign to `cluster1_r2_test`. 



### KMeans Cluster Analysis

Next perform a cluster analysis using ONLY variables that start with `cad0`-`cad9` and specify 6 clusters.  Set random_state to `99` for KMEANS algorithm. Add the variable `df_cluster['cluster']` to the origional dataframe to indicate the cluster membership. 




### 4. Clusters Continued

Then select only rows from df_clusterwhich you have assigned to cluster 1.  For only cluster 1 predict `y` with all variables using regression analysis. 

4a. Calculate the r2 for train and assign to `cluster2_r2_train`.

4b. calculate the r2 for test and assign to `cluster2_r2_test`. Like before set random_state to `99`.





#This runs all tests. 
import otter
grader = otter.Notebook()
grader.check_all()