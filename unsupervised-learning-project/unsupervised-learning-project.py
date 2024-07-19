# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#00b3e5;">Contents: Unsupervised Learning Project
# 1. [Part-A: Solution](#Part-A:-Solution)
# 2. [Part-B: Solution](#Part-B:-Solution)

# %% [markdown]
# # Part-A: Solution

# %% [markdown]
# - **DOMAIN:** Automobile
# - **CONTEXT:** The data concerns city-cycle fuel consumption in miles per gallon to be predicted in terms of 3 multivalued discrete and 5 continuous attributes.
# - **DATA DESCRIPTION:**
#  1. cylinders: multi-valued discrete 
#  2. displacement: continuous 
#  3. horsepower: continuous 
#  4. weight: continuous 
#  5. mpg: continuous
#  6. acceleration: continuous
#  7. model year: multi-valued discrete
#  8. origin: multi-valued discrete
#  9. car name: string (unique for each instance)
# - **PROJECT OBJECTIVE:** To understand K-means Clustering by applying on the Car Dataset to segment the cars into various categories.

# %%
# Import all the relevant libraries needed to complete the analysis, visualization, modeling and presentation
import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
sns.set_style('darkgrid')
# %matplotlib inline

from scipy import stats
from scipy.stats import zscore

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_samples, silhouette_score
from kmodes.kprototypes import KPrototypes

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Data Understanding & Exploration:

# %% [markdown]
# ### 1A. Read ‘Car name.csv’ as a DataFrame and assign it to a variable.

# %%
# CSV File 1
dfa1=pd.read_csv('Car name.csv')

# %%
dfa1.info()
dfa1.head()

# %% [markdown]
# ### 1B. Read ‘Car-Attributes.json as a DataFrame and assign it to a variable.

# %%
# JSON File 1
dfa2=pd.read_json('Car-Attributes.json')

# %%
dfa2.info()
dfa2.head()

# %% [markdown]
# ### 1C. Merge both the DataFrames together to form a single DataFrame

# %%
dfa=pd.concat([dfa2,dfa1],axis=1)

# %%
dfa.info()
dfa.head()

# %%
# Save this dataframe to csv, xlsx and json for general observation
# dfa.to_csv('mpg.csv', index=False)
# dfa.to_excel('mpg.xlsx', index = False)
# dfa.to_json('mpg.json', orient = 'split', compression = 'infer', index = 'true')

# %% [markdown]
# ### 1D. Print 5 point summary of the numerical features and share insights.

# %%
# Describe function generates descriptive statistics that summarize the central tendency, 
# dispersion and shape of a dataset’s distribution, excluding NaN values.

# This method tells us a lot of things about a dataset. One important thing is that 
# the describe() method deals only with numeric values. It doesn't work with any 
# categorical values. So if there are any categorical values in a column the describe() 
# method will ignore it and display summary for the other columns.
dfa.describe().T


# %% [markdown]
# **Observations:**
#
# - mpg:
#  - Mean and Median are nearly equal. Distribution might be normal.
#  - 75 % of values are less than 29, and maximum value is 46.
# - cyl:
#  - Mean and median are not equal. Skewness is expected.
#  - Range of values is small. It may be a categorical feature.
# - disp:
#  - Mean and Median are not equal. There is big  standard deviation. 
#  - Distribution is not normal because of big SD.
# - wt:
#  - Mean and Median are not equal. Some skewness is present.
#  - 75% of values are less than 3600, and maximum value is 5140.
# - acc:
#  - Mean and Median are nearly equal. Distribution might be normal.
#  - Not much deviation in the data.
# - yr:
#  - Mean and Median are nearly equal. Distribution might be normal.
#  - Range of values is small. It may be a categorical feature.
# - origin
#  - Mean and Median are not equal. Some skewness is present.
#  - Not much deviation in the data. Range of values is small.

# %% [markdown]
# ## 2. Data Preparation & Analysis:

# %% [markdown]
# ### 2A. Check and print feature-wise percentage of missing values present in the data and impute with the best suitable approach.

# %%
# Percentage of missing values

# df.isnull().sum()
# df.isna().sum()

def missing_check(df):
    total = df.isnull().sum().sort_values(ascending=False)   # total number of null values
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)  # percentage of values that are null
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # putting the above two together
    return missing_data # return the dataframe
missing_check(dfa)

# %% [markdown]
# **2H. Check for unexpected values in all the features and datapoints with such values.**

# %%
# NaN shows no missing values, but on careful data observation we can see "?" for hp values
dfa[dfa['hp']=="?"]

# %%
# Replace missing values with NaN
dfa = dfa.replace('?', np.nan)

dfa.isna().sum()

# %%
# Understand the median values of all the variables
dfa.median()

# %%
# Replace hp with median value
dfa['hp'].fillna((dfa['hp'].median()), inplace=True)

# %%
print('Now the data set has no missing values:')
dfa.isnull().sum()

# %% [markdown]
# ### 2B. Check for duplicate values in the data and impute with the best suitable approach.

# %%
# To take a look at the duplication in the DataFrame as a whole, just call the duplicated() method on 
# the DataFrame. It outputs True if an entire row is identical to a previous row.
dfa.duplicated().sum()

# %%
# Count the number of non-duplicates
(~dfa.duplicated()).sum()

# %% [markdown]
# ### 2C. Plot a pairplot for all features.

# %%
# Change the Datatype of quantitative features
col_cat=['hp']
    
#Function to convert the categorical to quantitative
def convert_to_cont(feature):
    dfa[feature]=pd.to_numeric(dfa[feature], errors='coerce')
    
for c in col_cat:
    convert_to_cont(c)

# %%
dfa.info()

# %%
# Pair plot for the numeric attributes
sns.pairplot(dfa, diag_kind='kde');

# %% [markdown]
# **Observations:**
#
# - From diagonal plots, we can see origin, cyl and disp has 3 peaks.
# - hp, wt and yr showing 2 peaks.
# - Other peaks are not so prominent.
# - acc, mpg have a near normal distribution.
# - mpg shows negative relationship with wt, hp and disp.
# - disp has positive relationship with hp, wt, and negative relationship with mpg.
# - acc has negative relationship with wt, hp and disp.
# - More corresponding relationships can be inferred from the above graph if required.

# %% [markdown]
# ### 2D. Visualize a scatterplot for ‘wt’ and ‘disp’. Datapoints should be distinguishable by ‘cyl’.

# %%
plt.figure(figsize=(15,8))
sns.scatterplot(data = dfa, x="wt", y="disp", hue="cyl")

# %% [markdown]
# ### 2E. Share insights for Q2.d.

# %% [markdown]
# **Observations:**
# - Positive correlation exists between wt and disp.
# - The correlation between wt and disp decreases as the number of cylinders increase.
# - Good clusters are visible for 4, 6, and 8 cylinders.
# - Presence of outliers affect the value of regression coefficients.

# %% [markdown]
# ### 2F. Visualize a scatterplot for ‘wt’ and ’mpg’. Datapoints should be distinguishable by ‘cyl’.

# %%
plt.figure(figsize=(15,8))
sns.scatterplot(data = dfa, x="wt", y="mpg", hue="cyl")

# %% [markdown]
# ### 2G. Share insights for Q2.f.

# %% [markdown]
# **Observations:**
# - Negative correlation exists between wt and mpg.
# - The correlation between wt and mpg increases as the number of cylinders increase.
# - Good clusters are visible for 4 and 8 cylinders.
# - Presence of outliers affect the value of regression coefficients.

# %% [markdown]
# ### 2H. Check for unexpected values in all the features and datapoints with such values.
# [Hint: ‘?’ is present in ‘hp’]

# %%
# For sake of simplicity and better EDA, This part has been completed in 2A above.

# %% [markdown]
# ### Quick EDA

# %% [markdown]
# #### Correlation Heatmap

# %%
# Visualize a heatmap to understand correlation between all features
plt.figure(dpi = 120,figsize= (5,4))
mask = np.triu(np.ones_like(dfa.corr()))
sns.heatmap(dfa.corr(),mask = mask, fmt = ".2f",annot=True,lw=1,cmap = 'plasma')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.title('Correlation Heatmap')
plt.show()

# %%
corr = dfa.corr()
corr

# %%
dfa.var()

# %% [markdown]
# #### Distribution and Outliers

# %%
# Single Box Plot
plt.figure(figsize=(20,8))
ax = sns.boxplot(data=dfa, orient="h", palette="Set2")

# %%
# mpg

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['mpg'],  ax=axes[0],color='Green')
sns.boxplot(x = 'mpg', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['mpg'],25),np.percentile(dfa['mpg'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['mpg'] if i < lower or i > upper]
print('{} Total Number of outliers in mpg: {}'.format('\033[1m',len(Outliers)))

# %%
# cyl

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['cyl'],  ax=axes[0],color='Green')
sns.boxplot(x = 'cyl', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['cyl'],25),np.percentile(dfa['cyl'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['cyl'] if i < lower or i > upper]
print('{} Total Number of outliers in cyl: {}'.format('\033[1m',len(Outliers)))

# %%
# disp

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['disp'],  ax=axes[0],color='Green')
sns.boxplot(x = 'disp', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['disp'],25),np.percentile(dfa['disp'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['disp'] if i < lower or i > upper]
print('{} Total Number of outliers in disp: {}'.format('\033[1m',len(Outliers)))

# %%
# hp

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['hp'],  ax=axes[0],color='Green')
sns.boxplot(x = 'hp', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['hp'],25),np.percentile(dfa['hp'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['hp'] if i < lower or i > upper]
print('{} Total Number of outliers in hp: {}'.format('\033[1m',len(Outliers)))

# %%
# wt

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['wt'],  ax=axes[0],color='Green')
sns.boxplot(x = 'wt', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['wt'],25),np.percentile(dfa['wt'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['wt'] if i < lower or i > upper]
print('{} Total Number of outliers in wt: {}'.format('\033[1m',len(Outliers)))

# %%
# acc

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['acc'],  ax=axes[0],color='Green')
sns.boxplot(x = 'acc', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['acc'],25),np.percentile(dfa['acc'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['acc'] if i < lower or i > upper]
print('{} Total Number of outliers in acc: {}'.format('\033[1m',len(Outliers)))

# %%
# yr

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['yr'],  ax=axes[0],color='Green')
sns.boxplot(x = 'yr', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['yr'],25),np.percentile(dfa['yr'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['yr'] if i < lower or i > upper]
print('{} Total Number of outliers in yr: {}'.format('\033[1m',len(Outliers)))

# %%
# origin

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['origin'],  ax=axes[0],color='Green')
sns.boxplot(x = 'origin', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['origin'],25),np.percentile(dfa['origin'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['origin'] if i < lower or i > upper]
print('{} Total Number of outliers in origin: {}'.format('\033[1m',len(Outliers)))

# %%
# car_name

f,axes=plt.subplots(1,2,figsize=(17,7))
dfa['car_name'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('car_name',data=dfa,ax=axes[1])
axes[0].set_title('car_name Pie Chart')
axes[1].set_title('car_name Bar Graph')
plt.show()

# %% [markdown]
# #### Remove Outliers

# %%
# Impute outliers with mean

col=['mpg','hp','acc']
for c in col:
    # Use the IQR method
    q25,q75=np.percentile(dfa[c],25),np.percentile(dfa[c],75)
    IQR=q75-q25
    Threshold=IQR*1.5
    lower,upper=q25-Threshold,q75+Threshold
    Outliers=[i for i in dfa[c] if i < lower or i > upper]
    print('{} Total Number of outliers in {} Before Imputing : {}'.format('\033[1m',c,len(Outliers)))
    # Mean of the column without considering the outliers
    dfa_include = dfa.loc[(dfa[c] >= lower) & (dfa[c] <= upper)]
    mean=int(dfa_include[c].mean())
    print('{} Mean of {} is {}'.format('\033[1m',c,mean))
    # Impute outliers with mean
    dfa[c]=np.where(dfa[c]>upper,mean,dfa[c])
    dfa[c]=np.where(dfa[c]<lower,mean,dfa[c])
    Outliers=[i for i in dfa[c] if i < lower or i > upper]
    print('{} Total Number of outliers in {} After Imputing : {}'.format('\033[1m',c,len(Outliers)))  
    print('\n')

# %%
dfa.head()

# %% [markdown]
# #### Normalize/Standardize the data with the best suitable approach.

# %%
dfa1=dfa.drop(['car_name'],axis=1)

# %%
# Using different scaling methods:
# scaler = RobustScaler()
# scaler = MinMaxScaler()
scaler = StandardScaler()
# mydata = mydata.apply(zscore)

cols_to_scale = ["mpg","cyl","disp","hp","wt","acc","yr","origin"]

dfa1[cols_to_scale] = scaler.fit_transform(dfa1[cols_to_scale])

# %%
dfa1.head()

# %% [markdown]
# ## 3. Clustering with all Features:

# %% [markdown]
# ### 3A. Apply K-Means clustering for 2 to 10 clusters.

# %%
# Train KMeans for 2 to 10 clusters

cluster_range = range(2,11)
inertia = []
silh_score = []
for num_clusters in cluster_range:
    clusters = KMeans(num_clusters, n_init = 10, random_state=0)
    clusters.fit(dfa1)
    labels = clusters.labels_
    centroids = clusters.cluster_centers_
    inertia.append(clusters.inertia_)
    silh_score.append(silhouette_score(dfa1, labels))
    
# Dataframe for num_clusters, cluster_errors and silh_score
clusters_dfa1 = pd.DataFrame({"num_clusters": cluster_range, "cluster_errors": inertia, "silh_score": silh_score})
clusters_dfa1

# %% [markdown]
# ### 3B. Plot a visual and find elbow point.

# %%
# Select K based on Inertia or Cluster Errors
plt.figure(figsize=(12,6))
plt.plot( clusters_dfa1.num_clusters, clusters_dfa1.cluster_errors, marker = "o" )

plt.xlabel('K')
plt.ylabel('Cluster Errors')
plt.plot(clusters_dfa1.num_clusters[1], clusters_dfa1.cluster_errors[1], 'ro')
plt.title('Selecting k with the Elbow Method')

# %% [markdown]
# Let's plot the silhouette score as a function of K :

# %%
# Select K based on Silhouette Score
plt.figure(figsize=(12,6))
plt.plot(clusters_dfa1.num_clusters, clusters_dfa1.silh_score, marker = "D", color = 'g')

plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.plot(clusters_dfa1.num_clusters[1], clusters_dfa1.silh_score[1], 'rd')
plt.title('Selecting k with Silhouette Score')

# %% [markdown]
# ### 3C. On the above visual, highlight which are the possible Elbow points.

# %%
# Consider the k=2 and k=3

# As you can see, there is an elbow at  𝑘=3 , which means that less clusters than that would be bad, 
# and more clusters would not help much and might cut clusters in half. So  𝑘=3  is a pretty good choice.

# %% [markdown]
# Another approach is to look at the silhouette score, which is the mean silhouette coefficient over all the instances. An instance's silhouette coefficient is equal to  (𝑏−𝑎)/max(𝑎,𝑏)  where  𝑎  is the mean distance to the other instances in the same cluster (it is the mean intra-cluster distance), and  𝑏  is the mean nearest-cluster distance, that is the mean distance to the instances of the next closest cluster (defined as the one that minimizes  𝑏 , excluding the instance's own cluster). The silhouette coefficient can vary between -1 and +1: a coefficient close to +1 means that the instance is well inside its own cluster and far from other clusters, while a coefficient close to 0 means that it is close to a cluster boundary, and finally a coefficient close to -1 means that the instance may have been assigned to the wrong cluster.

# %%
# Consider the k=2 and k=3

# As you can see, this visualization is much richer than the previous one: in particular, 
# although it confirms that  𝑘=2  is a very good choice, but it also underlines the fact 
# that  𝑘=3  is quite good as well.

# %% [markdown]
# ### 3D. Train a K-means clustering model once again on the optimal number of clusters.

# %%
# Train and fit K-means algorithm on the relevant K
kmeans = KMeans(n_clusters=3, n_init = 10, random_state=0)
kmeans.fit(dfa1)

# %%
# Check the number of datapoints in each cluster
labels = kmeans.labels_
counts = np.bincount(labels[labels>=0])
print(counts)

# %%
# let us check the cluster centers in each group
centroids = kmeans.cluster_centers_
centroid_dfa1 = pd.DataFrame(centroids, columns = list(dfa1) )
centroid_dfa1.transpose()

# %% [markdown]
# ### 3E. Add a new feature in the DataFrame which will have labels based upon cluster value.

# %%
# Add cluster numbers to original cars data
predictions = kmeans.predict(dfa1)

dfa["group"] = predictions
dfa['group'] = dfa['group'].astype('category')
dfa.head()

# %%
# Save this dataframe to csv, xlsx for general observation
# dfa.to_csv('mpg1.csv', index=False)
# dfa.to_excel('mpg1.xlsx', index = False)

# %%
# Add cluster numbers to scaled cars data
predictions = kmeans.predict(dfa1)

dfa1["group"] = predictions
dfa1['group'] = dfa['group'].astype('category')
dfa1.head()

# %%
# Visualize the clusters with respect to various attributes
dfa1.boxplot(by = 'group', layout=(3,4), figsize=(15, 10))

# %% [markdown]
# ### 3F. Plot a visual and color the datapoints based upon clusters.

# %%
# In my opinion, the best approach is to use multiple scatter plots, either in a matrix format or 
# by changing between variables. You can also consider using some data reduction method such as 
# PCA to consolidate your variables into a smaller number of factors.
plt.figure(figsize=(20,8))
sns.scatterplot(data=dfa, x="mpg", y="cyl", hue="group", palette="deep")

# %%
plt.figure(figsize=(20,8))
sns.scatterplot(data=dfa, x="mpg", y="origin", hue="group", palette="deep")

# %%
plt.figure(figsize=(20,8))
sns.scatterplot(data=dfa, x="mpg", y="hp", hue="group", palette="deep")

# %%
plt.figure(figsize=(20,8))
sns.scatterplot(data=dfa, x="mpg", y="wt", hue="group", palette="deep")

# %%
plt.figure(figsize=(20,8))
sns.scatterplot(data=dfa, x="wt", y="hp", hue="group", palette="deep")

# %% [markdown]
# ### 3G. Pass a new DataPoint and predict which cluster it belongs to.

# %%
# Initialize list elements
data = [10, 5, 100, 50, 2000, 21, 70, 1, 'toyota']
  
# Create the pandas DataFrame with column names provided explicitly
new_datapoint = pd.DataFrame([data], columns=['mpg','cyl','disp','hp','wt','acc','yr','origin','car_name'])
  
# Print dataframe.
new_datapoint

# %%
new_datapoint1=new_datapoint.drop(['car_name'],axis=1)

# %%
new_datapoint1.info()

# %%
# Add cluster numbers to the new datapoint created above
predictions = kmeans.predict(new_datapoint1)

new_datapoint["group"] = predictions
new_datapoint['group'] = new_datapoint['group'].astype('category')
new_datapoint.head()

# %% [markdown]
# **Improving the quality of clusters in next section...**

# %% [markdown]
# ## 3. Clustering with reduced Features:

# %%
dfa=dfa.drop(['group'],axis=1)
dfa.head()

# %%
# Use dummy variables for cyl and origin. We can also drop them.

# dfa1 = pd.get_dummies(data=dfa1, columns=['origin', 'cyl'])
# dfa1=dfa1.drop(["origin", "yr"],axis=1)

dfa1=dfa.drop(["yr","origin","car_name"],axis=1)
dfa1.head()

# Removing features based on following articles:
# https://towardsdatascience.com/interpretable-k-means-clusters-feature-importances-7e516eeb8d3c
# https://towardsdatascience.com/the-k-prototype-as-clustering-algorithm-for-mixed-data-type-categorical-and-numerical-fe7c50538ebb

# %%
# Using different scaling methods:
# scaler = RobustScaler()
# scaler = MinMaxScaler()
scaler = StandardScaler()
# mydata = mydata.apply(zscore)

cols_to_scale = ["mpg","cyl","disp","hp","wt","acc"]

dfa1[cols_to_scale] = scaler.fit_transform(dfa1[cols_to_scale])

# %%
dfa1.head()

# %% [markdown]
# ### 3A. Apply K-Means clustering for 2 to 10 clusters.

# %%
# Train KMeans for 2 to 10 clusters

cluster_range = range(2,11)
inertia = []
silh_score = []
for num_clusters in cluster_range:
    clusters = KMeans(num_clusters, n_init = 10, random_state=0)
    clusters.fit(dfa1)
    labels = clusters.labels_
    centroids = clusters.cluster_centers_
    inertia.append(clusters.inertia_)
    silh_score.append(silhouette_score(dfa1, labels))
    
# Dataframe for num_clusters and cluster_errors
clusters_dfa1 = pd.DataFrame({"num_clusters": cluster_range, "cluster_errors": inertia, "silh_score": silh_score})
clusters_dfa1

# %% [markdown]
# **With reduced set of features, we have lower cluster errors and higher Silhouette Scores.**

# %% [markdown]
# ### 3B. Plot a visual and find elbow point.

# %%
# Select K based on Inertia or Cluster Errors
plt.figure(figsize=(12,6))
plt.plot( clusters_dfa1.num_clusters, clusters_dfa1.cluster_errors, marker = "o" )

plt.xlabel('K')
plt.ylabel('Cluster Errors')
plt.plot(clusters_dfa1.num_clusters[1], clusters_dfa1.cluster_errors[1], 'ro')
plt.title('Selecting k with the Elbow Method')

# %% [markdown]
# Let's plot the silhouette score as a function of K :

# %%
# Select K based on Silhouette Score
plt.figure(figsize=(12,6))
plt.plot(clusters_dfa1.num_clusters, clusters_dfa1.silh_score, marker = "D", color = 'g')

plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.plot(clusters_dfa1.num_clusters[1], clusters_dfa1.silh_score[1], 'rd')
plt.title('Selecting k with Silhouette Score')

# %% [markdown]
# ### 3C. On the above visual, highlight which are the possible Elbow points.

# %%
# Consider the k=2 and k=3

# As you can see, there is an elbow at  𝑘=3, which means that less clusters than that would be bad, 
# and more clusters would not help much and might cut clusters in half. So  𝑘=3  is a pretty good choice.

# %% [markdown]
# Another approach is to look at the silhouette score, which is the mean silhouette coefficient over all the instances. An instance's silhouette coefficient is equal to  (𝑏−𝑎)/max(𝑎,𝑏)  where  𝑎  is the mean distance to the other instances in the same cluster (it is the mean intra-cluster distance), and  𝑏  is the mean nearest-cluster distance, that is the mean distance to the instances of the next closest cluster (defined as the one that minimizes  𝑏 , excluding the instance's own cluster). The silhouette coefficient can vary between -1 and +1: a coefficient close to +1 means that the instance is well inside its own cluster and far from other clusters, while a coefficient close to 0 means that it is close to a cluster boundary, and finally a coefficient close to -1 means that the instance may have been assigned to the wrong cluster.

# %%
# Consider the k=2 and k=3

# As you can see, this visualization is much richer than the previous one: in particular, 
# although it confirms that  𝑘=2  is a very good choice, but it also underlines the fact 
# that  𝑘=3  is quite good as well.

# %% [markdown]
# ### 3D. Train a K-means clustering model once again on the optimal number of clusters.

# %%
# Train and fit K-means algorithm on the relevant K
kmeans = KMeans(n_clusters=3, n_init = 10, random_state=0)
kmeans.fit(dfa1)

# %%
# Check the number of datapoints in each cluster
labels = kmeans.labels_
counts = np.bincount(labels[labels>=0])
print(counts)

# %%
# let us check the cluster centers in each group
centroids = kmeans.cluster_centers_
centroid_dfa1 = pd.DataFrame(centroids, columns = list(dfa1) )
centroid_dfa1.transpose()

# %% [markdown]
# ### 3E. Add a new feature in the DataFrame which will have labels based upon cluster value.

# %%
# Add cluster numbers to original cars data
predictions = kmeans.predict(dfa1)

dfa["group"] = predictions
dfa['group'] = dfa['group'].astype('category')
dfa.head()

# %%
#Save this dataframe to csv, xlsx for general observation
# dfa.to_csv('mpg1.csv', index=False)
# dfa.to_excel('mpg1.xlsx', index = False)

# %%
# Add cluster numbers to scaled cars data
predictions = kmeans.predict(dfa1)

dfa1["group"] = predictions
dfa1['group'] = dfa['group'].astype('category')
dfa1.head()

# %%
# Visualize the clusters with respect to various attributes
dfa1.boxplot(by = 'group', layout=(3,4), figsize=(15, 10))

# %% [markdown]
# ### 3F. Plot a visual and color the datapoints based upon clusters.

# %%
# In my opinion, the best approach is to use multiple scatter plots, either in a matrix format or 
# by changing between variables. You can also consider using some data reduction method such as 
# PCA to consolidate your variables into a smaller number of factors.
plt.figure(figsize=(20,8))
sns.scatterplot(data=dfa, x="mpg", y="cyl", hue="group", palette="deep")

# %%
plt.figure(figsize=(20,8))
sns.scatterplot(data=dfa, x="mpg", y="origin", hue="group", palette="deep")

# %%
plt.figure(figsize=(20,8))
sns.scatterplot(data=dfa, x="mpg", y="hp", hue="group", palette="deep")

# %%
plt.figure(figsize=(20,8))
sns.scatterplot(data=dfa, x="mpg", y="wt", hue="group", palette="deep")

# %%
plt.figure(figsize=(20,8))
sns.scatterplot(data=dfa, x="wt", y="hp", hue="group", palette="deep")

# %% [markdown]
# ### 3G. Pass a new DataPoint and predict which cluster it belongs to.

# %%
# Initialize list elements
data = [10, 5, 100, 50, 2000, 21, 70, 1, 'toyota']
  
# Create the pandas DataFrame with column names provided explicitly
new_datapoint = pd.DataFrame([data], columns=['mpg','cyl','disp','hp','wt','acc','yr','origin','car_name'])
  
# Print dataframe.
new_datapoint

# %%
new_datapoint1=new_datapoint.drop(["yr","origin","car_name"],axis=1)

# %%
new_datapoint1.info()

# %%
# Add cluster numbers to the new datapoint created above
predictions = kmeans.predict(new_datapoint1)

new_datapoint["group"] = predictions
new_datapoint['group'] = new_datapoint['group'].astype('category')
new_datapoint.head()

# %% [markdown]
# **We can do similar analysis for 2 groups as well to check if we get more clear distinction among groups.**

# %% [markdown]
# # Part-B: Solution

# %% [markdown]
# - **DOMAIN:** Automobile
# - **CONTEXT:** The purpose is to classify a given silhouette as one of three types of vehicle, using a set of features extracted from the silhouette. The vehicle may be viewed from one of many different angles.
# - **DATA DESCRIPTION:**
# The data contains features extracted from the silhouette of vehicles in different angles. Four "Corgie" model vehicles were used for the experiment: a double decker bus, Cheverolet van, Saab 9000 and an Opel Manta 400 cars. This particular combination of vehicles was chosen with the expectation that the bus, van and either one of the cars would be readily distinguishable, but it would be more difficult to distinguish between the cars.
#  - All the features are numeric i.e. geometric features extracted from the silhouette.
# - **PROJECT OBJECTIVE:** Apply dimensionality reduction technique – PCA and train a model and compare relative results.

# %%
# Import all the relevant libraries needed to complete the analysis, visualization, modeling and presentation
import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
sns.set_style('darkgrid')
# %matplotlib inline

from scipy import stats
from scipy.stats import zscore

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score 
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score
from sklearn.metrics import plot_precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, plot_roc_curve 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC

from sklearn.decomposition import PCA

# import xgboost as xgb
# from xgboost import plot_importance
# from catboost import CatBoostClassifier
# from lightgbm import LGBMClassifier

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import VotingClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Data Understanding and Cleaning:

# %% [markdown]
# ### 1A. Read ‘vehicle.csv’ and save as DataFrame.

# %%
# Read the CSV File 1
dfb=pd.read_csv('vehicle.csv')

# %%
dfb.info()
dfb.head()

# %%
# Analyze the distribution of the dataset
dfb.describe().T

# %%
dfb.columns


# %% [markdown]
# ### 1B. Check percentage of missing values and impute with correct approach.

# %%
# Percentage of missing values

# df.isnull().sum()
# df.isna().sum()

def missing_check(df):
    total = df.isnull().sum().sort_values(ascending=False)   # total number of null values
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)  # percentage of values that are null
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # putting the above two together
    return missing_data # return the dataframe
missing_check(dfb)

# %%
# Replace the missing values with mean values

cols_mean = ['compactness', 'circularity', 'pr.axis_aspect_ratio', 'max.length_aspect_ratio', 
             'pr.axis_rectangularity', 'skewness_about', 'skewness_about.2', 'class']

for cols in cols_mean:
    if(cols != 'class'): 
        dfb[cols] = dfb[cols].fillna(dfb[cols].mean())

# %%
# Replace the missing values with median values

cols_median = ['distance_circularity', 'radius_ratio', 'scatter_ratio', 'elongatedness', 
               'max.length_rectangularity','scaled_variance', 'scaled_variance.1', 
               'scaled_radius_of_gyration', 'scaled_radius_of_gyration.1',
               'skewness_about.1', 'hollows_ratio', 'class']

for cols in cols_median:
    if(cols != 'class'): 
        dfb[cols] = dfb[cols].fillna(dfb[cols].median())

# %%
dfb.info()

# %% [markdown]
# ### 1C. Visualize a Pie-chart and print percentage of values for variable ‘class’.

# %%
# Understand the target variable and check for imbalanced dataset

f,axes=plt.subplots(1,2,figsize=(17,7))
dfb['class'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('class',data=dfb,ax=axes[1])
axes[0].set_title('Response Variable Pie Chart')
axes[1].set_title('Response Variable Bar Graph')
plt.show()

# %%
# Group datapoints by class
dfb.groupby(["class"]).count()

# %% [markdown]
# **Insights:**
#
# - Class car has 50.7% of total values followed by Class bus as 25.8% and Class van as 23.5%.
# - The above graph shows that the data is biased towards datapoints having class value as car.
# - The number of data points of Class car is almost half the number of other two classes ie bus and van.
#
# **There is big imbalance in the target vector.**
#
# If the imbalanced data is not treated beforehand, then this will degrade the performance of the ML model. Most of the predictions will correspond to the majority class and treat the minority class of features as noise in the data and ignore them. This results in a high bias and low performance of the model.
#
# A widely adopted technique for dealing with highly unbalanced datasets is called re-sampling.
#
# **Two widely used re-sampling methods are:**
#
# - Under-sampling: It is the process where you randomly delete some of the observations from the majority class in order to match the numbers with the minority class.
# - Over-sampling: It is the process of generating synthetic data that tries to randomly generate a sample of the attributes from observations in the minority class
# - Here we will use oversampling because under-sampling may remove important information from the dataset

# %% [markdown]
# ### 1D. Check for duplicate rows in the data and impute with correct approach.

# %%
# To take a look at the duplication in the DataFrame as a whole, just call the duplicated() method on 
# the DataFrame. It outputs True if an entire row is identical to a previous row.
dfb.duplicated().sum()

# %%
# Count the number of non-duplicates
(~dfb.duplicated()).sum()

# %%
# Encode the Target Variable
# bus:0; car:1; van:2
le=LabelEncoder()
dfb['class']=le.fit_transform(dfb['class'])
dfb['class'].value_counts()

# %% [markdown]
# ### Quick EDA

# %% [markdown]
# #### Pairplot

# %%
sns.pairplot(dfb, hue="class")

# %% [markdown]
# #### Correlation Heatmap

# %%
# Checking Correlation Heatmap
plt.figure(dpi = 540,figsize= (30,25))
mask = np.triu(np.ones_like(dfb.corr()))
sns.heatmap(dfb.corr(),mask = mask, fmt = ".2f",annot=True,lw=1,cmap = 'plasma')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.title('Correlation Heatmap')
plt.show()

# %%
# Correlation of "class" with other features
plt.figure(figsize=(15,8))
dfb.corr()['class'].sort_values(ascending = False).plot(kind='bar')

# %%
corr = dfb.corr()
corr

# %%
dfb.var()

# %% [markdown]
# #### Distribution and Outliers

# %%
# Single Box Plot
plt.figure(figsize=(20,10))
ax = sns.boxplot(data=dfb, orient="h", palette="Set2")

# %%
# compactness

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['compactness'],  ax=axes[0],color='Green')
sns.boxplot(x = 'compactness', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['compactness'],25),np.percentile(dfb['compactness'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['compactness'] if i < lower or i > upper]
print('{} Total Number of outliers in compactness: {}'.format('\033[1m',len(Outliers)))

# %%
# circularity

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['circularity'],  ax=axes[0],color='Green')
sns.boxplot(x = 'circularity', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['circularity'],25),np.percentile(dfb['circularity'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['circularity'] if i < lower or i > upper]
print('{} Total Number of outliers in circularity: {}'.format('\033[1m',len(Outliers)))

# %%
# distance_circularity

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['distance_circularity'],  ax=axes[0],color='Green')
sns.boxplot(x = 'distance_circularity', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['distance_circularity'],25),np.percentile(dfb['distance_circularity'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['distance_circularity'] if i < lower or i > upper]
print('{} Total Number of outliers in distance_circularity: {}'.format('\033[1m',len(Outliers)))

# %%
# radius_ratio

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['radius_ratio'],  ax=axes[0],color='Green')
sns.boxplot(x = 'radius_ratio', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['radius_ratio'],25),np.percentile(dfb['radius_ratio'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['radius_ratio'] if i < lower or i > upper]
print('{} Total Number of outliers in radius_ratio: {}'.format('\033[1m',len(Outliers)))

# %%
# pr.axis_aspect_ratio

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['pr.axis_aspect_ratio'],  ax=axes[0],color='Green')
sns.boxplot(x = 'pr.axis_aspect_ratio', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['pr.axis_aspect_ratio'],25),np.percentile(dfb['pr.axis_aspect_ratio'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['pr.axis_aspect_ratio'] if i < lower or i > upper]
print('{} Total Number of outliers in pr.axis_aspect_ratio: {}'.format('\033[1m',len(Outliers)))

# %%
# max.length_aspect_ratio

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['max.length_aspect_ratio'],  ax=axes[0],color='Green')
sns.boxplot(x = 'max.length_aspect_ratio', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['max.length_aspect_ratio'],25),np.percentile(dfb['max.length_aspect_ratio'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['max.length_aspect_ratio'] if i < lower or i > upper]
print('{} Total Number of outliers in max.length_aspect_ratio: {}'.format('\033[1m',len(Outliers)))

# %%
# scatter_ratio

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['scatter_ratio'],  ax=axes[0],color='Green')
sns.boxplot(x = 'scatter_ratio', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['scatter_ratio'],25),np.percentile(dfb['scatter_ratio'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['scatter_ratio'] if i < lower or i > upper]
print('{} Total Number of outliers in scatter_ratio: {}'.format('\033[1m',len(Outliers)))

# %%
# elongatedness

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['elongatedness'],  ax=axes[0],color='Green')
sns.boxplot(x = 'elongatedness', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['elongatedness'],25),np.percentile(dfb['elongatedness'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['elongatedness'] if i < lower or i > upper]
print('{} Total Number of outliers in elongatedness: {}'.format('\033[1m',len(Outliers)))

# %%
# pr.axis_rectangularity

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['pr.axis_rectangularity'],  ax=axes[0],color='Green')
sns.boxplot(x = 'pr.axis_rectangularity', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['pr.axis_rectangularity'],25),np.percentile(dfb['pr.axis_rectangularity'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['pr.axis_rectangularity'] if i < lower or i > upper]
print('{} Total Number of outliers in pr.axis_rectangularity: {}'.format('\033[1m',len(Outliers)))

# %%
# max.length_rectangularity

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['max.length_rectangularity'],  ax=axes[0],color='Green')
sns.boxplot(x = 'max.length_rectangularity', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['max.length_rectangularity'],25),np.percentile(dfb['max.length_rectangularity'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['max.length_rectangularity'] if i < lower or i > upper]
print('{} Total Number of outliers in max.length_rectangularity: {}'.format('\033[1m',len(Outliers)))

# %%
# scaled_variance

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['scaled_variance'],  ax=axes[0],color='Green')
sns.boxplot(x = 'scaled_variance', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['scaled_variance'],25),np.percentile(dfb['scaled_variance'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['scaled_variance'] if i < lower or i > upper]
print('{} Total Number of outliers in scaled_variance: {}'.format('\033[1m',len(Outliers)))

# %%
# scaled_variance.1

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['scaled_variance.1'],  ax=axes[0],color='Green')
sns.boxplot(x = 'scaled_variance.1', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['scaled_variance.1'],25),np.percentile(dfb['scaled_variance.1'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['scaled_variance.1'] if i < lower or i > upper]
print('{} Total Number of outliers in scaled_variance.1: {}'.format('\033[1m',len(Outliers)))

# %%
# scaled_radius_of_gyration

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['scaled_radius_of_gyration'],  ax=axes[0],color='Green')
sns.boxplot(x = 'scaled_radius_of_gyration', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['scaled_radius_of_gyration'],25),np.percentile(dfb['scaled_radius_of_gyration'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['scaled_radius_of_gyration'] if i < lower or i > upper]
print('{} Total Number of outliers in scaled_radius_of_gyration: {}'.format('\033[1m',len(Outliers)))

# %%
# scaled_radius_of_gyration.1

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['scaled_radius_of_gyration.1'],  ax=axes[0],color='Green')
sns.boxplot(x = 'scaled_radius_of_gyration.1', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['scaled_radius_of_gyration.1'],25),np.percentile(dfb['scaled_radius_of_gyration.1'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['scaled_radius_of_gyration.1'] if i < lower or i > upper]
print('{} Total Number of outliers in scaled_radius_of_gyration.1: {}'.format('\033[1m',len(Outliers)))

# %%
# skewness_about

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['skewness_about'],  ax=axes[0],color='Green')
sns.boxplot(x = 'skewness_about', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['skewness_about'],25),np.percentile(dfb['skewness_about'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['skewness_about'] if i < lower or i > upper]
print('{} Total Number of outliers in skewness_about: {}'.format('\033[1m',len(Outliers)))

# %%
# skewness_about.1

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['skewness_about.1'],  ax=axes[0],color='Green')
sns.boxplot(x = 'skewness_about.1', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['skewness_about.1'],25),np.percentile(dfb['skewness_about.1'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['skewness_about.1'] if i < lower or i > upper]
print('{} Total Number of outliers in skewness_about.1: {}'.format('\033[1m',len(Outliers)))

# %%
# skewness_about.2

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['skewness_about.2'],  ax=axes[0],color='Green')
sns.boxplot(x = 'skewness_about.2', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['skewness_about.2'],25),np.percentile(dfb['skewness_about.2'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['skewness_about.2'] if i < lower or i > upper]
print('{} Total Number of outliers in skewness_about.2: {}'.format('\033[1m',len(Outliers)))

# %%
# hollows_ratio

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfb['hollows_ratio'],  ax=axes[0],color='Green')
sns.boxplot(x = 'hollows_ratio', data=dfb,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['hollows_ratio'],25),np.percentile(dfb['hollows_ratio'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['hollows_ratio'] if i < lower or i > upper]
print('{} Total Number of outliers in hollows_ratio: {}'.format('\033[1m',len(Outliers)))

# %% [markdown]
# #### Remove Outliers

# %%
# Impute outliers with mean

col=['radius_ratio',
'pr.axis_aspect_ratio', 
'max.length_aspect_ratio', 
'scaled_variance', 
'scaled_variance.1', 
'scaled_radius_of_gyration.1', 
'skewness_about', 
'skewness_about.1'
]

for c in col:
    # Use the IQR method
    q25,q75=np.percentile(dfb[c],25),np.percentile(dfb[c],75)
    IQR=q75-q25
    Threshold=IQR*1.5
    lower,upper=q25-Threshold,q75+Threshold
    Outliers=[i for i in dfb[c] if i < lower or i > upper]
    print('{} Total Number of outliers in {} Before Imputing : {}'.format('\033[1m',c,len(Outliers)))
    # Mean of the column without considering the outliers
    dfb_include = dfb.loc[(dfb[c] >= lower) & (dfb[c] <= upper)]
    mean=int(dfb_include[c].mean())
    print('{} Mean of {} is {}'.format('\033[1m',c,mean))
    # Impute outliers with mean
    dfb[c]=np.where(dfb[c]>upper,mean,dfb[c])
    dfb[c]=np.where(dfb[c]<lower,mean,dfb[c])
    Outliers=[i for i in dfb[c] if i < lower or i > upper]
    print('{} Total Number of outliers in {} After Imputing : {}'.format('\033[1m',c,len(Outliers)))  
    print('\n')

# %%
# Single Box Plot after removing outliers
plt.figure(figsize=(20,10))
ax = sns.boxplot(data=dfb, orient="h", palette="Set2")

# %%
dfb.head()

# %%
dfb.info()

# %% [markdown]
# ## 2. Data Preparation:

# %% [markdown]
# ### 2A. Split data into X and Y. [Train and Test optional]

# %%
# Arrange data into independent variables and dependent variables
X=dfb.drop(columns='class')
y=dfb['class'] # Target Vector

# %%
# Split X and y into training and test set in 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# %% [markdown]
# ### 2B. Standardize the Data.

# %%
# Using different scaling methods:
# scaler = RobustScaler()
# scaler = MinMaxScaler()
scaler = StandardScaler()

cols_to_scale = ['compactness', 'circularity', 'distance_circularity', 'radius_ratio',
       'pr.axis_aspect_ratio', 'max.length_aspect_ratio', 'scatter_ratio',
       'elongatedness', 'pr.axis_rectangularity', 'max.length_rectangularity',
       'scaled_variance', 'scaled_variance.1', 'scaled_radius_of_gyration',
       'scaled_radius_of_gyration.1', 'skewness_about', 'skewness_about.1',
       'skewness_about.2', 'hollows_ratio']

X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# %%
X_train.head()

# %%
X_test.head()

# %%
y_train.head()

# %%
y_test.head()

# %% [markdown]
# ## 3. Model Building:

# %% [markdown]
# ### 3A. Train a base Classification model using SVM.

# %%
# Create svm_model Object
svm_model = SVC()

# Training the model
svm_model.fit(X_train, y_train)

# Predict testing set
y_pred = svm_model.predict(X_test)

# %% [markdown]
# ### 3B. Print Classification metrics for train data.

# %%
# Classification Accuracy
print('Accuracy on Training data:',svm_model.score(X_train, y_train))
print('Accuracy on Test data:',svm_model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
df_cm = pd.DataFrame(cm, index = [i for i in ["bus","car","van"]],
                  columns = [i for i in ["bus","car","van"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %%
# Model comparison and K Fold Cross Validation test
model = svm_model
y_pred = model.predict(X_test)
y1_pred = model.predict(X_train)
accuracies = cross_val_score(estimator= model, X = X_train, y = y_train, cv=10)

precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = accuracy_score(y_train,y1_pred)*100
Test_Accuracy = accuracy_score(y_test,y_pred)*100
K_Fold_Mean_Accuracy = accuracies.mean()*100
Std_Deviation = accuracies.std()*100

base_1 = []
base_1.append(['SVM All Variables Base Model', Train_Accuracy, Test_Accuracy, K_Fold_Mean_Accuracy, Std_Deviation, precision,
               recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','K-Fold Mean Accuracy',
                                                'Std. Deviation', 'Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'],inplace=True,ascending=False)

# %% [markdown]
# ### 3C. Apply PCA on the data with 10 components.

# %%
# Scaling the complete feature matrix
X = scaler.fit_transform(X)

# %%
# Calculating the covariance between attributes after scaling
# Covariance indicates the level to which two variables vary together.
cov_matrix = np.cov(X,rowvar=False)
print('Covariance Matrix:')
print(cov_matrix)

# %%
# Use PCA on 10 components
pca10 = PCA(n_components=10, random_state=0)
pca10.fit(X)

# %%
# The eigen Values
print(pca10.explained_variance_)

# %%
# The eigen Vectors
print(pca10.components_)

# %%
# And the percentage of variation explained by each eigen Vector
print(pca10.explained_variance_ratio_)

# %%
# Variation explained by each component
plt.bar(list(range(1,11)),pca10.explained_variance_ratio_,alpha=0.5, align='center')
plt.ylabel('Variation explained')
plt.xlabel('Eigen Value')
plt.show()

# %% [markdown]
# ### 3D. Visualize Cumulative Variance Explained with Number of Components.

# %%
# Cumulative Variation explained by each component
plt.step(list(range(1,11)),np.cumsum(pca10.explained_variance_ratio_), where='mid')
plt.ylabel('Cumulative variation explained')
plt.xlabel('Eigen Value')
plt.show()

# %% [markdown]
# ### 3E. Draw a horizontal line on the above plot to highlight the threshold of 90%.

# %%
# Cumulative Variation explained by each component
# Red dashed line at 90% cumulative variation is explained by 5 principal components
plt.step(list(range(1,11)),np.cumsum(pca10.explained_variance_ratio_), where='mid')
plt.axhline(y=0.9, color='r', linestyle='--', lw=1)
plt.ylabel('Cumulative variation explained')
plt.xlabel('Eigen Value')
plt.show()

# Now 5 dimensions seems very reasonable. With 5 variables we can explain over 90% of the 
# variation in the original data!

# %% [markdown]
# ### 3F. Apply PCA on the data. This time Select Minimum Components with 90% or above variance explained.

# %%
# 5 principal components are able to explain more than 90% of variance in the data
pca5 = PCA(n_components=5)
pca5.fit(X)
print(pca5.components_)
print(pca5.explained_variance_ratio_)
Xpca5 = pca5.transform(X)

# %%
# Print the original features and the reduced features
print('Original number of features:', X.shape[1])
print('Reduced number of features:', Xpca5.shape[1])

# %%
# View the first 5 observations of the pca components
Xpca5_df = pd.DataFrame(data = Xpca5)
Xpca5_df.head()

# %% [markdown]
# ### 3G. Train SVM model on components selected from above step.

# %%
X_train_row, X_train_col = X_train.shape
print('The X_train comprises of', X_train_row, 'rows and', X_train_col, 'columns.')

# %%
X_test_row, X_test_col = X_test.shape
print('The X_test comprises of', X_test_row, 'rows and', X_test_col, 'columns.')

# %%
# Split the pca data into train and test ratio of 80:20
Xpca5_train, Xpca5_test, y_train, y_test = train_test_split(Xpca5, y, test_size=0.20, random_state=0)

# %%
Xpca5_train_row, Xpca5_train_col = Xpca5_train.shape
print('The Xpca5_train comprises of', Xpca5_train_row, 'rows and', Xpca5_train_col, 'columns.')

# %%
Xpca5_test_row,  Xpca5_test_col =  Xpca5_test.shape
print('The  Xpca5_test comprises of',  Xpca5_test_row, 'rows and',  Xpca5_test_col, 'columns.')

# %%
# Create svm_model Object
svm_model = SVC()

# Training the model
svm_model.fit(Xpca5_train, y_train)

# Predict testing set
y_pred = svm_model.predict(Xpca5_test)

# %% [markdown]
# ### 3H. Print Classification metrics for train data of above model and share insights.

# %%
# Classification Accuracy
print('Accuracy on Training data:',svm_model.score(Xpca5_train, y_train))
print('Accuracy on Test data:',svm_model.score(Xpca5_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
df_cm = pd.DataFrame(cm, index = [i for i in ["bus","car","van"]],
                  columns = [i for i in ["bus","car","van"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# **Evaluation metrics allow us to estimate errors to determine how well our models
# are performing:**
#
# > Accuracy: ratio of correct predictions over total predictions.
#
# > Precision: how often the classifier is correct when it predicts positive.
#
# > Recall: how often the classifier is correct for all positive instances.
#
# > F-Score: single measurement to combine precision and recall.

# %%
# Model comparison and K Fold Cross Validation test
model = svm_model
y_pred = model.predict(Xpca5_test)
y1_pred = model.predict(Xpca5_train)
accuracies = cross_val_score(estimator= model, X = Xpca5_train, y = y_train, cv=10)

precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = accuracy_score(y_train,y1_pred)*100
Test_Accuracy = accuracy_score(y_test,y_pred)*100
K_Fold_Mean_Accuracy = accuracies.mean()*100
Std_Deviation = accuracies.std()*100

# base_1 = []
base_1.append(['SVM 5 PCs with Base Model', Train_Accuracy, Test_Accuracy, K_Fold_Mean_Accuracy, Std_Deviation, precision,
               recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','K-Fold Mean Accuracy',
                                                'Std. Deviation', 'Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'],inplace=True,ascending=False)
model_comparison

# %% [markdown]
# **Insights:**
# - Both the models are giving good accuracy in addition to other metrics, but just by using 5 principal components (instead of 18 in the base model), we are able to capture a great amount of variation in the data.
# - The difference in performance metrics with PCA is even more noticeable as the number of dimensions increase.
# - The accuracy, precision, recall and F1 Score values are reduced after applying PCA because we have reduced the number of dimensions.
# - Despite giving less accuracy, precision and recall, this model is better as it has taken into consideration the relationship between the independent variables and reduced the columns which are highly correlated.

# %% [markdown]
# ## 4. Performance Improvement:

# %% [markdown]
# ### 4A. Train another SVM on the components out of PCA. Tune the parameters to improve performance.

# %%
# Using PCA with 6 components
pca6 = PCA(n_components=6)
pca6.fit(X)
print(pca6.components_)
print(pca6.explained_variance_ratio_)
Xpca6 = pca6.transform(X)

# %%
# Print the original features and the reduced features
print('Original number of features:', X.shape[1])
print('Reduced number of features:', Xpca6.shape[1])

# %%
# View the first 5 observations of the pca components
Xpca6_df = pd.DataFrame(data = Xpca6)
Xpca6_df.head()

# %%
X_train_row, X_train_col = X_train.shape
print('The X_train comprises of', X_train_row, 'rows and', X_train_col, 'columns.')

# %%
X_test_row, X_test_col = X_test.shape
print('The X_test comprises of', X_test_row, 'rows and', X_test_col, 'columns.')

# %%
# Split the pca data into train and test ratio of 80:20
Xpca6_train, Xpca6_test, y_train, y_test = train_test_split(Xpca6, y, test_size=0.20, random_state=0)

# %%
Xpca6_train_row, Xpca6_train_col = Xpca6_train.shape
print('The Xpca6_train comprises of', Xpca6_train_row, 'rows and', Xpca6_train_col, 'columns.')

 # %%
 Xpca6_test_row,  Xpca6_test_col =  Xpca6_test.shape
print('The  Xpca6_test comprises of',  Xpca6_test_row, 'rows and',  Xpca6_test_col, 'columns.')

# %% [markdown]
# #### Use SVM without Oversampling

# %%
# Create svm_model Object
svm_model = SVC()

# Training the model
svm_model.fit(Xpca6_train, y_train)

# Predict testing set
y_pred = svm_model.predict(Xpca6_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',svm_model.score(Xpca6_train, y_train))
print('Accuracy on Test data:',svm_model.score(Xpca6_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
df_cm = pd.DataFrame(cm, index = [i for i in ["bus","car","van"]],
                  columns = [i for i in ["bus","car","van"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %%
# Model comparison and K Fold Cross Validation test
model = svm_model
y_pred = model.predict(Xpca6_test)
y1_pred = model.predict(Xpca6_train)
accuracies = cross_val_score(estimator= model, X = Xpca6_train, y = y_train, cv=10)

precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = accuracy_score(y_train,y1_pred)*100
Test_Accuracy = accuracy_score(y_test,y_pred)*100
K_Fold_Mean_Accuracy = accuracies.mean()*100
Std_Deviation = accuracies.std()*100

# base_1 = []
base_1.append(['SVM 6 PCs w/o Oversampling', Train_Accuracy, Test_Accuracy, K_Fold_Mean_Accuracy, Std_Deviation, precision,
               recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','K-Fold Mean Accuracy',
                                                'Std. Deviation', 'Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'],inplace=True,ascending=False)

# %% [markdown]
# #### Use SVM with Oversampling

# %%
# Using SMOTE
smote=SMOTE(random_state=0)
X1, y1=smote.fit_resample(Xpca6_train, y_train)

#Training the model
svm_model.fit(X1, y1)

#Predict testing set
y_pred = svm_model.predict(Xpca6_test)

# Classification Accuracy
print('Accuracy on Training data:',svm_model.score(Xpca6_train, y_train))
print('Accuracy on Test data:',svm_model.score(Xpca6_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
df_cm = pd.DataFrame(cm, index = [i for i in ["bus","car","van"]],
                  columns = [i for i in ["bus","car","van"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %%
# Target vector is balanced after oversampling
print('After oversampling distribution of target vector:')
print(y1.value_counts())

# %%
# Model comparison and K Fold Cross Validation test
model = svm_model
y_pred = model.predict(Xpca6_test)
y1_pred = model.predict(X1)
accuracies = cross_val_score(estimator= model, X = X1, y = y1, cv=10)

precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = accuracy_score(y1,y1_pred)*100
Test_Accuracy = accuracy_score(y_test,y_pred)*100
K_Fold_Mean_Accuracy = accuracies.mean()*100
Std_Deviation = accuracies.std()*100

# base_1 = []
base_1.append(['SVM 6 PCs with Oversampling', Train_Accuracy, Test_Accuracy, K_Fold_Mean_Accuracy, Std_Deviation, precision,
               recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','K-Fold Mean Accuracy',
                                                'Std. Deviation', 'Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'],inplace=True,ascending=False)

# %% [markdown]
# #### Use automated search without Oversampling for hyper-parameters.

# %%
#Create svm_model Object
svm_model = SVC()

svm_model.fit(Xpca6_train, y_train)

# %%
# Exhaustive search over specified parameter values for an estimator.
# Important members are fit, predict.

# GridSearchCV implements a "fit" and a "score" method.
# It also implements "score_samples", "predict", "predict_proba",
# "decision_function", "transform" and "inverse_transform" if they are
# implemented in the estimator used.

# The parameters of the estimator used to apply these methods are optimized
# by cross-validated grid-search over a parameter grid.

grid_params = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

gs = RandomizedSearchCV(svm_model, grid_params, cv=10, random_state=0)
gs.fit(Xpca6_train, y_train)

# %%
pd.DataFrame({'param': gs.cv_results_["params"], 
              'score mean': gs.cv_results_["mean_test_score"], 
              'score s.d.': gs.cv_results_["std_test_score"],
             'rank': gs.cv_results_["rank_test_score"]})

# %%
print('Best Parameters:', gs.best_params_, 'mean score: ', gs.best_score_, sep='\n')

# %%
# Lets Build SVM with best params

svm_model = SVC(C=1, gamma=0.1, kernel= 'rbf')

svm_model.fit(Xpca6_train, y_train)
y_pred = svm_model.predict(Xpca6_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',svm_model.score(Xpca6_train, y_train))
print('Accuracy on Test data:',svm_model.score(Xpca6_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
df_cm = pd.DataFrame(cm, index = [i for i in ["bus","car","van"]],
                  columns = [i for i in ["bus","car","van"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %%
# Model comparison and K Fold Cross Validation test
model = svm_model
y_pred = model.predict(Xpca6_test)
y1_pred = model.predict(Xpca6_train)
accuracies = cross_val_score(estimator= model, X = Xpca6_train, y = y_train, cv=10)

precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = accuracy_score(y_train,y1_pred)*100
Test_Accuracy = accuracy_score(y_test,y_pred)*100
K_Fold_Mean_Accuracy = accuracies.mean()*100
Std_Deviation = accuracies.std()*100

# base_1 = []
base_1.append(['SVM 6 PCs with Hyperparameters', Train_Accuracy, Test_Accuracy, K_Fold_Mean_Accuracy, Std_Deviation, precision,
               recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','K-Fold Mean Accuracy',
                                                'Std. Deviation', 'Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'],inplace=True,ascending=False)

# %% [markdown]
# #### Use automated search with Oversampling for hyper-parameters.

# %%
#Create svm_model Object
svm_model = SVC()

# Using SMOTE
# smote=SMOTE(random_state=0)
# X1, y1=smote.fit_resample(Xpca6_train, y_train)

#Training the model
svm_model.fit(X1, y1)

# %%
# Exhaustive search over specified parameter values for an estimator.
# Important members are fit, predict.

# GridSearchCV implements a "fit" and a "score" method.
# It also implements "score_samples", "predict", "predict_proba",
# "decision_function", "transform" and "inverse_transform" if they are
# implemented in the estimator used.

# The parameters of the estimator used to apply these methods are optimized
# by cross-validated grid-search over a parameter grid.

grid_params = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

gs = RandomizedSearchCV(svm_model, grid_params, cv=10, random_state=0)
gs.fit(Xpca6_train, y_train)

# %%
pd.DataFrame({'param': gs.cv_results_["params"], 
              'score mean': gs.cv_results_["mean_test_score"], 
              'score s.d.': gs.cv_results_["std_test_score"],
             'rank': gs.cv_results_["rank_test_score"]})

# %%
print('Best Parameters:', gs.best_params_, 'mean score: ', gs.best_score_, sep='\n')

# %%
# Lets Build SVM with best params

svm_model = SVC(C=1, gamma=0.1, kernel= 'rbf')

svm_model.fit(X1, y1)
y_pred = svm_model.predict(Xpca6_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',svm_model.score(Xpca6_train, y_train))
print('Accuracy on Test data:',svm_model.score(Xpca6_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
df_cm = pd.DataFrame(cm, index = [i for i in ["bus","car","van"]],
                  columns = [i for i in ["bus","car","van"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# ### 4B. Share best Parameters observed from above step.

# %%
# Refer section 4A above for detailed analysis
# Refer section 4C below for model comparison
print('Best Parameters:', gs.best_params_, 'mean score: ', gs.best_score_, sep='\n')

# %% [markdown]
# ### 4C. Print Classification metrics for train data of above model and share relative improvement in performance in all the models along with insights.

# %%
# Final Table: Model comparison and K Fold Cross Validation test
model = svm_model
y_pred = model.predict(Xpca6_test)
y1_pred = model.predict(X1)
accuracies = cross_val_score(estimator= model, X = X1, y = y1, cv=10)

precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = accuracy_score(y1,y1_pred)*100
Test_Accuracy = accuracy_score(y_test,y_pred)*100
K_Fold_Mean_Accuracy = accuracies.mean()*100
Std_Deviation = accuracies.std()*100

# base_1 = []
base_1.append(['SVM 6 PCs with Hyperparameters and Oversampling', Train_Accuracy, Test_Accuracy, K_Fold_Mean_Accuracy, Std_Deviation, precision,
               recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','K-Fold Mean Accuracy',
                                                'Std. Deviation', 'Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'],inplace=True,ascending=False)
model_comparison

# %% [markdown]
# **Insights:**
# - As is evident from the above table, just by using 6 principal components instead of 18 in the base model, we are getting above 90% accuracy in addition to other improved performance metrics.
# - Some changes are also visible as we use oversampling to handle the imbalanced dataset.
# - Accuracy further improves with hyper-parameter tuning.
# - We are getting the best performance metrics by using both the Oversampling and Hyper-parameter tuning approach.
# - Majority of the variation in the dataset is captured with just the 6 components accompanied by a very good amount of dimensionality reduction.

# %% [markdown]
# ## 5. Data Understanding & Cleaning:

# %% [markdown]
# ### 5A. Explain pre-requisite/assumptions of PCA.

# %% [markdown]
# **There are some assumptions in PCA which are to be followed as they will lead to accurate functioning of this dimensionality reduction technique in ML. The assumptions in PCA are:**
# - There needs to be a linear relationship between all variables. The reason for this assumption is that a PCA is based on Pearson correlation coefficients, so there needs to be a linear relationship between the variables.
# - We should have sampling adequacy, which simply means that for PCA to produce a reliable result, large enough sample sizes are required. General rule of thumb: a minimum of 150 cases, or 5 to 10 cases per variable, has been recommended as a minimum sample size. There are a few methods to detect sampling adequacy: (1) the Kaiser-Meyer-Olkin (KMO) Measure of Sampling Adequacy for the overall data set; and (2) the KMO measure for each individual variable.
# - The data should be suitable for data reduction. We need to have adequate correlations between the variables in order for variables to be reduced to a smaller number of components.
# - We should not have significant outliers. Outliers are important because these can have a disproportionate influence on our results. 
# - We should have Interval-level measurement. All variables should be assessed on an interval or ratio level of measurement.
# - Technical implementations often assume no missing values.

# %% [markdown]
# ### 5B. Explain advantages and limitations of PCA.

# %% [markdown]
# **PCA offers multiple benefits, but it also suffers from certain shortcomings:**
#
# Advantages of PCA:
#
# - Easy to compute: PCA is based on linear algebra, which is computationally easy to solve by computers.
# - Speeds up other machine learning algorithms: Machine learning algorithms converge faster when trained on principal components instead of the original dataset.
# - Counteracts the issues of high-dimensional data: High-dimensional data causes regression-based algorithms to overfit easily. By using PCA beforehand to lower the dimensions of the training dataset, we prevent the predictive algorithms from overfitting.
# - PCA improves the data visualization for easy understanding.
#
# Disadvantages of PCA:
#
# - Data normalization must be done before applying PCA.
# - Low interpretability of principal components: Principal components are linear combinations of the features from the original data, but they are not as easy to interpret. For example, it is difficult to tell which are the most important features in the dataset after computing principal components. 
# - The trade-off between information loss and dimensionality reduction: Although dimensionality reduction is useful, it comes at a cost. Information loss is a necessary part of PCA. Balancing the trade-off between dimensionality reduction and information loss is unfortunately a necessary compromise that we have to make when using PCA.

# %% [markdown]
# # References:
#
# 1. [Towards Data Science](https://towardsdatascience.com)
# 2. [Kaggle. Kaggle Code](https://www.kaggle.com/code)
# 3. [KdNuggets](https://www.kdnuggets.com/)
# 4. [AnalyticsVidhya](https://www.analyticsvidhya.com/blog/)
# 5. [Wikipedia](https://en.wikipedia.org/)
# 6. [Numpy](https://numpy.org/)
# 7. [Pandas](https://pandas.pydata.org/)
# 8. [SciPy](https://scipy.org/)
# 9. [MatplotLib](https://matplotlib.org/)
# 10. [Seaborn](https://seaborn.pydata.org/)
# 11. [Python](https://www.python.org/)
# 12. [Plotly](https://plotly.com/)
# 13. [Bokeh](https://docs.bokeh.org/en/latest/)
# 14. [RStudio](https://www.rstudio.com/)
# 15. [MiniTab](https://www.minitab.com/en-us/)
# 16. [Anaconda](https://www.anaconda.com/)
