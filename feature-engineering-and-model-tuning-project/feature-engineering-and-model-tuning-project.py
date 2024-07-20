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
# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#00b3e5;">Contents: Featurisation and Model Tuning Project

# %% [markdown]
# # Solution

# %% [markdown]
# - **DOMAIN:** Semiconductor manufacturing process
# - **CONTEXT:** A complex modern semiconductor manufacturing process is normally under constant surveillance via the monitoring of signals/variables collected from sensors and or process measurement points. However, not all of these signals are equally valuable in a specific monitoring system. The measured signals contain a combination of useful information, irrelevant information as well as noise. Engineers typically have a much larger number of signals than are actually required. If we consider each type of signal as a feature, then feature selection may be applied to identify the most relevant signals. The Process Engineers may then use these signals to determine key factors contributing to yield excursions downstream in the process. This will enable an increase in process throughput, decreased time to learning and reduce the per unit production costs. These signals can be used as features to predict the yield type. And by analysing and trying out different combinations of features, essential signals that are impacting the yield type can be identified.
# - **DATA DESCRIPTION:** signal-data.csv : (1567, 592); The data consists of 1567 datapoints each with 591 features. The dataset presented in this case represents a selection of such features where each example represents a single production entity with associated measured features and the labels represent a simple pass/fail yield for in house line testing. Target column “ –1” corresponds to a pass and “1” corresponds to a fail and the data time stamp is for that specific test point.
# - **PROJECT OBJECTIVE:** We will build a classifier to predict the Pass/Fail yield of a particular process entity and analyse whether all the features are required to build the model or not.

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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_samples, silhouette_score
from kmodes.kprototypes import KPrototypes

import xgboost as xgb
from xgboost import plot_importance
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings("ignore")

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %% [markdown]
# ## 1. Import and understand the data.

# %% [markdown]
# ### 1A. Import ‘signal-data.csv’ as DataFrame.

# %%
# CSV File 1
dfa=pd.read_csv('signal-data.csv')

# %%
dfa.info()
dfa.head()

# %% [markdown]
# ### 1B. Print 5 point summary and share at least 2 observations.

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
# - Feature 0:
#  - Mean and Median are nearly equal. Distribution might be normal.
#  - 75 % of values are less than 3056, and maximum value is 3356.
# - Feature 589:
#  - Mean and median are not equal. Skewness is expected.
#  - Range of values is large.
#  - Distribution is not normal because of big SD.

# %% [markdown]
# ## 2. Data cleansing:

# %% [markdown]
# ### 2A. Write a for loop which will remove all the features with 20%+ Null values and impute rest with mean of the feature.

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


# %%
def rmissingvaluecol(dff, threshold):
    l = []
    l = list(dff.drop(dff.loc[:,list((100*(dff.isnull().sum()/len(dff.index)) >= threshold))].columns, 1).columns.values)
    print("# Columns having more than %s percent missing values: "%threshold, (dff.shape[1] - len(l)))
    print("Columns:\n", list(set(list((dff.columns.values))) - set(l)))
    return l


rmissingvaluecol(dfa,20) # Here threshold is 20% which means we are going to drop columns having more than 20% of missing values

# %%
l = rmissingvaluecol(dfa, 20)
dfa = dfa[l]

# %%
dfa.info()
dfa.head()

# %%
dfa.isnull().sum()

# %% [markdown]
# **Absence of a signal is assumed to be no signal in the dataset:**
# - So its better to replace the NaN values with zero.
# - Replacing NaN values with zero is giving a better performance metrics.

# %%
# Replace the NaN/NA with mean, median or zero (considering it as no signal)
# dfa.fillna(dfa.mean(),inplace = True)
dfa.fillna(0,inplace=True)

# %%
# Again, checking if there is any NULL values left
dfa.isnull().any().any()

# %% [markdown]
# ### 2B. Identify and drop the features which are having same value for all the rows.

# %%
# Drop the columns that have constant signal
cols = dfa.select_dtypes([np.number]).columns
std = dfa[cols].std()
cols_to_drop = std[std==0].index
dfa.drop(cols_to_drop, axis=1,inplace=True)
dfa.head()

# %%
dfa.info()

# %% [markdown]
# ### 2C. Drop other features if required using relevant functional knowledge. Clearly justify the same.

# %%
# Time is the id of the customer with corresponding details. This information may not be requried
# for analysis and modeling as the Time will be all unique values. So we can drop it safely.
dfa.drop(['Time'], axis=1, inplace=True)

# %%
# Label encode the target class with 0 and 1
dfa['Pass/Fail']=dfa['Pass/Fail'].replace([-1,1],[0,1])

# %% [markdown]
# ### 2D. Check for multi-collinearity in the data and take necessary action.

# %%
# Check for correlation and consider the features where correlation coeff > 0.7
plt.figure(figsize=(20,18))
corr=dfa.corr()
sns.heatmap(abs(corr>0.7),cmap="Reds");

# %%
# Make a copy of the dataset and drop the target class for easy EDA
dfa1=dfa.copy()
dfa1.drop(['Pass/Fail'],axis=1,inplace=True)

# %%
# Create correlation matrix
corr_matrix = dfa1.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Select features with correlation greater than 0.70
to_drop = [column for column in upper.columns if any(upper[column] > 0.70)]

# Drop features 
dfa1.drop(to_drop, axis=1, inplace=True)

# %%
row,column=dfa1.shape
print('After dropping the correlated features the dataset contains:', row, 'rows and', column, 'columns')

# %% [markdown]
# ### 2E. Make all relevant modifications on the data using both functional/logical reasoning/assumptions.

# %%
# Use boxplot to check for outliers
plt.figure(figsize=(50, 50))
col = 1
for i in dfa1.columns:
    plt.subplot(22, 10, col)
    sns.boxplot(dfa1[i],color='green')
    col += 1

# %%
# Replace the outliers with median
for i in dfa1.columns:
    q1 = dfa1[i].quantile(0.25)
    q3 = dfa1[i].quantile(0.75)
    iqr = q3 - q1
    
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    
    dfa1.loc[(dfa1[i] < low) | (dfa1[i] > high), i] = dfa1[i].median()

# %%
# Situation after removing the outliers with median
plt.figure(figsize=(50, 50))
col = 1
for i in dfa1.columns:
    plt.subplot(22, 10, col)
    sns.boxplot(dfa1[i],color='green')
    col += 1

# %% [markdown]
# ## 3. Data analysis & visualisation:

# %% [markdown]
# ### 3A. Perform a detailed univariate Analysis with appropriate detailed comments after each analysis.

# %%
# Check for distribution, skewness
dfa1.hist(bins = 30, figsize = (40, 40), color = 'green')
plt.show()

# %%
# Density plot to check for the distribution of features
plt.figure(figsize=(40, 40))
col = 1
for i in dfa1.columns:
    plt.subplot(22, 10, col)
    sns.distplot(dfa1[i], color = 'g')
    col += 1 

# %% [markdown]
# ### 3B. Perform bivariate and multivariate analysis with appropriate detailed comments after each analysis.

# %%
# Combine the dataset
y=dfa['Pass/Fail']
dfa1=pd.concat([dfa1,y],axis=1)

# %%
dfa1.info()

# %%
# Correlation of "Pass/Fail" with other features
# Open image in a new tab for details
plt.figure(figsize=(60,30))
dfa1.corr()['Pass/Fail'].sort_values(ascending = False).plot(kind='bar')

# %%
# As is evident, we may consider to drop the following features: '224','432','53','253','82','119','221'.
# dfa1.drop(['224','432','53','253','82','119','221'], axis=1, inplace=True)

# %%
# Understand the target variable and check for imbalanced dataset
f,axes=plt.subplots(1,2,figsize=(17,7))
dfa1['Pass/Fail'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('Pass/Fail',data=dfa1,ax=axes[1])
axes[0].set_title('Response Variable Pie Chart')
axes[1].set_title('Response Variable Bar Graph')
plt.show()

# %%
# Group datapoints by class
dfa1.groupby(["Pass/Fail"]).count()

# %% [markdown]
# **Insights from above graphs:**
#
# - Class 0 has 93.4% of total values followed by Class 1 as 6.6%.
# - The above graph shows that the data is biased towards datapoints having class value as 0.
# - The number of data points of Class 0 is almost 14 times of Class 1.
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

# %%
# Visualize a jointplot for ‘8’ and ‘9’ and share insights.
sns.jointplot(data = dfa1, x="8", y="9", kind = "reg")

# %% [markdown]
# **Observations:**
#
# - No correlation exists between 8 and 9.
# - Data distribution is almost like normal except some skewness.
# - Presence of outliers affect the value of regression coefficients.
# - Similar graphical observations can be replicated for variables of interest.

# %%
# Print the correlation coefficient between every pair of attributes
dfa1.corr()

# %%
# Checking Correlation Heatmap
# Open image in a new tab for details
plt.figure(dpi = 300,figsize= (100,90))
mask = np.triu(np.ones_like(dfa1.corr()))
sns.heatmap(dfa1.corr(),mask = mask, fmt = ".2f",annot=True,lw=1,cmap = 'plasma')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.title('Correlation Heatmap')
plt.show()

# %% [markdown]
# ## 4. Data pre-processing:

# %%
dfa1.info()
dfa1.head()

# %%
dfa1.isnull().sum()

# %%
dfa.isnull().any().any()

# %% [markdown]
# ### 4A. Segregate predictors vs target attributes.

# %%
# Create the features matrix and target vector
X=dfa1.drop(['Pass/Fail'], axis=1)
y=dfa1['Pass/Fail']

# %% [markdown]
# ### 4B. Check for target balancing and fix it if found imbalanced.

# %%
# Check for imbalanced dataset by numbers
y.value_counts()

# %%
# Check for imbalanced dataset by percentage
104/1463*100

# %% [markdown]
# Avoiding the following step, as plan is to use the Oversampling on the training data only.

# %%
# Using SMOTE; Create the oversampler. 
# smote=SMOTE(random_state=0)
# X1, y1=smote.fit_resample(X, y)

# Target vector is balanced after oversampling
# print('After oversampling distribution of target vector:')
# print(y1.value_counts())

# %% [markdown]
# ### 4C. Perform train-test split and standardise the data or vice versa if required.

# %%
# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
# Using different scaling methods:
# scaler = RobustScaler()
# scaler = MinMaxScaler()
scaler = StandardScaler()
# mydata = mydata.apply(zscore)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# X_train = X_train.apply(zscore)
# X_test = X_test.apply(zscore)

# %%
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# %% [markdown]
# ### 4D. Check if the train and test data have similar statistical characteristics when compared with original data.

# %%
# Please refer the section 7 in the end of this notebook.

# %% [markdown]
# ## 5. Model training, testing and tuning:

# %% [markdown]
# - A. Use any Supervised Learning technique to train a model.
# - B. Use cross validation techniques.
# - C. Apply hyper-parameter tuning techniques to get the best accuracy.
# - D. Use any other technique/method which can enhance the model performance.
# - E. Display and explain the classification report in detail.
# - F. Apply the above steps for all possible models that you have learnt so far.
#
# **Considering all the requirements in this section in following steps:**

# %% [markdown]
# ### 5A. Train a model using XGBoost.
# Base model is created in 5A, and the parameter tuning is done in 5B.

# %%
# Build the model
model = xgb.XGBClassifier(n_jobs=2, random_state=0, verbosity=0)

# Train the model
model.fit(X_train, y_train)
model_pred = model.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, model_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, model_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["0","1"]],
                  columns = [i for i in ["0","1"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %%
# Default Parameters used in the base model:
print(model)

# %% [markdown]
# ### 5B. Improve performance of the XGBoost as much as possible.

# %% [markdown]
# #### Case-1: Using oversampling over complete dataset

# %%
print('Before oversampling distribution of target vector:')
print(y.value_counts())

# %%
# Using SMOTE
# Create the oversampler. 
smt = SMOTE(random_state=0)
X1, y1 = smt.fit_sample(X, y)

# %%
# Target vector is balanced after oversampling
print('After oversampling distribution of target vector:')
print(y1.value_counts())

# %%
# Split X and y into training and test set in 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.20, random_state=0)

# %%
# Build the model
model = xgb.XGBClassifier(n_jobs=2, random_state=0, verbosity=0)

# Train the model
model.fit(X_train, y_train)
model_pred = model.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, model_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, model_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["0","1"]],
                  columns = [i for i in ["0","1"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# #### Case-2: Using oversampling over training dataset only

# %%
# Split X and y into training and test set in 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# %%
# Using SMOTE
# Create the oversampler.
smt = SMOTE(random_state=0)
X1, y1 = smt.fit_sample(X_train, y_train)

# %%
# Build the model
model = xgb.XGBClassifier(n_jobs=2, random_state=0, verbosity=0)

# Train the model
model.fit(X1, y1)
model_pred = model.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, model_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, model_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["0","1"]],
                  columns = [i for i in ["0","1"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# #### Case-3: Using hyperparameter tuning with Oversampling

# %%
# Build the model
model = xgb.XGBClassifier(n_jobs=2, random_state=0, verbosity=0)

params = {'max_depth': [3, 5, 6, 10, 15, 20],
          'learning_rate': [0.01, 0.1, 0.2, 0.3],
          'subsample': np.arange(0.5, 1.0, 0.1),
          'colsample_bytree': np.arange(0.4, 1.0, 0.1),
          'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
          'n_estimators': [100, 500, 1000],
          'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
          'gamma':[i/10.0 for i in range(0,5)],
          'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
          'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}

gs = RandomizedSearchCV(estimator=model, param_distributions=params, cv=5, verbose=10, n_jobs=2, random_state=0)
gs.fit(X1, y1)

# %%
pd.DataFrame({'param': gs.cv_results_["params"], 
              'score mean': gs.cv_results_["mean_test_score"], 
              'score s.d.': gs.cv_results_["std_test_score"],
              'rank': gs.cv_results_["rank_test_score"]})

# %%
print('Best Parameters:', gs.best_params_, 'mean score: ', gs.best_score_, sep='\n')

# %%
# Build the model
model = xgb.XGBClassifier(n_jobs=2, random_state=0, subsample=0.899999999999999, reg_lambda=1.0, reg_alpha=0.01, n_estimators=100, 
                          min_child_weight=0.5, max_depth=20, learning_rate=0.2, gamma=0.3, 
                          colsample_bytree=0.6, colsample_bylevel=0.8999999999999999, verbosity=0)

# Train the model
model.fit(X1, y1)
model_pred = model.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, model_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, model_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["0","1"]],
                  columns = [i for i in ["0","1"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# #### Case-4: Using hyperparameter tuning without oversampling

# %%
# Build the model
model = xgb.XGBClassifier(n_jobs=2, random_state=0, verbosity=0)

params = {'max_depth': [3, 5, 6, 10, 15, 20],
          'learning_rate': [0.01, 0.1, 0.2, 0.3],
          'subsample': np.arange(0.5, 1.0, 0.1),
          'colsample_bytree': np.arange(0.4, 1.0, 0.1),
          'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
          'n_estimators': [100, 500, 1000],
          'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
          'gamma':[i/10.0 for i in range(0,5)],
          'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
          'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}

gs = RandomizedSearchCV(estimator=model, param_distributions=params, cv=5, verbose=10, n_jobs=2, random_state=0)
gs.fit(X_train, y_train)

# %%
pd.DataFrame({'param': gs.cv_results_["params"], 
              'score mean': gs.cv_results_["mean_test_score"], 
              'score s.d.': gs.cv_results_["std_test_score"],
              'rank': gs.cv_results_["rank_test_score"]})

# %%
# Print the best parameters
print('Best Parameters:', gs.best_params_, 'mean score: ', gs.best_score_, sep='\n')

# %%
# Build the model
model = xgb.XGBClassifier(n_jobs=2, random_state=0, subsample=0.899999999999999, reg_lambda=50.0, reg_alpha=0.01, n_estimators=1000, 
                          min_child_weight=5.0, max_depth=20, learning_rate=0.1, gamma=0.3, 
                          colsample_bytree=0.5, colsample_bylevel=0.7999999999999999, verbosity=0)

# Train the model
model.fit(X_train, y_train)
model_pred = model.predict(X_test)

# %%
# Plot features importance chart
plt.figure(figsize=(100,90))
plot_importance(model)
plt.show()

# %%
# Classification Accuracy
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, model_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, model_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["0","1"]],
                  columns = [i for i in ["0","1"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# ### 5C. Consider all the possible models

# %% [markdown]
# #### Select  the best performing model without PCA

# %%
# Use K-Fold Cross Validation for model selection
# Define various classification models
LR_model=LogisticRegression(n_jobs=2, random_state=0)
KNN_model=KNeighborsClassifier(n_jobs=2)
GNB_model=GaussianNB()
# SVM_model_linear=SVC(kernel='linear',random_state=0)
# SVM_model_rbf=SVC(kernel='rbf',random_state=0)
# SVM_model_poly=SVC(kernel='poly',random_state=0)
RandomForest=RandomForestClassifier(n_jobs=2, random_state=0)
BaggingClassifier=BaggingClassifier(n_jobs=2, random_state=0)
AdaBoostClassifier=AdaBoostClassifier(random_state=0)
GBClassifier=GradientBoostingClassifier(random_state=0)
XGBClassifier=xgb.XGBClassifier(n_jobs=2, random_state=0, verbosity = 0)
LGBMClassifier=LGBMClassifier(n_jobs=2, random_state=0)
# CatBoostClassifier=CatBoostClassifier(thread_count=2, random_seed=0)

Hybrid = []
Hybrid.append(['RidgeClassifier',RidgeClassifier(random_state=0)])
Hybrid.append(['LogisticRegression',LogisticRegression(n_jobs=2, random_state=0)])
Hybrid.append(['SVM',SVC(random_state=0)])
Hybrid.append(['KNeigbors',KNeighborsClassifier(n_jobs=2)])
Hybrid.append(['GaussianNB',GaussianNB()])
Hybrid.append(['BernoulliNB',BernoulliNB()])
Hybrid.append(['DecisionTree',DecisionTreeClassifier(random_state=0)])

Hybrid_Ensemble=VotingClassifier(Hybrid, n_jobs=2)

# %%
# K Fold Cross Validation Scores

seed = 0

# Create models
models = []

models.append(('LR_Model', LR_model))
models.append(('KNN_Model', KNN_model))
models.append(('GNB_Model', GNB_model))
# models.append(('SVM_Linear', SVM_model_linear))
# models.append(('SVM_Rbf', SVM_model_rbf))
# models.append(('SVM_Poly', SVM_model_poly))
models.append(('RandomForest', RandomForest))
models.append(('BaggingClassifier', BaggingClassifier))
models.append(('AdaBoostClassifier', AdaBoostClassifier))
models.append(('GBClassifier', GBClassifier))
models.append(('XGBClassifier', XGBClassifier))
models.append(('LGBMClassifier', LGBMClassifier))
# models.append(('CatBoostClassifier', CatBoostClassifier))
models.append(('Hybrid_Ensemble', Hybrid_Ensemble))

# Evaluate each model in turn
results = []
names = []

# Use different metrics based on context
scoring = 'accuracy'
# scoring = 'precision'
# scoring = 'recall'
# scoring = 'f1'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10,random_state=seed,shuffle=True)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=2)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
# Boxplot for algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# %%
# Use SMOTE to handle the imbalanced dataset
# Create the oversampler.
# smote=SMOTE(random_state=0)
# X1, y1=smote.fit_resample(X_train, y_train)
# sampling_strategy=0.5

# Using SMOTE
smt = SMOTE(random_state=0)
X1, y1 = smt.fit_sample(X_train, y_train)

# Using random under sampling
# under= RandomUnderSampler(random_state=0)
# X1, y1 = under.fit_sample(X_train, y_train)

# # Using random over sampling
# over= RandomOverSampler(random_state=0)
# X1, y1 = over.fit_sample(X_train, y_train)

# # Using ADASYN
# oversample = ADASYN(random_state=0)
# X1, y1 = oversample.fit_resample(X_train, y_train)

# %%
base_1 = []
for m in range(len(models)):
    base_2 = []
    model = models[m][1]
    model.fit(X1,y1)
    y_pred = model.predict(X_test)
    y1_pred = model.predict(X1)
    cm = confusion_matrix(y_test,y_pred)
    accuracies = cross_val_score(estimator= model, X = X1, y = y1, cv=10)

# k-fOLD Validation
    roc = roc_auc_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    print(models[m][0],':')
    print(cm)
    print('Train Accuracy Score: ',accuracy_score(y1,y1_pred))
    print('Test Accuracy Score: ',accuracy_score(y_test,y_pred))
    print('K-Fold Validation Mean Accuracy: {:.2f} %;'.format(accuracies.mean()*100), 
          'Standard Deviation: {:.2f} %'.format(accuracies.std()*100))
    print('ROC AUC Score: {:.2f} %'.format(roc))
    print('Precision: {:.2f} %'.format(precision))
    print('Recall: {:.2f} %'.format(recall))
    print('F1 Score: {:.2f} %'.format(f1))
    print('Classification Report:')
    print(classification_report(y_test, y_pred)) 
    print('-'*60)
    base_2.append(models[m][0])
    base_2.append(accuracy_score(y1,y1_pred)*100)
    base_2.append(accuracy_score(y_test,y_pred)*100)
    base_2.append(accuracies.mean()*100)
    base_2.append(accuracies.std()*100)
    base_2.append(roc)
    base_2.append(precision)
    base_2.append(recall)
    base_2.append(f1)
    base_1.append(base_2)

# %%
model_comparison = pd.DataFrame(base_1,columns=['Model','Train_Accuracy','Test_Accuracy','K-Fold Mean Accuracy',
                                                'Std.Deviation','ROC_AUC','Precision','Recall','F1 Score'])

model_comparison.sort_values(by=['Recall','F1 Score'],inplace=True,ascending=False)
model_comparison

# %% [markdown]
# **Best Model: Gaussian Naive Bayes based on ROC_AUC, Precision, Recall and F1 Score**
#
# 1. Prediction of class Recall is the most important parameter to decide the best model for this problem. Gaussian NB has the highest Recall value here.
# 2. Both KNN and Hybrid_ensemble are next to the Gaussian NB in performance.
# 3. A lot of over-fitting is visible in a number of models like Random Forest, BaggingClassifier, XGB and LGBM.
# 4. A balanced dataset would further improve the performance.
# 5. Hyper-parameter tuning of the base model may improve the performance. This is continued below...

# %% [markdown]
# #### Using hyperparameter tuning without oversampling

# %%
# Build the model
model = GaussianNB()

params = {'var_smoothing': np.logspace(0,-9, num=100)
          }

gs = RandomizedSearchCV(estimator=model, param_distributions=params, cv=5, verbose=10, n_jobs=2, random_state=0)
gs.fit(X_train, y_train)

# %%
pd.DataFrame({'param': gs.cv_results_["params"], 
              'score mean': gs.cv_results_["mean_test_score"], 
              'score s.d.': gs.cv_results_["std_test_score"],
              'rank': gs.cv_results_["rank_test_score"]})

# %%
# Print the best parameters
print('Best Parameters:', gs.best_params_, 'mean score: ', gs.best_score_, sep='\n')

# %%
# Build the model
model = GaussianNB(var_smoothing=0.004328761281083057)

# Train the model
model.fit(X_train, y_train)
model_pred = model.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, model_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, model_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["0","1"]],
                  columns = [i for i in ["0","1"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# **We do not see much improvement in the model here.**

# %% [markdown]
# #### Using hyperparameter tuning with Oversampling

# %%
# Build the model
model = GaussianNB()

params = {'var_smoothing': np.logspace(0,-9, num=100)
          }

gs = RandomizedSearchCV(estimator=model, param_distributions=params, cv=5, verbose=10, n_jobs=2, random_state=0)
gs.fit(X1, y1)

# %%
pd.DataFrame({'param': gs.cv_results_["params"], 
              'score mean': gs.cv_results_["mean_test_score"], 
              'score s.d.': gs.cv_results_["std_test_score"],
              'rank': gs.cv_results_["rank_test_score"]})

# %%
print('Best Parameters:', gs.best_params_, 'mean score: ', gs.best_score_, sep='\n')

# %%
# Build the model
model = GaussianNB(var_smoothing=1.519911082952933e-07)

# Train the model
model.fit(X1, y1)
model_pred = model.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, model_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, model_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["0","1"]],
                  columns = [i for i in ["0","1"]])
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

# %% [markdown]
# **We do not see much improvement in the model here either. Base model for Gaussian NB is still the best performing model.**

# %% [markdown]
# ### 5D. Use PCA for various algorithms

# %%
# Load the data and pre-process for pca
dfb=pd.read_csv('signal-data.csv')
dfb=dfb.drop(['Time'],axis=1)

# Drop the columns that have constant signal
cols = dfb.select_dtypes([np.number]).columns
std = dfb[cols].std()
cols_to_drop = std[std==0].index
dfb.drop(cols_to_drop, axis=1,inplace=True)

# label encode the target class
dfb['Pass/Fail']=dfb['Pass/Fail'].replace([-1,1],[0,1])

# Replace the NaN/NA with zero and consider it as no signal
dfb.fillna(0,inplace=True)
row,column=dfb.shape
print('The dataset contains:', row, 'rows and', column, 'columns')

# %%
# Create the features matrix and target vector
X=dfb.drop(['Pass/Fail'], axis=1)
y=dfb['Pass/Fail']

# %% [markdown]
# #### Apply PCA on the data with all the components.

# %%
# Scaling the complete feature matrix
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %%
# Calculating the covariance between attributes after scaling
# Covariance indicates the level to which two variables vary together.
cov_matrix = np.cov(X,rowvar=False)
print('Covariance Matrix:')
print(cov_matrix)

# %%
# Use PCA on all components
pca474 = PCA(n_components=474, random_state=0)
pca474.fit(X)

# %%
# The eigen Values
print(pca474.explained_variance_)

# %%
# The eigen Vectors
print(pca474.components_)

# %%
# And the percentage of variation explained by each eigen Vector
print(pca474.explained_variance_ratio_)

# %%
# Variation explained by each component
plt.figure(figsize=(20,8))
plt.bar(list(range(1,475)),pca474.explained_variance_ratio_,alpha=0.5, align='center')
plt.ylabel('Variation explained')
plt.xlabel('Eigen Value')
plt.show()

# %% [markdown]
# #### Visualize Cumulative Variance Explained with Number of Components.

# %%
# Cumulative Variation explained by each component
plt.figure(figsize=(20,8))
plt.step(list(range(1,475)),np.cumsum(pca474.explained_variance_ratio_), where='mid')
plt.ylabel('Cumulative variation explained')
plt.xlabel('Eigen Value')
plt.show()

# %% [markdown]
# #### Draw a horizontal line on the above plot to highlight the threshold of 90%.

# %%
# Cumulative Variation explained by each component
# Red dashed line at 90% cumulative variation is explained by 120 principal components
plt.figure(figsize=(20,8))
plt.step(list(range(1,475)),np.cumsum(pca474.explained_variance_ratio_), where='mid')
plt.axhline(y=0.9, color='r', linestyle='--', lw=1)
plt.ylabel('Cumulative variation explained')
plt.xlabel('Eigen Value')
plt.show()

# Now 5 dimensions seems very reasonable. With 5 variables we can explain over 90% of the 
# variation in the original data!

# %% [markdown]
# #### Apply PCA on the data. This time Select Minimum Components with 90% or above variance explained.

# %%
# 120 principal components are able to explain more than 90% of variance in the data
pca120 = PCA(n_components=120)
pca120.fit(X)
print(pca120.components_)
print(pca120.explained_variance_ratio_)
Xpca120 = pca120.transform(X)

# %%
# Print the original features and the reduced features
print('Original number of features:', X.shape[1])
print('Reduced number of features:', Xpca120.shape[1])

# %%
# View the first 5 observations of the pca components
Xpca120_df = pd.DataFrame(data = Xpca120)
Xpca120_df.head()

# %% [markdown]
# #### Create train and test datasets

# %%
X_train_row, X_train_col = X_train.shape
print('The X_train comprises of', X_train_row, 'rows and', X_train_col, 'columns.')

# %%
X_test_row, X_test_col = X_test.shape
print('The X_test comprises of', X_test_row, 'rows and', X_test_col, 'columns.')

# %%
# Split the pca data into train and test ratio of 80:20
Xpca120_train, Xpca120_test, y_train, y_test = train_test_split(Xpca120, y, test_size=0.20, random_state=0)

# %%
Xpca120_train_row, Xpca120_train_col = Xpca120_train.shape
print('The Xpca120_train comprises of', Xpca120_train_row, 'rows and', Xpca120_train_col, 'columns.')

# %%
Xpca120_test_row,  Xpca120_test_col =  Xpca120_test.shape
print('The  Xpca120_test comprises of',  Xpca120_test_row, 'rows and',  Xpca120_test_col, 'columns.')

# %%
X_train=Xpca120_train
X_test=Xpca120_test

# %% [markdown]
# #### Select  the best performing model with PCA

# %%
# K Fold Cross Validation Scores

seed = 0

# Create models
models = []

models.append(('LR_Model', LR_model))
models.append(('KNN_Model', KNN_model))
models.append(('GNB_Model', GNB_model))
# models.append(('SVM_Linear', SVM_model_linear))
# models.append(('SVM_Rbf', SVM_model_rbf))
# models.append(('SVM_Poly', SVM_model_poly))
models.append(('RandomForest', RandomForest))
models.append(('BaggingClassifier', BaggingClassifier))
models.append(('AdaBoostClassifier', AdaBoostClassifier))
models.append(('GBClassifier', GBClassifier))
models.append(('XGBClassifier', XGBClassifier))
models.append(('LGBMClassifier', LGBMClassifier))
# models.append(('CatBoostClassifier', CatBoostClassifier))
models.append(('Hybrid_Ensemble', Hybrid_Ensemble))

# Evaluate each model in turn
results = []
names = []

# Use different metrics based on context
scoring = 'accuracy'
# scoring = 'precision'
# scoring = 'recall'
# scoring = 'f1'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10,random_state=seed,shuffle=True)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=2)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
# Boxplot for algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# %%
# Use SMOTE to handle the imbalanced dataset
# Create the oversampler.
# smote=SMOTE(random_state=0)
# X1, y1=smote.fit_resample(X_train, y_train)
# sampling_strategy=0.5

# Using SMOTE
smt = SMOTE(random_state=0)
X1, y1 = smt.fit_sample(X_train, y_train)

# Using random under sampling
# under= RandomUnderSampler(random_state=0)
# X1, y1 = under.fit_sample(X_train, y_train)

# # Using random over sampling
# over= RandomOverSampler(random_state=0)
# X1, y1 = over.fit_sample(X_train, y_train)

# # Using ADASYN
# oversample = ADASYN(random_state=0)
# X1, y1 = oversample.fit_resample(X_train, y_train)


# %%
base_1 = []
for m in range(len(models)):
    base_2 = []
    model = models[m][1]
    model.fit(X1,y1)
    y_pred = model.predict(X_test)
    y1_pred = model.predict(X1)
    cm = confusion_matrix(y_test,y_pred)
    accuracies = cross_val_score(estimator= model, X = X1, y = y1, cv=10)

# k-fOLD Validation
    roc = roc_auc_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    print(models[m][0],':')
    print(cm)
    print('Train Accuracy Score: ',accuracy_score(y1,y1_pred))
    print('Test Accuracy Score: ',accuracy_score(y_test,y_pred))
    print('K-Fold Validation Mean Accuracy: {:.2f} %;'.format(accuracies.mean()*100), 
          'Standard Deviation: {:.2f} %'.format(accuracies.std()*100))
    print('ROC AUC Score: {:.2f} %'.format(roc))
    print('Precision: {:.2f} %'.format(precision))
    print('Recall: {:.2f} %'.format(recall))
    print('F1 Score: {:.2f} %'.format(f1))
    print('Classification Report:')
    print(classification_report(y_test, y_pred)) 
    print('-'*60)
    base_2.append(models[m][0])
    base_2.append(accuracy_score(y1,y1_pred)*100)
    base_2.append(accuracy_score(y_test,y_pred)*100)
    base_2.append(accuracies.mean()*100)
    base_2.append(accuracies.std()*100)
    base_2.append(roc)
    base_2.append(precision)
    base_2.append(recall)
    base_2.append(f1)
    base_1.append(base_2)

# %%
model_comparison = pd.DataFrame(base_1,columns=['Model','Train_Accuracy','Test_Accuracy','K-Fold Mean Accuracy',
                                                'Std.Deviation','ROC_AUC','Precision','Recall','F1 Score'])

model_comparison.sort_values(by=['Recall','F1 Score'],inplace=True,ascending=False)
model_comparison

# %% [markdown]
# ## 6. Post Training and Conclusion:

# %% [markdown]
# ### 6A. Display and compare all the models designed with their train and test accuracies.

# %%
# This is answered in 5C above. Please refer for details.

# %% [markdown]
# ### 6B. Select the final best trained model along with your detailed comments for selecting this model.

# %%
# Create the features matrix and target vector
X=dfa1.drop(['Pass/Fail'], axis=1)
y=dfa1['Pass/Fail']

# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Using different scaling methods:
# scaler = RobustScaler()
# scaler = MinMaxScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Using SMOTE
# Create the oversampler.
smt = SMOTE(random_state=0)
X1, y1 = smt.fit_sample(X_train, y_train)

# %%
# Build the model
model = GaussianNB()

# Train the model
model.fit(X1, y1)
model_pred = model.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, model_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, model_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["0","1"]],
                  columns = [i for i in ["0","1"]])
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

# %% [markdown]
# **Considering the Class Recall as the most important parameter to decide the best model for this problem. We have the highest Recall value here:**
# - Class 0 predicted correctly for 86% of time. On similar lines for class 1 its 38%.
# - Using F1 Score: Precision and Recall is balanced for class 0 by 91% and for class 1 by 17%.
# - Precision, Recall, and F1 Score are highest for class 0 followed by class 1.
# - We have maximum F1 score for class 0, and minimum for class 1.

# %% [markdown]
# ### 6C. Pickle the selected model for future use.

# %%
# Import pickle Package
import pickle

# %%
# Save the ML model to a file in the current working directory
Pkl_Filename = "Pickle_GaussianNB_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)

# %%
# Load the ML Model back from the file
with open(Pkl_Filename, 'rb') as file:  
    Pickle_GaussianNB_Model = pickle.load(file)
    
Pickle_GaussianNB_Model

# %% [markdown]
# ### 6D. Write your conclusion on the results.

# %% [markdown]
# - We have tried multiple models Logistic Regression, KNN, GaussianNB, SVM, Random Forest, Bagging Classifier, AdaBoost Classifier, GB Classifier, XGB Classifier, LGBM Classifier, CatBoost Classifier.
# - Across methods GaussianNB performed the best while LGBM performed the worst.
# - We have tried four sampling techniques: SMOTE, Random Oversampling, Random Undersampling, and ADASYN. SMOTE performed better compared to others.
# - Different scaling methods like RobustScaler, MinMaxScaler, StandardScaler, Zscore were considered for model building.
# - We saw that for imbalanced classes accuracy and recall are inversely proportional to each other. Better recall models have lower accuracy and vice versa.
# - K-fold cross validation was used to compare various models.
# - As far as PCA is considered, the models does better without it.
# - Intervention of SME is vital to make sense of the dataset and various performance metrics.

# %% [markdown]
# ## 7. 4D. Check if the train and test data have similar statistical characteristics when compared with original data
# First covering: How (dis)similar are my train and test data?

# %%
# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
# loading test and train data
train = X_train
test = X_test

# %%
# adding a column to identify whether a row comes from train or not
train['is_train'] = 1
test['is_train'] = 0

# %%
# combining test and train data
df_combine = pd.concat([train, test], axis=0, ignore_index=True)
# dropping ‘target’ column as it is not present in the test
# df_combine = df_combine.drop(‘target’, axis =1)
y = df_combine['is_train'].values #labels
x = df_combine.drop('is_train', axis=1).values #covariates or our independent variables
tst, trn = test.values, train.values

# %%
m = RandomForestClassifier(n_jobs=2, max_depth=5, min_samples_leaf = 5)
predictions = np.zeros(y.shape) #creating an empty prediction array

# %%
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import roc_auc_score as AUC
skf = SKF(n_splits=20, shuffle=True, random_state=100)
for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
 X_train, X_test = x[train_idx], x[test_idx]
 y_train, y_test = y[train_idx], y[test_idx]
 
 m.fit(X_train, y_train)
 probs = m.predict_proba(X_test)[:, 1] #calculating the probability
 predictions[test_idx] = probs

# %%
print('ROC-AUC for train and test distributions:', AUC(y, predictions))

# %% [markdown]
# A high AUROC means that the model is performing well, and in this case it means that there is a big difference in distributions of predictor variables between the training and test set. Ideally, the distribution of the predictors for the training and test set should be the same, so you would want to get an AUROC that is close to 0.5.
#
# I think this situation would only be relevant in cases where you have your model deployed and you need to check if your model is still relevant over time. If you are building a new model you shouldn’t need to do something like this because the test data is randomly sampled from the dataset. Additionally, if you’re doing cross validation then there is even less reason to worry about something like that.

# %%
# Consider the distibution complete dataset X
plt.figure(figsize=(20,10))
sns.distplot(X);
plt.show()

# %%
# Consider the distibution complete dataset X_train
plt.figure(figsize=(20,10))
sns.distplot(X_train);
plt.show()

# %%
# Consider the distibution complete dataset X_test
plt.figure(figsize=(20,10))
sns.distplot(X_test);
plt.show()

# %%
# 5 Point summary for X
X.describe().T

# %%
# 5 Point summary for X_train
X_train_df=pd.DataFrame(X_train)
X_train_df.describe().T

# %%
# 5 Point summary for X_test
X_test_df=pd.DataFrame(X_test)
X_test_df.describe().T

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
