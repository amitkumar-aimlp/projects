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
# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#00b3e5;">Contents: Ensemble Techniques Project
# 1. [Part-A: Solution](#Part-A:-Solution)
# 2. [Part-B: Solution](#Part-B:-Solution)

# %% [markdown]
# # Part-A: Solution

# %% [markdown]
# - **DOMAIN:** Telecom
# - **CONTEXT:** A telecom company wants to use their historical customer data to predict behaviour to retain customers. You can analyse all relevant customer data and develop focused customer retention programs.
# - **DATA DESCRIPTION:** Each row represents a customer, each column contains customer’s attributes described on the column Metadata. The data set includes information about:
#  - Customers who left within the last month – the column is called Churn
#  - Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
#  - Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
#  - Demographic info about customers – gender, age range, and if they have partners and dependents
# - **PROJECT OBJECTIVE:** To Build a model that will help to identify the potential customers who have a higher probability to churn. This helps the company to understand the pinpoints and patterns of customer churn and will increase the focus on strategizing customer retention.

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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import plot_importance

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Data Understanding and Exploration:

# %% [markdown]
# ### 1A. Read ‘TelcomCustomer-Churn_1.csv’ as a DataFrame and assign it to a variable.

# %%
# CSV File 1
dfa1=pd.read_csv('TelcomCustomer-Churn_1.csv')

# %%
dfa1.info()
dfa1.head()

# %% [markdown]
# ### 1B. Read ‘TelcomCustomer-Churn_2.csv’ as a DataFrame and assign it to a variable.

# %%
# CSV File 2
dfa2=pd.read_csv('TelcomCustomer-Churn_2.csv')

# %%
dfa2.info()
dfa2.head()

# %% [markdown]
# ### 1C. Merge both the DataFrames on key ‘customerID’ to form a single DataFrame

# %%
# customerID is common in both the dataframes
dfa=dfa1.merge(dfa2, left_on='customerID', right_on='customerID')

# %%
dfa.info()
dfa.head()

# %% [markdown]
# ### 1D. Verify if all the columns are incorporated in the merged DataFrame by using simple comparison Operator in Python.

# %%
# By default, pd.merge gives you an inner merge so you only see the matched.
# Approach 1 is to do instead an outer merge
# And check which rows have np.nan for identifying those that could have missed out from an inner merge:

option1 = dfa1.merge(dfa2, on='customerID', how='outer')

# %%
print(option1[option1.isna().any(axis=1)])

# %%
dfa.columns


# %% [markdown]
# ## 2. Data Cleaning & Analysis:

# %% [markdown]
# ### 2A. Impute missing/unexpected values in the DataFrame.
# (This part will continue in 2B)

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
# Get a list of categories of categorical variable
print(dfa.gender.value_counts())
print(dfa.SeniorCitizen.value_counts())
print(dfa.Partner.value_counts())
print(dfa.Dependents.value_counts())
print(dfa.tenure.value_counts())
print(dfa.PhoneService.value_counts())
print(dfa.MultipleLines.value_counts())
print(dfa.InternetService.value_counts())
print(dfa.OnlineSecurity.value_counts())
print(dfa.OnlineBackup.value_counts())
print(dfa.DeviceProtection.value_counts())
print(dfa.TechSupport.value_counts())
print(dfa.StreamingTV.value_counts())
print(dfa.StreamingMovies.value_counts())
print(dfa.Contract.value_counts())
print(dfa.PaperlessBilling.value_counts())
print(dfa.PaymentMethod.value_counts())
print(dfa.MonthlyCharges.value_counts())
print(dfa.TotalCharges.value_counts())
print(dfa.Churn.value_counts())

# %% [markdown]
# ### 2B. Make sure all the variables with continuous values are of ‘Float’ type.
# For Example: MonthlyCharges, TotalCharges

# %%
dfa.info()

# %%
dfa.head().T

# %%
# Change the Datatype of quantitative features
col_cat=['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
    
#Function to convert the categorical to quantitative
def convert_to_cont(feature):
    dfa[feature]=pd.to_numeric(dfa[feature], errors='coerce')
    
for c in col_cat:
    convert_to_cont(c)

# %%
dfa.info()

# %%
dfa.isnull().sum()

# %%
dfa.fillna(dfa.mean(),inplace = True)

# %%
dfa.info()

# %%
# CustomerID is the id of the customer with corresponding details. This information may not be requried
# for analysis and modeling as the customerID will be all unique values. So we can drop it safely.
dfa.drop(['customerID'], axis=1, inplace=True)

# %%
dfa.info()

# %% [markdown]
# ### Quick EDA

# %%
# Describe function generates descriptive statistics that summarize the central tendency, 
# dispersion and shape of a dataset’s distribution, excluding NaN values.

# This method tells us a lot of things about a dataset. One important thing is that 
# the describe() method deals only with numeric values. It doesn't work with any 
# categorical values. So if there are any categorical values in a column the describe() 
# method will ignore it and display summary for the other columns.
dfa.describe().T

# %%
# Distribution of Continuous Features
int_feat = dfa.select_dtypes(exclude=['object','category']).columns
fig, ax = plt.subplots(nrows=2, ncols = 2, figsize=(15,8), constrained_layout=True)
ax=ax.flatten()
for c,i in enumerate(int_feat):
    sns.histplot(dfa[i], ax=ax[c], bins=10)
    ax[c].set_title(i)

# %%
# Distribution of Categorical Features
cat_cols = dfa.select_dtypes(include=['object','category']).columns
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(15,15), constrained_layout=True)
ax=ax.flatten()
for x,i in enumerate(cat_cols):
    sns.countplot(x=dfa[i], ax=ax[x])

# %%
# Visualize a pairplot with 2 classes distinguished by colors
sns.pairplot(dfa,hue='Churn', corner=True )

# %%
sns.heatmap(dfa.corr(), annot=True)

# %%
# Box Plot
plt.figure(figsize=(20,8))
ax = sns.boxplot(data=dfa, orient="h", palette="Set2")

# %%
# Distribution of Target Variable
count_no_churn = (dfa['Churn'] == 'No').sum()
print("Number of customers who didn't churn:",count_no_churn)
count_yes_churn = (dfa['Churn']== 'Yes').sum()
print("Number of customers who churned:",count_yes_churn)

# %%
fig, ax = plt.subplots(figsize=(20,8))
width = len(dfa['Churn'].unique())+6
fig.set_size_inches(width , 8)
ax=sns.countplot(data = dfa, x= 'Churn') 

for p in ax.patches: 
    ax.annotate(str((np.round(p.get_height()/len(dfa)*100,decimals=2)))+'%', 
    (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', 
    xytext=(0, 10), textcoords='offset points')


# %% [markdown]
# **Imbalanced dataset:**
#
# - The Target variable is not equally distributed, only 26.54% of customers have Churned.
# - The model may be biased towards the majority class i.e. Not Churned Customers.
# - Data can be balanced using the SMOTE.

# %% [markdown]
# ### 2C. Create a function that will accept a DataFrame as input and return pie-charts for all the appropriate Categorical features. Clearly show percentage distribution in the pie-chart

# %%
#Function to plot Pie-Charts for all categorical variables in the dataframe
def pie_charts_for_CategoricalVar(df,m):
    '''Takes in a dataframe(df_pie) and plots pie charts for all categorical 
    columns. m = number of columns required in grid'''
    
    #get all the column names in the dataframe
    a = []
    for i in df:
        a.append(i)
    
    #isolate the categorical variable names from a to b
    b = []
    for i in a:
        if (df[i].dtype.name) == 'object':
            b.append(i)
        
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.2)
    plt.suptitle("Pie-Charts for Categorical Variables in the dataframe", fontsize=18, y=0.95)
    
    # number of columns, as inputted while calling the function
    ncols = m
    # calculate number of rows
    nrows = len(b) // ncols + (len(b) % ncols > 0)
    
    # loop through the length of 'b' and keep track of index
    for n, i in enumerate(b):
        # add a new subplot iteratively using nrows and ncols
        ax = plt.subplot(nrows, ncols, n + 1)

        # filter df and plot 'i' on the new subplot axis
        df.groupby(i).size().plot(kind='pie', autopct='%.2f%%',ax=ax)
        
        ax.set_title(i.upper())
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.show()

pie_charts_for_CategoricalVar(dfa,4)

# %% [markdown]
# ### 2D. Share insights for Q2.C.

# %% [markdown]
# **Insights from the above pie charts:**
# 1. Most Features are not having the equitable distribution of various classes: Dependents, PhoneService, MultipleLines, TechSupport, StreamingTV,StreamingMovies, Contract, PaperlessBilling, Churn Etc.
# 2. Very few features have a balanced or almost balanced distributions of various classes: Gender, Partner, PaymentMethod Etc.
# 3. There is huge imbalance in target vector i.e. Churn:
#
# If the imbalanced data is not treated beforehand, then this will degrade the performance of the ML model. Most of the predictions will correspond to the majority class and treat the minority class of features as noise in the data and ignore them. This results in a high bias and low performance of the model.
#
# A widely adopted technique for dealing with highly unbalanced datasets is called re-sampling.
#
# **Two widely used re-sampling methods are:**
#
# - Under-sampling: It is the process where you randomly delete some of the observations from the majority class in order to match the numbers with the minority class.
#
# - Over-sampling: It is the process of generating synthetic data that tries to randomly generate a sample of the attributes from observations in the minority class
#
# - Here we will use oversampling because under-sampling may remove important information from the dataset

# %% [markdown]
# ### 2E. Encode all the appropriate Categorical features with the best suitable approach.

# %% [markdown]
# **There are a lot of Yes/No values, we can replace them with 1 or 0:**
#
# - PhoneService - is the telephone service connected (Yes, No) - (1,0)
# - MultipleLines - are multiple phone lines connected (Yes, No, No phone service) - (1,0,0)
# - InternetService - client's Internet service provider (DSL, Fiber optic, No) -(2,1,0)
# - OnlineSecurity - is the online security service connected (Yes, No, No internet service)-(1,0,0)
# - OnlineBackup - is the online backup service activated (Yes, No, No internet service)-(1,0,0)
# - DeviceProtection - does the client have equipment insurance (Yes, No, No internet service)-(1,0,0)
# - TechSupport - is the technical support service connected (Yes, No, No internet service)-(1,0,0)
# - StreamingTV - is the streaming TV service connected (Yes, No, No internet service)-(1,0,0)
# - StreamingMovies - is the streaming cinema service activated (Yes, No, No internet service)-(1,0,0)
# - Contract - type of customer contract (Month-to-month, One year, Two year) - (month-to-month - 1, One Year - 12, Two Year = 24)
# - PaperlessBilling - whether the client uses paperless billing (Yes, No) - (1,0)
# - PaymentMethod - payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
# - MonthlyCharges - current monthly payment
# - TotalCharges - the total amount that the client paid for the services for the entire time
# - Churn - whether there was a churn (Yes or No) - (1,0)

# %%
# Feature engineering - convert the object features to integer based on the category
dfa=dfa.replace('Yes',1)
dfa=dfa.replace('No',0)
dfa=dfa.replace('No internet service',0)
dfa=dfa.replace('No phone service',0)
dfa=dfa.replace('Fiber optic',2)
dfa=dfa.replace('DSL',1)
dfa=dfa.replace('Male',1)
dfa=dfa.replace('Female',0)
dfa=pd.get_dummies(data=dfa, columns=['Contract','PaymentMethod'],drop_first=True )

# %%
dfa.head()

# %% [markdown]
# ### 2F. Split the data into 80% train and 20% test.

# %%
# Create the features matrix and target vector
X=dfa.drop(['Churn'], axis=1)
y=dfa['Churn']

# %%
# Check for imbalanced dataset by numbers
y.value_counts()

# %%
# Check for imbalanced dataset by percentage
1869/5174*100

# %%
# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %% [markdown]
# ### 2G. Normalize/Standardize the data with the best suitable approach.

# %%
# Using different scaling methods:
# scaler = RobustScaler()
# scaler = MinMaxScaler()
scaler = StandardScaler()

cols_to_scale = ["MonthlyCharges","TotalCharges","tenure"]

X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# %%
X_train.head()

# %%
X_test.head()

# %%
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# %% [markdown]
# ## 3. Model building and Improvement:

# %% [markdown]
# ### 3A. Train a model using XGBoost. Also print best performing parameters along with train and test performance.
# Base model is created in 3A, and the parameter tuning is done in 3B. Considering both part 3A and 3B here.

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
df_cm = pd.DataFrame(cm, index = [i for i in ["Churn-No","Churn-Yes"]],
                  columns = [i for i in ["Churn-No","Churn-Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %%
# Default Parameters used in the base model:
print(model)

# %% [markdown]
# ### 3B. Improve performance of the XGBoost as much as possible. Also print best performing parameters along with train and test performance.

# %% [markdown]
# #### Case-1: Using oversampling over complete dataset

# %%
print('Before oversampling distribution of target vector:')
print(y.value_counts())

# %%
# Using SMOTE
# Create the oversampler. 
smote=SMOTE(random_state=0)
X1, y1=smote.fit_resample(X, y)

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
df_cm = pd.DataFrame(cm, index = [i for i in ["Churn-No","Churn-Yes"]],
                  columns = [i for i in ["Churn-No","Churn-Yes"]])
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
smote=SMOTE(random_state=0)
X1, y1=smote.fit_resample(X_train, y_train)

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
df_cm = pd.DataFrame(cm, index = [i for i in ["Churn-No","Churn-Yes"]],
                  columns = [i for i in ["Churn-No","Churn-Yes"]])
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
# **Considering the Prediction of customer churn (Recall) as the most important parameter to decide the best model for this problem. We have the highest Recall value here:**
# - Class 0 predicted correctly for 85% of time. On similar lines for class 1 its 54%.
# - Using F1 Score: Precision and Recall is balanced for class 0 by 84% and for class 1 by 56%.
# - Precision, Recall, and F1 Score are highest for class 0 followed by class 1.
# - We have maximum F1 score for class 0, and minimum for class 1.

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
model = xgb.XGBClassifier(n_jobs=2, random_state=0, subsample=0.899999999999999, reg_lambda=10.0, reg_alpha=1e-05, n_estimators=100, 
                          min_child_weight=1.0, max_depth=20, learning_rate=0.3, gamma=0.2, 
                          colsample_bytree=0.8999999999999999, colsample_bylevel=0.7, verbosity=0)

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
df_cm = pd.DataFrame(cm, index = [i for i in ["Churn-No","Churn-Yes"]],
                  columns = [i for i in ["Churn-No","Churn-Yes"]])
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
model = xgb.XGBClassifier(n_jobs=2, random_state=0, subsample=0.899999999999999, reg_lambda=50.0, reg_alpha=0.01, n_estimators=500, 
                          min_child_weight=3.0, max_depth=15, learning_rate=0.01, gamma=0.4, 
                          colsample_bytree=0.4, colsample_bylevel=0.5, verbosity=0)

# Train the model
model.fit(X_train, y_train)
model_pred = model.predict(X_test)

# %%
# Plot features importance chart
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
df_cm = pd.DataFrame(cm, index = [i for i in ["Churn-No","Churn-Yes"]],
                  columns = [i for i in ["Churn-No","Churn-Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# **Refer the Case-2 for best performing model above. Model tuning will continue in Part-B**

# %% [markdown]
# # Part-B: Solution

# %% [markdown]
# - **DOMAIN:** IT
# - **CONTEXT:** The purpose is to build a machine learning workflow that will work autonomously irrespective of Data and users can save efforts involved in building workflows for each dataset.
# - **PROJECT OBJECTIVE:** Build a machine learning workflow that will run autonomously with the csv file and return best performing model.
#
# - **STEPS AND TASK:**
#  1. Build a simple ML workflow which will accept a single ‘.csv’ file as input and return a trained base model that can be used for predictions. You can use 1 Dataset from Part 1 (single/merged).
#  2. Create separate functions for various purposes.
#  3. Various base models should be trained to select the best performing model.
#  4. Pickle file should be saved for the best performing model.
#
# - **Include best coding practices in the code:**
#  - Modularization
#  - Maintainability
#  - Well commented code etc.

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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, plot_roc_curve, precision_recall_curve, plot_precision_recall_curve, average_precision_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC

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
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Import and understand the Data:

# %%
# Using the complete dataset from Part-A

# CSV File 1
dfa1=pd.read_csv('TelcomCustomer-Churn_1.csv')
# CSV File 2
dfa2=pd.read_csv('TelcomCustomer-Churn_2.csv')
# customerID is common in both the dataframes
dfa=dfa1.merge(dfa2, left_on='customerID', right_on='customerID')

# CustomerID is the id of the customer with corresponding details. This information may not be requried
# for analysis and modeling as the customerID will be all unique values. So we can drop it safely.
dfa.drop(['customerID'], axis=1, inplace=True)

dfa.info()
dfa.head()

# %%
dfa.columns


# %% [markdown]
# ## 2. Automation of Data Pre-processing Task:

# %% [markdown]
# ### Clean Data

# %% [markdown]
# #### Remove the Missing Data
# We need to remove data where the value of any feature is nan or na or empty. Imputation is also a great strategy.

# %%
def remove_missing(df) : 
  remove = []
  for i, row in df.iterrows():
    if row.isna().values.any() : remove.append(i)
  df.drop(remove,axis=0,inplace=True)


# %% [markdown]
# #### Remove the Mismatch Data
# We need to remove data with mismatches: For e.g. a data point with a string value for a numerical feature. For this, we will check what data type is the majority for each feature and remove the data with a different data type for those features.
#
# Provision for exceptions: where we can specify features for which values can have different data types and we do not want to remove mismatches.

# %%
def remove_mismatch(df,exceptions=[]) : 
  for col in df : 
    if col in exceptions : continue
    df.reset_index(drop=True, inplace=True)
    s = [False]*len(df[col])
    for i,cell in enumerate(df[col]) : 
      try : n = int(cell)
      except : s[i] = True
    t = s.count(True)
    f = s.count(False)
    st = False
    if(t>f) : st = True
    remove = [i for i in range(len(df[col])) if s[i]!=st]
    df.drop(remove,axis=0,inplace=True)


# %% [markdown]
# #### Convert Numeric Data Stored as String to Numerical Form
#
# Sometimes Numeric Data (e.g. int, float) is stored as a String, this may lead to an error when we train our model or normalize our data. We need to identify such cases and convert them to their original numerical forms.

# %%
def str_to_num(df) : 
  for col in df : 
    try : df[col] = pd.to_numeric(df[col])
    except : pass


# %% [markdown]
# #### Single Function to Clean the Data

# %%
def clean(df,exceptions_mismatch=[]) : 
  remove_missing(df) 
  remove_mismatch(df,exceptions=exceptions_mismatch)
  str_to_num(df)


# %%
# Example
clean(dfa,exceptions_mismatch=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges', 'Churn'])

# %%
dfa.info()
dfa.head()

# %% [markdown]
# ### Encode Data
# - Label Encoding: Assign an integer to each unique value of a column/feature.
# - One Hot Encoding: Convert 1 column to n columns where n is the number of unique values in that column. Each new column represents a unique value in the original column and it contains either 0 or 1. So in each row, only one of the n columns will have the value 1 and the remaining n-1 columns will have the value 0.
#
# We are going to represent the type of encoding we want for each column using a dictionary, where the keys will be the column/feature names and their values will be the type of encoding we want.

# %%
# ‘None’ will mean One Hot Encoding
# ‘[]’ would mean Label Encoding without a given order
# ‘[a,b,c…]’ would mean Label Encoding with the list being the order
labels = {}
labels['gender'] = []
labels['SeniorCitizen'] = []
labels['Partner'] = []
labels['Dependents'] = []
# labels['tenure'] = []
labels['PhoneService'] = [] 
labels['MultipleLines'] = [] 
labels['InternetService'] = []
labels['OnlineSecurity'] = [] 
labels['OnlineBackup'] = []
labels['DeviceProtection'] = [] 
labels['TechSupport'] = []
labels['StreamingTV'] = [] 
labels['StreamingMovies'] = [] 
labels['PaperlessBilling'] = []
# labels['MonthlyCharges'] = [] 
# labels['TotalCharges'] = [] 
labels['Churn'] = []
labels['Contract'] = None
labels['PaymentMethod'] = None

# This way experimenting with different encoding will become very easy. 
# For eg., if we want to change the encoding of the ‘Type’ column from One Hot to Label, 
# we can do it by simply changing its value in the labels dictionary from None to [].

# %% [markdown]
# #### Label Encoding
# The function takes the column name and order as input.

# %%
# Lets say, df['col'] = ['b','a','b','c']

# order = []
 # Label Encoding with no given order
 # df['col'] = [0,1,0,2]
# order = ['a','b','c']
 # Label Encoding with given order
 # df['col'] = [1,0,1,2]
# order = ['a']
 # By giving only a few values in order we can keep remaining values as 'others'
 # df['col'] = [-1,0,-1,-1]

def encode_label(df,col,order=[]) :
  if(order==[]) : order = list(df[col].unique())
  for i,cell in enumerate(df[col]) : 
    try : 
      df.at[i,col] = order.index(df[col][i])
    except : 
      df.at[i,col] = -1


# %% [markdown]
# #### One Hot Encoding

# %%
# The function takes the column name as input.

# Lets say, df['col'] = ['b','a','b','c']

# After One Hot Encoding -
 # df['col_b'] = [1,0,1,0]
 # df['col_a'] = [0,1,0,0]
 # df['col_c'] = [0,0,0,1]

def encode_onehot(df,col) :
  k = {}
  n = df[col].shape[0]
  unique = df[col].unique()
  for unq in unique : k[unq] = [0]*n
  for i in range(n) :
    k[df.at[i,col]][i] = 1
  for unq in unique : df[f"{col}_{unq}"] = k[unq] 
  df.drop(col,axis=1,inplace=True)


# %% [markdown]
# #### Single Function For Encoding Data

# %%
def encode(df,cols) : 
  for col in cols.keys() : 
    if(cols[col] is None) : encode_onehot(df,col)
    else : encode_label(df,col,cols[col])


# %%
# Example
encode(dfa,labels)

# %%
dfa.info()
dfa.head()

# %%
# Convert the categorical Data type to Numerical for modeling
col=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'TotalCharges', 'Churn']

def convert_to_cont(feature):
    dfa[feature]=pd.to_numeric(dfa[feature], errors='coerce')
    
for c in col:
    convert_to_cont(c)
    
dfa.fillna(dfa.mean(),inplace = True)

# %%
dfa.info()
dfa.head()

# %%
dfa.columns

# %% [markdown]
# ### Quick EDA

# %%
# Correlation of "Churn" with other features
plt.figure(figsize=(15,8))
dfa.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')

# %%
# Checking Correlation Heatmap
plt.figure(dpi = 540,figsize= (30,25))
mask = np.triu(np.ones_like(dfa.corr()))
sns.heatmap(dfa.corr(),mask = mask, fmt = ".2f",annot=True,lw=1,cmap = 'plasma')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.title('Correlation Heatmap')
plt.show()


# %% [markdown]
# ### Normalize Data
# - Divide by Largest: Divide all the values in a column by the largest value in that column
# - Divide by Constant: Divide all the values in a column by a constant value (e.g. 255 in case of an image)
# - Divide by Constant x Largest: Divide all the values in a column by a given constant x largest value in that column
# - Min-Max Normalization: Subtracting the minimum value from all the values in a column and then dividing all the values by the largest value in that column (new min will be 0 and new max will be 1)
# - Mean Normalization: Subtracting the mean from all the values in a column and then dividing all the values by (largest-smallest).

# %% [markdown]
# #### Different Normalization Functions

# %%
# Dividing by largest
def normalize_dbl(df,cols,round=None) : 
  if(type(cols)!=list) : cols = [cols]
  for col in cols : 
    l = df[col].max()
    if round is None : df[col] = df[col].div(l)
    else : df[col] = df[col].div(l).round(round)
        
# Dividing by constant
def normalize_dbc(df,cols,round=None,c=1) :
  if(type(cols)!=list) : cols = [cols]
  for col in cols : 
    if round is None : df[col] = df[col].div(c)
    else : df[col] = df[col].div(c).round(round)
        
# Dividing by constant x largest
def normalize_dblc(df,cols,round=None,c=1) :
  if(type(cols)!=list) : cols = [cols]
  for col in cols : 
    l = df[col].max() * c
    if round is None : df[col] = df[col].div(l)
    else : df[col] = df[col].div(l).round(round)
        
# min-max normalization
def normalize_rescale(df,cols,round=None) :
  if(type(cols)!=list) : cols = [cols]
  for col in cols : 
    df[col] = df[col] - df[col].min()
    l = df[col].max()
    if round is None : df[col] = df[col].div(l)
    else : df[col] = df[col].div(l).round(round)
        
# mean normalization
def normalize_mean(df,cols,round=None) :
  if(type(cols)!=list) : cols = [cols]
  for col in cols : 
    mean = df[col].mean()
    l = df[col].max() - df[col].min()
    df[col] = df[col] - mean
    if round is None : df[col] = df[col].div(l)
    else : df[col] = df[col].div(l).round(round)


# %% [markdown]
# #### Single Function for Normalizing Data

# %%
def normalize(df,cols=None,kinds='dbl',round=None,c=1,exceptions=[]) :
  if(cols is None) : 
    cols = []
    for col in df : 
      if(pd.api.types.is_numeric_dtype(df[col])) : 
        if(max(df[col])>1 or min(df[col])<-1) : 
          if(col not in exceptions) : cols.append(col)
  if(type(cols)!=list) : cols = [cols]
  n = len(cols)
  if(type(kinds)!=list) : kinds = [kinds]*n
  for i,kind in enumerate(kinds) : 
    if(kind=='dbl') : normalize_dbl(df,cols[i],round)
    if(kind=='dbc') : normalize_dbc(df,cols[i],round,c)
    if(kind=='dblc') : normalize_dblc(df,cols[i],round,c)
    if(kind in ['min-max','rescale','scale']) : normalize_rescale(df,cols[i],round)
    if(kind=='mean') : normalize_mean(df,cols[i],round)


# %%
# Example

# We can vastly vary the overall normalizations by easily making changes in the 
# parameters of this function when we call it. This helps in experimenting with 
# different normalizations.

# Some examples of various ways in which we can normalize our data using this function

# If we want to normalize all columns (it detects numeric columns)
# normalize(df)

# If we want to normalize and round to 3 decimal places –
# normalize(df,round=3)

# If we want to normalize all columns by a kind other than dividing by largest –
# normalize(df,kinds='mean')

# If we want to normalize some columns with a kind and some columns with other kind –
# normalize(df,['Price','Horsepower'],'dbl')
# normalize(df,['AirBags','Cylinders'],'min-max')
# normalize(df,['RPM'],'dblc',c=1.25)

# OR
# normalize(df,['Price','AirBags','Cylinders','Horsepower','RPM'],['dbl','min-max','min-max','dbl','dblc'],c=1.25)

# If we want to normalize all columns except a few
# normalize(df,kinds='min-max',exceptions=['AirBags','RPM'],round=4)

normalize(dfa, kinds='mean', exceptions=['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'PaperlessBilling','Churn', 'Contract_Month-to-month', 
       'Contract_One year','Contract_Two year', 'PaymentMethod_Electronic check',
       'PaymentMethod_Mailed check', 'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)'])

# %%
dfa.info()
dfa.head()

# %% [markdown]
# ### Split Data
# We will use sklearn and make a function to split data where we won’t even need to mention if we are splitting our data into 2 or 3 portions.
#
# We also need to reset the index of x_train, x_test, etc., otherwise, we can face problems while iterating over them in the future.

# %%
from sklearn.model_selection import train_test_split
X = dfa.drop(['Churn'], axis=1)
y = dfa.loc[:,'Churn']


# %% [markdown]
# #### Split-1

# %%
def train_test(X,y,train_size=-1,test_size=-1) :

    if(train_size==-1) : train_size = 1-test_size

    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=train_size,random_state=0)

    X_train.reset_index(drop=True,inplace=True)

    X_test.reset_index(drop=True,inplace=True)

    y_train.reset_index(drop=True,inplace=True)

    y_test.reset_index(drop=True,inplace=True)

    return X_train,X_test,y_train,y_test


# %% [markdown]
# #### Split-2

# %%
def train_val_test(X,y,train_size=-1,val_size=-1,test_size=-1) :

    if(train_size==-1) : train_size = 1-val_size-test_size

    if(val_size==-1) : val_size = 1-train_size-test_size

    X_train,X_val,y_train,y_val = train_test_split(X,y,train_size=train_size,random_state=0)

    X_val,X_test,y_val,y_test = train_test_split(X_val,y_val,train_size=(val_size/(1-train_size)),random_state=0)

    X_train.reset_index(drop=True,inplace=True)

    X_val.reset_index(drop=True,inplace=True)

    X_test.reset_index(drop=True,inplace=True)

    y_train.reset_index(drop=True,inplace=True)

    y_val.reset_index(drop=True,inplace=True)

    y_test.reset_index(drop=True,inplace=True)

    return X_train,X_val,X_test,y_train,y_val,y_test


# %% [markdown]
# #### Single Function for Splitting Data
# If we pass two sizes in the function (eg. train_size & val_size) then it will be a three-way split, if we pass one size (eg. train_size) it will be a two-way split.

# %%
# Single Function
def split(X,y,train_size=-1,val_size=-1,test_size=-1) :
    if(train_size==-1 and val_size==-1) : return train_test(X,y,train_size=1-test_size)
    if(train_size==-1 and test_size==-1) : return train_test(X,y,train_size=1-val_size)
    if(val_size==-1 and test_size==-1) : return train_test(X,y,train_size=train_size)
    return train_val_test(X,y,train_size,val_size,test_size)


# %%
# Example for Split-1
# Train Validation Test Split:
X_train,X_val,X_test,y_train,y_val,y_test = split(X,y,train_size=0.7,val_size=0.15)

# %%
print(X_train.shape,X_val.shape,X_test.shape)
print(y_train.shape,y_val.shape,y_test.shape)

# %%
# Example for Split-2
# Train-Test Split:
X_train,X_test,y_train,y_test = split(X,y,train_size=0.80)

# %%
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# %% [markdown]
# ### Conclusion
# **We implemented preprocessing functions for Cleaning, Encoding, Normalizing, and Splitting the Data. We saw how organized preprocessing makes our job easier.**
#
# After Importing the data, we can pre-process it as per our needs in 4 lines. We can keep modifying the parameters to experiment with different preprocessing approaches.
# ```
# clean(df,exceptions_mismatch=[])
# encode(df,labels)
# normalize(df,cols=None,kinds='dbl',round=None,c=1,exceptions=[])
# split(X,y,train_size=-1,val_size=-1,test_size=-1)
# ```
# > Refer the above cells for customization of various parameters of these functions.

# %%
# Using the dataset from Part-A
# CSV File 1
dfa1=pd.read_csv('TelcomCustomer-Churn_1.csv')
# CSV File 2
dfa2=pd.read_csv('TelcomCustomer-Churn_2.csv')
# customerID is common in both the dataframes
dfa=dfa1.merge(dfa2, left_on='customerID', right_on='customerID')
# We can safely drop the customerID
dfa.drop(['customerID'], axis=1, inplace=True)

# Stacking all the functions for Cleaning, Encoding, Normalizing, and Splitting the Dataframe
# Single Function for Cleaning Data
clean(dfa,exceptions_mismatch=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges', 'Churn'])

# Single Function For Encoding Data
labels = {}
labels['gender'] = []
labels['SeniorCitizen'] = []
labels['Partner'] = []
labels['Dependents'] = []
# labels['tenure'] = []
labels['PhoneService'] = [] 
labels['MultipleLines'] = [] 
labels['InternetService'] = []
labels['OnlineSecurity'] = [] 
labels['OnlineBackup'] = []
labels['DeviceProtection'] = [] 
labels['TechSupport'] = []
labels['StreamingTV'] = [] 
labels['StreamingMovies'] = [] 
labels['PaperlessBilling'] = []
# labels['MonthlyCharges'] = [] 
# labels['TotalCharges'] = [] 
labels['Churn'] = []
labels['Contract'] = None
labels['PaymentMethod'] = None
encode(dfa,labels)

# Convert the categorical Data type to Numerical for modeling
col=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'TotalCharges', 'Churn']
def convert_to_cont(feature):
    dfa[feature]=pd.to_numeric(dfa[feature], errors='coerce')   
for c in col:
    convert_to_cont(c)
dfa.fillna(dfa.mean(),inplace = True)

# Single Function for Normalizing Data
normalize(dfa, kinds='mean', exceptions=['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'PaperlessBilling','Churn', 'Contract_Month-to-month', 
       'Contract_One year','Contract_Two year', 'PaymentMethod_Electronic check',
       'PaymentMethod_Mailed check', 'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)'])

# Create the Features Matrix and Target Vector
X = dfa.drop(['Churn'], axis=1)
y = dfa.loc[:,'Churn']

# Single Function for Splitting Data
X_train,X_test,y_train,y_test = split(X,y,train_size=0.80)

# %%
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# %%
X_train.head()

# %%
y_train.head()

# %% [markdown]
# ### Key Takeaways from the Automation of Data Pre-processing task:
# **Organized Preprocessing saves time and helps us preprocess different datasets and try different preprocessing without much code change:**
#
# - Data Cleaning: We need to remove (or replace) rows with na/nan values, remove rows with the wrong datatype for any feature, and convert numeric data stored as string format in the CSV/excel file back to its original form.
# - Data Encoding: We need to encode data as most ML models require numeric data. We implemented label encoding and one hot encoding.
# - Data Normalization: It helps in reducing bias towards a feature, and sometimes reduces computation time. We implemented 5 normalization techniques.
# - Data Splitting: We need to split our data into the train portion (for fitting the model) and the testing portion (for evaluating the model). Sometimes, we also split into a third portion – validation, which we use to find optimal parameters for our model.
#
# Other than preprocessing too, it’s good to keep our code organized, it helps in making changes later. We should also try to make universal functions taking the dataset as an argument rather than making hardcoded functions that will work only for the dataset we are using at that time.

# %% [markdown]
# ## 3. Model Building:

# %% [markdown]
# ### Simple Machine Learning Work flow
# Refer section "Conclusion" above to create Preprocessed X_train, X_test, y_train, y_test from a CSV File.

# %%
# Build the model
model = xgb.XGBClassifier(n_jobs=2, random_state=0, verbosity=0)

# Train the model
model.fit(X_train, y_train)
model_pred = model.predict(X_test)

# Classification Accuracy
print('')
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))
print('')

# Classification Report
print("Classification Report:\n",classification_report(y_test, model_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, model_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Churn-No","Churn-Yes"]],
                  columns = [i for i in ["Churn-No","Churn-Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

print('Default Parameters used in the base model:')
print(model)

# %% [markdown]
# ### Functional Approach: ML work flow template

# %%
# ML Workflow to select a best performing model
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot

# Get the dataset
def get_dataset():
	Xm, ym = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=0)
	return Xm, ym

# Get a list of models to evaluate
def get_models():
	models = dict()
	# define number of trees to consider
	n_trees = [10, 50, 100, 500, 1000, 5000]
	for n in n_trees:
		models[str(n)] = AdaBoostClassifier(random_state=0, n_estimators=n)
	return models

# Evaluate a given model using cross-validation
def evaluate_model(model, Xm, ym):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
	# evaluate the model and collect the results
	scores = cross_val_score(model, Xm, ym, scoring='accuracy', cv=cv, n_jobs=2)
	return scores

# Define dataset
Xm, ym = get_dataset()
# Get the models to evaluate
models = get_models()
# Evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	# Evaluate the model
	scores = evaluate_model(model, Xm, ym)
	# Store the results
	results.append(scores)
	names.append(name)
	# Summarize the performance along the way
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# Plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

# %% [markdown]
# ### Select  the best performing model

# %%
# Use K-Fold Cross Validation for model selection
# Define various classification models
BaggingClassifier=BaggingClassifier(n_jobs=2, random_state=0)
RandomForest=RandomForestClassifier(n_jobs=2, random_state=0)
AdaBoostClassifier=AdaBoostClassifier(random_state=0)
GBClassifier=GradientBoostingClassifier(random_state=0)
XGBClassifier=xgb.XGBClassifier(n_jobs=2, random_state=0, verbosity = 0)
LGBMClassifier=LGBMClassifier(n_jobs=2, random_state=0)
CatBoostClassifier=CatBoostClassifier(thread_count=2, random_seed=0)

Hybrid = []
Hybrid.append(['RidgeClassifier',RidgeClassifier(random_state=0)])
Hybrid.append(['Logistic Regression',LogisticRegression(n_jobs=2, random_state=0)])
Hybrid.append(['SVM',SVC(random_state=0)])
Hybrid.append(['KNeigbors',KNeighborsClassifier(n_jobs=2)])
Hybrid.append(['GaussianNB',GaussianNB()])
Hybrid.append(['BernoulliNB',BernoulliNB()])
Hybrid.append(['DecisionTree',DecisionTreeClassifier(random_state=0)])

Hybrid_ensemble=VotingClassifier(Hybrid, n_jobs=2)

# %%
# K Fold Cross Validation Scores

seed = 0

# Create models
models = []
models.append(('XGBClassifier', XGBClassifier))
models.append(('RandomForest', RandomForest))
models.append(('AdaBoostClassifier', AdaBoostClassifier))
models.append(('GBClassifier', GBClassifier))
models.append(('LGBMClassifier', LGBMClassifier))
models.append(('CatBoostClassifier', CatBoostClassifier))
models.append(('BaggingClassifier', BaggingClassifier))
models.append(('Hybrid_ensemble', Hybrid_ensemble))

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
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring, n_jobs=2)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
# Boxplot for algorithm comparison
fig = plt.figure(figsize=(12,5))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# %%
# Use SMOTE to handle the imbalanced dataset
# Create the oversampler.
smote=SMOTE(random_state=0)
X1, y1=smote.fit_resample(X_train, y_train)

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
# **Best Model: AdaBoostClassifier based on ROC_AUC, Precision, Recall and F1 Score**
#
# 1. Prediction of customer churn (Recall) is most important parameter to decide the best model for this problem. AdaBoostClassifier has the highest Recall value here.
# 2. Both Hybrid_ensemble and GBClassifier are next to the AdaBoostClassifier in performance.
# 3. XgBoost, CatBoost, and LightGBM are performing on similar scales. A balanced dataset may drastically change their performances.
# 4. Hyper-parameter tuning of the base model may improve the performance.

# %% [markdown]
# ### Using hyperparameter tuning without oversampling

# %%
from sklearn.ensemble import AdaBoostClassifier

# Build the model
model = AdaBoostClassifier(random_state=0)

# Other Parameters to be used:
# 'base_estimator': [RidgeClassifier(random_state=0),LogisticRegression(n_jobs=2, random_state=0),
#                    SVC(random_state=0), GaussianNB(), BernoulliNB(),DecisionTreeClassifier(random_state=0)]
params = {'n_estimators':list(range(2, 102, 2)), 
          'learning_rate':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
          'algorithm': ['SAMME', 'SAMME.R']}

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
model = AdaBoostClassifier(random_state=0, n_estimators=70, learning_rate=0.4, algorithm='SAMME.R')

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
df_cm = pd.DataFrame(cm, index = [i for i in ["Churn-No","Churn-Yes"]],
                  columns = [i for i in ["Churn-No","Churn-Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# **We do not see much improvement in the model here.**

# %% [markdown]
# ### Using hyperparameter tuning with Oversampling

# %%
# Build the model
model = AdaBoostClassifier(random_state=0)

params = {'n_estimators':list(range(2, 102, 2)), 
          'learning_rate':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
          'algorithm': ['SAMME', 'SAMME.R']}

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
model = AdaBoostClassifier(random_state=0, n_estimators=72, learning_rate=0.7, algorithm='SAMME.R')

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
df_cm = pd.DataFrame(cm, index = [i for i in ["Churn-No","Churn-Yes"]],
                  columns = [i for i in ["Churn-No","Churn-Yes"]])
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
# **Performance here is almost same as that of the base model in section 1.3.3.**

# %% [markdown]
# ## 4. Pickle file for the best performing model

# %%
# Final model from the above modeling process: Base Model without Hypertuning

# Build the model
model = AdaBoostClassifier(random_state=0)

# Train the model
model.fit(X1, y1)
model_pred = model.predict(X_test)

# Classification Accuracy
print('')
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))
print('')

# Classification Report
print("Classification Report:\n",classification_report(y_test, model_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, model_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Churn-No","Churn-Yes"]],
                  columns = [i for i in ["Churn-No","Churn-Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %%
# Import pickle Package
import pickle

# %%
# Save the ML model to a file in the current working directory
Pkl_Filename = "Pickle_AdaBoostClassifier_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)

# %%
# Load the ML Model back from the file
with open(Pkl_Filename, 'rb') as file:  
    Pickle_AdaBoostClassifier_Model = pickle.load(file)
    
Pickle_AdaBoostClassifier_Model

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
