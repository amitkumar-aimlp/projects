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
# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#00b3e5;">Contents: Supervised Learning Project
# 1. [Part-A: Solution](#Part-A:-Solution)
# 2. [Part-B: Solution](#Part-B:-Solution)

# %% [markdown]
# # Part-A: Solution

# %% [markdown]
# - **DOMAIN:** Medical
# - **CONTEXT:** Medical research university X is undergoing a deep research on patients with certain conditions. University has an internal AI team. Due to confidentiality the patient’s details and the conditions are masked by the client by providing different datasets to the AI team for developing a AIML model which can predict the condition of the patient depending on the received test results.
# - **DATA DESCRIPTION:** The data consists of biomechanics features of the patients according to their current conditions. Each patient is represented in the data set by six biomechanics attributes derived from the shape and orientation of the condition to their body part.
#  1. P_incidence
#  2. P_tilt
#  3. L_angle
#  4. S_slope
#  5. P_radius
#  6. S_Degree
#  7. Class
# - **PROJECT OBJECTIVE:** To Demonstrate the ability to fetch, process and leverage data to generate useful predictions by training Supervised Learning algorithms.

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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Data Understanding:

# %% [markdown]
# ### 1A. Read all the 3 CSV files as DataFrame and store them into 3 separate variables.

# %%
# CSV File 1, 2 and 3
dfa1=pd.read_csv('Part1+-+Normal.csv')

dfa2=pd.read_csv('Part1+-+Type_H.csv')

dfa3=pd.read_csv('Part1+-+Type_S.csv')

# %% [markdown]
# ### 1B. Print Shape and columns of all the 3 DataFrames.

# %%
dfa1.shape

# %%
dfa1.info()
dfa1.head()

# %%
dfa2.shape

# %%
dfa2.info()
dfa2.head()

# %%
dfa3.shape

# %%
dfa3.info()
dfa3.head()

# %% [markdown]
# ### 1C. Compare Column names of all the 3 DataFrames and clearly write observations

# %% [markdown]
# **Observations:**
#
# - All 3 datasets have common column names and datatypes.
# - Corresponding dataset rows are 100, 60, 150; all datasets have 7 columns.
# - Datatype of first 6 columns is float64, Datatype of last column is object.
# - There are no junk values in the dataset.
# - Considering the shape and dimensions of the datasets it would be easy to append or concatenate them to form a single dataframe.
# - First 6 features will form the features matrix, and the last column is the target vector.
# - Class is object we need to change the datatype of this column for better performance.
# -  Explore for null/missing values in the attributes and if required drop or impute values.

# %% [markdown]
# ### 1D. Print DataTypes of all the 3 DataFrames.

# %%
dfa1.dtypes

# %%
dfa2.dtypes

# %%
dfa3.dtypes

# %% [markdown]
# ### 1E. Observe and share variation in ‘Class’ feature of all the 3 DataFrames

# %%
dfa1['Class'].value_counts()

# %%
dfa2['Class'].value_counts()

# %%
dfa3['Class'].value_counts()

# %% [markdown]
# **Observations:**
# - Here we have three different classes in our dataset.
# - Here tp_s and Type_S; Normal and Nrmal; Type_H and type_h; represents same class.
# - There are some rows in which the target variable "Class" is not properly specified. We can use selection/subsetting tricks or Use .replace function to correct the data in 'Class' column.
# - All the variations in Class can be unified to create a correct target vector.

# %% [markdown]
# ## 2. Data Preparation and Exploration:

# %% [markdown]
# ### 2A. Unify all the variations in ‘Class’ feature for all the 3 DataFrames.
# For Example: ‘tp_s’, ‘Type_S’, ‘type_s’ should be converted to ‘type_s’

# %%
dfa1.loc[dfa1['Class']=='Nrmal','Class']='Normal'
dfa2.loc[dfa2['Class']=='type_h','Class']='Type_H'
dfa3.loc[dfa3['Class']=='tp_s','Class']='Type_S'

# %% [markdown]
# ### 2B. Combine all the 3 DataFrames to form a single DataFrame
# Checkpoint: Expected Output shape = (310,7)

# %%
# Concatenate pandas objects along a particular axis with optional set logic along the other axes.
dfa=pd.concat([dfa1,dfa2,dfa3],axis=0,ignore_index=True,sort=False) 
dfa.shape

# %%
# Understand and verify the Class variable
dfa['Class'].value_counts()

# %%
dfa['Class'].nunique()

# %% [markdown]
# ### 2C. Print 5 random samples of this DataFrame

# %%
# Return a random sample of items from an axis of object.
# You can use `random_state` for reproducibility.

dfa.sample(n=5)

# df.sample(frac = 0.5)
# df.sample(n = 5, replace = False)
# df.sample(n = 5, replace = True)
# df.sample(axis = 0)
# df.sample(axis = 1)
# df.sample(n = 5, random_state = 2)

# %%
dfa.sample(n=5)


# %% [markdown]
# ### 2D. Print Feature-wise percentage of Null values.

# %%
# There are no missing values in the dataset

# df.isnull().sum()
# df.isna().sum()

def missing_check(df):
    total = df.isnull().sum().sort_values(ascending=False)   # total number of null values
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)  # percentage of values that are null
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # putting the above two together
    return missing_data # return the dataframe
missing_check(dfa)

# %% [markdown]
# ### 2E. Check 5-point summary of the new DataFrame.

# %%
# Describe function generates descriptive statistics that summarize the central tendency, 
# dispersion and shape of a dataset’s distribution, excluding NaN values.

# This method tells us a lot of things about a dataset. One important thing is that 
# the describe() method deals only with numeric values. It doesn't work with any 
# categorical values. So if there are any categorical values in a column the describe() 
# method will ignore it and display summary for the other columns.

dfa.describe().transpose()

# %%
#Change the Data Type of 'Class' from object to category

dfa['Class']=dfa['Class'].astype('category') 

# %% [markdown]
# **Observations:**
# - P_incidence:
#  - Mean and Median are nearly equal. Distribution might be normal. 
#  - 75 % of values are less than 72, and maximum value is 129.
#
# - P_tilt:
#  - Mean and median are nearly equal. Distribution might be normal.
#  - It contains negative values; 75 % of values are less than 22, and maximum value is 49. Some right skewness is expected.
#
# - L_angle:
#  - Mean and Median are nearly equal. There is no deviation. Distribution might be normal.
#  - There might be few outliers because of the maximum value.
#
# - S_slope:
#  - Mean and Median are nearly equal.
#  - 75% of values are less than 52, and maximum value is 121.
#
# - P_radius:
#  - Distribution might be normal.
#  - Not much deviation in the data.
#
# - S_Degree:
#  - Mean is greater than Median so there might be right skewness in the data .
#  - We can see 75% of values are less than 41, and maximum value is 418;Outliers are expected.

# %% [markdown]
# ## 3. Data Analysis

# %% [markdown]
# ### 3A. Visualize a heatmap to understand correlation between all features

# %%
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
# ### 3B. Share insights on correlation.
# - Features having stronger correlation with correlation value.
# - Features having weaker correlation with correlation value.

# %% [markdown]
# **Observations:**
# - Corresponding Correlation is high between P_incidence and S_slope, L_angle.
# - S_degree and p_radius has negative correlation. Negative correlation is present.
# - Observe the independent variables variance and drop such variables having no variance or almost zero variance(variance < 0.1). They will be having almost no influence on the classification.

# %% [markdown]
# ### 3C. Visualize a pairplot with 3 classes distinguished by colors and share insights.

# %%
sns.pairplot(dfa,hue='Class')

# %%
sns.pairplot(dfa)

# sns.pairplot(dfa, hue="Class", diag_kind="hist")

# %% [markdown]
# **Insights:**
#
# - Along the diagonal, we can see the distribution of individual variables with histogram. 
# - Along the diagonal, we can see distribution of variables for three classes are not same.
# - It is evident that Type_S class is more compared to other two classes.
# - P_incidence has positive relationship with all variables except P_radius. Relationship is higher for S_slope and L_angle.
# - P_tilt has Higher Relationship with P_incidence and L_angle.There is no Relationship with S_slope and P_radius.
# - L_angle has positive Relationship with P_tilt, S_slope and S_Degree. It has no Relationship with P_radius.
# - S_slope has positive Relationship with L_angle and S_Degree.
# - P_radius has no Relationship with S_degree, P_tilt, L_angle.
# - S_degree has no strong positive Relationship with any of the variables.

# %% [markdown]
# ### 3D. Visualize a jointplot for ‘P_incidence’ and ‘S_slope’ and share insights.

# %%
sns.jointplot(data = dfa, x="P_incidence", y="S_slope", kind = "reg")

# %% [markdown]
# **Observations:**
# - Positive correlation exists between P_incidence and S_slope.
# - Data distribution is almost like normal except some skewness.
# - Presence of outliers affect the value of regression coefficients.

# %% [markdown]
# ### 3E. Visualize a boxplot to check distribution of the features and share insights.
# - Understand the data distribution for independent variables
# - Understand the outliers
# - Check for imbalanced dataset
# - Understand the data distribution with respect to the target vector

# %%
# Box Plot

plt.figure(figsize=(20,8))
ax = sns.boxplot(data=dfa, orient="h", palette="Set2")

# %% [markdown]
# **Insights (See below charts for more details):**
# - P_incidence
#  - Normal distribution is maintained with very less extreme values.
#  - Total number of outliers in column = 3.
# - P_tilt
#  - Data is Normally distributed and we can see one peakness in the center. Slight right skewness is present.
#  - We can see one outlier in negative end and few outliers in positive end.
# - L_angle
#  - Data distribution is normal.
#  - Little right skewness because of one outlier.
# - S_slope
#  - Data distribution is normal.
#  - Little right skewness because of one outlier.
# - P_radius
#  - Data is normally distributed
#  - We can see outliers at both the ends.
# - S_Degree
#  - There is Positive Skewness in the data.
#  - Distribution is hugely affected by Outliers.

# %%
# P_incidence

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['P_incidence'],  ax=axes[0],color='Green')
sns.boxplot(x = 'P_incidence', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['P_incidence'],25),np.percentile(dfa['P_incidence'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['P_incidence'] if i < lower or i > upper]
print('{} Total Number of outliers in P_incidence: {}'.format('\033[1m',len(Outliers)))

# %%
# P_tilt

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['P_tilt'],  ax=axes[0],color='Green')
sns.boxplot(x = 'P_tilt', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['P_tilt'],25),np.percentile(dfa['P_tilt'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['P_tilt'] if i < lower or i > upper]
print('{} Total Number of outliers in P_tilt: {}'.format('\033[1m',len(Outliers)))

# %%
# L_angle

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['L_angle'],  ax=axes[0],color='Green')
sns.boxplot(x = 'L_angle', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['L_angle'],25),np.percentile(dfa['L_angle'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['L_angle'] if i < lower or i > upper]
print('{} Total Number of outliers in L_angle: {}'.format('\033[1m',len(Outliers)))

# %%
# S_slope

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['S_slope'],  ax=axes[0],color='Green')
sns.boxplot(x = 'S_slope', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['S_slope'],25),np.percentile(dfa['S_slope'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['S_slope'] if i < lower or i > upper]
print('{} Total Number of outliers in S_slope: {}'.format('\033[1m',len(Outliers)))

# %%
# P_radius

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['P_radius'],  ax=axes[0],color='Green')
sns.boxplot(x = 'P_radius', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['P_radius'],25),np.percentile(dfa['P_radius'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['P_radius'] if i < lower or i > upper]
print('{} Total Number of outliers in P_radius: {}'.format('\033[1m',len(Outliers)))

# %%
# S_Degree

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['S_Degree'],  ax=axes[0],color='Green')
sns.boxplot(x = 'S_Degree', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['S_Degree'],25),np.percentile(dfa['S_Degree'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['S_Degree'] if i < lower or i > upper]
print('{} Total Number of outliers in S_Degree: {}'.format('\033[1m',len(Outliers)))

# %%
# Check for imbalanced dataset

f,axes=plt.subplots(1,2,figsize=(17,7))
dfa['Class'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('Class',data=dfa,ax=axes[1])
axes[0].set_title('Response Variable Pie Chart')
axes[1].set_title('Response Variable Bar Graph')
plt.show()

# %%
# Normal:0; Type_H:1; Type_S:2

dfa.groupby(["Class"]).count()

# %% [markdown]
# **Insights:**
# - Type_S Class has 48.4% of total values followed by Normal and Type_H Classes.
# - The ratio of distribution of three classes is 32:19:48, which shows imbalance of class feature i.e. Target Variable.
# - The above graph shows that the data is biased towards datapoints having class value as 2 -'Type _S'. 
# - The number of data points of Class 1 - 'Type_H' is almost half the number of 'Type_S patients.

# %%
# Understand the data distribution with respect to the target vector

plt.figure(figsize=(20,15))

plt.subplot(3, 2, 1)
plt.title('The relatio of P_incidence with "Class"')
sns.boxplot(x='Class', y='P_incidence', data= dfa)

plt.subplot(3, 2, 2)
plt.title('The relatio of P_tilt with "Class"')
sns.boxplot(x='Class', y='P_tilt', data= dfa)

plt.subplot(3, 2, 3)
plt.title('The relatio of L_angle with "Class"')
sns.boxplot(x='Class', y='L_angle', data= dfa)

plt.subplot(3, 2, 4)
plt.title('The relatio of S_slope with "Class"')
sns.boxplot(x='Class', y='S_slope', data= dfa)

plt.subplot(3, 2, 5)
plt.title('The relatio of P_radius with "Class"')
sns.boxplot(x='Class', y='P_radius', data= dfa)

plt.subplot(3, 2, 6)
plt.title('The relatio of S_Degree with "Class"')
sns.boxplot(x='Class', y='S_Degree', data= dfa)

plt.show()

# %% [markdown]
# **Observations:**
# - P_Incidence
#  - Normal Value is slightly higher than Type_H, and for Type_S Class Value is larger.
#
# - P_tilt
#  - Type_H is slightly higher than Normal Value, and for Type_S Class Value is larger.
#
# - L_angle 
#  - It has higher value for Type_S Class; and Normal class has higher values compared to type_H class.
#
# - S_slope 
#  - We can see huge values for Type_S class.
#
# - P_radius 
#  - Normal Class has more values. There are some extreme values for Type_s class.
#
# - S_Degree 
#  - We have large values for Type_S Class.

# %% [markdown]
# ### 3F. Test for significance of features
# (Additional Details)

# %%
# Use one-way anova to complete the statistical testing
col=['P_incidence','P_tilt','L_angle','S_slope','P_radius','S_Degree']
for i in col:
    print('{} Ho: Class types does not affect the {}'.format('\033[1m',i))
    print('{} H1: Class types affect the {}'.format('\033[1m',i))
    df_normal=dfa[dfa.Class=='Normal'][i]
    df_typeH=dfa[dfa.Class=='Type_H'][i]
    df_typeS=dfa[dfa.Class=='Type_S'][i]
    f_stats,p_value=stats.f_oneway(df_normal,df_typeH,df_typeS)
    print('{} F_stats: {}'.format('\033[1m',f_stats))
    print('{} p_value: {}'.format('\033[1m',p_value))
    if p_value < 0.05:  # Using significance level at 5%
        print('{} Reject Null Hypothesis. Class types has efect on {}'.format('\033[1m',i))
    else:
        print('{} Fail to Reject Null Hypothesis. Class types has no effect on {}'.format('\033[1m',i))
    print('\n')

# %% [markdown]
# We can see that Class type affects every independent variable.

# %% [markdown]
# ## 4. Model Building:

# %%
# Impute outliers with mean

col=['P_incidence','P_tilt','L_angle','S_slope','P_radius','S_Degree']
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
# Encode the Target Variable
# Normal: 0, Type_H: 1, Type_S: 2

le=LabelEncoder()
dfa['Class']=le.fit_transform(dfa['Class'])
dfa['Class'].value_counts()

# %%
# change datatype to category.

dfa['Class']=dfa['Class'].astype('category')

# %%
dfa.info()

# %% [markdown]
# ### 4A. Split data into X and Y.

# %%
# Split data into independent variables and dependent variables
# Create a base model without using scaling
X=dfa.drop(columns='Class')
y=dfa['Class'] #target

# %%
X.describe()

# %% [markdown]
# ### 4B. Split data into train and test with 80:20 proportion.

# %%
# Split X and y into training and test set in 80:20 ratio

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# %% [markdown]
# ### 4C. Train a Supervised Learning Classification base model using KNN classifier.

# %%
# Build knn with 5 neighbors
knn = KNeighborsClassifier(n_neighbors= 5, metric = 'euclidean')

# Train the model
knn.fit(X_train, y_train)
predicted_labels = knn.predict(X_test)

# %% [markdown]
# ### 4D. Print all the possible performance metrics for both train and test data.

# %%
# Classification Accuracy

print('Accuracy on Training data:',knn.score(X_train, y_train))
print('Accuracy on Test data:',knn.score(X_test, y_test))

# %% [markdown]
# **Observations:**
# 1. Training Accuracy is 87.50% and Testing Accuracy is 80.64%. Performance is less in test data.
# 2. This is due to some over-fitting of the data.

# %%
# Confusion Matrix

cm = confusion_matrix(y_test, predicted_labels, labels=[0,1,2])

df_cm = pd.DataFrame(cm, index = [i for i in ["Normal","Type_H","Type_S"]],
                  columns = [i for i in ["Normal","Type_H","Type_S"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')
plt.show()

# %% [markdown]
# **Observations:**
# - Our model predicts Type_S correctly most of the time followed by Type_H and Normal
# - Misclassification of labels are more when predicting Normal class
# - Since training dataset is slightly imbalanced, we can observe the misclassification error on test dataset.

# %%
# Total no of datapoints in Class variable
# Normal=0; 100
# Type_H=1; 60
# Type_S=2; 150
print("classification  Matrix:\n",classification_report(y_test,predicted_labels))

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
# **Observations:**
# - Class 0 predicted correctly for 77% of time. On similar lines for class 1 55% and class 2 93%.
# - Using F1 Score: Precision and Recall is balanced for class 0 by 76% and for class 1 by 55%.
# - Precision, Recall, and F1 Score are highest for class 2 followed by class 0 and class 1.
# - We have maximum F1 score for class 2, and minimum for class 1.

# %%
knn

# %% [markdown]
# ## 5. Performance Improvement:

# %% [markdown]
# **Strategy for Performance Improvement:**
# 1. Using different Scaling methods over dataset.
# 2. Using automated search for hyper-parameters.
# 3. Using manual search for hyper-parameters.
# 4. Model Selection approach with Scaling.
# 5. Model Selection approach without Scaling.

# %% [markdown]
# ### 5A. Experiment with various parameters to improve performance of the base model.
# (Optional: Experiment with various Hyperparameters - Research required)

# %% [markdown]
# #### 1. Using different Scaling methods over dataset.

# %%
# Create copy of dataset.
dfa_model = dfa.copy()

# Rescaling features matrix using various scaling methods:
# scaler = RobustScaler()
# scaler = MinMaxScaler()
scaler = StandardScaler()

features = [['P_incidence','P_tilt','L_angle','S_slope','P_radius','S_Degree']]
for feature in features:
    dfa_model[feature] = scaler.fit_transform(dfa_model[feature])
    
#Create KNN Object
knn = KNeighborsClassifier()

#Create x and y variable
X = dfa_model.drop(columns=['Class'])
y = dfa_model['Class']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#Training the model
knn.fit(X_train, y_train)

#Predict testing set
y_pred = knn.predict(X_test)

#Check performance using accuracy
print('Accuracy on Training data:',knn.score(X_train, y_train) )
print('Accuracy on Test data:',knn.score(X_test, y_test))

# Detailed Classification Report
print("classification  Matrix:\n",classification_report(y_test,y_pred))

# %%
knn

# %% [markdown]
# **Case-1 Observations:**
# - Accuracy improvement by 3% over base model.
# - Some Good improvements in Class precision.

# %% [markdown]
# #### 2. Using automated search for hyper-parameters.

# %%
# Build and train the model
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

# %%
# Exhaustive search over specified parameter values for an estimator.
# Important members are fit, predict.

# GridSearchCV implements a "fit" and a "score" method.
# It also implements "score_samples", "predict", "predict_proba",
# "decision_function", "transform" and "inverse_transform" if they are
# implemented in the estimator used.

# The parameters of the estimator used to apply these methods are optimized
# by cross-validated grid-search over a parameter grid.

grid_params = { 'n_neighbors' : [1,3,5,7,9,11,13,15,17,19],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}

gs = GridSearchCV(knn, grid_params, cv=10)
gs.fit(X_train, y_train)

# %%
gs.cv_results_['params']

# %%
gs.best_params_

# %%
gs.cv_results_['mean_test_score']

# %%
# Lets Build knn with best params

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', weights='distance')

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# %%
#Check performance using accuracy
print('Accuracy on Training data:',knn.score(X_train, y_train))
print('Accuracy on Test data:',knn.score(X_test, y_test))

# Detailed Classification Report
print("classification  Matrix:\n",classification_report(y_test,y_pred))

# %%
knn

# %% [markdown]
# **Case-2 Observations:**
# - 100% accuracy in training: Shows presence of huge overfittng in model.
# - Manual search for hyperparametes would be a good choice.
# - Accuracy improvement by 3% over base model.
# - Good improvements in Class precision.

# %% [markdown]
# #### 3. Using manual search for hyper-parameters.

# %%
# Optimize the value of k

train_score=[]
test_score=[]
for k in range(1,51):
    knn = KNeighborsClassifier(n_neighbors= k , metric = 'euclidean' ) 
    knn.fit(X_train, y_train)
    train_score.append(knn.score(X_train, y_train))
    test_score.append(knn.score(X_test, y_test))

# %%
# train_score vs. k

plt.plot(range(1,51),train_score)
plt.show()

# Here training accuracy decreases as k increases

# %%
# test_score vs. k

plt.plot(range(1,51),test_score)
plt.show()

# The accuracy is maximum when k is less than 50. So we can cap the value of k as less than 50.

# %%
# Check the performance of model for various k values; Consider the k upto 50
# Build knn with k neighbors

k=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49]
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i, metric = 'euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print('Accuracy on Training data for k {} is {}:'.format(i,knn.score(X_train, y_train)))
    print('Accuracy on Test data for k {} is {}:'.format(i,knn.score(X_test, y_test)))
    print("classification  Matrix:\n",classification_report(y_test,y_pred))

# %%
# For k=9, we can see a model with good overall performance metrics
# Lets Build knn with k=9, metric='euclidean' and weights='uniform'

knn = KNeighborsClassifier(n_neighbors=9, metric = 'euclidean')

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# %%
#Check performance using accuracy
print('Accuracy on Training data:',knn.score(X_train, y_train))
print('Accuracy on Test data:',knn.score(X_test, y_test))

# Detailed Classification Report
print("classification  Matrix:\n",classification_report(y_test,y_pred))

# %% [markdown]
# **Case-3 Observations:**
# - For k = 9, we can see a good model at corresponding train and test accuracies of 85.0% and 85.4%.
# - Accuracy improvement by 4% over base model.
# - Very Good improvements in Class precision, recall and F1 Scores.

# %% [markdown]
# #### 4. Model Selection approach with Scaling.

# %%
# Use K-Fold Cross Validation for model selection
# Define various classification models
LR_model=LogisticRegression()
KNN_model=KNeighborsClassifier(n_neighbors=9)
GN_model=GaussianNB()
svc_model_linear = SVC(kernel='linear',C=1,gamma=.6)
svc_model_rbf = SVC(kernel='rbf',degree=2,C=.009)
svc_model_poly  = SVC(kernel='poly',degree=2,gamma=0.1,C=.01)

# %%
# With Standard Scaler

seed = 0

# Create models
models = []
models.append(('LR', LR_model))
models.append(('KNN', KNN_model))
models.append(('NB', GN_model))
models.append(('SVM-linear', svc_model_linear))
models.append(('SVM-poly', svc_model_poly))
models.append(('SVM-rbf', svc_model_rbf))

# Evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10,random_state=seed,shuffle=True)
	cv_results = model_selection.cross_val_score(model,X,y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
# Boxplot for algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# %% [markdown]
# **Case-4 Observations:**
# - Accuracy is high for LR, NB and SVM-linear models. However the standard deviation is less for LR model.
# - LR is a better algorithm for this dataset because of high accuracy and less Standard deviation.

# %% [markdown]
# #### 5. Model Selection approach without Scaling.

# %%
# Without Standard Scaler

# Split X and y into training and test set in 80:20 ratio
dfa = dfa.copy()
X=dfa.drop(columns='Class')
y=dfa['Class'] #target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

seed = 0

# Create models
models = []
models.append(('LR', LR_model))
models.append(('KNN', KNN_model))
models.append(('NB', GN_model))
models.append(('SVM-linear', svc_model_linear))
models.append(('SVM-poly', svc_model_poly))
models.append(('SVM-rbf', svc_model_rbf))

# Evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed,shuffle=True)
	cv_results = model_selection.cross_val_score(model,X,y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Boxplot for algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# %% [markdown]
# **Case-5 Observations:**
# - Accuracy is high for LR, KNN, NB, SVM-linear, and SVM-poly models. However the standard deviation is less for SVM-ploy model.
# - Here SVM-linear is a better algorithm for this dataset because of high accuracy and some Standard deviation.

# %% [markdown]
# ### 5B. Clearly showcase improvement in performance achieved.
# For Example:
# - A. Accuracy: +15% improvement
# - B. Precision: +10% improvement.

# %% [markdown]
# **Observations:**
# 1. We can increase the Accuracy by 4% compared to the base model.
# 2. We can increase the Precision for Class-0, 1 and 2 by 5%, 12% and 3%.
# 3. We can increase the Recall for Class-0, 1 by 4%, 18%.
# 4. We can increase the F1 Score for Class-0, 1 and 2 by 4%, 15% and 1%.
# 5. SVM-linear is also a better algorithm for this dataset because of high accuracy and low Standard deviation.
# ![comp-1.png](attachment:comp-1.png)

# %% [markdown]
# ### 5C. Clearly state which parameters contributed most to improve model performance.

# %% [markdown]
# **Important parameters and methods:**
# 1. Number of neighbors; k=9.
# 2. The distance metric to use for modeling; metric = 'euclidean'.
# 3. Scaling method used for dataset; scaler = StandardScaler().
# 4. K-Fold cross validation approach for model selection.
# ![comp-2.png](attachment:comp-2.png)

# %% [markdown]
# # Part-B: Solution

# %% [markdown]
# - **DOMAIN:** Banking, Marketing
# - **CONTEXT:** A bank X is on a massive digital transformation for all its departments. Bank has a growing customer base whee majority of them are liability customers (depositors) vs borrowers (asset customers). The bank is interested in expanding the borrowers base rapidly to bring in more business via loan interests. A campaign that the bank ran in last quarter showed an average single digit conversion rate. Digital transformation being the core strength of the business strategy, marketing department wants to devise effective campaigns with better target marketing to increase the conversion ratio to double digit with same budget as per last campaign.
# - **DATA DESCRIPTION:**
#     1. Team: Team’s name
#     2. ID: Customer ID
#     3. Age: Customer’s approximate age.
#     4. CustomerSince: Customer of the bank since. [unit is masked]
#     5. HighestSpend: Customer’s highest spend so far in one transaction. [unit is masked]
#     6. ZipCode: Customer’s zip code.
#     7. HiddenScore: A score associated to the customer which is masked by the bank as an IP.
#     8. MonthlyAverageSpend: Customer’s monthly average spend so far. [unit is masked]
#     9. Level: A level associated to the customer which is masked by the bank as an IP.
#     10. Mortgage: Customer’s mortgage. [unit is masked]
#     11. Security: Customer’s security asset with the bank. [unit is masked]
#     12. FixedDepositAccount: Customer’s fixed deposit account with the bank. [unit is masked]
#     13. InternetBanking: if the customer uses internet banking.
#     14. CreditCard: if the customer uses bank’s credit card.
#     15. LoanOnCard: if the customer has a loan on credit card.
# - **PROJECT OBJECTIVE:** Build a Machine Learning model to perform focused marketing by predicting the potential customers who will convert using the historical dataset.

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Data Understanding and Preparation:

# %% [markdown]
# ### 1A. Read both the Datasets ‘Data1’ and ‘Data2’ as DataFrame and store them into two separate variables.

# %%
# Read the CSV File 1 and 2
dfb1=pd.read_csv('Part2+-+Data1.csv')

dfb2=pd.read_csv('Part2+-Data2.csv')

# %% [markdown]
# ### 1B. Print shape and Column Names and DataTypes of both the Dataframes.

# %%
dfb1.shape

# %%
dfb1.info()
dfb1.head()

# %%
dfb2.shape

# %%
dfb2.info()
dfb2.head()

# %% [markdown]
# ### 1C. Merge both the Dataframes on ‘ID’ feature to form a single DataFrame

# %%
# ID is common in both the dataframes
dfb=dfb1.merge(dfb2, left_on='ID', right_on='ID')

# %%
dfb.shape

# %%
dfb.head()

# %% [markdown]
# ### 1D. Change Datatype of below features to ‘Object’
# - ‘CreditCard’, ‘InternetBanking’, ‘FixedDepositAccount’, ‘Security’, ‘Level’, ‘HiddenScore’.
# - [Reason behind performing this operation:- Values in these features are binary i.e. 1/0. But DataType is ‘int’/’float’ which is not expected.]

# %%
dfb.dtypes

# %%
# Change the Datatype of categorical features
col=['HiddenScore','Level','Security','FixedDepositAccount','InternetBanking','CreditCard','LoanOnCard']
for c in col:
    dfb[c]=dfb[c].astype('category')

# %%
dfb.dtypes

# %% [markdown]
# **Observations:**
# - Final Dataframe has 14 columns and 5000 rows.
# - LoanOnCard is the target vector.
# - Explore for null/missing values in the attributes and if required drop or impute values.

# %% [markdown]
# ## 2. Data Exploration and Analysis:

# %% [markdown]
# ### 2A. Visualize distribution of Target variable ‘LoanOnCard’ and clearly share insights.

# %%
# Creat a side by side Pie and Bar chart
f,axes=plt.subplots(1,2,figsize=(17,7))
dfb['LoanOnCard'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0],shadow=True)
sns.countplot('LoanOnCard',data=dfb,ax=axes[1],order=[0,1])
axes[0].set_title('LoanOnCard Variable Pie Chart')
axes[1].set_title('LoanOnCard Variable Bar Graph')
plt.show()

# %% [markdown]
# **There is huge imbalance in target vector.**
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
# ### 2B. Check the percentage of missing values and impute if required.

# %%
dfb.info()


# %%
# Percentage of missing values

# dfb.isnull().sum()
# dfb.isna().sum()

def missing_check(df):
    total = df.isnull().sum().sort_values(ascending=False)   # total number of null values
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)  # percentage of values that are null
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # putting the above two together
    return missing_data # return the dataframe
missing_check(dfb)

# %%
# Target variable LoanOnCard have few missing records.
# As missing values are small, we can safely drop the missing records.
dfb.dropna(axis=0,inplace=True)

# %%
# All the missing records are dropped
dfb.isnull().sum()

# %% [markdown]
# ### 2C. Check for unexpected values in each categorical variable and impute with best suitable value.
# [Unexpected values means if all values in a feature are 0/1 then ‘?’, ‘a’, 1.5 are unexpected values which needs treatment ]

# %%
# Get a list of categories of categorical variable
# col=['HiddenScore','Level','Security','FixedDepositAccount','InternetBanking','CreditCard','LoanOnCard']
print(dfb.HiddenScore.value_counts())
print(dfb.Level.value_counts())
print(dfb.Security.value_counts())
print(dfb.FixedDepositAccount.value_counts())
print(dfb.InternetBanking.value_counts())
print(dfb.CreditCard.value_counts())
print(dfb.LoanOnCard.value_counts())

# %%
dfb[col].info()

# %% [markdown]
# **Observations:** The dataset is clean, and we do not have the unexpected values.

# %% [markdown]
# ### 2D. EDA: Exploratory Data Analysis 
# (Concise additional details for model building)

# %%
# ID and ZipCode columns are not useful in model building; So we can safely remove them
dfb.drop('ID',axis=1,inplace=True)
dfb.drop('ZipCode',axis=1,inplace=True)

# %%
# Dataframe after initial data cleaning
dfb.info()

# %%
# Data Summary for numerical features
dfb.describe()

# %%
# Data Summary for categorical features
# col=['HiddenScore','Level','Security','FixedDepositAccount','InternetBanking','CreditCard','LoanOnCard']

dfb[col].describe()

# %% [markdown]
# #### Distribution and outlier analysis of numerical features

# %%
# Age

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.boxplot(x = 'Age', data=dfb,  orient='h', ax=axes[1])
sns.distplot(dfb['Age'],  ax=axes[0])
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['Age'],25),np.percentile(dfb['Age'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['Age'] if i < lower or i > upper]
print('{} Total Number of outliers in Age: {}'.format('\033[1m',len(Outliers)))

# %%
# CustomerSince

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.boxplot(x = 'CustomerSince', data=dfb,  orient='h' , ax=axes[1])
sns.distplot(dfb['CustomerSince'],  ax=axes[0])
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['CustomerSince'],25),np.percentile(dfb['CustomerSince'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['CustomerSince'] if i < lower or i > upper]
print('{} Total Number of outliers in CustomerSince: {}'.format('\033[1m',len(Outliers)))

# %%
# HighestSpend

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.boxplot(x = 'HighestSpend', data=dfb,  orient='h', ax=axes[1])
sns.distplot(dfb['HighestSpend'],  ax=axes[0])
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['HighestSpend'],25),np.percentile(dfb['HighestSpend'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['HighestSpend'] if i < lower or i > upper]
print('{} Total Number of outliers in HighestSpend: {}'.format('\033[1m',len(Outliers)))

# %%
# MonthlyAverageSpend

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.boxplot(x = 'MonthlyAverageSpend', data=dfb,  orient='h' , ax=axes[1])
sns.distplot(dfb['MonthlyAverageSpend'],  ax=axes[0])
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['MonthlyAverageSpend'],25),np.percentile(dfb['MonthlyAverageSpend'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['MonthlyAverageSpend'] if i < lower or i > upper]
print('{} Total Number of outliers in MonthlyAverageSpend: {}'.format('\033[1m',len(Outliers)))

# %%
# Mortgage

# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.boxplot(x = 'Mortgage', data=dfb,  orient='h' , ax=axes[1])
sns.distplot(dfb['Mortgage'],  ax=axes[0])
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfb['Mortgage'],25),np.percentile(dfb['Mortgage'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfb['Mortgage'] if i < lower or i > upper]
print('{} Total Number of outliers in Mortgage: {}'.format('\033[1m',len(Outliers)))

# %% [markdown]
# #### Distribution of categorical features

# %%
# HiddenScore

f,axes=plt.subplots(1,2,figsize=(17,7))
dfb['HiddenScore'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0],shadow=True)
sns.countplot('HiddenScore',data=dfb,ax=axes[1],order=[1,2,4,3])
axes[0].set_title('HiddenScore Variable Pie Chart')
axes[1].set_title('HiddenScore Variable Bar Graph')
plt.show()

# %%
# Level

f,axes=plt.subplots(1,2,figsize=(17,7))
dfb['Level'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0],shadow=True)
sns.countplot('Level',data=dfb,ax=axes[1],order=[1,3,2])
axes[0].set_title('Level Variable Pie Chart')
axes[1].set_title('Level Variable Bar Graph')
plt.show()

# %%
# FixedDepositAccount

f,axes=plt.subplots(1,2,figsize=(17,7))
dfb['FixedDepositAccount'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0],shadow=True)
sns.countplot('FixedDepositAccount',data=dfb,ax=axes[1])
axes[0].set_title('FixedDepositAccount Variable Pie Chart')
axes[1].set_title('FixedDepositAccount Variable Bar Graph')
plt.show()

# %%
# InternetBanking

f,axes=plt.subplots(1,2,figsize=(17,7))
dfb['InternetBanking'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0],shadow=True)
sns.countplot('InternetBanking',data=dfb,ax=axes[1],order=[1,0])
axes[0].set_title('InternetBanking Variable Pie Chart')
axes[1].set_title('InternetBanking Variable Bar Graph')
plt.show()

# %%
# CreditCard

f,axes=plt.subplots(1,2,figsize=(17,7))
dfb['CreditCard'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0],shadow=True)
sns.countplot('CreditCard',data=dfb,ax=axes[1],order=[0,1])
axes[0].set_title('CreditCard Variable Pie Chart')
axes[1].set_title('CreditCard Variable Bar Graph')
plt.show()

# %%
# LoanOnCard

f,axes=plt.subplots(1,2,figsize=(17,7))
dfb['LoanOnCard'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0],shadow=True)
sns.countplot('LoanOnCard',data=dfb,ax=axes[1],order=[0,1])
axes[0].set_title('LoanOnCard Variable Pie Chart')
axes[1].set_title('LoanOnCard Variable Bar Graph')
plt.show()

# %% [markdown]
# #### Check for data imbalance in target vector

# %%
f,axes=plt.subplots(1,2,figsize=(17,7))
dfb['LoanOnCard'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0],shadow=True)
sns.countplot('LoanOnCard',data=dfb,ax=axes[1],order=[0,1])
axes[0].set_title('LoanOnCard Variable Pie Chart')
axes[1].set_title('LoanOnCard Variable Bar Graph')
plt.show()

# %%
# Non-Loan Holders:0; Loan Holders:1

dfb.groupby(["LoanOnCard"]).count()

# %% [markdown]
# **Balancing the Target Variable 'LoanOnCard' using SMOTE (Synthetic Minority Over-sampling Technique):**
#
# Imbalanced data is data in which observed frequencies are very different across the different possible values of a categorical variable. Basically, there are many observations of some type and very few of another type.
#
# SMOTE is a solution when we have imbalanced data.

# %% [markdown]
# #### Visualize a heatmap to understand correlation between all features

# %%
plt.figure(dpi = 120,figsize= (5,4))
mask = np.triu(np.ones_like(dfb.corr()))
sns.heatmap(dfb.corr(),mask = mask, fmt = ".2f",annot=True,lw=1,cmap = 'plasma')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.title('Correlation Heatmap')
plt.show()

# %% [markdown]
# #### Visualize a pairplot with 2 classes distinguished by colors

# %%
sns.pairplot(dfb,hue='LoanOnCard')

# %% [markdown]
# **Observations:**
# - There is no strong relation between features and target variable.
# - Age and CustomerSince are linearly related.
# - There is some relation between HighestSpend and MonthlyAverageSpend.
# - HighestSpend has a better correlation with LoanOnCard, a moderate correlation with FixedDepositAmount and MonthlyAverageSpend.
# - The features HiddenScore, Level, Mortgage, Security,InternetBanking has very low impact and do not reveal much of the information, hence can be dropped.

# %% [markdown]
# #### Test for significance of features

# %%
# List of Numerical Features
col_num = ['Age', 'CustomerSince', 'HighestSpend', 'MonthlyAverageSpend', 'Mortgage']

# List of Categorical Features
col_cat = ['HiddenScore', 'Level', 'Security', 'FixedDepositAccount', 'InternetBanking', 'CreditCard']

# %%
# Understand the mean values of target vector for all the Numerical Features
class_summary=dfb.groupby('LoanOnCard')
class_summary.mean().reset_index()

# %%
# Hypotheis Testing of continuous features with target vector

# Ho(Null Hypothesis):There is no significant difference in numerical features with different category of Target vector
# Ha(Alternate Hypothesis):There is significant difference in numerical features with different category of Target vector

# Use t test
for i in col_num:
    x = np.array(dfb[dfb.LoanOnCard == 0][i]) 
    y = np.array(dfb[dfb.LoanOnCard == 1][i])
    t, p_value  = stats.ttest_ind(x, y, axis = 0, equal_var=False) 
    print('{} P_Value:{}'.format('\033[1m',p_value))
    if p_value < 0.05:  # Setting significance level at 5%
        print('{} Reject Null Hypothesis. {} of Loan holders and Non-Loan holders are not same.'.format('\033[1m',i))
    else:
        print('{} Fail to Reject Null Hypothesis. {} of Loan holders and Non-Loan holders are same.'.format('\033[1m',i))
    print('\n')

# %% [markdown]
# **Observations:**
# - Age and CustomerSince features does not have effect on target variable.
# - Dropping of these columns can be considered during model building.

# %%
# Hypotheis Testing of categorcal features with target vector

# Ho(Null Hypothesis):There is no significant difference in categorical features with different category of Target vector
# Ha(Alternate Hypothesis):There is significant difference in categorical features with different category of Target vector

for i in col_cat:
    crosstab=pd.crosstab(dfb['LoanOnCard'],dfb[i])
    chi,p_value,dof,expected=stats.chi2_contingency(crosstab)
    print('{} P_Value:{}'.format('\033[1m',p_value))
    if p_value < 0.05:  # Setting our significance level at 5%
        print('{} Reject Null Hypothesis. There is significant difference in {} Feature for different category of target variable.'.format('\033[1m',i))
    else:
        print('{} Fail to Reject Null Hypothesis.There is no significant difference in {} Feature for different category of target variable.'.format('\033[1m',i))
    print('\n')

# %% [markdown]
# **Observations:**
#
# - Security, InternetBanking and CreditCard features does not have effect on target variable.
# - Dropping of these columns can be considered during model building.

# %% [markdown]
# ## 3. Data Preparation and Model Building

# %%
# Impute outliers with mean
col=['HighestSpend','MonthlyAverageSpend','Mortgage']

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
dfb.info()

# %% [markdown]
# ### 3A. Split data into X and Y.
# [Recommended to drop ID & ZipCode. LoanOnCard is target Variable]

# %%
# Arrange data into independent variables and dependent variables
X=dfb.drop(columns='LoanOnCard')
y=dfb['LoanOnCard'] # Target Vector

# %% [markdown]
# ### 3B. Split data into train and test. Keep 25% data reserved for testing.

# %%
# Split X and y into training and test set in 75:25 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# %% [markdown]
# ### 3C. Train a Supervised Learning Classification base model - Logistic Regression.

# %%
# Build the Logistic Regression model
logit = LogisticRegression()

# Train the model
logit.fit(X_train, y_train)
logit_pred = logit.predict(X_test)

# %% [markdown]
# ### 3D. Print evaluation metrics for the model and clearly share insights.

# %%
# Classification Accuracy
print('Accuracy on Training data:',logit.score(X_train, y_train))
print('Accuracy on Test data:',logit.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, logit_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, logit_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
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
# **Insights:**
# - Training set and testing set accuracies (Almost 90%) are balanced when model is built without sampling. Also accuracy is very good in this case.
# - Model is poor in predicting class 1 compared to class 0.
# - Accuracy is good but in this case we need to look on recall values.
# - Class 0 and Class 1 recall values are 98 and 58 respectively.
# - We don't have enough sample of class 1 to train the model.
# - We can improve the overall performance metrics with Oversampling, Hyper-parameter tuning and Model selection techniques.

# %% [markdown]
# ### 3E. Balance the data using the right balancing technique.
# - Check distribution of the target variable
# - Say output is class A : 20% and class B : 80%
# - Here you need to balance the target variable as 50:50.
# - Try appropriate method to achieve the same.

# %%
print('Before oversampling distribution of target vector:')
print(y.value_counts())

# %% [markdown]
# #### Using oversampling over complete dataset

# %%
# Using SMOTENC
# Create the oversampler. 
# For SMOTE-NC we need to pinpoint the column position for the categorical features in the dataset.
smote_nc=SMOTENC(categorical_features=[3,5,7,8,9,10],random_state=0)
X1, y1=smote_nc.fit_resample(X, y)

# %%
# Target vector is balanced after oversampling
print('After oversampling distribution of target vector:')
print(y1.value_counts())

# %%
# Split X and y into training and test set in 75:25 ratio
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.25, random_state=0)

# %%
# Build the Logistic Regression model
logit = LogisticRegression()

# Train the model
logit.fit(X_train, y_train)
logit_pred = logit.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',logit.score(X_train, y_train))
print('Accuracy on Test data:',logit.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, logit_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, logit_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# #### Using oversampling over training dataset only

# %%
# Split X and y into training and test set in 75:25 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# %%
# Using SMOTENC
# Create the oversampler. 
# For SMOTE-NC we need to pinpoint the column position for the categorical features in the dataset.
smote_nc=SMOTENC(categorical_features=[3,5,7,8,9,10],random_state=0)
X1, y1=smote_nc.fit_resample(X_train, y_train)

# %% [markdown]
# ### 3F. Again train the same previous model on balanced data.

# %%
# Build the Logistic Regression model
logit = LogisticRegression()

# Train the model
logit.fit(X1, y1)
logit_pred = logit.predict(X_test)

# %% [markdown]
# ### 3G. Print evaluation metrics and clearly share differences observed.

# %%
# Classification Accuracy
print('Accuracy on Training data:',logit.score(X_train, y_train))
print('Accuracy on Test data:',logit.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, logit_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, logit_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# **Insights:**
# - We can see there is decrease in test accuracy.
# - After oversampling only on training data we can see difference in values.
# - We are getting good recall values but the precision value is reduced.
# - Hyper-parameter tuning, variable selection, and model selection methods would be used to improve the performance.

# %% [markdown]
# ## 4. Performance Improvement:

# %% [markdown]
# ### 4A. Train a base model each for SVM, KNN.

# %% [markdown]
# ### 4B. Tune parameters for each of the models wherever required and finalize a model.
# (Optional: Experiment with various Hyperparameters - Research required)

# %% [markdown]
# ### Refer the relevant sections of SVM and KNN for 4A and 4B

# %%
# Modify the dataframe for SVM and KNN
dfb.info()

# %%
col=['HiddenScore','Level','Security','FixedDepositAccount','InternetBanking','CreditCard']
for c in col:
    dfb[c]=dfb[c].astype('int64')

# %%
dfb.info()

# %%
dfb.head()

# %% [markdown]
# #### SVM

# %% [markdown]
# #### Use SVM without Oversampling

# %%
# Create copy of dataset.
dfb_model = dfb.copy()

# Rescaling features matrix using various scaling methods:
# scaler = RobustScaler()
scaler = MinMaxScaler()
# scaler = StandardScaler()

features = [['Age','CustomerSince','HighestSpend','HiddenScore','MonthlyAverageSpend',
             'Level','Mortgage','Security','FixedDepositAccount','InternetBanking','CreditCard']]
for feature in features:
    dfb_model[feature] = scaler.fit_transform(dfb_model[feature])
    
#Create svm_model Object
svm_model = SVC()

#Create x and y variable
X = dfb_model.drop(columns=['LoanOnCard'])
y = dfb_model['LoanOnCard']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Training the model
svm_model.fit(X_train, y_train)

#Predict testing set
y_pred = svm_model.predict(X_test)

# Classification Accuracy
print('Accuracy on Training data:',svm_model.score(X_train, y_train))
print('Accuracy on Test data:',svm_model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# #### Use SVM with Oversampling

# %%
# Create copy of dataset.
dfb_model = dfb.copy()

# Rescaling features matrix using various scaling methods:
# scaler = RobustScaler()
scaler = MinMaxScaler()
# scaler = StandardScaler()

features = [['Age','CustomerSince','HighestSpend','HiddenScore','MonthlyAverageSpend',
             'Level','Mortgage','Security','FixedDepositAccount','InternetBanking','CreditCard']]
for feature in features:
    dfb_model[feature] = scaler.fit_transform(dfb_model[feature])
    
#Create svm_model Object
svm_model = SVC()

#Create x and y variable
X = dfb_model.drop(columns=['LoanOnCard'])
y = dfb_model['LoanOnCard']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Using SMOTE
smote=SMOTE(random_state=0)
X1, y1=smote.fit_resample(X_train, y_train)

#Training the model
svm_model.fit(X1, y1)

#Predict testing set
y_pred = svm_model.predict(X_test)

# Classification Accuracy
print('Accuracy on Training data:',svm_model.score(X_train, y_train))
print('Accuracy on Test data:',svm_model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# #### Use automated search without Oversampling for hyper-parameters.

# %%
# Build and train the model

# Create copy of dataset.
dfb_model = dfb.copy()

# Rescaling features matrix using various scaling methods:
# scaler = RobustScaler()
scaler = MinMaxScaler()
# scaler = StandardScaler()

features = [['Age','CustomerSince','HighestSpend','HiddenScore','MonthlyAverageSpend',
             'Level','Mortgage','Security','FixedDepositAccount','InternetBanking','CreditCard']]
for feature in features:
    dfb_model[feature] = scaler.fit_transform(dfb_model[feature])
    
#Create svm_model Object
svm_model = SVC()

#Create x and y variable
X = dfb_model.drop(columns=['LoanOnCard'])
y = dfb_model['LoanOnCard']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

svm_model.fit(X_train, y_train)

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

gs = GridSearchCV(svm_model, grid_params, cv=10)
gs.fit(X_train, y_train)

# %%
gs.cv_results_['params']

# %%
gs.best_params_

# %%
gs.cv_results_['mean_test_score']

# %%
# Lets Build SVM with best params

svm_model = SVC(C=10, gamma=1, kernel= 'poly')

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',svm_model.score(X_train, y_train))
print('Accuracy on Test data:',svm_model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# #### Use automated search with Oversampling for hyper-parameters.

# %%
# Build and train the model

# Create copy of dataset.
dfb_model = dfb.copy()

# Rescaling features matrix using various scaling methods:
# scaler = RobustScaler()
scaler = MinMaxScaler()
# scaler = StandardScaler()

features = [['Age','CustomerSince','HighestSpend','HiddenScore','MonthlyAverageSpend',
             'Level','Mortgage','Security','FixedDepositAccount','InternetBanking','CreditCard']]
for feature in features:
    dfb_model[feature] = scaler.fit_transform(dfb_model[feature])
    
#Create svm_model Object
svm_model = SVC()

#Create x and y variable
X = dfb_model.drop(columns=['LoanOnCard'])
y = dfb_model['LoanOnCard']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Using SMOTE
smote=SMOTE(random_state=0)
X1, y1=smote.fit_resample(X_train, y_train)

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

gs = GridSearchCV(svm_model, grid_params, cv=10)
gs.fit(X_train, y_train)

# %%
gs.cv_results_['params']

# %%
gs.best_params_

# %%
gs.cv_results_['mean_test_score']

# %%
# Lets Build SVM with best params

svm_model = SVC(C=10, gamma=1, kernel= 'poly')

svm_model.fit(X1, y1)
y_pred = svm_model.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',svm_model.score(X_train, y_train))
print('Accuracy on Test data:',svm_model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# #### KNN

# %% [markdown]
# #### Use KNN without Oversampling

# %%
# Create copy of dataset.
dfb_model = dfb.copy()

# Rescaling features matrix using various scaling methods:
#scaler = RobustScaler()
scaler = MinMaxScaler()
# scaler = StandardScaler()

features = [['Age','CustomerSince','HighestSpend','HiddenScore','MonthlyAverageSpend',
             'Level','Mortgage','Security','FixedDepositAccount','InternetBanking','CreditCard']]
for feature in features:
    dfb_model[feature] = scaler.fit_transform(dfb_model[feature])
    
#Create knn Object
knn = KNeighborsClassifier()

#Create x and y variable
X = dfb_model.drop(columns=['LoanOnCard'])
y = dfb_model['LoanOnCard']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Training the model
knn.fit(X_train, y_train)

#Predict testing set
y_pred = knn.predict(X_test)

# Classification Accuracy
print('Accuracy on Training data:',knn.score(X_train, y_train))
print('Accuracy on Test data:',knn.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# #### Use KNN with Oversampling

# %%
# Create copy of dataset.
dfb_model = dfb.copy()

# Rescaling features matrix using various scaling methods:
#scaler = RobustScaler()
scaler = MinMaxScaler()
# scaler = StandardScaler()

features = [['Age','CustomerSince','HighestSpend','HiddenScore','MonthlyAverageSpend',
             'Level','Mortgage','Security','FixedDepositAccount','InternetBanking','CreditCard']]
for feature in features:
    dfb_model[feature] = scaler.fit_transform(dfb_model[feature])
    
#Create knn Object
knn = KNeighborsClassifier()

#Create x and y variable
X = dfb_model.drop(columns=['LoanOnCard'])
y = dfb_model['LoanOnCard']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Using SMOTE
smote=SMOTE(random_state=0)
X1, y1=smote.fit_resample(X_train, y_train)

#Training the model
knn.fit(X1, y1)

#Predict testing set
y_pred = knn.predict(X_test)

# Classification Accuracy
print('Accuracy on Training data:',knn.score(X_train, y_train))
print('Accuracy on Test data:',knn.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# #### Use automated search without Oversampling for hyper-parameters.

# %%
# Build and train the model

# Create copy of dataset.
dfb_model = dfb.copy()

# Rescaling features matrix using various scaling methods:
# scaler = RobustScaler()
scaler = MinMaxScaler()
# scaler = StandardScaler()

features = [['Age','CustomerSince','HighestSpend','HiddenScore','MonthlyAverageSpend',
             'Level','Mortgage','Security','FixedDepositAccount','InternetBanking','CreditCard']]
for feature in features:
    dfb_model[feature] = scaler.fit_transform(dfb_model[feature])
    
#Create svm_model Object
knn = KNeighborsClassifier()

#Create x and y variable
X = dfb_model.drop(columns=['LoanOnCard'])
y = dfb_model['LoanOnCard']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

knn.fit(X_train, y_train)

# %%
# Exhaustive search over specified parameter values for an estimator.
# Important members are fit, predict.

# GridSearchCV implements a "fit" and a "score" method.
# It also implements "score_samples", "predict", "predict_proba",
# "decision_function", "transform" and "inverse_transform" if they are
# implemented in the estimator used.

# The parameters of the estimator used to apply these methods are optimized
# by cross-validated grid-search over a parameter grid.

grid_params = { 'n_neighbors' : [1,3,5,7,9,11,13,15,17,19],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}

gs = GridSearchCV(knn, grid_params, cv=10)
gs.fit(X_train, y_train)

# %%
gs.cv_results_['params']

# %%
gs.best_params_

# %%
gs.cv_results_['mean_test_score']

# %%
# Lets Build knn with best params

knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski', weights='distance')

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',knn.score(X_train, y_train))
print('Accuracy on Test data:',knn.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# This is a clear case of over-fitting; So we have to work with the manual search approach.

# %% [markdown]
# #### Use manual search for hyper-parameters.

# %%
# Optimize the value of k

train_score=[]
test_score=[]
for k in range(1,51):
    knn = KNeighborsClassifier(n_neighbors= k , metric = 'euclidean' ) 
    knn.fit(X_train, y_train)
    train_score.append(knn.score(X_train, y_train))
    test_score.append(knn.score(X_test, y_test))

# %%
# train_score vs. k

plt.plot(range(1,51),train_score)
plt.show()

# Here training accuracy decreases as k increases

# %%
# test_score vs. k

plt.plot(range(1,51),test_score)
plt.show()

# The accuracy is maximum when k is less than 50. So we can cap the value of k as less than 50.

# %%
# Check the performance of model for various k values; Consider the k upto 10
# Build knn with k neighbors

k=[1,3,5,7,9]
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i, metric = 'euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print('Accuracy on Training data for k {} is {}:'.format(i,knn.score(X_train, y_train)))
    print('Accuracy on Test data for k {} is {}:'.format(i,knn.score(X_test, y_test)))
    print("classification  Matrix:\n",classification_report(y_test,y_pred))

# %%
# For k=3, we can see a model with good overall performance metrics
# Lets Build knn with k=3, metric='euclidean' and weights='uniform'

knn = KNeighborsClassifier(n_neighbors=3, metric = 'euclidean')

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',knn.score(X_train, y_train))
print('Accuracy on Test data:',knn.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# #### Final KNN with Oversampling

# %%
# Create copy of dataset.
dfb_model = dfb.copy()

# Rescaling features matrix using various scaling methods:
#scaler = RobustScaler()
scaler = MinMaxScaler()
# scaler = StandardScaler()

features = [['Age','CustomerSince','HighestSpend','HiddenScore','MonthlyAverageSpend',
             'Level','Mortgage','Security','FixedDepositAccount','InternetBanking','CreditCard']]
for feature in features:
    dfb_model[feature] = scaler.fit_transform(dfb_model[feature])
    
#Create knn Object
knn = KNeighborsClassifier(n_neighbors=3, metric = 'euclidean')
#knn = KNeighborsClassifier(n_neighbors=3, metric = 'minkowski')

#Create x and y variable
X = dfb_model.drop(columns=['LoanOnCard'])
y = dfb_model['LoanOnCard']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Using SMOTE
smote=SMOTE(random_state=0)
X1, y1=smote.fit_resample(X_train, y_train)

#Training the model
knn.fit(X1, y1)

#Predict testing set
y_pred = knn.predict(X_test)

# Classification Accuracy
print('Accuracy on Training data:',knn.score(X_train, y_train))
print('Accuracy on Test data:',knn.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# ### 4C. Print evaluation metrics for final model.

# %% [markdown]
# #### Final model with all the relevant features

# %%
# Using the Automated search without Oversampling for Hyperparameters of SVM

# Create copy of dataset.
dfb_model = dfb.copy()

# Rescaling features matrix using various scaling methods:
# scaler = RobustScaler()
scaler = MinMaxScaler()
# scaler = StandardScaler()

features = [['Age','CustomerSince','HighestSpend','HiddenScore','MonthlyAverageSpend',
             'Level','Mortgage','Security','FixedDepositAccount','InternetBanking','CreditCard']]
for feature in features:
    dfb_model[feature] = scaler.fit_transform(dfb_model[feature])

#Create x and y variable
X = dfb_model.drop(columns=['LoanOnCard'])
y = dfb_model['LoanOnCard']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# %%
# Lets Build SVM with best params
svm_model = SVC(C=10, gamma=1, kernel= 'poly')

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',svm_model.score(X_train, y_train))
print('Accuracy on Test data:',svm_model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# #### Final model with all the significant features

# %%
# Using the Automated search without Oversampling for Hyperparameters of SVM

# Create copy of dataset.
dfb_model = dfb.copy()

# Rescaling features matrix using various scaling methods:
# scaler = RobustScaler()
scaler = MinMaxScaler()
# scaler = StandardScaler()

features = [['Age','CustomerSince','HighestSpend','HiddenScore','MonthlyAverageSpend',
             'Level','Mortgage','Security','FixedDepositAccount','InternetBanking','CreditCard']]
for feature in features:
    dfb_model[feature] = scaler.fit_transform(dfb_model[feature])

#Create x and y variable
X = dfb_model.drop(columns=['Age','CustomerSince','Security','InternetBanking','CreditCard','LoanOnCard'])
y = dfb_model['LoanOnCard']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# %%
# Lets Build SVM with best params
svm_model = SVC(C=10, gamma=1, kernel= 'poly')

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# %%
# Classification Accuracy
print('Accuracy on Training data:',svm_model.score(X_train, y_train))
print('Accuracy on Test data:',svm_model.score(X_test, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-Loan holders","Loan holders"]],
                  columns = [i for i in ["Non-Loan holders","Loan holders"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown]
# ### 4D. Share improvement achieved from base model to final model.

# %% [markdown]
# **Insights:**
#
# Refer the corresponding cells for additional details; Considering the base model as Logistic Regression and final model as SVM with significant features.
#
# 1. Accuracy improved from 95% to 98%.
# 2. Precision for class 0 improved from 96% to 98% and for class 1 improved from 75% to 91%.
# 3. Recall for class 0 improved from 98% to 99% and for class 1 improved from 58% to 81%.
# 4. F1 Score for class 0 improved from 97% to 99% and for class 1 improved from 66% to 86%.
# 5. Precision & Recall values are better in predicting the potential customers.
# 6. Banking domain prefers to see the Precision than the Recall as to avoid false negatives.
# 7. If the dataset contains equal samples of both the classes, better models can be built with higher accuracy, recall and precision values.
# 8. Few customers do not have credit card but those customers have loan on cards. We can avoid this data error.
#
# ![comp-3.png](attachment:comp-3.png)

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
