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

# %% [markdown] id="13395265"
# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#00b3e5;">Contents: Neural Networks Project
# 1. [Part-A: Solution](#Part-A:-Solution)
# 2. [Part-B: Solution](#Part-B:-Solution)

# %% [markdown] id="5f36f6dd"
# # Part-A: Solution

# %% [markdown] id="dedd939b"
# - **DOMAIN:** Electronics and Telecommunication
# - **CONTEXT:** A communications equipment manufacturing company has a product which is responsible for emitting informative signals.Company wants to build a machine learning model which can help the company to predict the equipment’s signal quality using various parameters.
# - **DATA DESCRIPTION:** The data set contains information on various signal tests performed:
#  1. Parameters: Various measurable signal parameters.
#  2. Signal_Quality: Final signal strength or quality
# - **PROJECT OBJECTIVE:** To build a classifier which can use the given parameters to determine the signal strength or quality.

# %% id="9b11c05c"
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

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings("ignore")

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %% colab={"base_uri": "https://localhost:8080/"} id="peP2nm91hndX" outputId="e4a342a1-949d-4961-892d-08346543ed8e"
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown] id="a3313f55"
# ## 1. Data import and Understanding:

# %% [markdown] id="fb411bd2"
# ### 1A. Read the ‘Signals.csv’ as DataFrame and import required libraries.

# %% id="ff8fc39f"
# Read and import the data
df=pd.read_csv('/content/drive/MyDrive/MGL/Project-DL/Signals.csv')

# %% colab={"base_uri": "https://localhost:8080/", "height": 597} id="7545a1df" outputId="e82647c4-4394-43ce-d6f6-c503bc149e07"
df.info()
df.head()


# %% [markdown] id="a0dc04f4"
# ### 1B. Check for missing values and print percentage for each attribute.

# %% colab={"base_uri": "https://localhost:8080/", "height": 426} id="1301611b" outputId="39f1b7c6-0cc5-4d5e-abea-4ae41e389658"
# Percentage of missing values

# df.isnull().sum()
# df.isna().sum()

def missing_check(df):
    total = df.isnull().sum().sort_values(ascending=False)   # total number of null values
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)  # percentage of values that are null
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # putting the above two together
    return missing_data # return the dataframe
missing_check(df)

# %% colab={"base_uri": "https://localhost:8080/"} id="72775fb2" outputId="a7450ea1-022e-4d38-b59d-1094a4fa7827"
print(df.Signal_Strength.value_counts())

# %% [markdown] id="97580215"
# ### 1C. Check for presence of duplicate records in the dataset and impute with appropriate method.

# %% colab={"base_uri": "https://localhost:8080/"} id="49d281c0" outputId="df60ba90-88bb-450a-8d1f-7cf7db831fb8"
# To take a look at the duplication in the DataFrame as a whole, just call the duplicated() method on 
# the DataFrame. It outputs True if an entire row is identical to a previous row.
df.duplicated().sum()

# %% colab={"base_uri": "https://localhost:8080/"} id="8b6ba294" outputId="8928d7b7-618f-4923-ecbd-ad7c3804a9ce"
# Count the number of non-duplicates
(~df.duplicated()).sum()

# %% colab={"base_uri": "https://localhost:8080/", "height": 485} id="bf738629" outputId="7a6f5153-190e-470b-deae-337079ad6bf5"
# Extract duplicate rows
df.loc[df.duplicated(), :]

# %% colab={"base_uri": "https://localhost:8080/"} id="dcdf4fd4" outputId="9349f56a-0d1b-4146-b6fc-01992ed0bdfc"
# Percentage of duplicate rows
DuplicatePercent = round((240/1599)*100, 4)
DuplicatePercent

# %% colab={"base_uri": "https://localhost:8080/"} id="30f8e7d9" outputId="4f1ebf2d-3c84-460f-e24d-c2d409c217c7"
df = df.drop_duplicates()
df.info()

# %% [markdown] id="eb91837f"
# ### 1D. Visualise distribution of the target variable

# %% colab={"base_uri": "https://localhost:8080/", "height": 459} id="0438b707" outputId="054e73ad-0b49-4b6a-aabb-678d1c538933"
# Understand the target variable and check for imbalanced dataset
f,axes=plt.subplots(1,2,figsize=(17,7))
df['Signal_Strength'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('Signal_Strength',data=df,ax=axes[1])
axes[0].set_title('Response Variable Pie Chart')
axes[1].set_title('Response Variable Bar Graph')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 330} id="969bbb89" outputId="000fa951-555b-4582-e299-3f62b8c2f6f6"
# Group datapoints by class
df.groupby(["Signal_Strength"]).count()

# %% [markdown] id="9463d47f"
# **Insights from above graphs:**
#
# - Class 5, 6, 7 are dominating total values followed by Class 3, 4, and 8.
# - The above graph shows that the data is biased towards data-points having class value as 5 and 6.
# - More can be inferred from the pie chart and bar graph above.
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
# - Here we may use oversampling because under-sampling may remove important information from the dataset

# %% [markdown] id="5105a239"
# ### 1E. Share insights from the initial data analysis (at least 2).

# %% colab={"base_uri": "https://localhost:8080/", "height": 426} id="6e58227d" outputId="28612869-2343-4b7c-b7d6-2fa1335fff1d"
# Describe function generates descriptive statistics that summarize the central tendency, 
# dispersion and shape of a dataset’s distribution, excluding NaN values.

# This method tells us a lot of things about a dataset. One important thing is that 
# the describe() method deals only with numeric values. It doesn't work with any 
# categorical values. So if there are any categorical values in a column the describe() 
# method will ignore it and display summary for the other columns.
df.describe().T

# %% [markdown] id="ff162913"
# **Observations:**
# - Parameter 1:
#  - Mean and Median are nearly equal. Distribution might be normal.
#  - 75 % of values are less than 9.2, and maximum value is 15.9.
# - Parameter 7:
#  - Mean and median are not equal. Skewness is expected.
#  - Range of values is large.
#  - Distribution is not normal because of big SD.

# %% [markdown] id="a509897d"
# ### Quick EDA

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="0e33a03f" outputId="367f2085-63a9-4d6e-fb3b-e410bd3e8602"
# Check for correlation and consider the features where correlation coeff > 0.7
plt.figure(figsize=(20,18))
corr=df.corr()
sns.heatmap(abs(corr>0.7),cmap="Reds");

# %% colab={"base_uri": "https://localhost:8080/", "height": 487} id="a050b2e6" outputId="5e683d10-4947-4cb2-a7e9-5b1f715c9599"
corr = df.corr()
corr

# %% id="94a6895b"
# Make a copy of the dataset and drop the target class for easy EDA
df1=df.copy()
df1.drop(['Signal_Strength'],axis=1,inplace=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="a9bcf3c9" outputId="68631407-c911-4b0c-d755-79e044d3ee84"
# Density plot to check for the distribution of features
plt.figure(figsize=(40, 40))
col = 1
for i in df1.columns:
    plt.subplot(4, 3, col)
    sns.distplot(df1[i], color = 'g')
    col += 1 

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="db682148" outputId="2c5c673a-f8b2-4297-da71-3fb1ed13a269"
# Use boxplot to check for outliers
plt.figure(figsize=(40, 40))
col = 1
for i in df1.columns:
    plt.subplot(4, 3, col)
    sns.boxplot(df1[i],color='green')
    col += 1

# %% id="20dfdf5c"
# Replace the outliers with median
for i in df1.columns:
    q1 = df1[i].quantile(0.25)
    q3 = df1[i].quantile(0.75)
    iqr = q3 - q1
    
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    
    df1.loc[(df1[i] < low) | (df1[i] > high), i] = df1[i].median()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="4dec411d" outputId="e54bb051-b4eb-4906-d7ad-1cf22166d2f8"
# Situation after removing the outliers with median
plt.figure(figsize=(40, 40))
col = 1
for i in df1.columns:
    plt.subplot(4, 3, col)
    sns.boxplot(df1[i],color='green')
    col += 1

# %% colab={"base_uri": "https://localhost:8080/"} id="802d8a82" outputId="f5d96fbb-efab-408d-f046-e366d95f3ea7"
# Combine the dataset
y=df['Signal_Strength']
df1=pd.concat([df1,y],axis=1)

df1.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 475} id="7d707923" outputId="20e87224-5325-4fd7-e2ff-14983cbfc1a3"
# Correlation of "Signal_Strength" with other features
# Open image in a new tab for details
plt.figure(figsize=(30,10))
df1.corr()['Signal_Strength'].sort_values(ascending = False).plot(kind='bar')

# %% [markdown] id="5579308b"
# ## 2. Data preprocessing:

# %% [markdown] id="8c19ed0e"
# ### 2A. Split the data into X & Y.

# %% id="4c72efef"
# Create the features matrix and target vector
X=df1.drop(['Signal_Strength'], axis=1)
y=df1['Signal_Strength']

# %% [markdown] id="eee0ed1e"
# ### 2B. Split the data into train & test with 70:30 proportion.

# %% id="7a014062"
# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# %% [markdown] id="895e5f33"
# ### 2C. Print shape of all the 4 variables and verify if train and test data is in sync.

# %% colab={"base_uri": "https://localhost:8080/"} id="afbe6f2f" outputId="745718f3-0960-40dc-f2a2-525d1cd99bd1"
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# %% [markdown] id="944ee94e"
# ### 2D. Normalise the train and test data with appropriate method.

# %% id="ab659401"
# Using different scaling methods:
# scaler = RobustScaler()
# scaler = MinMaxScaler()
scaler = StandardScaler()
# mydata = mydata.apply(zscore)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# X_train = X_train.apply(zscore)
# X_test = X_test.apply(zscore)

# %% [markdown] id="29ff4f16"
# ### 2E. Transform Labels into format acceptable by Neural Network.

# %% id="016ed97b"
# Encoding is used for categorical data
# We can consider 3,4,5,6,7,8 as categorical
# Use one hot encoding
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="96684799" outputId="2bb35a08-ad82-432c-a42a-862bf68e9dcb"
y_train.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="d4e31cac" outputId="dd924d54-1af5-410b-93aa-9cec97e02cdc"
y_test.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="00ca70f1" outputId="c78c1f67-45aa-4dde-d0c0-adb3df6dbc00"
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# %% [markdown] id="c891804a"
# ## 3. Model Training & Evaluation using Neural Network
# - 3A. Design a Neural Network to train a classifier.
# - 3B. Train the classifier using previously designed Architecture.
# - 3C. Plot 2 separate visuals.
#  - i. Training Loss and Validation Loss
#  - ii. Training Accuracy and Validation Accuracy
# - 3D. Design new architecture/update existing architecture in attempt to improve the performance of the model.
# - 3E. Plot visuals as in Q3.C and share insights about difference observed in both the models.

# %% id="67883d96"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import random
from tensorflow.keras import backend
random.seed(1)
np.random.seed(1) 
tf.random.set_seed(1)

# %% [markdown] id="Icc70s8M_8y8"
# ### Model-1: Base model

# %% id="vqTra0j21t4M"
backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

# %% id="o7yT7nu12mBZ"
# Initializing the ANN
model1 = Sequential()
# Input layer
model1.add(Dense(activation = 'relu', input_dim = 11, units=64))
#Add 1st hidden layer
model1.add(Dense(32, activation='relu'))
# Adding the output layer
model1.add(Dense(6, activation = 'sigmoid')) 

# %% id="-jBv61_F2oFo"
# Create optimizer with default learning rate
# Compile the model
model1.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

# %% colab={"base_uri": "https://localhost:8080/"} id="gto2k5Se2rwW" outputId="f243d695-7121-4262-e4d7-08ef6b387c64"
model1.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="LZr2viBM2sZX" outputId="471bc2be-406e-4485-8d31-4e25ff840816"
history1=model1.fit(X_train, y_train,           
          validation_split=0.2,
          epochs=50,
          batch_size=64,verbose=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 286} id="8hVFVKjJ2s6E" outputId="cb2fc3da-3cd9-4e21-ce67-42f310848c84"
# Capturing learning history per epoch
hist  = pd.DataFrame(history1.history)
hist['epoch'] = history1.epoch

# Plotting accuracy at different epochs
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model1.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="RvTL4eQPA7mE" outputId="a194d46e-b311-4879-ea48-f2fd37424b08"
# Capturing learning history per epoch
hist  = pd.DataFrame(history1.history)
hist['epoch'] = history1.epoch

# Plotting accuracy at different epochs
plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model1.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="2mSnDj1F6mtm" outputId="0d230611-2837-4a5b-fe3e-4216ab36adc7"
y_pred=model1.predict(X_test)
y_pred

# %% colab={"base_uri": "https://localhost:8080/", "height": 830} id="BcOxend26ljP" outputId="7eb67eec-a746-4bb2-80d3-6d8628fd4d61"
# Classification Accuracy
print('Loss and Accuracy on Training data:',model1.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model1.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), labels=[0,1,2,3,4,5])
df_cm = pd.DataFrame(cm, index = [i for i in ['3','4','5','6','7','8']],
                  columns = [i for i in ['3','4','5','6','7','8']])
plt.figure(figsize = (10,8))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown] id="4RjkiDiCvHpK"
# **Evaluation metrics allow us to estimate errors to determine how well our models are performing:**
#
# > Accuracy: ratio of correct predictions over total predictions.
#
# > Precision: how often the classifier is correct when it predicts positive.
#
# > Recall: how often the classifier is correct for all positive instances.
#
# > F-Score: single measurement to combine precision and recall.
#
# **Considering the Class Recall, Precision and F1 Score as the most important parameters to decide the best model for this problem. We have the highest values in Model-3 and Model-6 here. Please refer the Model-3 and Model-6 for the same.**

# %% [markdown] id="3oLRe3CnBTX5"
# ### Model-2: Improving Base Model
# Let's try to change the optimizer, tune the decision threshold, increase the layers and configure some other hyperparameters accordingly, in order to improve the model's performance.

# %% id="latyEJwj3naY"
backend.clear_session()
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

# %% id="Ilh6aYdnDWAM"
model2 = Sequential()

model2.add(Dense(256,activation='relu',kernel_initializer='he_uniform',input_dim = X_train.shape[1]))
# Adding the hidden and output layers
model2.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
model2.add(Dense(64,activation='relu',kernel_initializer='he_uniform'))
model2.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))
model2.add(Dense(6, activation = 'sigmoid'))
# Compiling the ANN with Adam optimizer and binary cross entropy loss function 
optimizer = tf.keras.optimizers.Adam(0.001)
model2.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

# %% colab={"base_uri": "https://localhost:8080/"} id="fNrsrPcdDlym" outputId="a06fd732-36ee-4496-a25c-ca23c78db0ef"
model2.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="Bsqi8v9EDonL" outputId="f472213e-2225-41b8-eecd-a92c0cb8df2c"
history2 = model2.fit(X_train,y_train,batch_size=64,epochs=50,verbose=1,validation_split = 0.2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 295} id="OuI0RQWTDykk" outputId="2659b1de-0a0d-4aa3-9f99-04dd91f6ec6a"
#Plotting Train Loss vs Validation Loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="POAWqA7MuoCc" outputId="7030f75b-d86b-485b-fe41-a490314f7547"
# Capturing learning history per epoch
hist  = pd.DataFrame(history2.history)
hist['epoch'] = history2.epoch

# Plotting accuracy at different epochs
plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model1.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="iNpUTgU5QF3B" outputId="8b7cd7a8-66e5-4e82-fb97-e4902a339fb8"
y_pred=model2.predict(X_test)
y_pred

# %% colab={"base_uri": "https://localhost:8080/", "height": 830} id="vOGhTWMJMyz2" outputId="208b9ada-9a84-4ca6-af99-f50c564d8476"
# Classification Accuracy
print('Loss and Accuracy on Training data:',model2.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model2.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), labels=[0,1,2,3,4,5])
df_cm = pd.DataFrame(cm, index = [i for i in ['3','4','5','6','7','8']],
                  columns = [i for i in ['3','4','5','6','7','8']])
plt.figure(figsize = (10,8))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown] id="Zx0KZAZHOEbH"
# ### Model-3: Using Batch Normalization technique

# %% id="oQs2RhjvOHky"
backend.clear_session()
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

# %% id="Hq-r37teOPtb"
model3 = Sequential()
model3.add(Dense(128,activation='relu',input_dim = X_train.shape[1]))
model3.add(BatchNormalization())
model3.add(Dense(64,activation='relu',kernel_initializer='he_uniform'))
model3.add(BatchNormalization())
model3.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))
model3.add(Dense(6, activation = 'sigmoid'))

# %% colab={"base_uri": "https://localhost:8080/"} id="_FaopHYvOvC0" outputId="7cb2a6e4-b78d-4654-e977-997e4905786d"
model3.summary()

# %% id="KvALh4l5Oxxa"
optimizer = tf.keras.optimizers.Adam(0.001)
model3.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

# %% colab={"base_uri": "https://localhost:8080/"} id="lkD5MWjzO2Xd" outputId="e2ecdd1f-baff-46a8-e9fd-4bca493d89f2"
history3 = model3.fit(X_train,y_train,batch_size=64,epochs=50,verbose=1,validation_split = 0.2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 295} id="J_oQNYWaO6XO" outputId="c4c497c7-278c-493e-fd10-f7cda4fba6ec"
#Plotting Train Loss vs Validation Loss
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 286} id="yjokd49Vutxz" outputId="bf52690e-76b5-4a1f-99ba-7382e588e6ef"
# Capturing learning history per epoch
hist  = pd.DataFrame(history3.history)
hist['epoch'] = history3.epoch

# Plotting accuracy at different epochs
plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model1.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="qGALVP6FQ28R" outputId="72e24cef-769e-49d4-95b0-6f2fab331722"
y_pred=model3.predict(X_test)
y_pred

# %% colab={"base_uri": "https://localhost:8080/", "height": 830} id="93Z1aHa6PSnd" outputId="d0a4be2c-480f-4c07-921f-30b5ac0384b6"
# Classification Accuracy
print('Loss and Accuracy on Training data:',model3.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model3.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), labels=[0,1,2,3,4,5])
df_cm = pd.DataFrame(cm, index = [i for i in ['3','4','5','6','7','8']],
                  columns = [i for i in ['3','4','5','6','7','8']])
plt.figure(figsize = (10,8))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown] id="tRX33HeEcgGs"
# ### Model-4: Using the Dropout technique

# %% id="zP8nALwOPWZs"
backend.clear_session()
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

# %% id="S35fyOqccu2g"
model4 = Sequential()
model4.add(Dense(256,activation='relu',input_dim = X_train.shape[1]))
model4.add(Dropout(0.2))
model4.add(Dense(128,activation='relu'))
model4.add(Dropout(0.2))
model4.add(Dense(64,activation='relu'))
model4.add(Dropout(0.2))
model4.add(Dense(32,activation='relu'))
model4.add(Dense(6, activation = 'sigmoid'))

# %% colab={"base_uri": "https://localhost:8080/"} id="2ewIXFEMc1On" outputId="4b6ffc74-3d94-4533-f022-63cf0d415c05"
model4.summary()

# %% id="Z4t8NRUIc2Q8"
optimizer = tf.keras.optimizers.Adam(0.001)
model4.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

# %% colab={"base_uri": "https://localhost:8080/"} id="dYAi3MPxc61J" outputId="30bd3f23-15d1-47bc-8585-ed4f8e60636f"
history4 = model4.fit(X_train,y_train,batch_size=64,epochs=50,verbose=1,validation_split = 0.2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 295} id="SasKaoQ1c9x6" outputId="e6f57388-f5d9-416c-9258-5ad4e29ca747"
#Plotting Train Loss vs Validation Loss
plt.plot(history4.history['loss'])
plt.plot(history4.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="D0xOpbkDuxyn" outputId="ef9f88c6-f4d8-4b53-fdea-5cd8fc248658"
# Capturing learning history per epoch
hist  = pd.DataFrame(history4.history)
hist['epoch'] = history4.epoch

# Plotting accuracy at different epochs
plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model1.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="o8DAqiCgdDh8" outputId="a0fcce90-5b12-49c4-fe12-9693b6c6a124"
y_pred=model4.predict(X_test)
y_pred

# %% colab={"base_uri": "https://localhost:8080/", "height": 830} id="7M8WmAZJdVg4" outputId="4816cf19-fc95-4340-8b23-035a72280371"
# Classification Accuracy
print('Loss and Accuracy on Training data:',model4.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model4.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), labels=[0,1,2,3,4,5])
df_cm = pd.DataFrame(cm, index = [i for i in ['3','4','5','6','7','8']],
                  columns = [i for i in ['3','4','5','6','7','8']])
plt.figure(figsize = (10,8))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown] id="uSiLIz41egip"
# ### Model-5: Hyperparameter Tuning
# Some important hyperparameters to look out for while optimizing neural networks are:
# - Type of Architecture
# - Number of Layers
# - Number of Neurons in a layer
# - Regularization hyperparameters
# - Learning Rate
# - Type of Optimizer
# - Dropout Rate

# %% id="lpWzsg-7dbd1"
backend.clear_session()
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)


# %% id="hoYtQSAReou4"
def create_model_v5(lr,batch_size):  
    np.random.seed(1337)
    model = Sequential()
    model.add(Dense(256,activation='relu',input_dim = X_train.shape[1]))
    model.add(Dropout(0.3))
    #model.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    #model.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))
    #model.add(Dropout(0.3))
    model.add(Dense(32,activation='relu')) 
    model.add(Dense(6, activation='sigmoid'))

    #compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer = optimizer,loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


# %% colab={"base_uri": "https://localhost:8080/"} id="3pmukB0TfBhI" outputId="7ae0771b-786a-4d91-95d6-c83bb2b70435"
model5 = KerasClassifier(build_fn=create_model_v5, verbose=1)

params = {'batch_size':[32, 64, 128],
          'lr':[0.01,0.1,0.001]}

gs = RandomizedSearchCV(estimator=model5, param_distributions=params, cv=5, verbose=10, n_jobs=2, random_state=0)
gs.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/", "height": 332} id="y_9rYBfUgPvU" outputId="5bf78154-7fcb-48a2-e272-d6b43165635b"
pd.DataFrame({'param': gs.cv_results_["params"], 
              'score mean': gs.cv_results_["mean_test_score"], 
              'score s.d.': gs.cv_results_["std_test_score"],
              'rank': gs.cv_results_["rank_test_score"]})

# %% colab={"base_uri": "https://localhost:8080/"} id="0vfdnxdPgQ-l" outputId="199e40e3-65c3-473d-e9b6-1de537176c4c"
print('Best Parameters:', gs.best_params_, 'mean score: ', gs.best_score_, sep='\n')

# %% colab={"base_uri": "https://localhost:8080/"} id="6HPtsMB2gZD2" outputId="96022cab-07e1-4bb7-f33a-e51921d5423a"
model5=create_model_v5(batch_size=32, lr=0.01)

model5.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="LC3b2QCJhEw2" outputId="f86663c4-ad00-4066-8bff-d248d44c642c"
optimizer = tf.keras.optimizers.Adam(0.01)
model5.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

history5=model5.fit(X_train, y_train, epochs=50, batch_size = 64, verbose=1,validation_split=0.2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 295} id="Q9bNPT1yhKD2" outputId="4f6ac72c-2e17-42fd-83c3-e6501cbe3983"
#Plotting Train Loss vs Validation Loss
plt.plot(history5.history['loss'])
plt.plot(history5.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="XnQBSRamu2Io" outputId="8d271d3a-7127-4338-ed85-a54a8eaa5193"
# Capturing learning history per epoch
hist  = pd.DataFrame(history5.history)
hist['epoch'] = history5.epoch

# Plotting accuracy at different epochs
plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model1.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="4ywBcAq3hn4z" outputId="38f8f83c-1464-42e1-fec9-208abada6af0"
y_pred=model5.predict(X_test)
y_pred

# %% colab={"base_uri": "https://localhost:8080/", "height": 830} id="6asGpzlih8mO" outputId="26b3870d-4c83-4af8-b71a-b8fad2ed7585"
# Classification Accuracy
print('Loss and Accuracy on Training data:',model5.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model5.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), labels=[0,1,2,3,4,5])
df_cm = pd.DataFrame(cm, index = [i for i in ['3','4','5','6','7','8']],
                  columns = [i for i in ['3','4','5','6','7','8']])
plt.figure(figsize = (10,8))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown] id="pgkmZMHoiLPV"
# ### Model-6: Using hyperparameter tuning with Oversampling

# %% id="6jrRquNcI-o9"
# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# %% id="LYZTQeiZiAwQ"
# Create the oversampler.
smote=SMOTE(random_state=0, k_neighbors=3)
X1, y1=smote.fit_resample(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="jRJwhYROBcXV" outputId="9c2d6945-1c1d-48ce-be0a-1a6ac0107291"
print('Before oversampling distribution of target vector:')
print(y1.value_counts())

# %% id="DE0TNOXSCCHI"
# Encoding is used for categorical data
# We can consider 3,4,5,6,7,8 as categorical
# Use one hot encoding
y1 = pd.get_dummies(y1)
y_test = pd.get_dummies(y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="hAfwDdSVCCHN" outputId="2ee1c962-a4d9-4a0d-a3bd-e595d466ed3e"
y1.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="TY8Cge6OCCHO" outputId="0c0da743-72a4-4088-941c-a243c54f3198"
y_test.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="_gXyqVF5CCHO" outputId="480d829d-5251-45e1-e346-a38049b85689"
print(X1.shape,X_test.shape)
print(y1.shape,y_test.shape)

# %% id="LmDefPN0BMjS"
backend.clear_session()
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

# %% colab={"base_uri": "https://localhost:8080/"} id="_E0QfePFBMjT" outputId="1069f777-26b1-4892-a470-605511cee4a8"
model6 = KerasClassifier(build_fn=create_model_v5, verbose=1)

params = {'batch_size':[32, 64, 128],
          'lr':[0.01,0.1,0.001]}

gs = RandomizedSearchCV(estimator=model6, param_distributions=params, cv=5, verbose=10, n_jobs=2, random_state=0)
gs.fit(X1, y1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 332} id="sNjuhEHEBMjT" outputId="caea90d4-d277-4bdb-a4b1-3a7c19d4f925"
pd.DataFrame({'param': gs.cv_results_["params"], 
              'score mean': gs.cv_results_["mean_test_score"], 
              'score s.d.': gs.cv_results_["std_test_score"],
              'rank': gs.cv_results_["rank_test_score"]})

# %% colab={"base_uri": "https://localhost:8080/"} id="eEK39mueBMjT" outputId="06a92228-17b5-4695-f7c3-e2b0ad1e9240"
print('Best Parameters:', gs.best_params_, 'mean score: ', gs.best_score_, sep='\n')

# %% colab={"base_uri": "https://localhost:8080/"} id="RN7Fv9_lBMjT" outputId="8e3f1b7d-b600-49ae-c57e-f025a068d67b"
model6=create_model_v5(batch_size=32, lr=0.01)

model6.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="ziriifuJBMjT" outputId="599654e9-d624-43a5-99ad-d3ac1581e9b2"
optimizer = tf.keras.optimizers.Adam(0.001)
model6.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

history6=model6.fit(X1, y1, epochs=50, batch_size = 64, verbose=1,validation_split=0.2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 295} id="WhoSoKwqBMjT" outputId="14130446-c746-4006-d59f-1805733bfb1b"
#Plotting Train Loss vs Validation Loss
plt.plot(history6.history['loss'])
plt.plot(history6.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="mR6VR-IPu7hN" outputId="3e864ae1-7eaf-4ddc-934c-a00550bff98c"
# Capturing learning history per epoch
hist  = pd.DataFrame(history6.history)
hist['epoch'] = history6.epoch

# Plotting accuracy at different epochs
plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model1.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="SoQzHNVXBMjT" outputId="e8d06081-9db2-4f3f-df5c-daecde22ab9f"
y_pred=model6.predict(X_test)
y_pred

# %% colab={"base_uri": "https://localhost:8080/", "height": 830} id="yvwPGMyqBMjU" outputId="e30ee436-4a68-4f0b-d058-efeefc92463a"
# Classification Accuracy
print('Loss and Accuracy on Training data:',model6.evaluate(X1, y1))
print('Loss and Accuracy on Test data:',model6.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), labels=[0,1,2,3,4,5])
df_cm = pd.DataFrame(cm, index = [i for i in ['3','4','5','6','7','8']],
                  columns = [i for i in ['3','4','5','6','7','8']])
plt.figure(figsize = (10,8))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown] id="5f36f6dd"
# # Part-B: Solution

# %% [markdown] id="dedd939b"
# - **DOMAIN:** Autonomous Vehicles
# - **CONTEXT:** 
#  - A Recognising multi-digit numbers in photographs captured at street level is an important component of modern-day map making. A classic example of a corpus of such street-level photographs is Google’s Street View imagery composed of hundreds of millions of geo-located 360-degree panoramic images.
#
#  - The ability to automatically transcribe an address number from a geo-located patch of pixels and associate the transcribed number with a known street address helps pinpoint, with a high degree of accuracy, the location of the building it represents. More broadly, recognising numbers in photographs is a problem of interest to the optical character recognition community.
#
#  - While OCR on constrained domains like document processing is well studied, arbitrary multi-character text recognition in photographs is still highly challenging. This difficulty arises due to the wide variability in the visual appearance of text in the wild on account of a large range of fonts, colours, styles, orientations, and character arrangements.
#
#  - The recognition problem is further complicated by environmental factors such as lighting, shadows, specularity, and occlusions as well as by image acquisition factors such as resolution, motion, and focus blurs. In this project, we will use the dataset with images centred around a single digit (many of the images do contain some distractors at the sides). Although we are taking a sample of the data which is simpler, it is more complex than MNIST because of the distractors.
# - **DATA DESCRIPTION:** 
#  - The SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with the minimal requirement on data formatting but comes from a significantly harder, unsolved, real-world problem (recognising digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images.
# ![Picture1.png](attachment:Picture1.png)
#  - Where the labels for each of this image are the prominent number in that image i.e. 2,6,7 and 4 respectively.
#  - The dataset has been provided in the form of h5py files. You can read about this file format here: https://docs.h5py.org/en/stable/
#  - Acknowledgement: Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits in Natural Images with Unsupervised
#  - Feature Learning NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011. PDF
# http://ufldl.stanford.edu/housenumbers as the URL for this site.
# - **PROJECT OBJECTIVE:** To build a digit classifier on the SVHN (Street View Housing Number) dataset.

# %% id="9b11c05c"
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

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings("ignore")

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %% colab={"base_uri": "https://localhost:8080/"} id="qM2-0wcYgGXW" outputId="aa03bdc0-0430-4854-c028-37d0f97fe345"
# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')

# %% [markdown] id="a3313f55"
# ## 1. Data Import and Exploration:

# %% [markdown] id="fb411bd2"
# ### 1A. Read the .h5 file and assign to a variable.

# %% id="ff8fc39f"
# Read the data
# h5py package: a Python interface to the HDF5 scientific data format.
import h5py
df = h5py.File('/content/drive/MyDrive/MGL/Project-DL/Autonomous_Vehicles_SVHN_single_grey1.h5', 'r')

# %% [markdown] id="a0dc04f4"
# ### 1B. Print all the keys from the .h5 file.

# %% colab={"base_uri": "https://localhost:8080/"} id="1301611b" outputId="0a8687d8-b87a-41cc-c185-bf202a2de1ce"
df.keys()

# %% [markdown] id="97580215"
# ### 1C. Split the data into X_train, X_test, y_train, y_test.

# %% id="6e58227d"
X_train = df['X_train'][:]
X_test = df['X_test'][:]

y_train = df['y_train'][:]
y_test = df['y_test'][:]

# X_val = df['X_val'][:]
# y_val = df['y_val'][:]

# %% [markdown] id="ec3f440a"
# Check the contents of features and labels of one example from the dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="dc6dc2b6" outputId="925d2106-c7ff-41bc-dbc3-138b3fb2c4d0"
X_train[:1]

# %% colab={"base_uri": "https://localhost:8080/"} id="1ab6df66" outputId="65c360fb-31cd-49ba-94bd-f901b227148f"
X_test[:1]

# %% colab={"base_uri": "https://localhost:8080/"} id="11782e12" outputId="b9d0b76e-4cd7-4839-c143-71766c33e325"
y_train[:1]

# %% colab={"base_uri": "https://localhost:8080/"} id="917cf6a1" outputId="69cb7107-1a83-42bd-dc8a-44272a64467e"
y_test[:1]

# %% [markdown] id="5579308b"
# ## 2. Data Visualisation and preprocessing:

# %% [markdown] id="8c19ed0e"
# ### 2A. Print shape of all the 4 data split into x, y, train, test to verify if x & y is in sync.

# %% colab={"base_uri": "https://localhost:8080/"} id="4c72efef" outputId="5fcf9434-c41b-47c0-9841-1227d85f6a85"
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# print(X_val.shape,y_val.shape)

# %% [markdown] id="eee0ed1e"
# ### 2B. Visualise first 10 images in train data and print its corresponding labels.

# %% colab={"base_uri": "https://localhost:8080/", "height": 95} id="7a014062" outputId="24879835-58ea-4e0a-9317-d945b0c2c52e"
# Label of the most prominent number in image
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train[i], cmap="gray")
    plt.axis('off')
plt.show()

print('Label for each of the above images: %s' % (y_train[0:10]))

# %% [markdown] id="895e5f33"
# ### 2C. Reshape all the images with appropriate shape update the data in same variable.

# %% colab={"base_uri": "https://localhost:8080/"} id="c6f60cf7" outputId="291eb274-1a79-4508-a352-8f3e0a01fd71"
# Need to reshape the X_train and X_test so that the same can be fed for model building. 
# Currently we have a 3D tensor and we need to feed a 2D tensor into the model.

X_train = X_train.reshape(X_train.shape[0], 1024, 1)
X_test = X_test.reshape(X_test.shape[0], 1024, 1)

print('Resized Training set', X_train.shape, y_train.shape)
print('Resized Test set', X_test.shape, y_test.shape)

# %% [markdown] id="944ee94e"
# ### 2D. Normalise the images i.e. Normalise the pixel values.

# %% id="ab659401"
# To normalize the data; We can divide it by 255 (Grayscale image can take values from 0-255)
# Normalize inputs from 0-255 to 0-1 range
X_train = X_train / 255.0
X_test = X_test / 255.0

# %% [markdown] id="29ff4f16"
# ### 2E. Transform Labels into format acceptable by Neural Network.

# %% id="9779f441"
# Encoding is used for categorical data
# We can consider 1 to 10 as categorical
# Use one hot encoding
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="fbaa7c6b" outputId="f4e0bc87-41ca-4dd3-a8e3-72ef8cc4cbdd"
y_train.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="64018542" outputId="75c7d868-6e56-4d7c-b6ed-7d1b911d58c7"
y_test.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="311cb0fa" outputId="7e0f3772-6588-4346-b479-81c8c5c67b9c"
# Final dataset shape
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# %% [markdown] id="cac4d272"
# ### 2F. Print total Number of classes in the Dataset.

# %% colab={"base_uri": "https://localhost:8080/"} id="016ed97b" outputId="a99c5abe-55d0-4132-f666-104912c9a28c"
num_classes = y_test.shape[1] 
print("The number of classes in this dataset are:",num_classes)

# %% [markdown] id="c891804a"
# ## 3. Model Training & Evaluation using Neural Network

# %% [markdown] id="bae33074"
# - 3A. Design a Neural Network to train a classifier.
# - 3B. Train the classifier using previously designed Architecture (Use best suitable parameters).
# - 3C. Evaluate performance of the model with appropriate metrics.
# - 3D. Plot the training loss, validation loss vs number of epochs and training accuracy, validation accuracy vs number of epochs plot and write your observations on the same.

# %% [markdown] id="4f22afed"
# ### Model-1: Base Model

# %% id="26a29880"
# Define model1
import keras
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# create model1
model1 = Sequential()  
# Multiple Dense units with Relu activation
model1.add(Dense(256, activation='relu',kernel_initializer='he_uniform', input_dim = X_train.shape[1])) 
model1.add(Dense(64, activation='relu',kernel_initializer='he_uniform'))
model1.add(Dense(64, activation='relu',kernel_initializer='he_uniform'))
model1.add(Dense(32, activation='relu',kernel_initializer='he_uniform'))
# For multiclass classification Softmax is used 
model1.add(Dense(num_classes, activation='softmax')) 

# %% id="320edf16"
# Compile model1
# RMS_prop=optimizers.RMSprop()   
# We can similarly use different optimizers like RMSprop, Adagrad and SGD 
# Loss function = Categorical cross entropy
adam = optimizers.Adam(lr=1e-3)
model1.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy']) 

# %% colab={"base_uri": "https://localhost:8080/"} id="3b1f93e8" outputId="3743ec6e-b32b-4e4e-fc3f-4cb30d250f13"
# Looking into our base model1
model1.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="328e42f8" outputId="58f70e46-6e8a-43d8-8a4f-e82b73b4dd8e"
# Fit the model1
history1=model1.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=200, verbose=2)

# %% colab={"base_uri": "https://localhost:8080/"} id="d8f4bd17" outputId="76b6587d-4092-4212-dfa6-36ff009c100a"
# predicting the model1 on test data
y_pred=model1.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="1061ef17" outputId="7be1c991-69ff-4d44-f637-fe2b3fe6330b"
# Capturing learning history per epoch
hist  = pd.DataFrame(history1.history)
hist['epoch'] = history1.epoch

# Plotting accuracy at different epochs
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model1.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 283} id="77fd77ad" outputId="b9656daa-3396-4eb3-8279-59863866f096"
# Capturing learning history per epoch
hist  = pd.DataFrame(history1.history)
hist['epoch'] = history1.epoch

# Plotting accuracy at different epochs
plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model1.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 900} id="QGr_Ybgnw7xX" outputId="9244b3fb-8d8d-4ea4-d618-3852d4cfa1d7"
# Classification Accuracy
print('Loss and Accuracy on Training data:',model1.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model1.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), labels=[0,1,2,3,4,5,6,7,8,9])
df_cm = pd.DataFrame(cm, index = [i for i in ['0','1','2','3','4','5','6','7','8','9']],
                  columns = [i for i in ['0','1','2','3','4','5','6','7','8','9']])
plt.figure(figsize = (10,8))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown] id="DmZNeAV19Muk"
# **Evaluation metrics allow us to estimate errors to determine how well our models are performing:**
#
# > Accuracy: ratio of correct predictions over total predictions.
#
# > Precision: how often the classifier is correct when it predicts positive.
#
# > Recall: how often the classifier is correct for all positive instances.
#
# > F-Score: single measurement to combine precision and recall.
#
# **Considering the Class Recall, Precision, F1 Score and Accuracy as the most important parameters to decide the best model for this problem. We have the highest values in Model-3. Please refer the Model-3 for the same.**

# %% [markdown] id="c331fb1d"
# ### Model-2: Improving the Base Model
# Using Dropout and Batch Normalization

# %% id="9060cec0"
# Define model2

from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout

# create model2
model2 = Sequential()  
# Multiple Dense units with Relu activation
model2.add(Dense(256, activation='relu',kernel_initializer='he_uniform',input_dim=X_train.shape[1])) 
model2.add(Dense(64, activation='relu',kernel_initializer='he_uniform'))
model2.add(Dense(64, activation='relu',kernel_initializer='he_uniform'))
model2.add(Dense(32, activation='relu',kernel_initializer='he_uniform'))

model2.add(Dropout(0.5))
model2.add(BatchNormalization())
# For multiclass classification Softmax is used 
model2.add(Dense(num_classes, activation='softmax')) 

# %% id="e8e8f8e9"
# Compile model2
# RMS_prop=optimizers.RMSprop()   
# We can similarly use different optimizers like RMSprop, Adagrad and SGD 
# Loss function = Categorical cross entropy
adam = optimizers.Adam(lr=1e-3)
model2.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy']) 

# %% colab={"base_uri": "https://localhost:8080/"} id="25b3d977" outputId="c40d0544-cb52-4a3e-bf30-b0b9ae8a04a1"
# Looking into our base model2
model2.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="5478d6c2" outputId="8e6a8aa9-be24-4ee3-d45a-bf2b87c309bb"
# Fit the model2
history2=model2.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=200, verbose=2)

# %% colab={"base_uri": "https://localhost:8080/"} id="bd194b51" outputId="459cb72e-313e-46fe-aa26-0a591d1a406c"
# predicting the model2 on test data
y_pred=model2.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="dZOptWdANA5Q" outputId="09923e26-ba1d-406d-e77e-00a6169f3241"
# Capturing learning history per epoch
hist  = pd.DataFrame(history2.history)
hist['epoch'] = history2.epoch

# Plotting accuracy at different epochs
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model2.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="wTJQ04zdNA5R" outputId="df55ab4b-3970-43c2-e4a7-47ad4da9f6c6"
# Capturing learning history per epoch
hist  = pd.DataFrame(history2.history)
hist['epoch'] = history2.epoch

# Plotting accuracy at different epochs
plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model2.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 900} id="EFjD4ka4NA5R" outputId="37dcc147-e1e4-46f4-e841-fc463bb45a2d"
# Classification Accuracy
print('Loss and Accuracy on Training data:',model2.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model2.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), labels=[0,1,2,3,4,5,6,7,8,9])
df_cm = pd.DataFrame(cm, index = [i for i in ['0','1','2','3','4','5','6','7','8','9']],
                  columns = [i for i in ['0','1','2','3','4','5','6','7','8','9']])
plt.figure(figsize = (10,8))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown] id="nCcTDRUUUTVi"
# ### Model-3: Using Hyperparameter Tuning

# %% id="_QQ6QE2NulGv"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import random
from tensorflow.keras import backend
random.seed(1)
np.random.seed(1) 
tf.random.set_seed(1)


# %% id="eujCbgKnPdz9"
def create_model(lr,batch_size):  
    model= Sequential()
    model.add(BatchNormalization(input_shape=((1024,)))) 
    model.add(Dense(256,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))
    # compile the model
    sgd = optimizers.Adam(learning_rate = lr)
    model.compile(loss=losses.categorical_crossentropy,optimizer=sgd,metrics=['accuracy'])
    return model


# %% colab={"base_uri": "https://localhost:8080/"} id="h7sdjyqEOW4E" outputId="0bd44978-e7ea-4523-e558-b25967a56bfd"
model3 = KerasClassifier(build_fn=create_model, verbose=1)

params = {'batch_size':[32, 64, 128],
          'lr':[0.01,0.1,0.001]}

gs = RandomizedSearchCV(estimator=model3, param_distributions=params, cv=5, verbose=10, n_jobs=2, random_state=0)
gs.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/", "height": 332} id="QymSQwgKOZOQ" outputId="627d6a7e-8979-4605-92b7-242288ce9f70"
pd.DataFrame({'param': gs.cv_results_["params"], 
              'score mean': gs.cv_results_["mean_test_score"], 
              'score s.d.': gs.cv_results_["std_test_score"],
              'rank': gs.cv_results_["rank_test_score"]})

# %% colab={"base_uri": "https://localhost:8080/"} id="GqMJ4_anOZLn" outputId="b06927cf-67ae-4350-bf71-5771a9eab483"
print('Best Parameters:', gs.best_params_, 'mean score: ', gs.best_score_, sep='\n')

# %% colab={"base_uri": "https://localhost:8080/"} id="-hHCgs7cOZJI" outputId="a284abc2-792f-48a7-b996-3b3adbb68031"
model3=create_model(batch_size=64, lr=0.001)

model3.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="9cuPQidFOZGx" outputId="6b89e199-7466-4a5d-ba7a-b7cc269ebeb3"
optimizer = tf.keras.optimizers.Adam(0.001)
model3.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

history3=model3.fit(X_train, y_train, epochs=50, batch_size = 200, verbose=1, validation_split=0.2)

# %% colab={"base_uri": "https://localhost:8080/"} id="jGlfY-gSx7m1" outputId="fbb9d516-4aeb-405e-c08b-88100a600ad1"
# predicting the model1 on test data
y_pred=model3.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="pK7R_5D7x7m6" outputId="48396ddc-dcfb-4a5d-ad1d-c7235da0ca8a"
# Capturing learning history3 per epoch
hist  = pd.DataFrame(history3.history)
hist['epoch'] = history3.epoch

# Plotting accuracy at different epochs
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model3.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="xW6Q8hDGx7m6" outputId="ba9f6d1c-e6fa-45e3-df2f-7a432fc0321d"
# Capturing learning history3 per epoch
hist  = pd.DataFrame(history3.history)
hist['epoch'] = history3.epoch

# Plotting accuracy at different epochs
plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model3.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 900} id="uL6ObsRWx7m7" outputId="a749400c-a6b4-4631-97b7-de73de892b60"
# Classification Accuracy
print('Loss and Accuracy on Training data:',model3.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model3.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), labels=[0,1,2,3,4,5,6,7,8,9])
df_cm = pd.DataFrame(cm, index = [i for i in ['0','1','2','3','4','5','6','7','8','9']],
                  columns = [i for i in ['0','1','2','3','4','5','6','7','8','9']])
plt.figure(figsize = (10,8))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown] id="QX0HExXiVpIy"
# ### Model-4: Using Regularization Technique

# %% id="sx1xZolnWYGK"
#Importing Libraries
import tensorflow
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, optimizers
import math


# %% id="ltd1M9jPVsKh"
def model(iterations, lr, Lambda, verb=0, eval_test=False):
    scores=[]
    learning_rate=lr
    hidden_nodes=256
    output_nodes=10
    iterations=iterations
    # For early stopping of model.
    callbacks=tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    #model
    model = Sequential()
    model.add(Dense(500, input_shape=(1024,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(hidden_nodes,activation='relu'))
    model.add(Dense(hidden_nodes,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(hidden_nodes,activation='relu'))
    model.add(Dense(hidden_nodes,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(hidden_nodes,activation='relu'))
    model.add(Dense(hidden_nodes,activation='relu'))
    model.add(Dense(output_nodes, activation='softmax', kernel_regularizer=regularizers.l2(Lambda)))
    # adam optmizer with custom learning rate
    adam= optimizers.Adam(lr=learning_rate)
    #Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    #Fit the model
    model.fit(X_train,y_train, validation_data=(X_test,y_test),epochs=iterations,
              batch_size=500, verbose=verb, callbacks=[callbacks])
    
    if eval_test == True:
        score = model.evaluate(X_train,y_train, verbose=0)
        scores.append(score)
        score = model.evaluate(X_test,y_test, verbose=0)
        scores.append(score)
        score = model.evaluate(X_test,y_test, verbose=0)
        scores.append(score)
        return scores
    else:
        score = model.evaluate(X_test,y_test, verbose=(verb+1)%2)
        return score


# %% colab={"base_uri": "https://localhost:8080/"} id="jaB45zVVXVhp" outputId="a9c79629-2c1d-461c-aa87-63ce2a8cceac"
# Using low learning rate and zero regularization.
iterations = 1
lr=0.0001
Lambda=0
score=model(iterations, lr, Lambda)
print(f'\nLoss is {score[0]} and Accuracy is {score[1]}')

# %% colab={"base_uri": "https://localhost:8080/"} id="QuNtE0nNXVgI" outputId="b47d5635-b90c-4517-f467-e881d4035960"
# Increasing the learning Rate and zero regularization
iterations = 1
lr=1e3
Lambda=0
score=model(iterations, lr, Lambda)
print(f'\nLoss is {score[0]} and Accuracy is {score[1]}')

# %% colab={"base_uri": "https://localhost:8080/"} id="f6IlD8rSXVeP" outputId="add493b8-8c8d-47c6-f1bb-633ea35ccdde"
# Using regularization
iterations = 50
lr=1e-4
Lambda=1e-7
score=model(iterations, lr, Lambda)
print(f'Loss is {score[0]} and Accuracy is {score[1]}')

# %% colab={"base_uri": "https://localhost:8080/"} id="37SfYHLgXVaz" outputId="a7f701be-1a02-400b-a3b4-a7018413ae58"
iterations = 10
lr=2
Lambda=1e-2
score=model(iterations, lr, Lambda)
print(f'Loss is {score[0]} and Accuracy is {score[1]}')

# %% colab={"base_uri": "https://localhost:8080/"} id="6Gs-1db6XVXu" outputId="4767b401-5ad4-40e4-a20f-2e1e83a43167"
# Using random values in a range (Wide)
import math
results =[]
for i in range(10):
    lr=math.pow(10, np.random.uniform(-4.0,1.0))
    Lambda = math.pow(10, np.random.uniform(-7,-2))
    iterations = 30
    score=model(iterations, lr, Lambda)
    result=f'Loss is {score[0]} and Accuracy is {score[1]} with learing rate {lr} and Lambda {Lambda}\n'
    print(result)
    results.append(result)

# %% colab={"base_uri": "https://localhost:8080/"} id="QBt7DEG7XVVV" outputId="0d5e6f43-499b-4d21-a08d-4160c52c1f91"
# Using random values in a range (Narrow)
import math
results =[]
for i in range(10):
    lr=math.pow(10, np.random.uniform(-4.0,-2.0))
    Lambda = math.pow(10, np.random.uniform(-5,-3))
    iterations = 50
    score=model(iterations, lr, Lambda)
    result=f'Loss is {score[0]} and Accuracy is {score[1]} with learing rate {lr} and Lambda {Lambda}\n'
    print(result)
    results.append([result,[score[0],score[1],lr,Lambda]])

# %% id="FJ9tk2EqXVSM"
# Final Values
lr= 0.001346
Lambda= 0.000988
iterations = 100
eval_test= True
scores = model(iterations, lr, Lambda,verb=0, eval_test=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="iUTvWKObZIYY" outputId="8265ab78-64db-4daa-f4e6-e5a43b881f77"
print(f'Training Dataset Loss is {scores[0][0]} Accuracy is {scores[0][1]}\n')
print(f'Validation Dataset Loss is {scores[1][0]} Accuracy is {scores[1][1]}\n')
print(f'Test Dataset Loss is {scores[2][0]} Accuracy is {scores[2][1]}\n')

# %% [markdown] id="NYJl55kwJsIt"
# **Insights from above models:**
# - There are around 10 classes in the dataset which represent digits from 0-9.
#
# - We trained a Neural Network with dense hidden layers of different number of units and are able to achieve a final test accuracy of 83% (Using Hyperparameter Tuning).
#
# - Also we notice that after a certain point the model begins to overfit on our dataset as is clear from the plots above where the validation loss begins to increase after certain point and validation accuracy begins to decrease.
#
# - Thus, with this amount of accuracy we are able to distinguish between the different digits in this dataset.

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
# 17. [PapersWithCode](https://paperswithcode.com/)
