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

# %% [markdown] id="1Ran63qgZmhf"
# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#00b3e5;">Contents: Capstone Project
# 1. [Milestone-1](#Milestone-1)
# 2. [Milestone-2](#Milestone-2)

# %% [markdown] id="6iUFe9RPaQEM"
# # Problem Statement: Project Specs

# %% [markdown] id="ISw9F8Nyb_QP"
# - **DOMAIN:** Health Care
# - **CONTEXT:** Computer vision can be used in health care for identifying diseases. In Pneumonia detection we need to detect Inflammation
# of the lungs. In this challenge, you’re required to build an algorithm to detect a visual signal for pneumonia in medical
# images. Specifically, your algorithm needs to automatically locate lung opacities on chest radiographs.
# - **Data Description:**
#  - In the dataset, some of the features are labeled “Not Normal No Lung Opacity”. This extra third class indicates that while pneumonia was
# determined not to be present, there was nonetheless some type of abnormality on the image and oftentimes this finding may mimic the
# appearance of true pneumonia. 
#  - Dicom original images: - Medical images are stored in a special format called DICOM files (*.dcm). They
# contain a combination of header metadata as well as underlying raw image arrays for pixel data.
#  - Dataset has been attached along with this project. Please use the same for this capstone project.
#  - Original link to the dataset : https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data [ for your reference
# only ]. You can refer to the details of the dataset in the above link
#  - Acknowledgements: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview/acknowledgements.
# - **PROJECT OBJECTIVE:** Design a DL based algorithm for detecting pneumonia.
#
# - **Project Tasks: Milestone-1**
#  - Input: Context and Dataset
#  - Step 1: Import the data.
#  - Step 2: Map training and testing images to its classes.
#  - Step 3: Map training and testing images to its annotations.
#  - Step 4: Preprocessing and Visualisation of different classes
#  - Step 5: Display images with bounding box.
#  - Step 6: Design, train and test basic CNN models for classification.
#  - Step 7: Interim report
# - Submission: Interim report, Jupyter Notebook with all the steps in Milestone-1
# - **Project Tasks: Milestone-2**
#  - Input: Preprocessed output from Milestone-1
#  - Step 1: Fine tune the trained basic CNN models for classification.
#  - Step 2: Apply Transfer Learning model for classification.
#  - Step 3: Design, train and test RCNN & its hybrids based object detection models to impose the bounding box or mask over the area of interest.
#  - Step 4: Pickle the model for future prediction.
#  - Step 5: Final Report
# - Submission: Final report, Jupyter Notebook with all the steps in Milestone-1 and Milestone-2

# %% [markdown] id="RPLKgd03UcAo"
# # Milestone-1

# %% colab={"base_uri": "https://localhost:8080/"} id="n6Uq0QdizSj8" outputId="2a8faf70-1dc1-4d70-ee9c-3752c745d7b6"
# Notebook timestamp
from datetime import datetime
dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f"Updated {dt_string} (GMT)")

# %% id="Ibf2TzQHvC6n"
# NB start time
import time
a = time.time()

# %% colab={"base_uri": "https://localhost:8080/"} id="uqC6egREfeu8" outputId="5a1ddad9-c4aa-45b9-aff0-dc53b8480963"
# !pip install ipython-autotime
# %load_ext autotime

# %% colab={"base_uri": "https://localhost:8080/"} id="QCsp7E0U3ksH" outputId="774eee32-c771-451e-ca74-3ff7b21611d8"
# Import all the relevant libraries needed to complete the analysis, visualization, modeling and presentation
import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
sns.set_style('darkgrid')

from scipy import stats
from scipy.stats import zscore

from sklearn import preprocessing
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
from sklearn.metrics import f1_score
# from sklearn.metrics import plot_precision_recall_curve, average_precision_score


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC

# from sklearn.decomposition import PCA
# from scipy.cluster.hierarchy import dendrogram, linkage
# from scipy.cluster.hierarchy import fcluster
# from sklearn.cluster import KMeans 
# from sklearn.metrics import silhouette_samples, silhouette_score

# import xgboost as xgb
# from xgboost import plot_importance
# from lightgbm import LGBMClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# import re
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.stem.snowball import SnowballStemmer
# import pandas_profiling as pp

# import gensim
# import logging

# import cv2
# from google.colab.patches import cv2_imshow
# from glob import glob
# import itertools

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Conv2DTranspose
from tensorflow.keras.layers import Activation, GlobalMaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate, Reshape
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K

from tensorflow import keras
from keras.utils.np_utils import to_categorical  
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

import warnings
warnings.filterwarnings("ignore")

import random
from zipfile import ZipFile

# Import pickle Package
import pickle

# Set random_state
random_state = 42

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %% colab={"base_uri": "https://localhost:8080/"} id="8y-qaIh_5yJv" outputId="b3a65b94-6c37-4a73-ba5a-90921c973510"
# Install pydicom package
# !pip install pydicom

# %% colab={"base_uri": "https://localhost:8080/"} id="ckyuzld_5CP3" outputId="a5930962-0404-40f5-b7cd-b1d59530ca5a"
# Import other relevant packages
from numpy import asarray

import cv2
import glob
import time
import pydicom as dcm
import skimage
import math

from matplotlib.patches import Rectangle
from skimage import feature, filters
from PIL import Image

from functools import partial
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm, tqdm_notebook

# Tensorflow / Keras
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from keras import models
from keras import layers

from tensorflow.keras.applications import MobileNet, MobileNetV2, VGG19, EfficientNetV2B3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input 
import tensorflow.keras.utils as pltUtil
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input as resnetProcess_input

sns.set_style('whitegrid')
np.warnings.filterwarnings('ignore')

# %% colab={"base_uri": "https://localhost:8080/"} id="ie8jaqaw1fZw" outputId="38435f43-c517-4901-a9af-ef0e0fefd9df"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="PBtTvUXO1fYZ" outputId="39d64e20-3f9a-4e6a-860c-eb5276c5f6e6"
# Current working directory
# %cd "/content/drive/MyDrive/MGL/Project-Capstone/"

# # List all the files in a directory
# for dirname, _, filenames in os.walk('path'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# %% colab={"base_uri": "https://localhost:8080/"} id="ug6w7boO18Ae" outputId="7c8ffe63-cdc1-408c-97dd-4bcc4dcc7149"
# # Path of the data file
# path = 'rsna-pneumonia-detection-challenge.zip'

# # Unzip files in the current directory

# with ZipFile (path,'r') as z:
#   z.extractall() 
# print("Training zip extraction done!")

# %% colab={"base_uri": "https://localhost:8080/"} id="4c-ieSHs1fUL" outputId="7e48368b-0d4c-4543-bddf-d333f240ec08"
# List files in the directory
# !ls

# %% [markdown] id="Lbl1MiuBlAvq"
# ## EXPLORATORY DATA ANALYSIS

# %% [markdown] id="qDkhZL1BbzM2"
# ## Load the data

# %% colab={"base_uri": "https://localhost:8080/"} id="4iJ_CFaS0XR_" outputId="3e421e92-257d-47a6-c91b-724e426c5c7d"
class_info_df = pd.read_csv('stage_2_detailed_class_info.csv')
train_labels_df = pd.read_csv('stage_2_train_labels.csv') 

# %% colab={"base_uri": "https://localhost:8080/"} id="jpG1HGsJ-cC8" outputId="1da479ea-1cbd-4f85-a5c9-03f666b7732a"
print(f"Detailed class info -  Rows: {class_info_df.shape[0]}, Columns: {class_info_df.shape[1]}")
print(f"Train labels -  Rows: {train_labels_df.shape[0]}, Columns: {train_labels_df.shape[1]}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 380} id="IthjVitHAfzP" outputId="1dce21fb-9482-45bb-e0b6-3405729f274a"
class_info_df.sample(10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 380} id="0c6wgqgSAkte" outputId="82e3ac17-f20f-4c25-dae6-8daf381c7169"
train_labels_df.sample(10)


# %% colab={"base_uri": "https://localhost:8080/", "height": 129} id="eiOPP0TIAkq0" outputId="7cbfd53b-d5a7-4feb-ea70-bd4cfa885fb3"
# Understand the missing data points
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return np.transpose(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']))

missing_data(train_labels_df)

# %% colab={"base_uri": "https://localhost:8080/", "height": 129} id="oJrO2117AkoY" outputId="b60a27b0-692b-4239-b2b9-9a9e8fa02cde"
missing_data(class_info_df)

# %% colab={"base_uri": "https://localhost:8080/"} id="rjymWhLDvkvj" outputId="9badba8f-edf4-431c-c3bb-e560ce65c476"
# Duplicate patients in the class_info_df dataset
duplicateClassRowsDF = class_info_df[class_info_df.duplicated(['patientId'])]
duplicateClassRowsDF.shape

## There are 3543 duplicates similar to the train_labels_df dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 129} id="pv0TbX51vke7" outputId="bf9096fc-9c77-455f-d64c-9eb41a44a554"
duplicateClassRowsDF.head(2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 129} id="IQ-R4WELvkdD" outputId="de7f18aa-5454-4325-c86d-dae6f4143dd1"
## The same patient id has the same class even though they are duplicate
class_info_df[class_info_df.patientId=='00704310-78a8-4b38-8475-49f4573b2dbb']

# %% colab={"base_uri": "https://localhost:8080/"} id="i29AYPsHgclg" outputId="b0a5f77a-b6f3-4327-bd8b-8688a5ffb907"
# Duplicate patients in the train_labels_df dataset
duplicateRowsDF = train_labels_df[train_labels_df.duplicated(['patientId'])]
duplicateRowsDF.shape

## There are 3543 duplicates similar to the class_info_df dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 129} id="8hPP6MI1gclh" outputId="3a948214-3db4-42a4-a8c9-df9a604c79e0"
duplicateRowsDF.head(2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 129} id="i9dvl8WXgclh" outputId="962744eb-a1bd-4745-a88e-8860c8448e84"
## Examining one of the patient id which is duplicate , we can see that the x,y, widht and height is not the same
## This indicates that the same patient has two bounding boxes in the same dicom image
train_labels_df[train_labels_df.patientId=='00436515-870c-4b36-a041-de91049b9ab4']

# %% colab={"base_uri": "https://localhost:8080/"} id="WkipgQkQB7FI" outputId="cf0c5f22-53b1-4df1-fff2-114ad6982701"
# Clear the matplotlib plotting backend
# %matplotlib inline
plt.close('all')

# %% colab={"base_uri": "https://localhost:8080/", "height": 475} id="xCpCHyyoB7Ci" outputId="c291e6b5-b9d2-4564-bc64-857973b1e3a9"
# Understand the 'class' variable
f,axes=plt.subplots(1,2,figsize=(17,7))
class_info_df['class'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('class',data=class_info_df,ax=axes[1])
axes[0].set_title('Pie Chart for class')
axes[1].set_title('Bar Graph for class')
plt.show()

# %% [markdown] id="CJA1ue8EyG7d"
# ## Merge the data

# %% colab={"base_uri": "https://localhost:8080/"} id="2HPSZ__2B7AM" outputId="9d026786-2014-472b-99a5-c0de7987c432"
# Let's merge the two datasets using Patient ID as the merge criteria.
train_class_df = train_labels_df.merge(class_info_df, left_on='patientId', right_on='patientId', how='inner')

# %% colab={"base_uri": "https://localhost:8080/"} id="JEyrDxeEFVvR" outputId="c0743be1-2e28-45b3-e2b0-ecafeea2ba8f"
train_class_df.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 224} id="rmfyWc32FVik" outputId="81b2aaf3-6294-4ead-ac44-e9fc6ac936df"
train_class_df.sample(5)

# %% colab={"base_uri": "https://localhost:8080/", "height": 421} id="PkrOhsiqGAkh" outputId="ff17a05a-e9ec-428d-c51c-c38926460900"
# Let's plot the number of examinations for each class detected, grouped by Target value.
fig, ax = plt.subplots(nrows=1,figsize=(12,6))
tmp = train_class_df.groupby('Target')['class'].value_counts()
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
sns.barplot(ax=ax,x = 'Target', y='Exams',hue='class',data=df)
plt.title("Chest exams class and Target")
plt.show()

# %% [markdown] id="Ajl91x-gzeoY"
# All chest examinations with Target = 1 (pathology detected) are associated with class: Lung Opacity.
#
# The chest examinations with Target = 0 (no pathology detected) are either of class: Normal or class: No Lung Opacity/Not Normal.
#
# Make a note for modeling: we are using the 2 classes instead of 3.

# %% colab={"base_uri": "https://localhost:8080/", "height": 752} id="0i8LoabEGAiA" outputId="1f96e69f-ee7a-48d3-d8b7-1d6d3be3326a"
# Density plots for class - Lung Opacity (Target class = 1)
target1 = train_class_df[train_class_df['Target']==1]
sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(2,2,figsize=(12,12))
sns.distplot(target1['x'],kde=True,bins=50, color="red", ax=ax[0,0])
sns.distplot(target1['y'],kde=True,bins=50, color="blue", ax=ax[0,1])
sns.distplot(target1['width'],kde=True,bins=50, color="green", ax=ax[1,0])
sns.distplot(target1['height'],kde=True,bins=50, color="magenta", ax=ax[1,1])
locs, labels = plt.xticks()
plt.tick_params(axis='both', which='major', labelsize=12)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="PVrKOuaAGAfd" outputId="81a50c50-6ab9-4ac7-c683-ee49c6febedb"
# We will show a sample of center points (xc, yc) superimposed with the corresponding sample of the rectangles.
fig, ax = plt.subplots(1,1,figsize=(7,7))
target_sample = target1.sample(2000)
target_sample['xc'] = target_sample['x'] + target_sample['width'] / 2
target_sample['yc'] = target_sample['y'] + target_sample['height'] / 2
plt.title("Centers of Lung Opacity rectangles (brown) over rectangles (yellow)\nSample size: 2000")
target_sample.plot.scatter(x='xc', y='yc', xlim=(0,1024), ylim=(0,1024), ax=ax, alpha=0.8, marker=".", color="brown")
for i, crt_sample in target_sample.iterrows():
    ax.add_patch(Rectangle(xy=(crt_sample['x'], crt_sample['y']),
                width=crt_sample['width'],height=crt_sample['height'],alpha=3.5e-3, color="yellow"))
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 359} id="J0YBuNd4dcQY" outputId="27c9f429-b75c-4fc7-8364-1fa0e42a747a"
corr = train_class_df.corr()
plt.figure(figsize=(12,5))

sns.heatmap(corr,annot=True)

## There is high corelation between widht and height

# %% [markdown] id="oSZXmwnVNOTe"
# ## Explore the DICOM data

# %% colab={"base_uri": "https://localhost:8080/"} id="iQ3ar-7qGAct" outputId="7cbdceb8-6833-47fe-8a11-f55e23001766"
# The files names are the patients IDs.
image_sample_path = os.listdir('stage_2_train_images')[:5]
print(image_sample_path)

# GDrive I/O Problem; Run after

# %% colab={"base_uri": "https://localhost:8080/"} id="ncyPT-_RGAaX" outputId="61fd9b84-efd9-48be-ddd2-a474307ec4c5"
image_train_path = os.listdir('stage_2_train_images')
image_test_path = os.listdir('stage_2_test_images')
print("Number of images in train set:", len(image_train_path),"\nNumber of images in test set:", len(image_test_path))

# %% [markdown] id="t8PmD5s02MxC"
# We have a reduced number of images in the training set (26684), compared with the number of images in the train_df data (30227).
#
# We may have duplicated entries in the train and class datasets.

# %% colab={"base_uri": "https://localhost:8080/"} id="8GkCN_ZmGAXv" outputId="e8a93f10-2f3a-4a02-865e-dc31d789a064"
print("Unique patientId in  train_class_df: ", train_class_df['patientId'].nunique())

# %% [markdown] id="GSNSOvy_2oBb"
# It confirms that the number of unique patientId's are equal with the number of DICOM images in the train set.

# %% colab={"base_uri": "https://localhost:8080/"} id="zn0h9f9dbyAu" outputId="c275ef38-1d0a-4186-f45a-78469e3d88c4"
# Duplicate patients in the dataset
duplicateRowsDF = train_class_df[train_class_df.duplicated(['patientId'])]
duplicateRowsDF.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 129} id="bkj3oHHSbx3U" outputId="0bc941bb-458b-4b6c-9684-e4610c396587"
duplicateRowsDF.head(2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 192} id="6NWYAwZUcNi8" outputId="f814ba45-3562-49ec-d627-8b93727e4ed5"
## Examining one of the patient id which is duplicate , we can see that the x,y, widht and height is not the same
## This indicates that the same patient has two bounding boxes in the same dicom image
train_class_df[train_class_df.patientId=='00436515-870c-4b36-a041-de91049b9ab4']

# %% colab={"base_uri": "https://localhost:8080/", "height": 192} id="RYUu-hbocNf-" outputId="c806376d-8643-4495-f317-69725da50e64"
train_class_df[train_class_df.patientId=='00704310-78a8-4b38-8475-49f4573b2dbb']

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} id="X_9iWpw9GAVY" outputId="0eaffa35-aafa-4fc6-ce07-0afdde98eaf2"
# Understand the target, class and duplicate entries
tmp = train_class_df.groupby(['patientId','Target', 'class'])['patientId'].count()
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
tmp = df.groupby(['Exams','Target','class']).count()
df2 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
df2.columns = ['Exams', 'Target','Class', 'Entries']
df2

# %% colab={"base_uri": "https://localhost:8080/", "height": 421} id="NMLvZFXDGAS0" outputId="151603ba-2377-4110-97f9-86144345bfd0"
fig, ax = plt.subplots(nrows=1,figsize=(12,6))
sns.barplot(ax=ax,x = 'Target', y='Entries', hue='Exams',data=df2)
plt.title("Chest exams class and Target")
plt.show()

# %% [markdown] id="sMv3Cxg6PyQo"
# ## Dicom metadata

# %% colab={"base_uri": "https://localhost:8080/"} id="5nsFM7ApGAQ4" outputId="7dacbeb9-9adc-498a-f626-3ea5392114cf"
samplePatientID = list(train_class_df[:3].T.to_dict().values())[0]['patientId']
samplePatientID = samplePatientID+'.dcm'
dicom_file_path = os.path.join("stage_2_train_images/",samplePatientID)
dicom_file_dataset = dcm.read_file(dicom_file_path)
dicom_file_dataset


# %% [markdown] id="OANhCRci3UfP"
# We can see that some attributes may have a higher predictive value:
#
# - Patient sex;
# - Patient age;
# - Modality;
# - Body part examined;
# - View position;
# - Rows & Columns;
# - Pixel Spacing.

# %% [markdown] id="X7UfrlpK36B-"
# ## Plot DICOM images

# %% colab={"base_uri": "https://localhost:8080/"} id="ucSiz2X9GAPV" outputId="9a2461da-45ad-4868-c2f6-71364c886bf8"
# Plot DICOM images with Target = 1
def show_dicom_images(data):
    img_data = list(data.T.to_dict().values())
    f, ax = plt.subplots(3,3, figsize=(16,18))
    for i,data_row in enumerate(img_data):
        patientImage = data_row['patientId']+'.dcm'
        imagePath = os.path.join("stage_2_train_images/",patientImage)
        data_row_img_data = dcm.read_file(imagePath)
        modality = data_row_img_data.Modality
        age = data_row_img_data.PatientAge
        sex = data_row_img_data.PatientSex
        data_row_img = dcm.dcmread(imagePath)
        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title('ID: {}\nModality: {} Age: {} Sex: {} Target: {}\nClass: {}\nWindow: {}:{}:{}:{}'.format(
                data_row['patientId'],
                modality, age, sex, data_row['Target'], data_row['class'], 
                data_row['x'],data_row['y'],data_row['width'],data_row['height']))
    plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="cyRxnFTHGALv" outputId="def611d6-8bdb-41cd-ffcc-786ce30fdd76"
show_dicom_images(train_class_df[train_class_df['Target']==1].sample(9))


# %% colab={"base_uri": "https://localhost:8080/"} id="23pp1M_KGAJx" outputId="f3d5387c-ed9b-4f44-96f1-7dc0c19c6329"
# Images with bounding boxes
def show_dicom_images_with_boxes(data):
    img_data = list(data.T.to_dict().values())
    f, ax = plt.subplots(3,3, figsize=(16,18))
    for i,data_row in enumerate(img_data):
        patientImage = data_row['patientId']+'.dcm'
        imagePath = os.path.join("stage_2_train_images/",patientImage)
        data_row_img_data = dcm.read_file(imagePath)
        modality = data_row_img_data.Modality
        age = data_row_img_data.PatientAge
        sex = data_row_img_data.PatientSex
        data_row_img = dcm.dcmread(imagePath)
        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title('ID: {}\nModality: {} Age: {} Sex: {} Target: {}\nClass: {}'.format(
                data_row['patientId'],modality, age, sex, data_row['Target'], data_row['class']))
        rows = train_class_df[train_class_df['patientId']==data_row['patientId']]
        box_data = list(rows.T.to_dict().values())
        for j, row in enumerate(box_data):
            ax[i//3, i%3].add_patch(Rectangle(xy=(row['x'], row['y']),
                        width=row['width'],height=row['height'], 
                        color="yellow",alpha = 0.1))   
    plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="m4cIaQUSGAHA" outputId="dff3d9e8-209c-4d49-840c-5647ca9fc9d2"
show_dicom_images_with_boxes(train_class_df[train_class_df['Target']==1].sample(9))

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="R4-g0uqBGAEi" outputId="fea61c88-7e68-4bd8-b747-faf1b69056a2"
# Plot DICOM images with Target = 0
show_dicom_images(train_class_df[train_class_df['Target']==0].sample(9))

# %% [markdown] id="F327k_wzcSKf"
# ## Add meta information from the dicom data

# %% [markdown] id="S79g4_3Z4rr1"
# ### Train dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="RWoZGxdIZAWE" outputId="b875baee-8d5a-41b8-9fd0-ddd6ed739790"
vars = ['Modality', 'PatientAge', 'PatientSex', 'BodyPartExamined', 'ViewPosition', 'ConversionType', 'Rows', 'Columns', 'PixelSpacing']

def process_dicom_data(data_df, data_path):
    for var in vars:
        data_df[var] = None
    image_names = os.listdir(data_path)
    for i, img_name in tqdm_notebook(enumerate(image_names)):
        imagePath = os.path.join(data_path,img_name)
        data_row_img_data = dcm.read_file(imagePath)
        idx = (data_df['patientId']==data_row_img_data.PatientID)
        data_df.loc[idx,'Modality'] = data_row_img_data.Modality
        data_df.loc[idx,'PatientAge'] = pd.to_numeric(data_row_img_data.PatientAge)
        data_df.loc[idx,'PatientSex'] = data_row_img_data.PatientSex
        data_df.loc[idx,'BodyPartExamined'] = data_row_img_data.BodyPartExamined
        data_df.loc[idx,'ViewPosition'] = data_row_img_data.ViewPosition
        data_df.loc[idx,'ConversionType'] = data_row_img_data.ConversionType
        data_df.loc[idx,'Rows'] = data_row_img_data.Rows
        data_df.loc[idx,'Columns'] = data_row_img_data.Columns  
        data_df.loc[idx,'PixelSpacing'] = str.format("{:4.3f}",data_row_img_data.PixelSpacing[0]) 


# %% colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["bf45a2343940464cb78775d62b778b66", "26df0f4f858940ba84fa25840deb1a06", "4caaf992f50d49f683675ace3bad2633", "04d478caf5d847868f33a07f1afc9f0a", "88519ede8e8d4fceaa0663bc9d534ee8", "b51cabb713704ca2bdd825d37cd2238f", "535c77efecd4452a9c71000c02ac1d1f", "a2963009abe741f08d9df3c02859f4f4", "8855d5c29c6947129a8f7b8183130b9e", "5ee56c53683d4bc3bd02a6feb9095119", "e97aa43bbd804d9b87780ebd7be9c47c"]} id="PVpE9o9RZATe" outputId="8d71bbff-7782-426c-8657-5dd9a450610b"
process_dicom_data(train_class_df,'stage_2_train_images/')

# max time = 24 min; Run the notebook again if it takes more time

# %% colab={"base_uri": "https://localhost:8080/"} id="orwhl8SHLH2S" outputId="e4b3df77-d548-4dcd-8f32-62861f147faf"
train_class_df.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="W6mc9FE7hEiE" outputId="41bae586-19c9-4563-a834-30b849b50c97"
train_class_df.sample(10)

# %% [markdown] id="j2Pb46Lq4zBj"
# ### Test dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="o9fEH6aBZARL" outputId="dd9a4fd9-55f5-4a31-9791-091ca6dfce65"
test_class_df = pd.read_csv('stage_2_sample_submission.csv')

# %% colab={"base_uri": "https://localhost:8080/"} id="qIQtvbFFKW5C" outputId="28365102-1403-4cd7-ab9c-eb4112d4ebbb"
test_class_df.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 380} id="rAb7N8JzZAPO" outputId="de7a1eaa-04d7-417b-d2fc-aae4d05098ac"
test_class_df.sample(10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["45e955e8593842958669d615ae8db634", "a1b4346d2e1949069024ea36bc2f7cb0", "68cded045e874c5c807ba2c6ecc7549d", "84d4ac7da1424becb16d2c3ae0375697", "4b8d54cf108a48f09165ffb7decf6660", "6eeee5a0510c4d9ca2b16191671399f5", "62e9e9917f0d4d62aff04d55024cff41", "fa9a9fafe16b4256be7e8b6644a584b2", "2096287bcdf94ce796a6854ce8ce784c", "e6af5747525a41df8913f7fea78c1aa1", "1b91de72602f4a0aab1b91c54f57df85"]} id="xQgdxqZEZANb" outputId="9aa19d2d-d9a3-469c-d798-f457d3658278"
test_class_df = test_class_df.drop('PredictionString',1)
process_dicom_data(test_class_df,'stage_2_test_images/')

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="b5RvrsEHZAKc" outputId="7f3c88db-5add-4efc-cc15-9546d8f19842"
test_class_df.sample(10)

# %% [markdown] id="Dh6RuX3A47Yq"
# ## Modality Attribute

# %% colab={"base_uri": "https://localhost:8080/"} id="Z2BHMwbOZAIm" outputId="1d6994e9-a55e-41f3-dcb8-b497a011582a"
print("Modalities: train:",train_class_df['Modality'].unique(), "test:", test_class_df['Modality'].unique())

# %% [markdown] id="SuakOPiN5IEn"
# ## BodyPartExamined Attribute

# %% colab={"base_uri": "https://localhost:8080/"} id="5L9pXG0gZAGl" outputId="26b0ca47-94b3-40dc-91e2-d6bea67e7662"
print("Body Part Examined: train:",train_class_df['BodyPartExamined'].unique(), "test:", test_class_df['BodyPartExamined'].unique())

# %% colab={"base_uri": "https://localhost:8080/"} id="xhJVmDwf06YJ" outputId="569d4fd1-808c-40b8-f939-7d4be88009e0"
print("Body Part Examined (train):", train_class_df['BodyPartExamined'].value_counts())
print()
print("Body Part Examined (test):", test_class_df['BodyPartExamined'].value_counts())

# %% [markdown] id="Mvc0dPKt5O-p"
# ## ViewPosition Attribute

# %% colab={"base_uri": "https://localhost:8080/"} id="LSIwa40aZABZ" outputId="4ba3bc31-1f97-4d10-88b3-2a9de77c3618"
print("View Position: train:",train_class_df['ViewPosition'].unique(), "test:", test_class_df['ViewPosition'].unique())

# %% [markdown] id="AoYSerN95bBv"
# ### Train dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 475} id="HGiGrOyFY_-5" outputId="fcb1d5f5-4d9a-4dad-e58b-e835cac9cf9d"
f,axes=plt.subplots(1,2,figsize=(17,7))
train_class_df['ViewPosition'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('ViewPosition',data=train_class_df,ax=axes[1])
axes[0].set_title('Pie Chart for ViewPosition')
axes[1].set_title('Bar Graph for ViewPosition')
plt.show()


# %% colab={"base_uri": "https://localhost:8080/"} id="ajY1lRoKY_8I" outputId="cc0071ab-2daa-4689-ea84-113e633d53d5"
def plot_window(data, color_point, color_window, text):
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    plt.title("Centers of Lung Opacity rectangles over rectangles\n{}".format(text))
    data.plot.scatter(x='xc', y='yc', xlim=(0,1024), ylim=(0,1024), ax=ax, alpha=0.8, marker=".", color=color_point)
    for i, crt_sample in data.iterrows():
        ax.add_patch(Rectangle(xy=(crt_sample['x'], crt_sample['y']),
            width=crt_sample['width'],height=crt_sample['height'],alpha=3.5e-3, color=color_window))
    plt.show()


# %% colab={"base_uri": "https://localhost:8080/"} id="kC7pUh03jghC" outputId="5a67e0d7-4b8d-4f33-a2cf-ec3adb997c67"
target1 = train_class_df[train_class_df['Target']==1]

target_sample = target1.sample(2000)
target_sample['xc'] = target_sample['x'] + target_sample['width'] / 2
target_sample['yc'] = target_sample['y'] + target_sample['height'] / 2

target_ap = target_sample[target_sample['ViewPosition']=='AP']
target_pa = target_sample[target_sample['ViewPosition']=='PA']

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="HSU2-fxMjge9" outputId="1c7b1e8c-9e29-4c25-adcb-ecb8ea3143dd"
plot_window(target_ap,'green', 'yellow', 'Patient View Position: AP')

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="9yjh-lQXjgbr" outputId="2b27d9b8-45c2-400f-8798-bb64559f2a76"
plot_window(target_pa,'blue', 'red', 'Patient View Position: PA')

# %% [markdown] id="3nmYOzZW5lNB"
# ### Test dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 475} id="OzxR2Di5jgY3" outputId="83349a5d-9133-43fa-ad54-2d742011c0c5"
f,axes=plt.subplots(1,2,figsize=(17,7))
test_class_df['ViewPosition'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('ViewPosition',data=test_class_df,ax=axes[1])
axes[0].set_title('Pie Chart for ViewPosition')
axes[1].set_title('Bar Graph for ViewPosition')
plt.show()

# %% [markdown] id="3gEYR11r5xlm"
# ## ConversionType Attribute

# %% colab={"base_uri": "https://localhost:8080/"} id="oZj90eg1jgWo" outputId="104f92a1-c0ec-43fd-d6f2-b1b6248f6761"
print("Conversion Type: train:",train_class_df['ConversionType'].unique(), "test:", test_class_df['ConversionType'].unique())

# %% colab={"base_uri": "https://localhost:8080/"} id="6wkkrYH_2Jgi" outputId="f6783753-fc77-4756-9026-6a7857816b86"
print("Conversion Type (train):", train_class_df['ConversionType'].value_counts())
print()
print("Conversion Type (test):", test_class_df['ConversionType'].value_counts())

# %% [markdown] id="LM0rfTaE56ut"
# ## Rows and Columns

# %% colab={"base_uri": "https://localhost:8080/"} id="9KturLiSjxsN" outputId="4190863f-2e84-4f5e-b400-ba500ae723e6"
print("Rows: train:",train_class_df['Rows'].unique(), "test:", test_class_df['Rows'].unique())
print("Columns: train:",train_class_df['Columns'].unique(), "test:", test_class_df['Columns'].unique())

# %% colab={"base_uri": "https://localhost:8080/"} id="GmFgE6eA2hza" outputId="28046d15-3b71-448d-a4fa-beb2279c9ad9"
print("Rows: train:", train_class_df['Rows'].value_counts())
print()
print("Columns: train:", test_class_df['Rows'].value_counts())

# %% colab={"base_uri": "https://localhost:8080/"} id="G3MbfjE231hT" outputId="3a94e9bd-63d6-4344-dae3-8d139535bfb5"
print("Rows: train:", train_class_df['Columns'].value_counts())
print()
print("Columns: train:", test_class_df['Columns'].value_counts())

# %% [markdown] id="4mW-FapTkc3i"
# ## PatientAge Attribute

# %% [markdown] id="R9LFulluklcN"
# ### Train dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="SqlaIvQGjxpn" outputId="291db91a-9a07-4a3f-f1bc-415da74bbe81"
tmp = train_class_df.groupby(['Target', 'PatientAge'])['patientId'].count()
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
tmp = df.groupby(['Exams','Target', 'PatientAge']).count()
df2 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()

# %% colab={"base_uri": "https://localhost:8080/"} id="ZMge-wIxjxmq" outputId="ee1897b3-f64a-4cf6-ebbe-aa98e23c2615"
tmp = train_class_df.groupby(['class', 'PatientAge'])['patientId'].count()
df1 = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
tmp = df1.groupby(['Exams','class', 'PatientAge']).count()
df3 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="JEIp1ak3jxkD" outputId="91abb2c2-7a2e-4d74-a855-9fe3036fbfe2"
fig, (ax) = plt.subplots(nrows=1,figsize=(16,6))
sns.barplot(ax=ax, x = 'PatientAge', y='Exams', hue='Target',data=df2)
plt.title("Train set: Chest exams Age and Target")
plt.xticks(rotation=90)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="umMG-xJXk1Kl" outputId="0a52c907-9257-4d3c-97a7-9646981da130"
fig, (ax) = plt.subplots(nrows=1,figsize=(16,6))
sns.barplot(ax=ax, x = 'PatientAge', y='Exams', hue='class',data=df3)
plt.title("Train set: Chest exams Age and class")
plt.xticks(rotation=90)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="gcRLcH17k1I9" outputId="518e4208-ffc4-4b5a-a7d9-f4c441d69f0c"
target_age1 = target_sample[target_sample['PatientAge'] < 20]
target_age2 = target_sample[(target_sample['PatientAge'] >=20) & (target_sample['PatientAge'] < 35)]
target_age3 = target_sample[(target_sample['PatientAge'] >=35) & (target_sample['PatientAge'] < 50)]
target_age4 = target_sample[(target_sample['PatientAge'] >=50) & (target_sample['PatientAge'] < 65)]
target_age5 = target_sample[target_sample['PatientAge'] >= 65]

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="LoY5uzJak1Gu" outputId="02aa11c8-e9f3-4db2-f8a7-99a584813094"
plot_window(target_age1,'blue', 'red', 'Patient Age: 1-19 years')

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="Pu3A2D17k1E-" outputId="06c34162-3f0d-4003-d75a-8b55465582cd"
plot_window(target_age2,'blue', 'red', 'Patient Age: 20-34 years')

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="xyYd62Y4k1Dx" outputId="b29b5170-3d5e-48ea-e72b-bb20d0a785f6"
plot_window(target_age3,'blue', 'red', 'Patient Age: 35-49 years')

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="jV2wRfrnk1Bq" outputId="d3632b04-7b6c-4598-8769-4147f447eb45"
plot_window(target_age4,'blue', 'red', 'Patient Age: 50-65 years')

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="Vsv9n3R7k09R" outputId="157af957-4e08-4c4c-b368-4f102b8e54eb"
plot_window(target_age5,'blue', 'red', 'Patient Age: 65+ years')

# %% [markdown] id="jcDJbdPHlsqC"
# ### Test dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="ylJEwkkuk07A" outputId="f0568a23-b753-46ac-9f92-0a8b635a029e"
fig, (ax) = plt.subplots(nrows=1,figsize=(16,6))
sns.countplot(test_class_df['PatientAge'], ax=ax)
plt.title("Test set: Patient Age")
plt.xticks(rotation=90)
plt.show()

# %% [markdown] id="BGKteTiAl1Pl"
# ## PatientGender Attribute

# %% [markdown] id="KEGrhbo3mPBI"
# ### Train dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 421} id="_KLdMxjfk04m" outputId="33d51a69-c57e-4e33-d5a2-8fd7d941ee0e"
tmp = train_class_df.groupby(['Target', 'PatientSex'])['patientId'].count()
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
tmp = df.groupby(['Exams','Target', 'PatientSex']).count()
df2 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
fig, ax = plt.subplots(nrows=1,figsize=(6,6))
sns.barplot(ax=ax, x = 'PatientSex', y='Exams', hue='Target',data=df2)
plt.title("Train set: Patient Sex and Target")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 421} id="ePztOiDil5oo" outputId="987a2557-0ceb-4eb0-91cf-ddcd71fb870a"
tmp = train_class_df.groupby(['class', 'PatientSex'])['patientId'].count()
df1 = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
tmp = df1.groupby(['Exams','class', 'PatientSex']).count()
df3 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
fig, (ax) = plt.subplots(nrows=1,figsize=(6,6))
sns.barplot(ax=ax, x = 'PatientSex', y='Exams', hue='class',data=df3)
plt.title("Train set: Patient Sex and class")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="cRORmN7-l5lp" outputId="3d379eac-a70e-4d3a-ec16-6baed7001a8c"
target_female = target_sample[target_sample['PatientSex']=='F']
target_male = target_sample[target_sample['PatientSex']=='M']

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="CDXZPBN6l5jL" outputId="072d762e-1b43-46c6-a77a-805dba8f74e8"
plot_window(target_female,"red", "magenta","Patients Sex: Female")

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="QLKIQ675l5g5" outputId="4012f019-0f75-47cb-93c3-5b26072f6168"
plot_window(target_male,"darkblue", "blue", "Patients Sex: Male")

# %% [markdown] id="1PI417FMmpM9"
# ### Test dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 475} id="a6Jen3JemjG8" outputId="98b059a2-c68e-4d9b-f080-c26e740078fe"
f,axes=plt.subplots(1,2,figsize=(17,7))
test_class_df['PatientSex'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('PatientSex',data=test_class_df,ax=axes[1])
axes[0].set_title('Pie Chart for PatientSex')
axes[1].set_title('Bar Graph for PatientSex')
plt.show()

# %% [markdown] id="t0NbyjI4SSSY"
# ## PixelSpacing

# %% [markdown] id="2K59crk8S_j1"
# ### Train dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="-lZzeGDnSO_k" outputId="353bc4f1-cd8e-4b2c-e18a-bfa3a713dc86"
train_class_df['PixelSpacing'].value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 314} id="lOBdIpDcSO6H" outputId="044a103c-6062-40e5-db4a-280b6bc56e82"
sns.countplot(x='PixelSpacing',hue='Target',data=train_class_df)

# %% colab={"base_uri": "https://localhost:8080/", "height": 314} id="lTjVq4VKSO81" outputId="b4fa398b-1cc7-42e5-953e-502312bb2952"
sns.countplot(x='PixelSpacing',hue='class',data=train_class_df)

# %% [markdown] id="EboI1fC5S7Ow"
# ### Test dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="DizZmgaDS4me" outputId="2008b946-5224-48ac-c39b-3e29ef9bd936"
test_class_df['PixelSpacing'].value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 317} id="qCxua-mhS4me" outputId="b029d20d-359c-4826-f9f5-8a739b205788"
sns.countplot(x='PixelSpacing',hue='PatientSex',data=test_class_df)

# %% colab={"base_uri": "https://localhost:8080/", "height": 317} id="eRoIT6k3S4me" outputId="030f1a30-2951-41ed-ae50-e6bdb6a31caa"
sns.countplot(x='PixelSpacing',hue='PatientSex',data=test_class_df)

# %% [markdown] id="LIMmGYc4S2t6"
# **Modeling for Classification and Object Detection:**
#
# Model Building​: Build a pneumonia detection model starting from basic CNN.
# - Prepare the data for model building. Split data into train and validation sets.
# - Create the network and compile the model. 
# - Print the model summary. 
# - Create the train and the validation generator and fit the model.
#
# Model Evaluation: ​Test the model to understand the performance using the right evaluation metrics.
# - Evaluate the model using the plots for the loss, accuracy and other metrics identified.
# - Predict on a batch of images and provide visualizations of the same
# - Identify the areas of improvement for the next iterations.

# %% [markdown] id="cF6jKHKZh-tQ"
# ## MODELING: CLASSIFICATION

# %% [markdown] id="wWsOhpw20Kha"
# ## 1. Prepare the data.

# %% colab={"base_uri": "https://localhost:8080/"} id="mUOmVkOrRqFC" outputId="3dd15a72-82bd-4254-e955-b18f89afa55b"
# File paths from the G Drive
trainImagesPath = "stage_2_train_images"
testImagesPath = "stage_2_test_images"

labelsPath = "stage_2_train_labels.csv"
classInfoPath = "stage_2_detailed_class_info.csv"

# Read the labels and classinfo
labels = pd.read_csv(labelsPath)
details = pd.read_csv(classInfoPath)

# %% colab={"base_uri": "https://localhost:8080/"} id="2OJtk3rFRqCm" outputId="1b96f075-e9c9-4628-cfa5-38f7d0d429a9"
labels.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 380} id="WyR4vZ0kRp_0" outputId="16016d8d-e7f9-4bf8-f657-2fbe37c1bcdf"
labels.sample(10)

# %% colab={"base_uri": "https://localhost:8080/"} id="h1f9wHhmRp9M" outputId="a9895a7b-1272-4d35-a08d-b2cf7f292b6b"
details.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 380} id="eAattqASRp7L" outputId="456d539d-3084-4bc3-c2bc-96f3abc3b3e9"
details.sample(10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 224} id="aUF5gpzr02uZ" outputId="40d50e38-ea46-47ee-d234-b9e0518636b5"
# Concatenate the two datasets - 'labels' and 'details':
train_data = pd.concat([labels, details['class']], axis = 1)

train_data.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="aC9wFVq-02r5" outputId="83db00b8-9676-4bb3-aae2-2911d1f0940c"
train_data.info()

# %% colab={"base_uri": "https://localhost:8080/"} id="WwidewoI02pE" outputId="bb13ab80-2ab6-41e3-c604-a98f39d9b027"
# Using 1000 samples from each class
# Increase the dataset size later
train_data1 = train_data.groupby('class', group_keys=False).apply(lambda x: x.sample(1000))

# %% colab={"base_uri": "https://localhost:8080/"} id="hq7zo5hJ02mp" outputId="33a1c4c9-504b-4811-9b0f-444ee1881081"
# Check the training dataset with class distribution (data balancing)
train_data1["class"].value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} id="UtTS-DyW02kK" outputId="f82f47a4-48d8-4854-bf66-eb33e29e254d"
# Final train data info
train_data1.info()

# %% colab={"base_uri": "https://localhost:8080/"} id="zxpRkICg02hc" outputId="299105c2-a46f-449e-a55a-c724ceb1bc57"
# Create the arrays
images = []
ADJUSTED_IMAGE_SIZE = 128 # Other image sizes to try later: [(224, 224), (384, 384), (512, 512), (640, 640)]
imageList = []
classLabels = []       
labels = []
originalImage = []

# Function to read the image from the path and reshape the image to smaller size
def readAndReshapeImage(image):
    img = np.array(image).astype(np.uint8)
    # Resize the image
    res = cv2.resize(img,(ADJUSTED_IMAGE_SIZE,ADJUSTED_IMAGE_SIZE), interpolation = cv2.INTER_LINEAR)
    return res

# Read the image and resize the image
def populateImage(rowData):
    for index, row in rowData.iterrows():
        patientId = row.patientId
        classlabel = row["class"]
        dcm_file = 'stage_2_train_images/'+'{}.dcm'.format(patientId)
        dcm_data = dcm.read_file(dcm_file)
        img = dcm_data.pixel_array
        # Convert the images to 3 channels as the dicom image pixels do not have colour class
        if len(img.shape) != 3 or img.shape[2] != 3:
            img = np.stack((img,) * 3, -1)
        imageList.append(readAndReshapeImage(img))
        # originalImage.append(img)
        classLabels.append(classlabel)
    tmpImages = np.array(imageList)
    tmpLabels = np.array(classLabels)
        # originalImages = np.array(originalImage)
    return tmpImages,tmpLabels


# %% colab={"base_uri": "https://localhost:8080/"} id="ToJxBvNg02e7" outputId="cd6d0893-9734-4bcc-ae60-7b41f00c6862"
# %%time
## Read the images into numpy arrays
images, labels = populateImage(train_data1)

# %% colab={"base_uri": "https://localhost:8080/"} id="tHHdARBB02cd" outputId="c1f60363-b2a7-40a5-d5cd-d28480c927f2"
# Sample images
images[:1]

# %% colab={"base_uri": "https://localhost:8080/"} id="ZxbOrG-s02aj" outputId="e168da57-24be-475b-c532-065e803b8a3b"
# Sample labels
labels[:10]

# %% colab={"base_uri": "https://localhost:8080/"} id="hIEgrUud02Xk" outputId="ec5d4ea5-5cd3-4fc9-8be8-5b2b26f29d37"
# Shape of the dataset (X, yc)
# The image is of 128*128 with 3 channels
images.shape , labels.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 303} id="GGLs3DmJ02VT" outputId="eaf05a3d-27a3-4260-d774-0dab52d6b577"
# Check the converted image 
plt.imshow(images[12])

# %% colab={"base_uri": "https://localhost:8080/"} id="5pQTIVaD02Si" outputId="48024aa5-c331-42bd-bdba-994331814800"
# Check the unique labels
np.unique(labels), len(np.unique(labels))

# %% colab={"base_uri": "https://localhost:8080/"} id="0INQYbm302Qf" outputId="20c3118f-26d9-4151-c100-5ba3eb509ca1"
# Encode the labels
from sklearn.preprocessing import LabelBinarizer
enc = LabelBinarizer()
yc = enc.fit_transform(labels)

# %% colab={"base_uri": "https://localhost:8080/"} id="DnD0ZgJJ02Nj" outputId="a6301857-e3c3-493e-ccb0-65cb71ea1a44"
yc[:10]

# %% colab={"base_uri": "https://localhost:8080/"} id="30cXes6lW-O3" outputId="e75dbee6-7f6e-4770-cd76-42183577dea6"
# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, yc, test_size=0.3, random_state=0)

# %% colab={"base_uri": "https://localhost:8080/"} id="4AQL8J--K0Ur" outputId="f2d00108-1c1c-49df-bc41-8c1d373c6731"
# Function to plot the various metrics across the epochs
"""
@Description: This function plots our metrics for our models across epochs
@Inputs: The history of the fitted model
@Output: Plots for accuracy, precision, recall, AUC, and loss
"""
def plottingScores(hist):
    fig, ax = plt.subplots(1, 5, figsize=(25, 3))
    ax = ax.ravel()

    for i, met in enumerate(['accuracy', 'precision', 'recall', 'AUC', 'loss']):
        ax[i].plot(hist.history[met])
        ax[i].plot(hist.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['train', 'val'])


# %% colab={"base_uri": "https://localhost:8080/"} id="kBuccAmAK0SE" outputId="71ede8ff-731f-42d7-de5f-15847220f8d7"
# Metrics to evaluate the model
METRICS = ['accuracy', 
           tf.keras.metrics.Precision(name='precision'), 
           tf.keras.metrics.Recall(name='recall'), 
           tf.keras.metrics.AUC(name='AUC')]


# %% [markdown] id="WNPgDIESeXPO"
# **Callbacks API: A callback is an object that can perform actions at various stages of training (e.g. at the start or end of an epoch, before or after a single batch, etc):**
#
# **1. ModelCheckpoint callback:** is used in conjunction with training using model.fit() to save a model or weights (in a checkpoint file) at some interval, so the model or weights can be loaded later to continue the training from the state saved.
#
# A few options this callback provides include:
#
# - Whether to only keep the model that has achieved the "best performance" so far, or whether to save the model at the end of every epoch regardless of performance.
# - Definition of 'best'; which quantity to monitor and whether it should be maximized or minimized.
# - The frequency it should save at. Currently, the callback supports saving at the end of every epoch, or after a fixed number of training batches.
# - Whether only weights are saved, or the whole model is saved.
#
# **2. EarlyStopping callback:** Stop training when a monitored metric has stopped improving.
#
# Assuming the goal of a training is to minimize the loss. With this, the metric to be monitored would be 'loss', and mode would be 'min'. A model.fit() training loop will check at end of every epoch whether the loss is no longer decreasing, considering the min_delta and patience if applicable. Once it's found no longer decreasing, model.stop_training is marked True and the training terminates.
#
# The quantity to be monitored needs to be available in logs dict. To make it so, pass the loss or metrics at model.compile().
#
# **3. LearningRateScheduler callback:** Learning rate scheduler.
#
# At the beginning of every epoch, this callback gets the updated learning rate value from schedule function provided at __init__, with the current epoch and current learning rate, and applies the updated learning rate on the optimizer.
#
# **4. ReduceLROnPlateau callback:** Reduce learning rate when a metric has stopped improving.
#
# Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.

# %% colab={"base_uri": "https://localhost:8080/"} id="Rf2EZQZrK0NE" outputId="c649d1dd-2b9b-4a0c-a1cd-6c8e7b1291de"
# Define the callback, checkpoint and early_stopping
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 20)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("cr_model.h5", save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)


# %% [markdown] id="4tQ3jgVTPT70"
# ## Model-1: Base Model

# %% colab={"base_uri": "https://localhost:8080/"} id="p5806SKzK0KX" outputId="ea326250-beef-41c4-a030-e7508b3aa5aa"
# Function to create a simple fully connected NN
def fcnn_model():
    # Basic model with a flattening layer followed by 2 dense layers
    model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape = (128, 128, 3)), 
                tf.keras.layers.Dense(128, activation = "relu"), 
                tf.keras.layers.Dense(3, activation = "softmax")
                ])
    
    return model


# %% colab={"base_uri": "https://localhost:8080/"} id="jvAYDjU9K0Hm" outputId="2a4b3d1e-8081-487f-9cdf-26bf41068c51"
# Build our FCNN model and compile
model1 = fcnn_model()
model1.summary()
model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=METRICS)

# %% colab={"base_uri": "https://localhost:8080/"} id="am_HAmaqPgq0" outputId="7b84c20e-7b0d-48af-912c-551632bbbf20"
# Fit the model
H = model1.fit(X_train, y_train,  
                          epochs = 30,
                          batch_size = 128,
                          validation_split = 0.2, 
                          # class_weight = classWeight, 
                          verbose = 1,
                          callbacks = [checkpoint_cb, early_stopping_cb, lr_scheduler])

# %% colab={"base_uri": "https://localhost:8080/", "height": 234} id="onYgPjGFPgov" outputId="8846758a-3564-4059-c993-23d593950866"
# Evaluate and display results
results = model1.evaluate(X_test, y_test) # Evaluate the model on test data
results = dict(zip(model1.metrics_names,results))

print(results)
plottingScores(H)

# %% colab={"base_uri": "https://localhost:8080/"} id="luI_Elu1U1mQ" outputId="88ee3d77-9cf8-477a-ccbf-d3a14444e52d"
y_pred = model1.predict(X_test)
# y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/"} id="IQ4pnhX3ZdxX" outputId="c496ba9c-4862-47ae-de98-3ddd84880259"
y_pred[:10]

# %% colab={"base_uri": "https://localhost:8080/", "height": 847} id="JMaE7AIQU1kX" outputId="202fc923-85fa-46d7-dda2-edc75299cf96"
# Classification Report
print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
df_cm = pd.DataFrame(cm, index = [i for i in ['0', '1', '2']],  
                         columns = [i for i in ['0', '1', '2']])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="BCieRYjQU1iD" outputId="23c1954a-adea-4b03-fc4b-3d34d1a50ef4"
# Model comparison
precision = precision_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
recall = recall_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
f1 = f1_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
AUC = results['AUC']

Train_Accuracy = model1.evaluate(X_train, y_train)
Test_Accuracy = model1.evaluate(X_test, y_test)

base_1 = []
base_1.append(['Model-1: Base Model', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1, AUC])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score', 'AUC'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="3y-nb57zg9be" outputId="71c09a95-b11c-43b4-ba64-9b485f6f02d5"
# Training history dataframe
hist = pd.DataFrame(H.history)

# Understand the various Metrics with epochs
hist

# %% colab={"base_uri": "https://localhost:8080/", "height": 318} id="Ntk10ew9g9Lg" outputId="432f5434-4567-4334-b405-846fbe065582"
hist.describe()


# %% [markdown] id="wRtKORSGKO0Z"
# ## Model-2: CNN-1

# %% colab={"base_uri": "https://localhost:8080/"} id="KE3zmr3AKSE1" outputId="751e5df6-dca3-49be-9022-ff4644e136bb"
# Function to create a complex NN model
def cnn_model1():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding = 'valid', activation = 'relu', input_shape=(128, 128, 3)), #  convolutional layer
        tf.keras.layers.MaxPool2D(pool_size=(2,2)), # flatten output of conv
        
        tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding = 'valid', activation = 'relu'), #  convolutional layer
        tf.keras.layers.MaxPool2D(pool_size=(2,2)), # flatten output of conv
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'valid'),
        tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Flatten(), # flatten output of conv
        tf.keras.layers.Dense(512, activation = "relu"), # hidden layer
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation = "relu"), #  output layer
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation = "softmax")])
    
    return model


# %% colab={"base_uri": "https://localhost:8080/"} id="IlQvC8BpLJso" outputId="35951afb-817f-42ac-81c1-3ff8a26497df"
# Build and compile the model
model2 = cnn_model1()
model2.summary()
model2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=METRICS)

# %% colab={"base_uri": "https://localhost:8080/"} id="FTxFFfwYLJd7" outputId="38e5994e-3d43-4cdf-9fb3-47eb2cd2ba43"
# Fit the model
H = model2.fit(X_train, y_train,  
                      epochs=30, 
                      validation_split = 0.20, 
                      batch_size=128,
                      # class_weight=classWeight,
                      callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
                      verbose=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 251} id="DfsF7RWELJbg" outputId="81e2f5ae-5e15-4d74-e8fb-08188dc0a60f"
# Evaluate the model results and put into a dict
results = model2.evaluate(X_test, y_test)
results = dict(zip(model2.metrics_names,results))

print(results)
plottingScores(H)

# %% colab={"base_uri": "https://localhost:8080/"} id="o3Yx6tQ_Acqy" outputId="e129dada-4aae-4125-8143-703973665321"
y_pred = model2.predict(X_test)
# y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 847} id="qncOmZHEAcq0" outputId="42fae7ea-a08e-47b6-8b06-95d8b3528dcd"
# Classification Report
print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
df_cm = pd.DataFrame(cm, index = [i for i in ['0', '1', '2']],  
                         columns = [i for i in ['0', '1', '2']])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="76TlGRU1Acq1" outputId="3a8f09cb-38be-4017-fc94-9519ff1ebf99"
# Model comparison
precision = precision_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
recall = recall_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
f1 = f1_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
AUC = results['AUC']

Train_Accuracy = model2.evaluate(X_train, y_train)
Test_Accuracy = model2.evaluate(X_test, y_test)

# base_1 = []
base_1.append(['Model-2: CNN-1', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1, AUC])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score', 'AUC'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)


# %% [markdown] id="CqUZz5mwOm3x"
# ## Model-3: CNN-2

# %% colab={"base_uri": "https://localhost:8080/"} id="alZY2ZIkOm32" outputId="09750a40-88ad-4c10-adac-4a30fc577452"
# Function to create a complex NN model
def cnn_model2():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(128, 128, 3)), #  convolutional layer
        tf.keras.layers.MaxPool2D(pool_size=(2,2)), # flatten output of conv
        
        tf.keras.layers.Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'), #  convolutional layer
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)), # flatten output of conv
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='valid'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.3),
        

        tf.keras.layers.Flatten(), # Flattening
        
        # Full Connection
        tf.keras.layers.Dense(64, activation='relu'), # hidden layer
        tf.keras.layers.Dropout(0.5), # Dropout
        tf.keras.layers.Dense(3, activation='softmax')]) #  output layer
    
    return model

# %% colab={"base_uri": "https://localhost:8080/"} id="9RLEujsMOm32" outputId="e7fe3f32-6a6e-4e88-ac13-9f8b1300b80f"
# Build and compile the model
model3 = cnn_model2()
model3.summary()
model3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=METRICS)

# %% colab={"base_uri": "https://localhost:8080/"} id="cmcJOhj8Om32" outputId="584ec747-d879-4d06-f99b-ba516cbef7fd"
# Fit the model
H = model3.fit(X_train, y_train,  
                      epochs=30, 
                      validation_split = 0.20, 
                      batch_size=128,
                      # class_weight=classWeight,
                      callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
                      verbose=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 251} id="QL1VM5SYOm33" outputId="daa0529d-0179-4276-caa9-b2d1aff3c835"
# Evaluate the model results and put into a dict
results = model3.evaluate(X_test, y_test)
results = dict(zip(model3.metrics_names,results))

print(results)
plottingScores(H)

# %% colab={"base_uri": "https://localhost:8080/"} id="guPmHPsLOm33" outputId="81ce0f9a-9105-4962-84d2-7d7f0d297b22"
y_pred = model3.predict(X_test)
# y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 847} id="4z64sgSyOm33" outputId="9f2da4d9-44cd-4ab7-8d4c-6a18ef00b33c"
# Classification Report
print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
df_cm = pd.DataFrame(cm, index = [i for i in ['0', '1', '2']],  
                         columns = [i for i in ['0', '1', '2']])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="OF12OSCQOm34" outputId="70b48402-d7e5-4241-e0ed-87519cfbe0db"
# Model comparison
precision = precision_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
recall = recall_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
f1 = f1_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
AUC = results['AUC']

Train_Accuracy = model3.evaluate(X_train, y_train)
Test_Accuracy = model3.evaluate(X_test, y_test)

# base_1 = []
base_1.append(['Model-3: CNN-2', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1, AUC])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score', 'AUC'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)


# %% [markdown] id="CWzbTz5xTeNK"
# # Milestone-2

# %% [markdown] id="ZGwD_QYRMv1m"
# ## Model-4: MobileNetV2

# %% colab={"base_uri": "https://localhost:8080/"} id="OsF7pPd6qT06" outputId="adff36b0-3bc5-490d-f48d-bd9d72655f45"
# # Pre-process the data for VGG16 model
# X_train = preprocess_input(X_train) 
# X_test = preprocess_input(X_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="KxfluEzwLJY0" outputId="8d05d826-24e7-4bf4-82c9-b861d66c6a24"
# Function to create NN with transfer learning
def mn_model():
    model = tf.keras.Sequential([
        tf.keras.applications.MobileNetV2(include_top = False, weights="imagenet", input_shape=(128, 128, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(3, activation = 'softmax')])
    
    model.layers[0].trainable = False
    
    return model


# %% colab={"base_uri": "https://localhost:8080/"} id="ojvQjhIpLJWi" outputId="4def7b90-bff2-46a8-9c9e-037388e7f5c3"
# Build and compile the mobile net model
model4 = mn_model()
model4.summary()
model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=METRICS)

# %% colab={"base_uri": "https://localhost:8080/"} id="BUdxCGnkM1K6" outputId="38214eaf-85cf-4600-e503-e30ac3243f7e"
H = model4.fit(X_train, y_train,  
                          epochs = 30, 
                          validation_split = 0.20, 
                          # class_weight = classWeight,
                          batch_size = 64,
                          callbacks = [checkpoint_cb, early_stopping_cb, lr_scheduler])

# %% colab={"base_uri": "https://localhost:8080/", "height": 251} id="VdHoAFrIM1HO" outputId="63236d08-808c-49d4-c943-60846d4ee282"
# Show results and print graphs
results = model4.evaluate(X_test, y_test)
results = dict(zip(model4.metrics_names,results))

print(results)
plottingScores(H)

# %% colab={"base_uri": "https://localhost:8080/"} id="xuOdi9VDHqgR" outputId="ae5be7cf-45ba-4497-a2a3-ef3ed3edc619"
y_pred = model4.predict(X_test)
# y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 847} id="XETpn8lkHqgS" outputId="ef2ffdf8-2eae-4e7b-c59d-8198a72f8e33"
# Classification Report
print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
df_cm = pd.DataFrame(cm, index = [i for i in ['0', '1', '2']],  
                         columns = [i for i in ['0', '1', '2']])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="-jRWQvPCHqgT" outputId="82e76d4a-0613-4f37-9993-5bddd4b8aac4"
# Model comparison
precision = precision_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
recall = recall_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
f1 = f1_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
AUC = results['AUC']

Train_Accuracy = model4.evaluate(X_train, y_train)
Test_Accuracy = model4.evaluate(X_test, y_test)

# base_1 = []
base_1.append(['Model-4: MobileNetV2', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1, AUC])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score', 'AUC'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)


# %% [markdown] id="mAtARdr3nyeL"
# ## Model-5: VGG16

# %% colab={"base_uri": "https://localhost:8080/"} id="eeUI9pEUnyeQ" outputId="8b4added-b9be-4f4f-89a8-3c46373463a0"
# Function to create NN with transfer learning
def vgg16_model():
    model = tf.keras.Sequential([
        tf.keras.applications.VGG16(include_top = False, weights="imagenet", input_shape=(128, 128, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(3, activation = 'softmax')])
    
    model.layers[0].trainable = False
    
    return model


# %% colab={"base_uri": "https://localhost:8080/"} id="KZj9v3NmnyeQ" outputId="17c20a7d-55ce-444a-d440-7594eef55406"
# Build and compile the model
model5 = vgg16_model()
model5.summary()
model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=METRICS)

# %% colab={"base_uri": "https://localhost:8080/"} id="53HWEPr8nyeQ" outputId="8d797f41-d9e7-4c3d-f2a2-b09fa950b109"
H = model5.fit(X_train, y_train,  
                          epochs = 10, 
                          validation_split = 0.20, 
                          # class_weight = classWeight,
                          batch_size = 64,
                          callbacks = [checkpoint_cb, early_stopping_cb, lr_scheduler])
# t = 80min

# %% colab={"base_uri": "https://localhost:8080/", "height": 251} id="uklGzFu8nyeQ" outputId="336c7f85-2cfa-40a6-cd36-d9c926c57fa2"
# Show results and print graphs
results = model5.evaluate(X_test, y_test)
results = dict(zip(model5.metrics_names,results))

print(results)
plottingScores(H)

# %% colab={"base_uri": "https://localhost:8080/"} id="3Y3hIQZ0nyeQ" outputId="1b1e32e3-74e1-4b3b-ef01-67b6b17dd766"
y_pred = model5.predict(X_test)
# y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 847} id="36DczWjQnyeR" outputId="42c2182e-4587-402d-fb45-d762ebe8ec16"
# Classification Report
print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
df_cm = pd.DataFrame(cm, index = [i for i in ['0', '1', '2']],  
                         columns = [i for i in ['0', '1', '2']])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="tR26OgQinyeR" outputId="df1ca093-4c52-4907-843e-5045e7eae9b7"
# Model comparison
precision = precision_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
recall = recall_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
f1 = f1_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
AUC = results['AUC']

Train_Accuracy = model5.evaluate(X_train, y_train)
Test_Accuracy = model5.evaluate(X_test, y_test)

# base_1 = []
base_1.append(['Model-5: VGG16', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1, AUC])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score', 'AUC'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)


# %% [markdown] id="tFmgpTVqIGM2"
# ## Model-6: ResNet50

# %% colab={"base_uri": "https://localhost:8080/"} id="QMk3VAKSIGM2" outputId="a04520fe-7400-4ce1-a4a1-cbcf74a5f2ed"
# Function to create NN with transfer learning
def resnet50_model():
    model = tf.keras.Sequential([
        tf.keras.applications.ResNet50(include_top = False, weights="imagenet", input_shape=(128, 128, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(3, activation = 'softmax')])
    
    model.layers[0].trainable = False
    
    return model


# %% colab={"base_uri": "https://localhost:8080/"} id="HkDhXtw3IGM3" outputId="e01a095a-4039-4aad-ef8d-bd4a70dd3716"
# Build and compile the model
model6 = resnet50_model()
model6.summary()
model6.compile(optimizer='adam', loss='categorical_crossentropy', metrics=METRICS)

# %% colab={"base_uri": "https://localhost:8080/"} id="sncTl7HUIGM3" outputId="d8557f03-3041-4ee0-d607-4062cb56b8e9"
H = model6.fit(X_train, y_train,  
                          epochs = 10, 
                          validation_split = 0.20, 
                          # class_weight = classWeight,
                          batch_size = 64,
                          callbacks = [checkpoint_cb, early_stopping_cb, lr_scheduler])

# %% colab={"base_uri": "https://localhost:8080/", "height": 251} id="29ukK36VIGM3" outputId="94b7c1c9-6279-4a38-92a5-0c376e7c3e9f"
# Show results and print graphs
results = model6.evaluate(X_test, y_test)
results = dict(zip(model6.metrics_names,results))

print(results)
plottingScores(H)

# %% colab={"base_uri": "https://localhost:8080/"} id="1ZpIoPQ4IGM3" outputId="51bd737d-a9a4-490e-ccb9-e084efafa3ce"
y_pred = model6.predict(X_test)
# y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 847} id="nlj6cAJrIGM3" outputId="de8975e8-d98e-44ee-e166-5dfbc9256e87"
# Classification Report
print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
df_cm = pd.DataFrame(cm, index = [i for i in ['0', '1', '2']],  
                         columns = [i for i in ['0', '1', '2']])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="6bfttl3AIGM3" outputId="331c3650-741e-4e0e-e6dc-41892d55cd90"
# Model comparison
precision = precision_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
recall = recall_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
f1 = f1_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
AUC = results['AUC']

Train_Accuracy = model6.evaluate(X_train, y_train)
Test_Accuracy = model6.evaluate(X_test, y_test)

# base_1 = []
base_1.append(['Model-6: ResNet50', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1, AUC])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score', 'AUC'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)


# %% [markdown] id="7zuIqGDkTZs0"
# ## Model-7: EfficientNetV2B3

# %% colab={"base_uri": "https://localhost:8080/"} id="yrR7EaOPTZs0" outputId="0ecf6398-b766-4bdf-a1e7-fa5f1e776c2d"
# Function to create NN with transfer learning
def enetv2b3_model():
    model = tf.keras.Sequential([
        tf.keras.applications.EfficientNetV2B3(include_top = False, weights="imagenet", input_shape=(128, 128, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(3, activation = 'softmax')])
    
    model.layers[0].trainable = False
    
    return model


# %% colab={"base_uri": "https://localhost:8080/"} id="TINxjyYHTZs0" outputId="b9bd87f9-aaca-4629-ecaf-14b2e8dbd11a"
# Build and compile the model
model7 = enetv2b3_model()
model7.summary()
model7.compile(optimizer='adam', loss='categorical_crossentropy', metrics=METRICS)

# %% colab={"base_uri": "https://localhost:8080/"} id="HIUQx8BjTZs0" outputId="da9cd85d-d706-42e7-e1b9-0e4f44772f3c"
H = model7.fit(X_train, y_train,  
                          epochs = 10, 
                          validation_split = 0.20, 
                          # class_weight = classWeight,
                          batch_size = 64,
                          callbacks = [checkpoint_cb, early_stopping_cb, lr_scheduler])

# %% colab={"base_uri": "https://localhost:8080/", "height": 313} id="g6gAaIMaTZs1" outputId="1890d54b-2db7-4af0-a7a1-73dc943aff75"
# Show results and print graphs
results = model7.evaluate(X_test, y_test)
results = dict(zip(model7.metrics_names,results))

print(results)
plottingScores(H)

# %% colab={"base_uri": "https://localhost:8080/"} id="AWTsoIKETZs1" outputId="2ec06e93-5921-4c6f-a218-1c28cc993605"
y_pred = model7.predict(X_test)
# y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 847} id="ZrIQhjYZTZs1" outputId="807039c9-0fda-4bea-e26b-053d214463b2"
# Classification Report
print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
df_cm = pd.DataFrame(cm, index = [i for i in ['0', '1', '2']],  
                         columns = [i for i in ['0', '1', '2']])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="rvHIrfe-TZs1" outputId="11631672-dd1e-46cb-eab9-b514ffefc65b"
# Model comparison
precision = precision_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
recall = recall_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
f1 = f1_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
AUC = results['AUC']

Train_Accuracy = model7.evaluate(X_train, y_train)
Test_Accuracy = model7.evaluate(X_test, y_test)

# base_1 = []
base_1.append(['Model-7: EfficientNetV2B3', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1, AUC])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score', 'AUC'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)


# %% [markdown] id="3FpBhzoc7Wvl"
# ## Model-8: CNN-1 (Tuned)
# Hyperparameter tuning for Model-2: CNN-1

# %% colab={"base_uri": "https://localhost:8080/"} id="sHzoaCxumVkt" outputId="7e9e7e22-c914-4bce-a877-c76ea65d4dc1"
# # Function to create and compile model, required for KerasClassifier
def cnn_model(lr, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding = 'valid', activation = 'relu', input_shape=(128, 128, 3)), # convolutional layer
    tf.keras.layers.MaxPool2D(pool_size=(2,2)), # flatten output of conv
        
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding = 'valid', activation = 'relu'), # convolutional layer
    tf.keras.layers.MaxPool2D(pool_size=(2,2)), # flatten output of conv
    tf.keras.layers.Dropout(0.3),
        
    tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'valid'),
    tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'valid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Flatten(), # flatten output of conv
    tf.keras.layers.Dense(512, activation = "relu"), # hidden layer
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation = "relu"), #  output layer
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation = "softmax")])
    # compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=METRICS)  
    return model


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Cn8oqZfMmViW" outputId="c9a968cd-7c12-4c1c-ce14-f9ee1b64d5f4"
# Create the model
model = KerasClassifier(build_fn = cnn_model, verbose=1)

# Grid search for batch size and learning rate
params = {'batch_size':[32, 64, 128],
          'lr':[0.01,0.1,0.001]}
          
# Grid search for batch size, learning rate, optimizers, activation, epochs, layers, neurons, etc.
# params = {'batch_size':[32, 64, 128],
#           'optimizers':['rmsprop', 'adam', 'SGD'],
#           'activation':['relu', 'sigmoid', 'tanh'],
#           'epochs':[50, 100, 150],
#           'lr':[0.01, 0.1, 0.001]}

gs = RandomizedSearchCV(estimator=model, param_distributions=params, cv=5, verbose=10, random_state=0) # For local machine n_jobs=2
gs.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/", "height": 349} id="iMHlSGZHmVe0" outputId="9bfad680-6a3e-4980-9837-729dfdde7b1c"
# Summarize the results in a dataframe
pd.DataFrame({'param': gs.cv_results_["params"], 
              'score mean': gs.cv_results_["mean_test_score"], 
              'score s.d.': gs.cv_results_["std_test_score"],
              'rank': gs.cv_results_["rank_test_score"]})

# %% colab={"base_uri": "https://localhost:8080/"} id="5i32QEVimVcb" outputId="6ac3e887-a0ef-4f0b-c3c1-b732bdcd4e8b"
# Select the best parameters
print('Best Parameters:', gs.best_params_, 'mean score: ', gs.best_score_, sep='\n')

# %% colab={"base_uri": "https://localhost:8080/"} id="RPe9BS37mmjC" outputId="c1343556-18ad-4d72-a353-ccea14d5ad02"
# Build and compile the model with best params
model8 = cnn_model(lr = gs.best_params_['lr'], batch_size = gs.best_params_['batch_size']) #Change this with best params
model8.summary()

opt = tf.keras.optimizers.Adam(gs.best_params_['lr'])
model8.compile(optimizer=opt, loss="categorical_crossentropy", metrics=METRICS)

# %% colab={"base_uri": "https://localhost:8080/"} id="-SpQM_OS6A0P" outputId="5d73d7aa-63b7-42aa-dc34-28ab34786b7f"
# Fit the model
H = model8.fit(X_train, y_train,  
                      epochs=30, 
                      validation_split = 0.20, 
                      batch_size=gs.best_params_['batch_size'], # Change this with best params
                      # class_weight=classWeight,
                      callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
                      verbose=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 251} id="lE3Glh8g6A0V" outputId="381eff16-0307-49bc-da00-e802864dbbe4"
# Evaluate the model results and put into a dict
results = model8.evaluate(X_test, y_test)
results = dict(zip(model8.metrics_names,results))

print(results)
plottingScores(H)

# %% colab={"base_uri": "https://localhost:8080/"} id="txKZaGwq6A0V" outputId="1fc5aeac-cc82-41bc-ae31-d6d257425500"
y_pred = model8.predict(X_test)
# y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 847} id="4uJm9iO46A0V" outputId="f3c44900-d560-42b9-8582-e2bae5865564"
# Classification Report
print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
df_cm = pd.DataFrame(cm, index = [i for i in ['0', '1', '2']],  
                         columns = [i for i in ['0', '1', '2']])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="IHZwHPhP6A0V" outputId="f635482a-deda-4138-927b-9fa06e4480f5"
# Model comparison
precision = precision_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
recall = recall_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
f1 = f1_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average='macro')
AUC = results['AUC']

Train_Accuracy = model8.evaluate(X_train, y_train)
Test_Accuracy = model8.evaluate(X_test, y_test)

# base_1 = []
base_1.append(['Model-8: CNN-1 (Tuned)', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1, AUC])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score', 'AUC'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% [markdown] id="7yPDafNQYrY5"
# ## Model Comparison - Classification

# %% colab={"base_uri": "https://localhost:8080/", "height": 318} id="An7aicmTS1js" outputId="8f423767-06ab-448d-fcef-70a5ab2b6bee"
# Sumarize the results of modeling in a dataframe; Datapoints used = 3000; Epochs = 10/20; ADJUSTED_IMAGE_SIZE = 128
model_comparison

# %% [markdown] id="gZt_UAwN-gg8"
# Evaluation metrics allow us to estimate errors to determine how well our models are performing:
# - Accuracy: ratio of correct predictions over total predictions.
# - Precision: how often the classifier is correct when it predicts positive.
# - Recall: how often the classifier is correct for all positive instances.
# - F-Score: single measurement to combine precision and recall.

# %% colab={"base_uri": "https://localhost:8080/", "height": 638} id="n0rKINx0wP1E" outputId="85a275c4-e426-43f1-f2a7-f66db5dcc0fc"
# Bar graph for Model Vs. F1 Score
plt.figure(figsize=(20,10))
sns.barplot(data=model_comparison, x="Model", y="F1 Score")
plt.title("Model Comparison: Model Vs. F1 Score")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 656} id="OrRVt_aIATeb" outputId="bdc5a4a1-26e8-4038-8bf0-4c4d7262e977"
# Bar graph for Model Vs. Metrics
model_comparison[['Model', 'Precision', 'Recall', 'F1 Score', 'AUC']].plot(kind='bar', x = 'Model', rot = 0, sort_columns = 'Model', figsize=(20,10))
plt.title("Model Comparison: Model Vs. Metrics")
plt.xlabel("Model")
plt.ylabel("Metrics")

# %% [markdown] id="9DkDZHxJi-Pz"
# ## MODELING: OBJECT DETECTION
# Model selection was a challenge as both localization and classification was needed into one model. We did some experimentation with Faster R-CNN architecture. After spending sometime on that we realized that for semantic segmentation (binary classification) U-Net is  better and simpler than Faster R-CNN. We found that U-Net is also widely used in medical applications and AI models. Hence we switched to the U-Net. Facing errors while importing the Matterplot/Mask_RCNN package.
#
# Due to lack of time, we could not try RCNN, Yolo and CheXNet. But for our learning purpose, we would like to try them after completing this project.
#
# https://paperswithcode.com/method/chexnet

# %% [markdown] id="vAbMy6ESkCC1"
# ## 1. Prepare the data.

# %% colab={"base_uri": "https://localhost:8080/"} id="2M2qd6q2kCC2" outputId="3cc8517c-c35e-47c8-8cf0-37a3b907d28f"
# File paths from the G Drive
trainImagesPath = "stage_2_train_images"
testImagesPath = "stage_2_test_images"

labelsPath = "stage_2_train_labels.csv"
classInfoPath = "stage_2_detailed_class_info.csv"

# Read the labels and classinfo
labels = pd.read_csv(labelsPath)
details = pd.read_csv(classInfoPath)

# %% colab={"base_uri": "https://localhost:8080/"} id="4zHnEhv9kCC2" outputId="d13da8a2-4df5-4c44-a66a-fc8752c615ce"
labels.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 380} id="nohAdRhmkCC2" outputId="9dbbf894-6f7d-422c-bf52-6d3903865ad3"
labels.sample(10)

# %% colab={"base_uri": "https://localhost:8080/"} id="6T08Jl-IkCC2" outputId="80cceb9a-518e-401c-abbe-c10dff17609e"
details.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 380} id="gusE4EAukCC2" outputId="4c13c7c0-8fce-4084-c5f1-bacfdc3c9505"
details.sample(10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 224} id="_cVvfybCkCC2" outputId="b8b8b4e8-43f6-48c8-d194-1d7a9565f84e"
# Concatenate the two datasets - 'labels' and 'details':
train_data = pd.concat([labels, details['class']], axis = 1)

train_data.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="J-YxOQR8kCC2" outputId="5f3d7e6b-d450-4198-d342-94d7117bc3f8"
train_data.info()

# %% colab={"base_uri": "https://localhost:8080/"} id="yNnvqJrrkCC3" outputId="6fce3ed9-babd-4b14-ec05-5619eeb6a771"
# Use the 2000 datapoints for training and 2000 for testing
# Use full datasets later
traind = train_data[0:2000]
testd = train_data[2000:4000]

traind.fillna(0, inplace=True)
testd.fillna(0, inplace=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="kSy8PvOskCC3" outputId="4648e295-c268-4b86-d78e-b7062d33a048"
# Check the distribution of the Target variable
traind.Target.value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} id="VToYllfLYuHM" outputId="435220dc-04f1-4a13-d05c-3d63fb9acc30"
# Check the distribution of the Target variable
testd.Target.value_counts()

# %% [markdown] id="j-6VjExXQE1U"
# Define Generator class: This class handles large dataset. By creating batches, the resource usage is minimised. This generator class returns both souce image and masked image. Masked images are generated using the bounderies present in the label file.

# %% colab={"base_uri": "https://localhost:8080/"} id="FYdkAkMWIE7u" outputId="b1c89d8b-3764-42b5-b1bf-75d798f4258d"
# Keras data generator class; Refer the below articles for details:
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c

# Manage the memory by reducing the batch size; 8, 10, 16, 32, 64
BATCH_SIZE = 6
# Image size to be scaled; 128 × 128, 224 x 224, 256 × 256, and 512 × 512
IMAGE_SIZE = 224

# Actual Image size 
IMG_WIDTH = 1024
IMG_HEIGHT = 1024

class TrainGenerator(Sequence):

    def __init__(self,  _labels):       
        self.pids = _labels["patientId"].to_numpy()
        self.coords = _labels[["x", "y", "width", "height"]].to_numpy()
        self.coords = self.coords * IMAGE_SIZE / IMG_WIDTH        

    def __len__(self):
        return math.ceil(len(self.coords) / BATCH_SIZE)

    """
    The contrast of an image is enhanced when various shades in the image becomes more distinct.
    We can do so by darkening the shades of the darker pixels and vice versa. 
    This is equivalent to widening the range of pixel intensities. To have a good contrast, 
    the following histogram characteristics are desirable:

    1) the pixel intensities are uniformly distributed across the full range of values (each intensity value is equally probable), and
    2) the cumulative histogram is increasing linearly across the full intensity range.

    Histogram equalization modifies the distribution of pixel intensities to achieve these characteristics.
    """
    def __doHistogramEqualization(self,img):
        # Pre processing Histogram equalization
        histogram_array = np.bincount(img.flatten(), minlength=256)
        # Normalize
        num_pixels = np.sum(histogram_array)
        histogram_array = histogram_array/num_pixels
        # Normalized cumulative histogram
        chistogram_array = np.cumsum(histogram_array)
        """
        STEP 2: Pixel mapping lookup table
        """
        transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
        """
        STEP 3: Transformation
        """
        img_list = list(img.flatten())

        # Transform pixel values to equalize
        eq_img_list = [transform_map[p] for p in img_list]

        # Reshape and write back into img_array
        img = np.reshape(np.asarray(eq_img_list), img.shape)

        return img

    def __getitem__(self, idx): # Get a batch
        batch_coords = self.coords[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE] # Image coords
        batch_pids = self.pids[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE] # Image pids    
        batch_images = np.zeros((len(batch_pids), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        batch_masks = np.zeros((len(batch_pids), IMAGE_SIZE, IMAGE_SIZE))
        for _indx, _pid in enumerate(batch_pids):
            _path = 'stage_2_train_images/'+'{}.dcm'.format(_pid)
            _imgData = dcm.read_file(_path)

            img = _imgData.pixel_array 
            # img = np.stack((img,)*3, axis=-1) # Expand grayscale image to contain 3 channels

            # Resize image
            resized_img = cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE), interpolation = cv2.INTER_AREA)
            
            resized_img = self.__doHistogramEqualization(resized_img)
        
            batch_images[_indx][:,:,0] = preprocess_input(np.array(resized_img[:,:], dtype=np.float32)) 
            batch_images[_indx][:,:,1] = preprocess_input(np.array(resized_img[:,:], dtype=np.float32)) 
            batch_images[_indx][:,:,2] = preprocess_input(np.array(resized_img[:,:], dtype=np.float32)) 
            x = int(batch_coords[_indx, 0])
            y = int(batch_coords[_indx, 1])
            width = int(batch_coords[_indx, 2])
            height = int(batch_coords[_indx, 3])
            
            batch_masks[_indx][y:y+height, x:x+width] = 1

        return batch_images, batch_masks


# %% colab={"base_uri": "https://localhost:8080/"} id="xxSQw3aXIE3l" outputId="367f610d-8de9-4256-87b7-97098b40e60a"
traind = TrainGenerator(traind)
testd = TrainGenerator(testd)


# %% colab={"base_uri": "https://localhost:8080/"} id="EratW5X0IE0n" outputId="f940ee1c-cc0b-4c43-84eb-a7ea52b9f4ed"
# Function to show the images with mask
def showMaskedImage(_imageSet, _maskSet, _index) :
    maskImage = _imageSet[_index]

    maskImage[:,:,0] = _maskSet[_index] * _imageSet[_index][:,:,0]
    maskImage[:,:,1] = _maskSet[_index] * _imageSet[_index][:,:,1]
    maskImage[:,:,2] = _maskSet[_index] * _imageSet[_index][:,:,2]

    plt.imshow(maskImage[:,:,0])


# %% colab={"base_uri": "https://localhost:8080/", "height": 321} id="OV5LXec-IEyk" outputId="8cc25ad5-6fdf-4c5d-86c8-52ccf904d6ee"
# Sample pre-processed image from the TrainGenerator class
imageSet0 = traind[1][0][1]
plt.imshow(imageSet0)

# %% colab={"base_uri": "https://localhost:8080/", "height": 286} id="886iz91DIEvk" outputId="73d2b23e-a1c6-47b8-cc52-7b2dbe67aa58"
# Masks for the same
imageSet0 = traind[2][0]
maskSet0 = traind[2][1]    
showMaskedImage(imageSet0, maskSet0, 5)


# %% [markdown] id="axw0OCJCd7a0"
# ## Performance Metrics

# %% [markdown] id="keBze1fnZWpv"
# **Dice Coefficient (F1 Score) and IoU:**
#
# Simply put, the Dice Coefficient is 2 * the Area of Overlap divided by the total number of pixels in both images.
#
# The Dice coefficient is very similar to the IoU. They are positively correlated, meaning if one says model A is better than model B at segmenting an image, then the other will say the same. Like the IoU, they both range from 0 to 1, with 1 signifying the greatest similarity between predicted and truth.
#
# Intersection-Over-Union (IoU, Jaccard Index): The Intersection-Over-Union (IoU), also known as the Jaccard Index, is one of the most commonly used metrics in semantic segmentation. The IoU is a very straightforward metric that’s extremely effective.
#
# In conclusion, the most commonly used metrics for semantic segmentation are the IoU and the Dice Coefficient.

# %% colab={"base_uri": "https://localhost:8080/"} id="4LpWtwAIHoNa" outputId="ad4711f7-d7dd-42e9-a9b0-a6db1e99283f"
# Function to calculate the Dice Coef.
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + tf.keras.backend.epsilon()) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + tf.keras.backend.epsilon())

# Dice Loss
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# # IoU Coef; Use the Keras metrics for MeanIoU
# def iou_coef(y_true, y_pred, smooth=1):
#   intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
#   union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
#   iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
#   return iou

# F1 Score: Taken from old keras source code
def f1_score(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# %% colab={"base_uri": "https://localhost:8080/"} id="NKkGF05mkCC5" outputId="4754db7c-86cc-444a-b54e-8065546c5dd2"
# Function to plot the various metrics across the epochs
"""
@Description: This function plots our metrics for our models across epochs
@Inputs: The history of the fitted model
@Output: Plots for accuracy, precision, recall, AUC, and loss
"""
def plottingScores(hist):
    fig, ax = plt.subplots(1, 8, figsize=(30, 4))
    ax = ax.ravel()

    for i, met in enumerate(['accuracy', 'precision', 'recall', 'f1_score', 'AUC', 'loss', 'dice_coef', 'mean_iou']):
        ax[i].plot(hist.history[met])
        ax[i].plot(hist.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['train', 'val'])


# %% colab={"base_uri": "https://localhost:8080/"} id="mLC8ngqwkCC5" outputId="e69ff35b-05c2-4e5e-e93c-69d9b13ea30d"
# Metrics to evaluate the model
METRICS = ['accuracy', 
           tf.keras.metrics.Precision(name='precision'), 
           tf.keras.metrics.Recall(name='recall'), 
           f1_score,
           tf.keras.metrics.AUC(name='AUC'),
           dice_coef,
           tf.keras.metrics.MeanIoU(num_classes=2, name='mean_iou')]


# %% colab={"base_uri": "https://localhost:8080/"} id="4n3Fge07kCC5" outputId="adaf14bd-b7de-4291-af81-e6d94ce047e8"
# Define the callback, checkpoint and early_stopping
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 20)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("unet_model.h5", save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# %% [markdown] id="JUXzrhBmkCC5"
# ## Model-9: UNet (MobileNet)

# %% colab={"base_uri": "https://localhost:8080/"} id="QeJpg1aghVxY" outputId="afc8ee78-0d1c-4262-87a7-5afadea4c16e"
LR = 1e-4
ALPHA = 1.0

# Define the UNet with MobileNet
def create_unet_mobilenet(trainable=True):
    """Function to create UNet architecture with MobileNet.
        
    Arguments:
        trainable -- Flag to make layers trainable. Default value is 'True'.
    """
    # Get all layers with 'imagenet' weights
    model = MobileNet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, alpha=ALPHA, weights="imagenet") 
    # Top layer is last layer of the model
    
    # Make all layers trainable
    for layer in model.layers:
        layer.trainable = trainable

    # Add all the UNET layers here
    convLayer_112by112 = model.get_layer("conv_pw_1_relu").output
    convLayer_56by56 = model.get_layer("conv_pw_3_relu").output
    convLayer_28by28 = model.get_layer("conv_pw_5_relu").output
    convLayer_14by14 = model.get_layer("conv_pw_11_relu").output
    convLayer_7by7 = model.get_layer("conv_pw_13_relu").output
    # The last layer of mobilenet model is of dimensions (7x7x1024)

    # Start upsampling from 7x7 to 14x14 ...up to 224x224 to form UNET
    # concatinate with the original image layer of the same size from MobileNet
    x = Concatenate()([UpSampling2D()(convLayer_7by7), convLayer_14by14])
    x = Concatenate()([UpSampling2D()(x), convLayer_28by28])
    x = Concatenate()([UpSampling2D()(x), convLayer_56by56])
    x = Concatenate()([UpSampling2D()(x), convLayer_112by112])
    x = UpSampling2D(name="unet_last")(x) # upsample to 224x224

    # Add classification layer
    x = Conv2D(1, kernel_size=1, activation="sigmoid", name="masks")(x)
    x = Reshape((IMAGE_SIZE, IMAGE_SIZE))(x) 
    
    return Model(inputs=model.input, outputs=x)


# %% colab={"base_uri": "https://localhost:8080/"} id="Qm-taIWkhVsV" outputId="f449d079-70d9-4aa6-848d-35016c17715e"
# Build the model 
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
model9 = create_unet_mobilenet(input_shape)
model9.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="Z-h-CLe2hVpx" outputId="3258ffcf-41fa-4e14-a134-10cf188989d2"
model9.compile(optimizer="adam", loss = dice_loss, metrics=METRICS)

# %% colab={"base_uri": "https://localhost:8080/"} id="Xh006vsVhVnC" outputId="c09cc2dc-8de9-4408-9174-6af76c32b436"
# Fit the model
train_steps = len(traind)//BATCH_SIZE
valid_steps = len(testd)//BATCH_SIZE

if len(traind) % BATCH_SIZE != 0:
    train_steps += 1
if len(testd) % BATCH_SIZE != 0:
    valid_steps += 1


H = model9.fit(traind, epochs=20,
                        steps_per_epoch=train_steps,
                        validation_data=testd,
                        callbacks = [checkpoint_cb, early_stopping_cb, lr_scheduler],
                        use_multiprocessing=True,
                        workers=4,
                        validation_steps=valid_steps,                      
                        shuffle=True)
                        # class_weight = classWeight; Not supported for 3+ Dimensions

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="UQ3iFCmSkCC6" outputId="733ae55d-e413-4ccc-973c-8014af0ade75"
# Evaluate and display results
results = model9.evaluate(testd) # Evaluate the model on test data
results = dict(zip(model9.metrics_names, results))

print(results)
plottingScores(H)

# %% colab={"base_uri": "https://localhost:8080/"} id="yQWyUdotFacC" outputId="8d0edeab-f85e-46ab-dfcd-fbb3c9fa5409"
# Prepare the test data: Use random 20 images
testd1 = train_data[15000:15020]
testd1.fillna(0, inplace=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="RJOFXzo7FaZo" outputId="17be8850-e9e4-4fff-fd49-914cfc7a49e7"
# Check target distrubution in test dataset (Equal distribution)
testd1.Target.value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} id="eJ0OgDOqFaXC" outputId="64edd608-d0ea-4bd4-c456-c677dda8630d"
# Set the TrainGenerator for test data
testd1_datagen = TrainGenerator(testd1)

# %% colab={"base_uri": "https://localhost:8080/"} id="LlgMk8RkFaUs" outputId="37cdff22-0596-4ade-bea9-7fac98b97047"
# Evaluate the model
test_steps = (len(testd1_datagen)//BATCH_SIZE)
if len(testd1_datagen) % BATCH_SIZE != 0:
    test_steps += 1

model9.evaluate(testd1_datagen)

# %% colab={"base_uri": "https://localhost:8080/"} id="jMXBlN4GFaSF" outputId="e1ff7516-5392-4c3c-fcb4-c0509ac5b21d"
# Prdict the test data that we have
pred_mask = model9.predict(testd1_datagen)

testd1 = testd1.reset_index()

# %% colab={"base_uri": "https://localhost:8080/"} id="QlxjkiWdFaPr" outputId="48e3827a-d188-447a-cf33-d948d7c7bacc"
# Calculate the y_test, y_pred, tmpImages, tmpMask, originalMask

y_pred = []
y_test = []
imageList = []
predMaskTemp = []
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def getPredictions(testd1):
    masks = np.zeros((int(testd1.shape[0]), IMAGE_HEIGHT, IMAGE_WIDTH))

    for index, row in testd1.iterrows():
        patientId = row.patientId
        # print(patientId)

        classlabel = row["Target"]
        dcm_file = 'stage_2_train_images/'+'{}.dcm'.format(patientId)
        dcm_data = dcm.read_file(dcm_file)
        img = dcm_data.pixel_array
        resized_img = cv2.resize(img,(IMAGE_HEIGHT,IMAGE_WIDTH), interpolation = cv2.INTER_LINEAR)
        predMaskTemp.append(pred_mask[index])
        iou = (pred_mask[index] > 0.5) * 1.0
        y_pred.append((1 in iou) * 1)
        imageList.append(resized_img)
        y_test.append(classlabel)
        x_scale = IMAGE_HEIGHT / 1024
        y_scale = IMAGE_WIDTH / 1024

        if(classlabel == 1):
            x = int(np.round(row['x'] * x_scale))
            y = int(np.round(row['y'] * y_scale))
            w = int(np.round(row['width'] * x_scale))
            h = int(np.round(row['height'] * y_scale))
            masks[index][y:y+h, x:x+w] = 1
    tmpImages = np.array(imageList)
    tmpMask = np.array(predMaskTemp)
    originalMask = np.array(masks)
    return (y_test, y_pred, tmpImages, tmpMask, originalMask)


# %% colab={"base_uri": "https://localhost:8080/"} id="Gc2mSPXTFaNB" outputId="f3196484-04af-4901-a0f9-f81d6007b232"
# Create predictions map
y_test, y_pred, imagelist, maskList, originalMask = getPredictions(testd1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 304} id="OeevCKgHFaKc" outputId="4d3dba0d-0132-419a-981e-eb7453dcfb01"
# Pick a random image
dcm_file = 'stage_2_train_images/'+'{}.dcm'.format('9358d1c5-ba61-4150-a233-41138208a3f9')
dcm_data = dcm.read_file(dcm_file)
img = dcm_data.pixel_array
plt.imshow(imagelist[12])

# %% colab={"base_uri": "https://localhost:8080/", "height": 800} id="qR64cQdiHS4S" outputId="567f7be3-ff0e-4b65-9a24-1e19353bff29"
# Visualize the train and output data 

fig = plt.figure(figsize=(15, 15))

a = fig.add_subplot(1, 4, 1)
imgplot = plt.imshow(imagelist[1])
a.set_title('Original Images ',fontsize=20)

a = fig.add_subplot(1, 4, 2)
imgplot = plt.imshow(imagelist[12])

a = fig.add_subplot(1, 4, 3)
imgplot = plt.imshow(imagelist[13])

a = fig.add_subplot(1, 4, 4)
imgplot = plt.imshow(imagelist[15])

fig = plt.figure(figsize=(15, 15))
a = fig.add_subplot(1, 4, 1)

imgplot = plt.imshow(originalMask[1])
a.set_title('Oringial Mask (Truth) ',fontsize=20)
a.set_xlabel('Pneumonia {}:'.format(y_test[1]), fontsize=20)

a = fig.add_subplot(1, 4, 2)
imgplot = plt.imshow(originalMask[12])
a.set_xlabel('Pneumonia {}:'.format(y_test[12]), fontsize=20)

a = fig.add_subplot(1, 4, 3)
imgplot = plt.imshow(originalMask[13])
a.set_xlabel('Pneumonia {}:'.format(y_test[13]), fontsize=20)

a = fig.add_subplot(1, 4, 4)
imgplot = plt.imshow(originalMask[15])
a.set_xlabel('Pneumonia {}:'.format(y_test[15]), fontsize=20)

fig = plt.figure(figsize=(15, 15))
a = fig.add_subplot(1, 4, 1)
a.set_title('Predicted Mask  ',fontsize=20)
imgplot = plt.imshow(maskList[1])
a.set_xlabel('Pneumonia {}:'.format(y_pred[1]), fontsize=20)

a = fig.add_subplot(1, 4, 2)
imgplot = plt.imshow(maskList[12])
a.set_xlabel('Pneumonia {}:'.format(y_pred[12]), fontsize=20)

a = fig.add_subplot(1, 4, 3)
imgplot = plt.imshow(maskList[13])
a.set_xlabel('Pneumonia {}:'.format(y_pred[13]), fontsize=20)

a = fig.add_subplot(1, 4, 4)
imgplot = plt.imshow(maskList[15])
a.set_xlabel('Pneumonia {}:'.format(y_pred[15]), fontsize=20)

# %% colab={"base_uri": "https://localhost:8080/", "height": 829} id="upjOxEqfkCC8" outputId="04fa7257-7296-48e3-d81d-b56f34184b53"
# Classification Report for test sample
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix for test sample
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index = [i for i in ['0', '1']],  
                         columns = [i for i in ['0', '1']])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="0CxffOYr0rDm" outputId="1f5ede3d-93fc-4bbc-cf73-f22aa2d0156f"
# Training history dataframe
hist = pd.DataFrame(H.history)

# Model comparison
Train_Accuracy = hist['accuracy'].mean()
# Test_Accuracy = model.evaluate(testd)

precision = results['precision']
recall = results['recall']
f1 = results['f1_score']
AUC = results['AUC']
Dice_Coef = results['dice_coef']
MeanIoU = results['mean_iou']

base_1 = []
base_1.append(['Model-9: UNet (MobileNet)', Train_Accuracy, results['accuracy'], precision, recall, f1, AUC, Dice_Coef, MeanIoU])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score', 'AUC', 'Dice Coef', 'MeanIoU'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% colab={"base_uri": "https://localhost:8080/", "height": 758} id="sZUARLUMAArN" outputId="06bd3249-88af-4220-8099-30c3e62ec77f"
# Understand the various Metrics with epochs
hist

# %% colab={"base_uri": "https://localhost:8080/", "height": 382} id="hq-yXKxiAItK" outputId="dfbaf5f1-10d4-434f-df24-e449fe067e30"
hist.describe()


# %% [markdown] id="Y_HH7i94R-eZ"
# ## Model-10: UNet (ResNet50)

# %% colab={"base_uri": "https://localhost:8080/"} id="uKKFCoUGlwKP" outputId="9c1f8786-640d-45d0-cdf0-3a396fe9369c"
# Function to create part of UNet architecture with ResNet50
def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    """
    Adds convolutional layer followed by Batch Normalization and Activation layers.
        
    Arguments:
        prevlayer -- previous layer of the convolution block        
        filters -- number of filters for convolution
        prefix -- prefix for the layer name
        strides -- convolution stride. Default is 1x1.
    """
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    
    # Returns the built layers of the block.
    return conv


# %% colab={"base_uri": "https://localhost:8080/"} id="kan3zbqtSGKv" outputId="5f6f6c5e-5727-4da3-d965-f20a6da96430"
# Define the UNet with ResNet50
def create_unet_resnet50(trainable=True):
    """Function to create UNet architecture with ResNet50.
        
    Arguments:
        trainable -- Flag to make layers trainable. Default value is 'True'.
    """
    resnetLayers = ResNet50(weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False) # Load pre-trained Resnet
    # Top layer is last layer of the model

    for layer in resnetLayers.layers:
        layer.trainable = trainable

    # Add all the UNet layers here
    convLayer_112by112 = resnetLayers.get_layer("conv1_relu").output
    convLayer_56by56 = resnetLayers.get_layer("conv2_block3_out").output # conv2_block3_2_relu
    convLayer_28by28 = resnetLayers.get_layer("conv3_block4_out").output # conv3_block4_2_relu
    convLayer_14by14 = resnetLayers.get_layer("conv4_block6_out").output # conv4_block6_2_relu
    convLayer_7by7 = resnetLayers.get_layer("conv5_block3_out").output # conv5_block3_2_relu
    # The last layer of resnet model(conv5_block3_out) is of dimensions (7x7x2048)
    # Start upsampling from 7x7 to 14x14 ...up to 224x224 to form UNet
    # concatinate with the original image layer of the same size from ResNet50
    up14by14 = Concatenate()([UpSampling2D()(convLayer_7by7), convLayer_14by14])
    upConvLayer_14by14 = conv_block_simple(up14by14, 256, "upConvLayer_14by14_1")
    upConvLayer_14by14 = conv_block_simple(upConvLayer_14by14, 256, "upConvLayer_14by14_2")
    
    up28by28 = Concatenate()([UpSampling2D()(upConvLayer_14by14), convLayer_28by28])
    upConvLayer_28by28 = conv_block_simple(up28by28, 256, "upConvLayer_28by28_1")
    upConvLayer_28by28 = conv_block_simple(upConvLayer_28by28, 256, "upConvLayer_28by28_2")
     
    up56by56 = Concatenate()([UpSampling2D()(upConvLayer_28by28), convLayer_56by56])
    upConvLayer_56by56 = conv_block_simple(up56by56, 256, "upConvLayer_56by56_1")
    upConvLayer_56by56 = conv_block_simple(upConvLayer_56by56, 256, "upConvLayer_56by56_2")    
    
    up112by112 = Concatenate()([UpSampling2D()(upConvLayer_56by56), convLayer_112by112])
    upConvLayer_112by112 = conv_block_simple(up112by112, 256, "upConvLayer_112by112_1")
    upConvLayer_112by112 = conv_block_simple(upConvLayer_112by112, 256, "upConvLayer_112by112_2")   
    
    up224by224 = UpSampling2D(name="unet_last")(upConvLayer_112by112) # upsample to 224x224
    upConvLayer_224by224 = conv_block_simple(up224by224, 256, "upConvLayer_224by224_1")
    upConvLayer_224by224 = conv_block_simple(upConvLayer_224by224, 256, "upConvLayer_224by224_2")
    # Add classification layer
    upConvLayer_224by224 = Conv2D(1, kernel_size=1, activation="sigmoid", name="masks")(upConvLayer_224by224)
    upConvLayer_224by224 = Reshape((IMAGE_SIZE, IMAGE_SIZE))(upConvLayer_224by224) 

    return Model(inputs=resnetLayers.input, outputs=upConvLayer_224by224)


# %% colab={"base_uri": "https://localhost:8080/"} id="h4cuPUnuSGKv" outputId="24fbe7a5-ce6e-4cf5-fb60-390907fe1763"
# Build the model 
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
model10 = create_unet_resnet50(input_shape)
model10.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="F6cLzDdjSGKv" outputId="271b0b71-ea34-4901-fdcb-552731e0552c"
# Compile the model
model10.compile(optimizer="adam", loss = dice_loss, metrics=METRICS)

# %% colab={"base_uri": "https://localhost:8080/"} id="5eQTOasuSGKw" outputId="1c46d012-4d40-476f-97d9-7b036af43a3c"
# Fit the model
train_steps = len(traind)//BATCH_SIZE
valid_steps = len(testd)//BATCH_SIZE

if len(traind) % BATCH_SIZE != 0:
    train_steps += 1
if len(testd) % BATCH_SIZE != 0:
    valid_steps += 1

H = model10.fit(traind, epochs=20,
                        steps_per_epoch=train_steps,
                        validation_data=testd,
                        callbacks = [checkpoint_cb, early_stopping_cb, lr_scheduler],
                        use_multiprocessing=True,
                        workers=4,
                        validation_steps=valid_steps,                      
                        shuffle=True)
                        # class_weight = classWeight; Not supported for 3+ Dimensions

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="hFpUgAA5SGKw" outputId="c40a793c-d644-47ce-b55e-7276d216baa0"
# Evaluate and display results
results = model10.evaluate(testd) # Evaluate the model on test data
results = dict(zip(model10.metrics_names, results))

print(results)
plottingScores(H)

# %% colab={"base_uri": "https://localhost:8080/"} id="13cn1S5XSGKw" outputId="0cd505ef-a1f8-4a68-aca5-3d698052e649"
# Prepare the test data: Use random 20 images
testd1 = train_data[15000:15020]
testd1.fillna(0, inplace=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="dabKPFU3SGKw" outputId="5bdad8cb-04ef-4754-8708-261748dee748"
# Check target distrubution in test dataset (Equal distribution)
testd1.Target.value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} id="kh9Tg7zzSGKw" outputId="cab319c0-3b68-4d3a-962c-16106089ab1f"
# Set the TrainGenerator for test data
testd1_datagen = TrainGenerator(testd1)

# %% colab={"base_uri": "https://localhost:8080/"} id="Ip1q7dcSSGKw" outputId="a8a552b1-064c-4138-a26b-dff4904f6f1a"
# Evaluate the model
test_steps = (len(testd1_datagen)//BATCH_SIZE)
if len(testd1_datagen) % BATCH_SIZE != 0:
    test_steps += 1

model10.evaluate(testd1_datagen)

# %% colab={"base_uri": "https://localhost:8080/"} id="q_-BXzm6SGKw" outputId="3b934bc4-20dd-4fe9-b3ac-8e1fc8fa2ae9"
# Predict the test data that we have
pred_mask = model10.predict(testd1_datagen)

testd1 = testd1.reset_index()

# %% colab={"base_uri": "https://localhost:8080/"} id="IKeBfLH9SGKw" outputId="0d5949be-8f30-4df0-c0a9-bceab5ca5437"
# Calculate the y_test, y_pred, tmpImages, tmpMask, originalMask

y_pred = []
y_test = []
imageList = []
predMaskTemp = []
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def getPredictions(testd1):
    masks = np.zeros((int(testd1.shape[0]), IMAGE_HEIGHT, IMAGE_WIDTH))

    for index, row in testd1.iterrows():
        patientId = row.patientId
        # print(patientId)

        classlabel = row["Target"]
        dcm_file = 'stage_2_train_images/'+'{}.dcm'.format(patientId)
        dcm_data = dcm.read_file(dcm_file)
        img = dcm_data.pixel_array
        resized_img = cv2.resize(img,(IMAGE_HEIGHT,IMAGE_WIDTH), interpolation = cv2.INTER_LINEAR)
        predMaskTemp.append(pred_mask[index])
        iou = (pred_mask[index] > 0.5) * 1.0
        y_pred.append((1 in iou) * 1)
        imageList.append(resized_img)
        y_test.append(classlabel)
        x_scale = IMAGE_HEIGHT / 1024
        y_scale = IMAGE_WIDTH / 1024

        if(classlabel == 1):
            x = int(np.round(row['x'] * x_scale))
            y = int(np.round(row['y'] * y_scale))
            w = int(np.round(row['width'] * x_scale))
            h = int(np.round(row['height'] * y_scale))
            masks[index][y:y+h, x:x+w] = 1
    tmpImages = np.array(imageList)
    tmpMask = np.array(predMaskTemp)
    originalMask = np.array(masks)
    return (y_test, y_pred, tmpImages, tmpMask, originalMask)


# %% colab={"base_uri": "https://localhost:8080/"} id="4E8gQcV1SGKw" outputId="1db89e33-50e3-4fcd-8e3c-aa3773dd421d"
# Create the predictions map
y_test, y_pred, imagelist, maskList, originalMask = getPredictions(testd1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 304} id="XdJfhByrSGKw" outputId="02f3e144-365d-42cd-b2c5-5c39c9b0fe15"
# Pick a random image
dcm_file = 'stage_2_train_images/'+'{}.dcm'.format('9358d1c5-ba61-4150-a233-41138208a3f9')
dcm_data = dcm.read_file(dcm_file)
img = dcm_data.pixel_array
plt.imshow(imagelist[12])

# %% colab={"base_uri": "https://localhost:8080/", "height": 800} id="tfXsnsXJSGKw" outputId="0f1f3a9d-a6e6-428c-b806-07ad06e202aa"
# Visualize the train and output data 

fig = plt.figure(figsize=(15, 15))

a = fig.add_subplot(1, 4, 1)
imgplot = plt.imshow(imagelist[1])
a.set_title('Original Images ',fontsize=20)

a = fig.add_subplot(1, 4, 2)
imgplot = plt.imshow(imagelist[12])

a = fig.add_subplot(1, 4, 3)
imgplot = plt.imshow(imagelist[13])

a = fig.add_subplot(1, 4, 4)
imgplot = plt.imshow(imagelist[15])

fig = plt.figure(figsize=(15, 15))
a = fig.add_subplot(1, 4, 1)

imgplot = plt.imshow(originalMask[1])
a.set_title('Oringial Mask (Truth) ',fontsize=20)
a.set_xlabel('Pneumonia {}:'.format(y_test[1]), fontsize=20)

a = fig.add_subplot(1, 4, 2)
imgplot = plt.imshow(originalMask[12])
a.set_xlabel('Pneumonia {}:'.format(y_test[12]), fontsize=20)

a = fig.add_subplot(1, 4, 3)
imgplot = plt.imshow(originalMask[13])
a.set_xlabel('Pneumonia {}:'.format(y_test[13]), fontsize=20)

a = fig.add_subplot(1, 4, 4)
imgplot = plt.imshow(originalMask[15])
a.set_xlabel('Pneumonia {}:'.format(y_test[15]), fontsize=20)

fig = plt.figure(figsize=(15, 15))
a = fig.add_subplot(1, 4, 1)
a.set_title('Predicted Mask  ',fontsize=20)
imgplot = plt.imshow(maskList[1])
a.set_xlabel('Pneumonia {}:'.format(y_pred[1]), fontsize=20)

a = fig.add_subplot(1, 4, 2)
imgplot = plt.imshow(maskList[12])
a.set_xlabel('Pneumonia {}:'.format(y_pred[12]), fontsize=20)

a = fig.add_subplot(1, 4, 3)
imgplot = plt.imshow(maskList[13])
a.set_xlabel('Pneumonia {}:'.format(y_pred[13]), fontsize=20)

a = fig.add_subplot(1, 4, 4)
imgplot = plt.imshow(maskList[15])
a.set_xlabel('Pneumonia {}:'.format(y_pred[15]), fontsize=20)

# %% colab={"base_uri": "https://localhost:8080/", "height": 829} id="U_5mLLb-SGKx" outputId="abf6222d-eb39-4232-82ea-460d463f18d5"
# Classification Report for test sample
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix for test sample
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index = [i for i in ['0', '1']],  
                         columns = [i for i in ['0', '1']])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="OibbCUDiSGKx" outputId="ada2553d-7f08-40b6-b4b4-5f5eea6386be"
# Training history dataframe
hist = pd.DataFrame(H.history)

# Model comparison
Train_Accuracy = hist['accuracy'].mean()
# Test_Accuracy = model.evaluate(testd)

precision = results['precision']
recall = results['recall']
f1 = results['f1_score']
AUC = results['AUC']
Dice_Coef = results['dice_coef']
MeanIoU = results['mean_iou']

# base_1 = []
base_1.append(['Model-10: UNet (ResNet50)', Train_Accuracy, results['accuracy'], precision, recall, f1, AUC, Dice_Coef, MeanIoU])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score', 'AUC', 'Dice Coef', 'MeanIoU'])
model_comparison.sort_values(by=['Dice Coef','MeanIoU'], inplace=True, ascending=False)

# %% [markdown] id="I1VD06dg5aCN"
# ## Model Comparison - Object Detection

# %% colab={"base_uri": "https://localhost:8080/", "height": 129} id="e2sxhM2TC6Vq" outputId="f2831753-ec7d-4d11-ee73-191364224e8c"
# Sumarize the results of modeling in a dataframe; Datapoints used = 4000; Epochs = 10/20; ADJUSTED_IMAGE_SIZE = 224
model_comparison

# %% [markdown] id="p7din9Wj-7cg"
# Dice Coefficient (F1 Score) and IoU:
#
# Simply put, the Dice Coefficient is 2 * the Area of Overlap divided by the total number of pixels in both images.
#
# The Dice coefficient is very similar to the IoU. They are positively correlated, meaning if one says model A is better than model B at segmenting an image, then the other will say the same. Like the IoU, they both range from 0 to 1, with 1 signifying the greatest similarity between predicted and truth.
#
# Intersection-Over-Union (IoU, Jaccard Index): The Intersection-Over-Union (IoU), also known as the Jaccard Index, is one of the most commonly used metrics in semantic segmentation. The IoU is a very straightforward metric that’s extremely effective.

# %% colab={"base_uri": "https://localhost:8080/", "height": 638} id="0Ehd4VsQkCDB" outputId="5b3190e8-8430-478b-8728-06c360ab0e1d"
# Bar graph for Model Vs. F1 Score
plt.figure(figsize=(20,10))
sns.barplot(data=model_comparison, x="Model", y="MeanIoU")
plt.title("Model Comparison: Model Vs. MeanIoU")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 656} id="rOE_B29TkCDB" outputId="b6ff6cec-e264-42f8-c360-dc64289ad5d7"
# Bar graph for Model Vs. Metrics
model_comparison[['Model', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Dice Coef', 'MeanIoU']].plot(kind='bar', x = 'Model', rot = 0, sort_columns = 'Model', figsize=(20,10))
plt.title("Model Comparison: Model Vs. Metrics")
plt.xlabel("Model")
plt.ylabel("Metrics")

# %% [markdown] id="v9XHLlwUSJzM"
# ## Pickle the model

# %% colab={"base_uri": "https://localhost:8080/"} id="ktpgWaakSZzS" outputId="6cf54be3-6739-498e-d33d-c389824c7527"
# Save the Classification model to a file in the current working directory
Pkl_Filename = "Pickle_Classification_EfficientNetV2B3_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model7, file)

# # Load the DL Model back from the file
# with open(Pkl_Filename, 'rb') as file:  
#     Pickle_Classification_EfficientNetV2B3_Model = pickle.load(file)
    
# Pickle_Classification_EfficientNetV2B3_Model

# %% colab={"base_uri": "https://localhost:8080/"} id="T3eVVVIgSZww" outputId="1fc20c39-0f5d-42d8-b2bb-d1d2a9ff3181"
# Save the Object Detection model to a file in the current working directory
Pkl_Filename = "Pickle_ObjectDetection_UNetMobileNet_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model9, file)

# # Load the DL Model back from the file
# with open(Pkl_Filename, 'rb') as file:  
#     Pickle_ObjectDetection_UNetMobileNet_Model = pickle.load(file)
    
# Pickle_ObjectDetection_UNetMobileNet_Model

# %% [markdown] id="55MGp5BcklCG"
# # Project Summary:
# 1. We understood the value of CRISP-DM project management framework in the realm of data science domain.
# 2. We have completed the EDA, and analysed the important and relevant attributes of our dataset in greater details.
# 3. We have done the pre-processing of the DICOM images, and collected additional information from the images. In the process, we learned to access, analyse and process the images for data modelling.
# 4. We have built the various CNN models for classification with and without transfer learning. In the process, we learned to use various state-of-the art pre trained models with ImageNet weights.
# 5. We have built UNET models for semantic segmentation of images.
# 6. We learned to differentiate and make use of various evaluation metrics to understand the quality of our models.
# 7. Given the time and computing resources, we can also experiment with Mask RCNN, Yolo, CheXNet and other more advanced models.
# 8. We can improve the Dice Coef. and MeanIoU by image augmentation, increasing the sample size, image size and hyperparameter tuning.
# 9. We would have done some up sampling of the 'Lung Opacity' class using augmentations. This would have helped us improving Recall value for 'Lung Opacity' class.
# 10. Classification error to understand the level of confidence of our various metrics is also a great factor to consider while quoting the performance metrics of various models.
# 11. Understanding Kappa, MCC and mAP are also great factors to consider for model evaluation measures.
# 12. We can also implement the Bayesian optimization techniques for hyperparameter tuning.
# 13. Overall, it was a great experience working on this project and we learned a lot in the journey.
#
#

# %% [markdown] id="VSKPDabt4axO"
# # System Resources Used
# https://colab.research.google.com/drive/151805XTDg--dgHb3-AXJCpnWaqRhop_2#scrollTo=gsqXZwauphVV

# %% colab={"base_uri": "https://localhost:8080/"} id="ln9b03OsmjEB" outputId="517c6d6b-426a-4cf1-8a93-5127819742d1"
# !lscpu |grep 'Model name'

# %% colab={"base_uri": "https://localhost:8080/"} id="SVx9OHXf3gQX" outputId="8ce21e2a-70e8-4a7e-b469-2a567543735f"
# Memory that we can use
# max memory used = 8GB
# !free -h --si | awk  '/Mem:/{print $2}'

# %% colab={"base_uri": "https://localhost:8080/"} id="3ZZprdn33rzb" outputId="49fbd067-e5e8-4cf5-f0de-f6223f8b7272"
# if it had turbo boost it would've shown Min and Max MHz also but it is only showing current frequency 
# this means it always operates at shown frequency
# !lscpu | grep "MHz"

# %% colab={"base_uri": "https://localhost:8080/"} id="4qbWTjz-3lw9" outputId="c8d20968-0f8f-48f7-85f8-753f968c1744"
# hard disk space that we can use
# !df -h / | awk '{print $4}'

# %% colab={"base_uri": "https://localhost:8080/"} id="agcCa-N65K4q" outputId="8807a01f-f1f0-487e-fdee-2e14a98ea7b6"
# no.of cores each processor is having 
# !lscpu | grep 'Core(s) per socket:'

# %% colab={"base_uri": "https://localhost:8080/"} id="0gRmNlOw5Ku0" outputId="5f083095-2600-474f-b86f-bfc65f48e43f"
# no.of threads each core is having
# !lscpu | grep 'Thread(s) per core'

# %% colab={"base_uri": "https://localhost:8080/"} id="10ZEz5I_5Kk9" outputId="c81b0aee-8d86-4bff-dae6-ad6b29239152"
# no.of sockets i.e available slots for physical processors
# !lscpu | grep 'Socket(s):'

# %% colab={"base_uri": "https://localhost:8080/"} id="8iHgc-gh5ahw" outputId="286e4077-4de3-4475-eb6e-d835a83ff3d4"
# !lscpu | grep "L3 cache" 

# %% colab={"base_uri": "https://localhost:8080/"} id="40_IHARL5nWg" outputId="380edf0a-15d1-4190-973d-b8d037e829ad"
# use this command to see GPU activity while doing Deep Learning tasks, 
# for this command 'nvidia-smi' and for above one to work, go to 
# 'Runtime > change runtime type > Hardware Accelerator > GPU'
# !nvidia-smi

# %% colab={"base_uri": "https://localhost:8080/"} id="2a2sXE4W5v9y" outputId="42b15d54-aa10-4242-a464-b59f1a29e9e0"
# GPU count and name
# !nvidia-smi -L

# %% colab={"base_uri": "https://localhost:8080/"} id="3zp5i4x86Nba" outputId="59a04db5-ce89-4e53-a698-c1cee9460bd4"
# Timestamp
# !rm /etc/localtime
# !ln -s /usr/share/zoneinfo/Asia/Bangkok /etc/localtime
# !date

# %% colab={"base_uri": "https://localhost:8080/"} id="rn6cMo_Yunra" outputId="d545de9b-167d-462a-cfaf-114b98b13207"
# NB end time
b = time.time()
# Total NB Runtime
# print(f"Total NB Run Time (min.): {(b-a)/60}")

# %% [markdown] id="s1epDLXh3-BS"
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
