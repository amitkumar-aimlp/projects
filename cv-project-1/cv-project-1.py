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
# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#00b3e5;">Contents: Computer Vision Project - 1
# 1. [Part-A: Solution](#Part-A:-Solution)
# 2. [Part-B: Solution](#Part-B:-Solution)

# %% [markdown] id="5f36f6dd"
# # Part-A: Solution

# %% [markdown] id="dedd939b"
# - **DOMAIN:** Botanical Research
# - **CONTEXT:** University X is currently undergoing some research involving understanding the characteristics of plant and plant seedlings at various stages of growth. They already have have invested on curating sample images. They require an automation which can create a classifier capable of determining a plant's species from a photo.
# - **DATA DESCRIPTION:** The dataset comprises of images from 12 plant species.
# Source: https://www.kaggle.com/c/plant-seedlings-classification/data.
# - **PROJECT OBJECTIVE:** To create a classifier capable of determining a plant's species from a photo.

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
from sklearn.metrics import plot_precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, plot_roc_curve 

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

import cv2
from google.colab.patches import cv2_imshow
from glob import glob
import itertools

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation,GlobalMaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam,RMSprop
from keras.utils.np_utils import to_categorical  
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings("ignore")

import random

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %% colab={"base_uri": "https://localhost:8080/"} id="peP2nm91hndX" outputId="f56f2f90-326c-4096-8103-b36c93a0cb70"
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown] id="a3313f55"
# ## 1. Import and Understand the data

# %% [markdown] id="fb411bd2"
# ### 1A. Extract ‘plant-seedlings-classification.zip’ into new folder (unzipped) using python.

# %% id="ff8fc39f"
# Set the path to the dataset folder. (The dataset contains image folder: "train")
train_path = "/content/drive/MyDrive/MGL/Project-CV-1/plant-seedlings-classification.zip"

# Extract the files from dataset to train_temp
from zipfile import ZipFile
with ZipFile(train_path, 'r') as zip:
  zip.extractall('/content/drive/MyDrive/MGL/Project-CV-1/unzip-psc')

# %% [markdown] id="a0dc04f4"
# ### 1B. Map the images from train folder with train labels to form a DataFrame.
# - Using Numpy array instead of dataframe as it is faster for image processing.
# - Converting images into numpy arrays is important, as the model accepts inputs in the form of numpy arrays.

# %% colab={"base_uri": "https://localhost:8080/"} id="7545a1df" outputId="7dd6d26f-5124-4154-83d0-52f9961f368d"
# The path to all images in training set. (* means include all folders and files.)
path = "/content/drive/MyDrive/MGL/Project-CV-1/unzip-psc/plant-seedlings-classification/train/*/*.png"  
files = glob(path)

# Initialize empty list to store the image data as numbers.
X = []
# Initialize empty list to store the labels of images
y = [] 
j = 1
num = len(files)

# Collect images and labels in X and y
for img in files:
    '''
    Append the image data to X list.
    Append the labels to y list.
    '''
    print(str(j) + "/" + str(num), end="\r")
    # Get image (with resizing to 128 x 128)
    X.append(cv2.resize(cv2.imread(img), (128, 128))) 
    # Get image label (folder name contains the class to which the image belongs) 
    y.append(img.split('/')[-2])  
    j += 1

# Train images set
X = np.asarray(X) 
# Train labels set 
y = pd.DataFrame(y)  

# %% id="DH9WDhcOzYGc"
# # Save the data
# np.save('/content/drive/MyDrive/MGL/Project-CV-1/X.npy', X)
# y.to_pickle('/content/drive/MyDrive/MGL/Project-CV-1/X.pkl')

# %% id="K033V__a5MRe"
# # Load the data
# X = np.load('/content/drive/MyDrive/MGL/Project-CV-1/X.npy')
# y = pd.read_pickle('/content/drive/MyDrive/MGL/Project-CV-1/X.pkl')

# %% colab={"base_uri": "https://localhost:8080/"} id="1301611b" outputId="29578fdc-7c76-490c-a04b-32010c69578e"
print(X.shape)
print(y.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="72775fb2" outputId="1d4cce71-8bda-4926-d8b5-5aa46858daa7"
y[0].value_counts()


# %% [markdown] id="97580215"
# ### 1C. Write a function that will select n random images and display images along with its species.

# %% colab={"base_uri": "https://localhost:8080/", "height": 520} id="4iI-PKee7GGd" outputId="ff8d424f-d751-4e8f-b7c0-33ea783c362c"
def random_images():
  r = random.randint(1, 200)

  f = plt.figure(figsize=(20, 20))

  f.add_subplot(2, 6, 1)
  plt.imshow(X[0+r])
  plt.title(y[0][0+r])

  f.add_subplot(2, 6, 2)
  plt.imshow(X[400+r])
  plt.title(y[0][400+r])

  f.add_subplot(2, 6, 3)
  plt.imshow(X[750+r])
  plt.title(y[0][750+r])
  
  f.add_subplot(2, 6, 4)
  plt.imshow(X[1300+r])
  plt.title(y[0][1300+r])

  f.add_subplot(2, 6, 5)
  plt.imshow(X[1700+r])
  plt.title(y[0][1700+r])

  f.add_subplot(2, 6, 6)
  plt.imshow(X[1600+r])
  plt.title(y[0][1600+r])

  f.add_subplot(1, 6, 1)
  plt.imshow(X[2400+r])
  plt.title(y[0][2400+r])

  f.add_subplot(1, 6, 2)
  plt.imshow(X[2630+r])
  plt.title(y[0][2630+r])

  f.add_subplot(1, 6, 3)
  plt.imshow(X[3000+r])
  plt.title(y[0][3000+r])
  
  f.add_subplot(1, 6, 4)
  plt.imshow(X[3200+r])
  plt.title(y[0][3200+r])

  f.add_subplot(1, 6, 5)
  plt.imshow(X[3538+r])
  plt.title(y[0][3538+r])

  f.add_subplot(1, 6, 6)
  plt.imshow(X[4500+r])
  plt.title(y[0][4500+r])

# Using the function to display 12 random images
random_images()

# %% [markdown] id="5579308b"
# ## 2. Data preprocessing

# %% [markdown] id="8c19ed0e"
# ### 2A. Create X & Y from the DataFrame.

# %% id="4c72efef"
# This is completed in part 1B above.

# %% [markdown] id="eee0ed1e"
# ### 2B. Encode labels of the images.

# %% colab={"base_uri": "https://localhost:8080/"} id="7a014062" outputId="3a15e536-96c2-4d8b-a35f-12fbe956829e"
# Using LabelEncoder from preprocessing
labels = preprocessing.LabelEncoder()
labels.fit(y[0])
print('Classes'+str(labels.classes_))

# %% colab={"base_uri": "https://localhost:8080/"} id="hXo6erNcRc9G" outputId="23fdd5dc-83d4-4d40-f239-c81885086a3f"
encodedLabel = labels.transform(y[0])
convertedLabels = np_utils.to_categorical(encodedLabel)
classes = convertedLabels.shape[1]
print(str(classes))

# %% [markdown] id="895e5f33"
# ### 2C. Unify shape of all the images.

# %% id="afbe6f2f"
# This is completed in part 1B above.
# All images have been resized to (128, 128, 3).

# %% [markdown] id="944ee94e"
# ### 2D. Normalise all the images.

# %% id="ab659401"
# Normalization of images
X = X.astype('float32') / 255.0

# %% [markdown] id="c891804a"
# ## 3. Model training
# - 3A. Split the data into train and test data.
# - 3B. Create new CNN architecture to train the model
# - 3C. Train the model on train data and validate on test data.
# - 3D. Select a random image and print actual label and predicted label for the same.

# %% [markdown] id="nOzjUHGwE8Af"
# > Due to time constraint, I am using small number of epochs, and training only the model-1. Similar process can be replicated for other complex models (From Model-2 to Model-5).

# %% colab={"base_uri": "https://localhost:8080/"} id="eicONwJPK22L" outputId="3a4dfa8a-cefb-4c30-883a-151276e17354"
# Shape of the dataset
print(X.shape)
print(y.shape)

# %% id="vqTra0j21t4M"
# Split X and y into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, convertedLabels, test_size=0.3, random_state=0, stratify=convertedLabels)

# %% colab={"base_uri": "https://localhost:8080/"} id="o7yT7nu12mBZ" outputId="98a9a2a8-b571-4e1a-d190-a631ddf8e0d9"
# Shape of train and test datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="gto2k5Se2rwW" outputId="c2f26f86-8af7-431b-ca52-cd2972ee9d4e"
labels.classes_[0], labels.classes_[1], labels.classes_[11]

# %% colab={"base_uri": "https://localhost:8080/"} id="AWGuYJd67yQA" outputId="1205e1d1-c1fd-4aa7-98b5-6edf0d30f125"
labels.classes_

# %% colab={"base_uri": "https://localhost:8080/", "height": 285} id="kQGuiuJBaFCh" outputId="b7a08753-1af3-4854-bcf6-ca1f72420d08"
plt.imshow(X_train[0])

# %% colab={"base_uri": "https://localhost:8080/"} id="-jBv61_F2oFo" outputId="7da96d8f-0611-4b95-cc6e-ae01ea38a6eb"
y_train[0]

# %% colab={"base_uri": "https://localhost:8080/", "height": 285} id="hJhLadsr3xmU" outputId="e21265c2-6824-4cd9-9e75-75d114f2064a"
plt.imshow(X_test[0])

# %% colab={"base_uri": "https://localhost:8080/"} id="b2bPK6213xbX" outputId="6c5b50ec-8465-46d4-de89-595f52ab95f5"
y_test[0]

# %% [markdown] id="sW-LZLBWpitu"
# ### Model-1
# - 2 convolution layers ( filters=64 / 128 , kernel_size=(3, 3) activation='relu')
# - MaxPool2D((2, 2)
# - Dropout(0.25)
# - Flatten
# - 2 dense layers (128 / 64, activation='relu')
# - Dropout(0.25)
# - loss='categorical_crossentropy', optimizer='adam'
# - model compile with ImageDataGenerator to minimize overfitting.
# - shuffle = True

# %% id="OwgJ6PhgoVgN"
generator = ImageDataGenerator(rotation_range = 180,
                               zoom_range = 0.2,
                               width_shift_range = 0.2,
                               height_shift_range = 0.2,
                               horizontal_flip = True,
                               vertical_flip = True)
generator.fit(X_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="2HbX2Xp2oVek" outputId="fe8eaca5-f509-42f3-97fa-11943142d996"
model1 = Sequential()

model1.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
model1.add(MaxPool2D((2, 2)))
model1.add(Dropout(0.25))

model1.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model1.add(MaxPool2D((2, 2)))
model1.add(Dropout(0.25))

model1.add(Flatten())

model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.25))

model1.add(Dense(64, activation='relu'))
model1.add(Dropout(0.25))

model1.add(Dense(classes, activation='softmax'))

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model1.summary()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="-OfBP-ruoVc4" outputId="ffe5e1a9-8fea-4198-b5df-45325180ad38"
history1 = model1.fit(generator.flow(X_train,y_train,batch_size=200), epochs=50, verbose=2, shuffle=True, validation_data=(X_test,y_test))
pd.DataFrame(history1.history)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="8hVFVKjJ2s6E" outputId="b9c47121-a297-4399-98fe-7f483476ee55"
# Capturing learning history per epoch
hist  = pd.DataFrame(history1.history)
hist['epoch'] = history1.epoch

# Plotting accuracy at different epochs
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model1.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 283} id="RvTL4eQPA7mE" outputId="ca9096e8-9afd-4136-99e7-c9b33aa7504f"
# Capturing learning history per epoch
hist  = pd.DataFrame(history1.history)
hist['epoch'] = history1.epoch

# Plotting accuracy at different epochs
plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model1.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="2mSnDj1F6mtm" outputId="91b277ae-2880-4f96-f98c-0bae8a4ba865"
y_pred=model1.predict(X_test)
y_pred

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="BcOxend26ljP" outputId="dc6d29cd-0cec-4bb7-e22b-cd4aa8397b66"
# Classification Accuracy
print("Classification Accuracy:")
print('Loss and Accuracy on Training data:',model1.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model1.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
df_cm = pd.DataFrame(cm, index = [i for i in [labels.classes_]],  
                         columns = [i for i in [labels.classes_]])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
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
# **Considering the Class Recall, Precision, F1 Score and Accuracy as the most important parameters to decide the best model for this problem. We may have higher values in the below models here. Please refer the other models for the same.**
# - A decent performance is visible from the above metrics.
# - As evident, Model-1 is not overfitting, we can use more complex models to improve the performance.
# - A Balanced dataset will be more benefcial in our case.
# - The number of features in the image dataset are huge compared to the number of examples, this is a challenge for DNNs and CNNs.
# - To overcome overfitting, we can use data augmentation, batch normalization, and transfer learning.

# %% [markdown] id="9BSF2-kXVsdh"
# ### Predicting the class of a single image:

# %% colab={"base_uri": "https://localhost:8080/", "height": 303} id="D7U5cQDNQMNc" outputId="8926f40a-08e0-419a-ee4d-c4c782cc4f9c"
# Using test images from the given dataset
# Test Image as n1
n1 = 51
plt.imshow(X_test[n1])

# y_pred=model1.predict(X_test)

# Actual Class
print("Actual Class: ",labels.classes_[y_test.argmax(axis=1)[n1]])
# Predicted Class
print("Predicted Class: ",labels.classes_[y_pred.argmax(axis=1)[n1]])

# %% [markdown]
# **Additional Models; Not part of the Project Part-A:**

# %% [markdown] id="EtNMJVcopn23"
# ### Model-2
# - 3 convolution layers (filters=64/128/128 , kernel_size=(3, 3) activation='relu')
# - MaxPool2D((2, 2),
# - Dropout(0.25)
# - Flatten
# - 1 dense layer (256, activation='relu')
# - Dropout(0.5)
# - loss='categorical_crossentropy', optimizer='adam'
# - model compile with ImageDataGenerator to minimize overfitting.
# - shuffle = True

# %% id="latyEJwj3naY"
# generator = ImageDataGenerator(rotation_range = 180,
#                                zoom_range = 0.2,
#                                width_shift_range = 0.2,
#                                height_shift_range = 0.2,
#                                horizontal_flip = True,
#                                vertical_flip = True)
# generator.fit(X_train)

# %% id="Ilh6aYdnDWAM"
# model2 = Sequential()

# model2.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
# model2.add(MaxPool2D((2, 2)))
# model2.add(Dropout(0.25))

# model2.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
# model2.add(MaxPool2D((2, 2)))
# model2.add(Dropout(0.25))

# model2.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
# model2.add(MaxPool2D((2, 2)))
# model2.add(Dropout(0.25))

# model2.add(Flatten())

# model2.add(Dense(256, activation='relu'))
# model2.add(Dropout(0.5))

# model2.add(Dense(classes, activation='softmax'))

# model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model2.summary()

# %% id="fNrsrPcdDlym"
# history2 = model2.fit(generator.flow(X_train,y_train,batch_size=64),epochs=100, verbose=2,shuffle=True,validation_data=(X_test,y_test))
# pd.DataFrame(history2.history)

# %% id="OuI0RQWTDykk"
# # Plotting Train Loss vs Validation Loss
# plt.plot(history2.history['loss'])
# plt.plot(history2.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# %% id="POAWqA7MuoCc"
# # Capturing learning history per epoch
# hist  = pd.DataFrame(history2.history)
# hist['epoch'] = history2.epoch

# # Plotting accuracy at different epochs
# plt.plot(hist['accuracy'])
# plt.plot(hist['val_accuracy'])
# plt.legend(("train" , "valid") , loc =0)

# # Printing results
# results = model2.evaluate(X_test, y_test)

# %% id="iNpUTgU5QF3B"
# y_pred=model2.predict(X_test)
# y_pred

# %% id="vOGhTWMJMyz2"
# # Classification Accuracy
# print("Classification Accuracy:")
# print('Loss and Accuracy on Training data:',model2.evaluate(X_train, y_train))
# print('Loss and Accuracy on Test data:',model2.evaluate(X_test, y_test))
# print()

# # Classification Report
# print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# # Confusion Matrix
# print("Confusion Matrix Chart:")
# cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
# df_cm = pd.DataFrame(cm, index = [i for i in [labels.classes_]],  
#                          columns = [i for i in [labels.classes_]])
# plt.figure(figsize = (12,10))
# sns.heatmap(df_cm, annot=True, fmt='g')
# plt.title('Confusion Matrix')
# plt.ylabel('Actual Label')
# plt.xlabel('Predicted Label')
# plt.show()

# %% [markdown] id="YiO013Dqpvvo"
# ### Model-3
# - 4 convolution layers (filters=64/64/128/256, kernel_size=(3, 3) activation='relu')
# - MaxPool2D((2, 2)
# - Dropout(0.25)
# - GlobalMaxPool2D
# - Flatten
# - 2 dense layers (256 / 256, activation='relu')
# - Dropout(0.25)
# - loss='categorical_crossentropy', optimizer='adam'
# - model compile with ImageDataGenerator to minimize overfitting.
# - shuffle = True

# %% id="oQs2RhjvOHky"
# model3 = Sequential()

# model3.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
# model3.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model3.add(MaxPool2D((2, 2)))
# model3.add(Dropout(0.25))

# model3.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
# model3.add(MaxPool2D((2, 2)))
# model3.add(Dropout(0.25))

# model3.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
# model3.add(MaxPool2D((2, 2)))
# model3.add(Dropout(0.25))

# model3.add(GlobalMaxPool2D())

# model3.add(Flatten())

# model3.add(Dense(256, activation='relu'))
# model3.add(Dropout(0.25))

# model3.add(Dense(256, activation='relu'))
# model3.add(Dropout(0.25))

# model3.add(Dense(classes, activation='softmax'))

# model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model3.summary()

# %% id="Hq-r37teOPtb"
# history3 = model3.fit(generator.flow(X_train,y_train,batch_size=64),epochs=200, verbose=2,shuffle=True,validation_data=(X_test,y_test))
# pd.DataFrame(history3.history)

# %% id="J_oQNYWaO6XO"
# # Plotting Train Loss vs Validation Loss
# plt.plot(history3.history['loss'])
# plt.plot(history3.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# %% id="yjokd49Vutxz"
# # Capturing learning history per epoch
# hist  = pd.DataFrame(history3.history)
# hist['epoch'] = history3.epoch

# # Plotting accuracy at different epochs
# plt.plot(hist['accuracy'])
# plt.plot(hist['val_accuracy'])
# plt.legend(("train" , "valid") , loc =0)

# # Printing results
# results = model3.evaluate(X_test, y_test)

# %% id="qGALVP6FQ28R"
# y_pred=model3.predict(X_test)
# y_pred

# %% id="93Z1aHa6PSnd"
# # Classification Accuracy
# print("Classification Accuracy:")
# print('Loss and Accuracy on Training data:',model3.evaluate(X_train, y_train))
# print('Loss and Accuracy on Test data:',model3.evaluate(X_test, y_test))
# print()

# # Classification Report
# print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# # Confusion Matrix
# print("Confusion Matrix Chart:")
# cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
# df_cm = pd.DataFrame(cm, index = [i for i in [labels.classes_]],  
#                          columns = [i for i in [labels.classes_]])
# plt.figure(figsize = (12,10))
# sns.heatmap(df_cm, annot=True, fmt='g')
# plt.title('Confusion Matrix')
# plt.ylabel('Actual Label')
# plt.xlabel('Predicted Label')
# plt.show()

# %% [markdown] id="eEsu5LSwp1Aw"
# ### Model-4
# - 6 convolution layers (filters=64/64/128/128/256/256, kernel_size=(3, 3) activation='relu')
# - MaxPool2D((2, 2)
# - Dropout(0.25)
# - GlobalMaxPool2D
# - Flatten
# - 3 dense layers (256/256/256, activation='relu')
# - Dropout(0.25)
# - loss='categorical_crossentropy', optimizer='adam'
# - model compile with ImageDataGenerator to minimize overfitting.
# - shuffle = True

# %% id="2ewIXFEMc1On"
# generator = ImageDataGenerator(rotation_range = 180,
#                                zoom_range = 0.2,
#                                width_shift_range = 0.2,
#                                height_shift_range = 0.2,
#                                horizontal_flip = True,
#                                vertical_flip = True)
# generator.fit(X_train)

# %% id="S35fyOqccu2g"
# model4 = Sequential()

# model4.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
# model4.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model4.add(MaxPool2D((2, 2)))
# model4.add(Dropout(0.25))

# model4.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
# model4.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
# model4.add(MaxPool2D((2, 2)))
# model4.add(Dropout(0.25))

# model4.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
# model4.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
# model4.add(MaxPool2D((2, 2)))
# model4.add(Dropout(0.25))

# model4.add(GlobalMaxPool2D())

# model4.add(Flatten())

# model4.add(Dense(256, activation='relu'))
# model4.add(Dropout(0.25))

# model4.add(Dense(256, activation='relu'))
# model4.add(Dropout(0.25))

# model4.add(Dense(256, activation='relu'))
# model4.add(Dropout(0.25))

# model4.add(Dense(classes, activation='softmax'))

# model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model4.summary()

# %% id="Z4t8NRUIc2Q8"
# history4 = model4.fit(generator.flow(X_train,y_train,batch_size=64),epochs=200, verbose=2,shuffle=True,validation_data=(X_test,y_test))
# pd.DataFrame(history4.history)

# %% id="SasKaoQ1c9x6"
# #Plotting Train Loss vs Validation Loss
# plt.plot(history4.history['loss'])
# plt.plot(history4.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# %% id="D0xOpbkDuxyn"
# # Capturing learning history per epoch
# hist  = pd.DataFrame(history4.history)
# hist['epoch'] = history4.epoch

# # Plotting accuracy at different epochs
# plt.plot(hist['accuracy'])
# plt.plot(hist['val_accuracy'])
# plt.legend(("train" , "valid") , loc =0)

# # Printing results
# results = model4.evaluate(X_test, y_test)

# %% id="o8DAqiCgdDh8"
# y_pred=model4.predict(X_test)
# y_pred

# %% id="7M8WmAZJdVg4"
# # Classification Accuracy
# print("Classification Accuracy:")
# print('Loss and Accuracy on Training data:',model4.evaluate(X_train, y_train))
# print('Loss and Accuracy on Test data:',model4.evaluate(X_test, y_test))
# print()

# # Classification Report
# print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# # Confusion Matrix
# print("Confusion Matrix Chart:")
# cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
# df_cm = pd.DataFrame(cm, index = [i for i in [labels.classes_]],  
#                          columns = [i for i in [labels.classes_]])
# plt.figure(figsize = (12,10))
# sns.heatmap(df_cm, annot=True, fmt='g')
# plt.title('Confusion Matrix')
# plt.ylabel('Actual Label')
# plt.xlabel('Predicted Label')
# plt.show()

# %% [markdown] id="a53aDqfDp5ca"
# ### Model-5

# %% id="vHH3RsuUAqDG"
# # Initialize the object of ImageDataGenerator
# datagen= tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
#                                                          width_shift_range=0.2,
#                                                          height_shift_range=0.2,
#                                                          zoom_range=[0.4,1.5],
#                                                          horizontal_flip=True,
#                                                          vertical_flip=True)

# datagen.fit(X_train)

# %% id="m9M359cXAp0o"
# # Initialize and Build the Model
# #Clear any previous model from memory
# tf.keras.backend.clear_session()

# #Initialize model
# model5 = tf.keras.models.Sequential()

# #Add 1st Conv Layer
# model5.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))

# #Add 2nd Conv Layer
# model5.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))

# #normalize data
# model5.add(tf.keras.layers.BatchNormalization())

# #Add Max Pool layer
# model5.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

# #Add 3rd Conv Layer
# model5.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu'))

# #normalize data
# model5.add(tf.keras.layers.BatchNormalization())

# #Add Max Pool layer
# model5.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

# #Add 4th Conv Layer
# model5.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu'))

# #normalize data
# model5.add(tf.keras.layers.BatchNormalization())

# #Add Max Pool layer
# model5.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

# #Add Global Max Pool layer
# model5.add(tf.keras.layers.GlobalMaxPool2D())

# #Flatten the data
# model5.add(tf.keras.layers.Flatten())

# #Add 1st dense layer
# model5.add(tf.keras.layers.Dense(128, activation='relu'))

# #normalize data
# model5.add(tf.keras.layers.BatchNormalization())

# #Add Dropout
# model5.add(tf.keras.layers.Dropout(0.3))

# #Add 2nd dense layer
# model5.add(tf.keras.layers.Dense(128, activation='relu'))

# #normalize data
# model5.add(tf.keras.layers.BatchNormalization())

# #Add Dropout
# model5.add(tf.keras.layers.Dropout(0.3))

# #Add Output Layer
# model5.add(tf.keras.layers.Dense(12, activation='softmax'))

# %% id="e8lIm-Y_Apoz"
# # Compile the model
# # Specify Loss and Optimizer
# model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %% id="i6zYDTYnAozA"
# #Model Summary
# model5.summary()

# %% id="Zrcr_NijB5_P"
# history5 = model5.fit_generator(datagen.flow(X_train, y_train, batch_size=200), 
#                     epochs=50, validation_data=(X_test, y_test), verbose = 2)

# %% id="WhoSoKwqBMjT"
# #Plotting Train Loss vs Validation Loss
# plt.plot(history5.history['loss'])
# plt.plot(history5.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# %% id="mR6VR-IPu7hN"
# # Capturing learning history per epoch
# hist  = pd.DataFrame(history5.history)
# hist['epoch'] = history5.epoch

# # Plotting accuracy at different epochs
# plt.plot(hist['accuracy'])
# plt.plot(hist['val_accuracy'])
# plt.legend(("train" , "valid") , loc =0)

# # Printing results
# results = model5.evaluate(X_test, y_test)

# %% id="SoQzHNVXBMjT"
# y_pred=model5.predict(X_test)
# y_pred

# %% id="yvwPGMyqBMjU"
# # Classification Accuracy
# print("Classification Accuracy:")
# print('Loss and Accuracy on Training data:',model5.evaluate(X_train, y_train))
# print('Loss and Accuracy on Test data:',model5.evaluate(X_test, y_test))
# print()

# # Classification Report
# print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# # Confusion Matrix
# print("Confusion Matrix Chart:")
# cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
# df_cm = pd.DataFrame(cm, index = [i for i in [labels.classes_]],  
#                          columns = [i for i in [labels.classes_]])
# plt.figure(figsize = (12,10))
# sns.heatmap(df_cm, annot=True, fmt='g')
# plt.title('Confusion Matrix')
# plt.ylabel('Actual Label')
# plt.xlabel('Predicted Label')
# plt.show()

# %% [markdown] id="0dd7c2e4"
# # Part-B: Solution

# %% [markdown] id="e624389a"
# - **DOMAIN:** Botanical Research
# - **CONTEXT:** University X is currently undergoing some research involving understanding the characteristics of flowers. They already have invested on curating sample images. They require an automation which can create a classifier capable of determining a flower’s species from a photo.
# - **DATA DESCRIPTION:** The dataset comprises of images from 17 plant species.
# - **PROJECT OBJECTIVE:** To experiment with various approaches to train an image classifier to predict type of flower from the image.

# %% id="26856bb6"
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
from sklearn.metrics import plot_precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, plot_roc_curve 

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

import cv2
from google.colab.patches import cv2_imshow
from glob import glob
import itertools

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalMaxPool2D, GlobalAveragePooling2D
from tensorflow.keras import Input

from keras.utils.np_utils import to_categorical  
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %% colab={"base_uri": "https://localhost:8080/"} id="qM2-0wcYgGXW" outputId="bc60b120-d429-4ec7-ee5c-679920d99e1e"
# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')

# %% [markdown] id="f0150bbf"
# ## 1. Import and Understand the data

# %% [markdown] id="406e16b3"
# ### 1A. Import and read oxflower17 dataset from tflearn and split into X and Y while loading.
# Hint: It can be imported from tflearn.datasets. If tflearn is not installed, install it.
# It can be loaded using: x, y = oxflower17.load_data()

# %% colab={"base_uri": "https://localhost:8080/"} id="LitVHgnMIFig" outputId="948078a4-1264-4719-9cdf-b874766a93e2"
pip install tflearn

# %% colab={"base_uri": "https://localhost:8080/"} id="060c9a76" outputId="18c60a8f-0b70-475a-bfa7-964217291118"
import tflearn
import tflearn.datasets.oxflower17 as oxflower17

# Read the data
import tflearn.datasets.oxflower17 as oxflower17
X, y = oxflower17.load_data(one_hot=True, resize_pics=(128,128))
# X, y = oxflower17.load_data(resize_pics=(224,224))
# np.savez_compressed('oxflower17', X=X, Y=y)

# %% [markdown] id="3c191e96"
# ### 1B. Print Number of images and shape of the images.

# %% colab={"base_uri": "https://localhost:8080/"} id="d4386fcf" outputId="bfc4a97e-97d2-439b-b99c-f71baa4cb3cf"
print(X.shape)
print(y.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="3MJx_0tmyUYM" outputId="3b4cdacc-80c8-4ee6-c5e0-a4b6a742d6ad"
X[0].min(), X[0].max()

# %% colab={"base_uri": "https://localhost:8080/"} id="z-JkUpjMQ2Vg" outputId="ca8b474f-4c81-423d-958a-6d85433a4185"
X

# %% [markdown] id="f7078165"
# ### 1C. Print count of each class from y.

# %% colab={"base_uri": "https://localhost:8080/"} id="4iGonaRqUT5G" outputId="d96461f9-d35f-4685-f424-c5d5ee0e61d6"
# One-hot-encoded numpy array
y

# %% colab={"base_uri": "https://localhost:8080/"} id="8ueYIL3T4gY2" outputId="1be8a686-bd3e-4abd-d5d3-9db26c844b39"
# No of Unique rows in y
np.unique(y, axis=0)

# %% colab={"base_uri": "https://localhost:8080/"} id="7-YHtzsF7WY3" outputId="75fcbf52-cc8c-4f7d-d964-94efcbc63e3a"
# Converting y into a dataframe
ydf = pd.DataFrame(y, columns = [i for i in range(0,17)])
print(ydf)
print(type(ydf))

# %% colab={"base_uri": "https://localhost:8080/"} id="CP0Y6JrDAXrC" outputId="8e343c2c-9974-4d08-bd88-a433c9b65f36"
# Reverse one-hot-encoding in pandas
ydf.idxmax(axis=1)

# %% colab={"base_uri": "https://localhost:8080/"} id="r2ffddogA73d" outputId="d76eca41-a48f-48fe-897f-fa9f788d7540"
# Number of images in each class
ydf.idxmax(axis=1).value_counts()

# %% [markdown] id="HvLX5ewZVgPN"
# Looks like the data is well balanced; Good for model building.

# %% [markdown] id="6c9d21cd"
# ## 2. Image Exploration & Transformation 
# [Learning purpose - Not related to final model]

# %% colab={"base_uri": "https://localhost:8080/"} id="X6uy5GrWXsO3" outputId="104b98ff-44b7-402c-b7db-2c5e8099d525"
# y as numpy array with numeric labels
y1 = ydf.idxmax(axis=1).to_numpy()
y1


# %% [markdown] id="f60c081a"
# ### 2A. Display 5 random images

# %% colab={"base_uri": "https://localhost:8080/", "height": 256} id="SSeNRo54QZti" outputId="5e3be5fc-fb65-4f38-bd60-1c29b80500b0"
# Visualize the images along with their labels
def show_img(count, data, label):
  
    fig, axs = plt.subplots(1, count, figsize=(20, 16))
    for i in range(0, count):
        axs[i].imshow(data[i], label=label[i])
        axs[i].set_title('Label: {}'.format(label[i]))
    plt.show()
        
show_img(5, X, y1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="5b4b3a3d" outputId="f6a91bee-33b1-4af8-d8de-750b5d82e826"
# Displaying an image from each class (Total classes = 17 with labels from 0 to 16)
cols = 5
rows = int(np.ceil(len(np.unique(y1))/cols))

fig, ax = plt.subplots(rows, cols, figsize=(20,20))
for i in np.unique(y1):
  col = i % 5
  row = int(i/5)

  ax[row][col].imshow(X[i])
  ax[row][col].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
  ax[row][col].set_title(f"Flower Label: {i}",{'fontsize':15})
plt.show() 

# %% [markdown] id="50f7b095"
# ### 2B. Select any image from the dataset and assign it to a variable.

# %% colab={"base_uri": "https://localhost:8080/", "height": 612} id="Xg7DRLyPP0o3" outputId="240fbfe5-9266-4bab-dcbd-93b239ad7a84"
# Selecting an image from the dataset
path = '/content/17flowers/jpg/9/image_0721.jpg'

# Using the cv2_imshow
from google.colab.patches import cv2_imshow
img1 = cv2.imread(path, cv2.IMREAD_COLOR)
# cv2_imshow(img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Using the imshow
plt.figure(figsize=(20, 10))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(img1)

# %% [markdown] id="0ec12f83"
# ### 2C. Transform the image into grayscale format and display the same.

# %% colab={"base_uri": "https://localhost:8080/", "height": 612} id="c6f60cf7" outputId="ec1a9e0a-949b-4b59-df88-58e369f8c749"
img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# cv2_imshow(img2)

# Using the imshow
plt.figure(figsize=(20, 10))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img2)

# %% [markdown] id="49d82951"
# ### 2D. Apply a filter to sharpen the image and display the image before and after sharpening.

# %% colab={"base_uri": "https://localhost:8080/", "height": 563} id="ab2b3409" outputId="fe7a49da-ce25-4ed5-a3bb-6532b4c5873b"
# # Original Image
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# Sharpened Image
img3 = cv2.imread(path, cv2.IMREAD_COLOR)

kernel1 = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
img3 = cv2.filter2D(src=img3, ddepth=-1, kernel=kernel1)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
# cv2_imshow(img1)
# cv2_imshow(img3)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(img1)
ax.set_title('Original Image')

ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(img3)
ax.set_title('Sharpened Image')

# %% [markdown] id="c55fb083"
# ### 2E. Apply a filter to blur the image and display the image before and after blur.

# %% colab={"base_uri": "https://localhost:8080/", "height": 563} id="9779f441" outputId="f2b7b0b5-b99e-492e-c2c9-337744796c3e"
# # Original Image
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# Sharpened Image
img4 = cv2.imread(path, cv2.IMREAD_COLOR)

img4 = cv2.blur(src=img4, ksize=(6,6))
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
# cv2_imshow(img1)
# cv2_imshow(img4)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(img1)
ax.set_title('Original Image')

ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(img4)
ax.set_title('Blurred Image')

# %% [markdown] id="cac4d272"
# ### 2F. Display all the 4 images from above questions besides each other to observe the difference.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1235b29b" outputId="d31fb1d0-3d55-484b-90b2-18f7905159ee"
# Original Image and Sharpened Image
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(img1)
ax.set_title('Original Image')

ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(img3)
ax.set_title('Sharpened Image')

# Original Image and Blurred Image
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(img1)
ax.set_title('Original Image')

ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(img4)
ax.set_title('Blurred Image')

# %% [markdown] id="0a8e1a66"
# ## 3. Model training and Tuning:

# %% [markdown] id="s2Vw_11eq8Na"
# > Due to time constraint, I am using small number of epochs, and training only the selected models. Similar process can be replicated for other complex models (Model-4 and Model-5).

# %% [markdown] id="4f22afed"
# ### 3A. Split the data into train and test with 80:20 proportion.

# %% colab={"base_uri": "https://localhost:8080/"} id="_yPajUNvTVVE" outputId="187db907-d9fb-496d-c540-b7da709bae87"
# Reshaping the data to train using the SVM
print(X.shape)
print(y.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="G1LtFAi2SbOS" outputId="a604a3c7-4d66-46da-d192-d699f494445f"
X1 = X.reshape(X.shape[0], 128 * 128 *3)
X1.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="0vrx7vXln3FI" outputId="315d633f-fb42-4928-847e-02010b077da9"
y1

# %% colab={"base_uri": "https://localhost:8080/"} id="h7jJStFWn8BL" outputId="eaf7ec0e-97b3-4e9f-e4bc-95330fc3d5aa"
y1[1]

# %% colab={"base_uri": "https://localhost:8080/"} id="eWqS4T6cbWAa" outputId="e118d064-90ab-46f6-ef42-fa8bb3b27551"
y1.shape

# %% id="B6aKkDPhdA6I"
# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=0)

# %% colab={"base_uri": "https://localhost:8080/"} id="lZXu7_AKeAco" outputId="7fc8a5cd-abdf-4795-d278-bb6874003b19"
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# %% [markdown] id="c331fb1d"
# ### 3B. Train a model using any Supervised Learning algorithm and share performance metrics on test data.

# %% id="J8tyPrCGeAIl"
# Using SVM from SkLearn
from sklearn.svm import SVC

# %% id="b_JP12tYd_8q"
#Create svm_model Object
model1 = SVC()

#Training the model
model1.fit(X_train, y_train)

#Predict testing set
y_pred = model1.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="iRjKdjKvYiR9" outputId="4d6dc5bd-78a2-47cb-cb26-1a30ce06bd3e"
# Classification Accuracy
print("Classification Accuracy:")
print('Accuracy on Training data:',model1.score(X_train, y_train))
print('Accuracy on Test data:',model1.score(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index = [i for i in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']],  
                         columns = [i for i in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
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
# **Considering the Class Recall, Precision, F1 Score and Accuracy as the most important parameters to decide the best model for this problem. We have the highest values in Model-3. Please refer the Model-3 for the same. (Part 3D)**

# %% [markdown] id="nCcTDRUUUTVi"
# ### 3C. Train a model using Neural Network and share performance metrics on test data.

# %% id="XxwVCJCZe5mB"
# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=0)

# %% colab={"base_uri": "https://localhost:8080/"} id="sKdV2BKVe5Xx" outputId="f07135d4-f054-445b-b1e3-a51c0b42b383"
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# %% id="aiqYDL6Pl26S"
model2 = Sequential()

model2.add(BatchNormalization())
model2.add(Dense(256,activation='relu',kernel_initializer='he_uniform',input_dim = X_train.shape[1]))
model2.add(Dropout(0.4))

model2.add(BatchNormalization())
model2.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
model2.add(Dropout(0.4))

model2.add(BatchNormalization())
model2.add(Dense(64,activation='relu',kernel_initializer='he_uniform'))
model2.add(Dropout(0.4))

model2.add(BatchNormalization())
model2.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))
model2.add(Dropout(0.4))

model2.add(Dense(y_train.shape[1], activation = 'sigmoid'))

# Compiling the ANN with Adam optimizer and binary cross entropy loss function 
optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model2.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

# %% colab={"base_uri": "https://localhost:8080/"} id="__PaPbgil21S" outputId="49836778-458c-41e5-bb50-17deb2cead32"
# Looking into our base model2
model2.build(X_train.shape)
model2.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="vMfzM-iYsZcE" outputId="3ecefa6b-dbd1-45cd-ecf6-d345c80ef90a"
# Fit the model2
history2=model2.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=200, batch_size=200, verbose=2, shuffle=True)

# %% id="7cB7GN2Bsda7"
# predicting the model2 on test data
y_pred=model2.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 265} id="UZkwnFyJsnoj" outputId="0ebf61ea-31b0-4e22-8102-bf7adcf0b229"
# Capturing learning history per epoch
hist  = pd.DataFrame(history2.history)
hist['epoch'] = history2.epoch

# Plotting accuracy at different epochs
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model2.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 265} id="57OFznIbsdYE" outputId="6d0736b4-92d9-4d09-ad0e-ec560e89b555"
# Capturing learning history per epoch
hist  = pd.DataFrame(history2.history)
hist['epoch'] = history2.epoch

# Plotting accuracy at different epochs
plt.plot(hist['acc'])
plt.plot(hist['val_acc'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model2.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="VVUnPAfesdVn" outputId="ae63ad9f-3e97-4072-a8be-4d7259968765"
# Classification Accuracy
print("Classification Accuracy:")
print('Loss and Accuracy on Training data:',model2.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model2.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
df_cm = pd.DataFrame(cm, index = [i for i in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']],  
                         columns = [i for i in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown] id="QX0HExXiVpIy"
# ### 3D. Train a model using a basic CNN and share performance metrics on test data.

# %% id="xoU_IQip9Uhp"
# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %% colab={"base_uri": "https://localhost:8080/"} id="BrTlgXBs9UGe" outputId="2f958507-02c3-4ede-a681-b0fac294865f"
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# %% id="XpftR74nJVju"
generator = ImageDataGenerator(rotation_range = 180,
                               zoom_range = 0.2,
                               width_shift_range = 0.2,
                               height_shift_range = 0.2,
                               horizontal_flip = True,
                               vertical_flip = True)
generator.fit(X_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="Imu60xTR77Yg" outputId="069db0d6-a60c-4b68-a4a3-fc4193f94295"
model3 = Sequential()

model3.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
model3.add(MaxPool2D((2, 2)))
# model3.add(Dropout(0.25))

model3.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model3.add(MaxPool2D((2, 2)))
# model3.add(Dropout(0.25))

model3.add(Flatten())

model3.add(Dense(128, activation='relu'))
# model3.add(Dropout(0.25))

model3.add(Dense(64, activation='relu'))
# model3.add(Dropout(0.25))

model3.add(Dense(17, activation='softmax'))

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model3.summary()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="vk1LHKvn77M6" outputId="a767bee5-81db-4f57-ed77-14abb34c8dec"
history3 = model3.fit(generator.flow(X_train,y_train, batch_size=200), epochs=100, verbose=2, shuffle=True, validation_data=(X_test,y_test))
pd.DataFrame(history3.history)

# %% colab={"base_uri": "https://localhost:8080/", "height": 267} id="E1eSCeFg77Ey" outputId="d64a44b0-2a24-44cd-dada-49579567d146"
# Capturing learning history per epoch
hist  = pd.DataFrame(history3.history)
hist['epoch'] = history3.epoch

# Plotting accuracy at different epochs
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model3.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 265} id="DX_njaUf769F" outputId="7f1cefa6-e688-4a08-ad07-3c545c1a5082"
# Capturing learning history per epoch
hist  = pd.DataFrame(history3.history)
hist['epoch'] = history3.epoch

# Plotting accuracy at different epochs
plt.plot(hist['acc'])
plt.plot(hist['val_acc'])
plt.legend(("train" , "valid") , loc =0)

# Printing results
results = model3.evaluate(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="XYC8zFZexgN6" outputId="4a84b095-2b0b-47e4-9216-0b6fe043e6d8"
y_pred=model3.predict(X_test)
y_pred

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="mAM9_pgu8agn" outputId="7c98d98a-c024-4117-fa24-3617d3bf9af5"
# Classification Accuracy
print("Classification Accuracy:")
print('Loss and Accuracy on Training data:',model3.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model3.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
df_cm = pd.DataFrame(cm, index = [i for i in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']],  
                         columns = [i for i in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown] id="bd9491e0"
# ### 3E. Predict the class/label of image ‘Prediction.jpg’ using best performing model and share predicted label.

# %% id="u8P8w3A33HjR"
# Function to read images from the given path and predict the image class
import matplotlib.image as mpimg
import keras.utils as image

def imagepredict(path):
    img=mpimg.imread(path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    # Prediction
    img = image.load_img(path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32')/255
    pred = np.argmax(model3.predict(x))
    print()
    print("Predicted Class of the above image is: {}.".format(pred))


# %% colab={"base_uri": "https://localhost:8080/", "height": 283} id="JaWJzltB3HVd" outputId="92f78986-2bef-4259-e2b4-718244664035"
path = '/content/drive/MyDrive/MGL/Project-CV-1/Prediction.jpg'
imagepredict(path)

# %% colab={"base_uri": "https://localhost:8080/", "height": 320} id="ASWVVzi2yFZN" outputId="5d33055f-086e-4fc1-ba87-38e707f9e1bb"
# Using test images from the given dataset
# Test Image as n1
n1 = 21
plt.imshow(X_test[n1])

# y_pred=model1.predict(X_test)

# Actual Class
print("Actual Class:",[y_test.argmax(axis=1)[n1]])
# Predicted Class
print("Predicted Class:",[y_pred.argmax(axis=1)[n1]])
print()

# %% [markdown]
# **Additional Models; Not part of the Project Part-B:**

# %% [markdown] id="Ey3WPA2i69Bs"
# ### CNN with VGG-16
# - Flatten
# - 2 dense layers (256, activation='relu')
# - Dropout(0.5)
# - loss='categorical_crossentropy', optimizer='adam'
# - model compile with ImageDataGenerator to minimize overfitting.
# - shuffle = True

# %% id="BMLynQzv65g2"
# generator = ImageDataGenerator(rotation_range = 180,
#                                zoom_range = 0.2,
#                                width_shift_range = 0.2,
#                                height_shift_range = 0.2,
#                                horizontal_flip = True,
#                                vertical_flip = True)
# generator.fit(X_train)

# %% id="am2tbAis65Ye"
# from keras.applications.vgg16 import VGG16

# # Initialize the VGG-16 model
# # Remove the final layer of the model and add 12 classess of plant seedlings
# # Input images: 128px by 128px.

# prior_model = VGG16(weights='imagenet',include_top=False, input_shape=(128,128,3))

# # Lets create our model

# model4 = Sequential()

# # Here we add a all the VGG16 as a layer

# model4.add(prior_model)

# %% id="ruHAPtE565QR"
# model4.summary()

# %% id="BzCUetk665Hv"
# model4.layers[0].summary()

# %% id="17ofVutH64_r"
# model4.add(Flatten())
# model4.add(Dense(256,activation='relu'))
# model4.add(Dropout(0.5))
# model4.add(Dense(17, activation='softmax'))

# %% id="pjahujgw6423"
# model4.summary()

# %% id="p6Mli0kz64uG"
# # Looping over each layers in layer 0 to freeze them
# for layers in model4.layers[0].layers: 
#   layers.trainable = False

# # Freezing layer 0 as well for good measure
# model4.layers[0].trainable = False 

# %% id="WmRbZQ4o64jU"
# model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %% id="36X3eNhQ8Yuw"
# history4 = model4.fit(generator.flow(X_train,y_train,batch_size=300), epochs=100, verbose=2, shuffle=True, validation_data=(X_test,y_test))
# pd.DataFrame(history4.history)

# %% [markdown] id="WhcolXOtZPCS"
# ### CNN with InceptionV3
# - Flatten
# - 2 dense layers (1024, activation='relu')
# - Dropout(0.5)
# - loss='categorical_crossentropy', optimizer='adam'
# - model compile with ImageDataGenerator to minimize overfitting.
# - shuffle = True

# %% id="zuOgTTRpYwCn"
# from keras.applications.inception_v3 import InceptionV3

# # Initialize the InceptionV3 model
# # Remove the final layer of the model and add 12 classess of plant seedlings
# # Input images: 128px by 128px.

# prior_model = InceptionV3(weights='imagenet',include_top=False, input_shape=(128,128,3))

# # Lets create our model

# model5 = Sequential()

# # Here we add a all the InceptionV3 as a layer

# model5.add(prior_model)

# %% id="HsOOu5BOYv_z"
# model5.summary()

# %% id="VWlVoITEYv9R"
# model5.layers[0].summary()

# %% id="JYUvEBclYv6T"
# model5.add(Flatten())

# model5.add(Dense(1024, activation='relu'))
# model5.add(Dropout(0.5))

# model5.add(Dense(17, activation='softmax'))

# model5.summary()

# %% id="UyM1Ojz5Yv23"
# # Looping over each layers in layer 0 to freeze them
# for layers in model5.layers[0].layers: 
#   layers.trainable = False

# # freezing layer 0 as well for good measure
# model5.layers[0].trainable = False 

# %% id="TbtYtnrlYvyf"
# model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %% id="_e2Nx7TMYvvg"
# history5 = model5.fit(generator.flow(X_train,y_train,batch_size=300), epochs=100, verbose=2,shuffle=True,validation_data=(X_test,y_test))
# pd.DataFrame(history5.history)

# %% [markdown] id="cbeb381f"
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
