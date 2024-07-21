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
# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#00b3e5;">Contents: NLP Project - 2
# 1. [Part-A: Solution](#Part-A:-Solution)
# 2. [Part-B: Solution](#Part-B:-Solution)

# %% [markdown] id="5f36f6dd"
# # Part-A: Solution

# %% [markdown] id="dedd939b"
# - **DOMAIN:** Digital content and entertainment industry
# - **CONTEXT:** The objective of this project is to build a text classification model that analyses the customer's sentiments based on their reviews in the IMDB database. The model uses a complex deep learning model to buildan embedding layer followed by a classification algorithm to analyse the sentiment of the customers.
# - **Data Description:** The Dataset of 50,000 movie reviews from IMDB, labelled by sentiment (positive/negative). Reviews have been preprocessed, and each review is encoded as a sequence of word indexes (integers). For convenience, the words are indexed by their frequency in the dataset, meaning the for that has index 1 is the most frequent word. Use the first 20 words from each review to speed up training, using a max vocabulary size of 10,000. As a convention, "0" does not stand for a specific word, but instead is used to encode any unknown word.
# - **PROJECT OBJECTIVE:** To Build a sequential NLP classifier which can use input text parameters to determine the customer sentiments.

# %% id="9b11c05c"
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
from sklearn.metrics import plot_precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, plot_roc_curve 

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

import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
import pandas_profiling as pp

import gensim
import logging

# import cv2
# from google.colab.patches import cv2_imshow
# from glob import glob
# import itertools

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, GlobalMaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad

from tensorflow.keras.applications import MobileNetV2
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

# Set random_state
random_state = 42

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %% colab={"base_uri": "https://localhost:8080/"} id="peP2nm91hndX" outputId="71ae08d2-e931-4251-9c5b-5f63700d810e"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="dzBl_FQExInw" outputId="e61f4e51-68a7-41c4-e49e-26064d9e1a5d"
# Current working directory
# %cd "/content/drive/MyDrive/MGL/Project-NLP-2/"

# # List all the files in a directory
# for dirname, _, filenames in os.walk('path'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# %% colab={"base_uri": "https://localhost:8080/"} id="JR6VDtpLxKAU" outputId="3833d871-39ec-47e4-f63c-31e7961f10bb"
# List files in the directory
# !ls

# %% [markdown] id="a3313f55"
# ## 1. Import and analyse the data set.
# - Use `imdb.load_data()` method
# - Get train and test set
# - Take 10000 most frequent words

# %% [markdown] id="IE21UtIbWnYq"
# ### Quick EDA for complete dataset

# %% id="2uiB6LCbWyRC"
# # Path of the data file
# path = 'IMDB Dataset.csv.zip'

# # Unzip files in the current directory

# with ZipFile (path,'r') as z:
#   z.extractall() 
# print("Training zip extraction done!")

# %% id="jAdKLFsYWyOk"
# Import the dataset
df = pd.read_csv('IMDB Dataset.csv')

# %% colab={"base_uri": "https://localhost:8080/"} id="Mk8l9fRqWyMK" outputId="cf3098c2-44e2-4095-b808-8be74ad3d4f7"
df.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="qL_KMrfDWyKA" outputId="15612876-8a41-4bca-8cd2-8da5b55d88d9"
df.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="rPBDYejbWyID" outputId="10a94c27-f0b0-4c88-de9f-f9ac77d88940"
# pd.set_option('display.max_colwidth', None)
df.head()

# %% id="Qlgt9VCmdyt_"
# Clear the matplotlib plotting backend
# %matplotlib inline
plt.close('all')

# %% colab={"base_uri": "https://localhost:8080/", "height": 458} id="jnTvdrdSWyGn" outputId="f123d2f3-4f96-4e36-894e-364c947a5d51"
# Understand the 'sentiment' the target vector
f,axes=plt.subplots(1,2,figsize=(17,7))
df['sentiment'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('sentiment',data=df,ax=axes[1])
axes[0].set_title('Pie Chart for sentiment')
axes[1].set_title('Bar Graph for sentiment')
plt.show()

# %% [markdown] id="SaE5p3-meFes"
# The dataset consists of two groups:
# - 25000 positive reviews
# - 25000 negative reviews
#
# Its evident that the dataset is very well balanced. This is a very favourable situation for a classification task.

# %% colab={"base_uri": "https://localhost:8080/", "height": 442} id="GMA1RMJhWyER" outputId="5226b343-ab5a-43d8-efa2-fb959237ca49"
# Visualize word cloud of random positive and negative review

# Choose randomly a positive review and a negative review
ind_positive = random.choice(list(df[df['sentiment'] == 'positive'].index))
ind_negative = random.choice(list(df[df['sentiment'] == 'negative'].index))

review_positive = df['review'][ind_positive]
review_negative = df['review'][ind_negative]

print('Positive review: ', review_positive)
print('\n')
print('Negative review: ', review_negative)
print('\n')

from wordcloud import WordCloud
cloud_positive = WordCloud().generate(review_positive)
cloud_negative = WordCloud().generate(review_negative)

plt.figure(figsize = (20,15))
plt.subplot(1,2,1)
plt.imshow(cloud_positive)
plt.title('Positive review')

plt.subplot(1,2,2)
plt.imshow(cloud_negative)
plt.title('Negative review')
plt.show()

# %% id="_m1Re8Nd6bu7"
# Text Cleaning
import re

def remove_url(text):
    url_tag = re.compile(r'https://\S+|www\.\S+')
    text = url_tag.sub(r'', text)
    return text

def remove_html(text):
    html_tag = re.compile(r'<.*?>')
    text = html_tag.sub(r'', text)
    return text

def remove_punctuation(text): 
    punct_tag = re.compile(r'[^\w\s]')
    text = punct_tag.sub(r'', text) 
    return text

def remove_special_character(text):
    special_tag = re.compile(r'[^a-zA-Z0-9\s]')
    text = special_tag.sub(r'', text)
    return text

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text    
    
def clean_text(text):
    text = remove_url(text)
    text = remove_html(text)
    text = remove_punctuation(text)
    text = remove_special_character(text)
    text = remove_emojis(text)
    text = text.lower()
    
    return text


# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="bojFPwgs6bsf" outputId="8f3a0675-6f59-45c5-e7dc-0397d4842a6b"
df['processed'] = df['review'].apply(lambda x: clean_text(x))
df['label'] = df['sentiment'].apply(lambda x: 0 if x == 'negative' else 1)
df.head()

# %% id="b77h-2iFFuer"
# df = df.sample(n=1000, random_state = 0)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="bWhrkpof6bqH" outputId="2a1de604-2aee-444c-efe5-f1456f2199f7"
# Create the features matrix and target vector
df1=df[['processed', 'label']]
df1.head()

# %% id="Ar9ZizJ96bn_"
# Split the data for training and testing
# To be used in the transformers (BERT)
train, test = train_test_split(df1, test_size=0.5, random_state=0)

# %% [markdown] id="LY7maY7Hi1J4"
# ### Using the imdb.load_data() method

# %% colab={"base_uri": "https://localhost:8080/"} id="wWojkALIzpBw" outputId="7d15685d-5fc8-4319-d90c-7d4602009673"
# Loading the IMDB dataset
# The argument num_words=10000 keeps the top 10,000 most frequently occurring words in the training data. 
# The rare words are discarded to keep the size of the data manageable.

top_words = 10000
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(path="imdb.npz",
                                                      num_words=top_words)

# %% colab={"base_uri": "https://localhost:8080/"} id="VU0fHEYUZGtV" outputId="5ce2129f-4acd-4d27-f87a-437bf74e703e"
X_train

# %% colab={"base_uri": "https://localhost:8080/"} id="tSa8Vr4FZOOt" outputId="e792b460-56b5-4820-cb32-80e59951107b"
y_train

# %% [markdown] id="5579308b"
# ## 2. Perform relevant sequence adding on the data.

# %% [markdown] id="c891804a"
# ## 3. Perform following data analysis:
# - Print shape of features and labels
# - Print value of any one feature and it's label

# %% [markdown] id="e4db61f8"
# ## 4. Decode the feature value to get original sentence

# %% [markdown] id="-A-EqLOn_RV5"
# ### Considering the above 2nd, 3rd, and 4th parts together in below code cells:

# %% [markdown] id="Z3gFJq8M-0vN"
# Let's take a moment to understand the format of the data. The dataset comes preprocessed: each example is an array of integers representing the words of the movie review. Each label is an integer value of either 0 or 1, where 0 is a negative review, and 1 is a positive review.

# %% colab={"base_uri": "https://localhost:8080/"} id="E_m_bSFD0DIP" outputId="be5e9dda-c849-43bc-eabd-62e7f620a489"
# Shape of training data
print("X_train: {}, y_train: {}".format(len(X_train),len(y_train)))

# %% colab={"base_uri": "https://localhost:8080/"} id="AeIbiOOr0DF2" outputId="c30f14d2-d1d5-4db7-a07b-0e442c302ae2"
# Shape of test data
print("X_test: {}, y_test: {}".format(len(X_test),len(y_test)))

# %% colab={"base_uri": "https://localhost:8080/"} id="UxqHCCPB0DDm" outputId="24e32ce5-eac9-4235-b51a-2d52ebef8f70"
# The text of reviews have been converted to integers, where each integer represents a specific word in a dictionary. 
# Looking at the first review
print(X_train[0])
print(y_train[0])

# %% colab={"base_uri": "https://localhost:8080/"} id="fGZyPCEt1USy" outputId="a55fa375-890f-472d-bb77-6b88f90abb61"
# Movie reviews may be different lengths. The below code shows the number of words in the first and second reviews. 
# Since inputs to a NN/RNN must be the same length, we'll need to resolve this later.
len(X_train[0]), len(X_train[1])

# %% colab={"base_uri": "https://localhost:8080/"} id="ZUxXW_WD5UEj" outputId="56bc4f95-5b11-4520-e3bd-21063aa06854"
# Convert integers back to text: Here, we'll create a helper function to query a dictionary object that contains the integer to string mapping:

# A dictionary mapping words to an integer index
imdb = keras.datasets.imdb
word_index = imdb.get_word_index()

# The first indices are reserved

word_index = {k:(v+3) for k,v in word_index.items()}

word_index["<PAD>"] =0
word_index["<START>"]=1
word_index["<UNK>"]=2 #unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text])


# %% colab={"base_uri": "https://localhost:8080/", "height": 140} id="x9kA7iiS5X_y" outputId="95aa59bc-6b3e-4ef0-b7b1-b69861c3b1b0"
decode_review(X_train[0])

# %% colab={"base_uri": "https://localhost:8080/", "height": 140} id="ogHexlet5UB9" outputId="245635f7-f2e2-4c67-bf42-9aff15f8447b"
decode_review(X_train[1])

# %% [markdown] id="_WR1LV0l24pU"
# The reviews (intteger arrays) must be converted to tensors before fed into the neural network. This conversion can be done in many ways:
#
# - One-hot-encode the arrays to convert them into vectors of 0s and 1s. For example, the sequence [1, 5, 6] would become a 10,000-dimensional vector that is all zeros except for indices at 1, 5 and 6, which are ones. Then, make this the first layer in our network—a Dense layer—that can handle floating point vector data. This approach is memory intensive, though, requiring a num_words * num_reviews size matrix.
#
#
# - Another method, we can pad the arrays so that they all have the same length, then create an integer tensor of shape max_length * num_reviews. We can use an embedding layer capable of handling this shape as the first layer in our network. Since the movie reviews must be the same length, we will use the pad_sequences function to standardize the lengths.

# %% id="Vk_qCwIwY3X9"
# pad_sequences is used to ensure that all sequences in a list have the same length. By default this is done by padding 0 in the beginning
# of each sequence until each sequence has the same length as the longest sequence.

#Since the sequences have different lengtht, then we use padding method to put all sequences to the same length. 
#The parameter "maxlen" sets the maximum length of the output sequence. 

# If length of the input sequence is larger than "maxlen", then it is trunced to keep only #maxlen words, (truncating = 'pre': keep the 
# previous part of the sequence; truncating = 'post': keep the posterior part of the sequence).

# If length of the input sequence is smaller than "maxlen", then 0 elements will be padded into the previous part of sequence 
# (if padding = 'pre' - by defaut) or in the tail of the sequence (if padding = 'post').

max_length = 256
trunc_type = 'post'

X_train = keras.preprocessing.sequence.pad_sequences(X_train, value=word_index["<PAD>"],padding="post",maxlen = max_length, truncating = trunc_type)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, value=word_index["<PAD>"],padding="post",maxlen = max_length, truncating = trunc_type)

# %% colab={"base_uri": "https://localhost:8080/"} id="f4aDGh3B9Rgq" outputId="aef40534-ab5e-4e45-cac6-ce08ccac0dd4"
# Check the length of reviews again
len(X_train[0]), len(X_train[1])

# %% colab={"base_uri": "https://localhost:8080/"} id="p-nCyqRy9Rd3" outputId="b8f3108e-c0bb-40e7-fd9f-e679241982ab"
# Check the first review after padding
X_train[0]

# %% [markdown] id="fd09209e"
# ## 5. Design, train, tune and test a sequential model.
# Hint: The aim here Is to import the text, process it such a way that it can be taken as an inout to the ML/NN classifiers. Be analytical and experimental here in trying new approaches to design the best model.

# %% [markdown] id="qM9e_RKdNQKl"
# ### ANN

# %% colab={"base_uri": "https://localhost:8080/"} id="ytU93f2ENU-B" outputId="5ae088cf-453b-4f0c-fc8a-ccc38a21de41"
# Input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000
embedding_dim = 16
max_length = 256

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="_7AUoxBoNU7i" outputId="4a253c46-57b6-4527-f3db-f6a8a307c68e"
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

H = model.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_test, y_test))

# %% colab={"base_uri": "https://localhost:8080/", "height": 336} id="ZM9MDue0NU4z" outputId="496144e4-3787-4829-b00e-c214e0b2934d"
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.plot(H.history['accuracy'], label = 'Train')
plt.plot(H.history['val_accuracy'], label = 'Validation')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(H.history['loss'], label = 'Train')
plt.plot(H.history['val_loss'], label = 'Validation')
plt.legend()
plt.title('Loss')

plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="x8d94k3DNU2V" outputId="dee47aed-5455-4493-a900-00aee7763fdc"
y_pred_proba = model.predict(X_test)
y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 916} id="JC1Tcb0gQjxj" outputId="7b4d9ab6-dd63-4aea-f20b-26fae2394e41"
# Classification Accuracy
print("Classification Accuracy:")
print('Loss and Accuracy on Training data:',model.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% colab={"base_uri": "https://localhost:8080/"} id="Gursahpi2Kw1" outputId="1e060ed7-455a-465c-c86f-f933c903940d"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.evaluate(X_train, y_train)
Test_Accuracy = model.evaluate(X_test, y_test)

base_1 = []
base_1.append(['ANN', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% colab={"base_uri": "https://localhost:8080/"} id="7aqgzf_JPAL1" outputId="31cb7cb1-26d0-434c-e205-3439b0ea9227"
# An approach for predicted and actual labels
for i in range(5):
  print(decode_review(X_test[i]))
  pred = model.predict(X_test[i].reshape(1, 256))
  print('Prediction prob = ', pred, '\t Actual =', y_test[i])

# %% [markdown] id="44D3J9d9s4EB"
# ### RNN

# %% colab={"base_uri": "https://localhost:8080/"} id="glQU8RH3s4EG" outputId="89ea19a5-d0e0-4733-dc37-3038b72e4a12"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, AveragePooling1D, Bidirectional, LSTM, SimpleRNN, Dense

vocab_size = 10000
embedding_dim = 16
max_length = 256

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length = max_length))
model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(AveragePooling1D(pool_size = 2))
# model.add(Bidirectional(SimpleRNN(32, dropout = 0.5)))
model.add((SimpleRNN(32, dropout = 0.5)))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

H = model.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_test, y_test), verbose=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 336} id="22BiAMyGs4EG" outputId="6ddc2ffc-a030-445f-c0f7-13a812cedf79"
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.plot(H.history['accuracy'], label = 'Train')
plt.plot(H.history['val_accuracy'], label = 'Validation')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(H.history['loss'], label = 'Train')
plt.plot(H.history['val_loss'], label = 'Validation')
plt.legend()
plt.title('Loss')

plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="kLy0qIFIs4EG" outputId="ef48080c-096d-487c-8dac-fed4bf35a4cc"
y_pred_proba = model.predict(X_test)
y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 916} id="GBHq-Hazs4EG" outputId="3822ddbf-c726-46fa-edb0-3be7a8191c34"
# Classification Accuracy
print("Classification Accuracy:")
print('Loss and Accuracy on Training data:',model.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% colab={"base_uri": "https://localhost:8080/"} id="2hItcohz8SU2" outputId="20c291be-686f-4030-a825-9074059f56ee"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.evaluate(X_train, y_train)
Test_Accuracy = model.evaluate(X_test, y_test)

# base_1 = []
base_1.append(['RNN', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% [markdown] id="EbbNvck_wYmZ"
# ### GRU

# %% colab={"base_uri": "https://localhost:8080/"} id="-auTLUgowYme" outputId="8cc10c4c-64bf-41c3-9b49-c82c230e1ed6"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, AveragePooling1D, Bidirectional, LSTM, SimpleRNN, GRU, Dense

vocab_size = 10000
embedding_dim = 16
max_length = 256

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length = max_length))
model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(AveragePooling1D(pool_size = 2))
# model.add(Bidirectional(GRU(32, dropout = 0.5)))
model.add((GRU(32, dropout = 0.5)))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

H = model.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_test, y_test), verbose=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 336} id="vXo5bYVowYmf" outputId="64b6d5c6-d996-4c19-a00c-06dad769dfa0"
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.plot(H.history['accuracy'], label = 'Train')
plt.plot(H.history['val_accuracy'], label = 'Validation')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(H.history['loss'], label = 'Train')
plt.plot(H.history['val_loss'], label = 'Validation')
plt.legend()
plt.title('Loss')

plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="Hu_nOFhQwYmf" outputId="7932928b-5c48-49c1-dd3d-e4774ba4bc4b"
y_pred_proba = model.predict(X_test)
y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 916} id="ErCaClj5wYmf" outputId="90f5ddf4-8106-4f4f-ebe9-d3c2aecd9720"
# Classification Accuracy
print("Classification Accuracy:")
print('Loss and Accuracy on Training data:',model.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% colab={"base_uri": "https://localhost:8080/"} id="TT6EgBTW-wMb" outputId="f8d064b3-6694-4701-a7d5-eebd628eb89a"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.evaluate(X_train, y_train)
Test_Accuracy = model.evaluate(X_test, y_test)

# base_1 = []
base_1.append(['GRU', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% [markdown] id="fch5cpxtV358"
# ### LSTM

# %% colab={"base_uri": "https://localhost:8080/"} id="a7288dc7" outputId="a993517f-94ea-497d-b73a-5bd2f39708ac"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, AveragePooling1D, Bidirectional, LSTM, Dense

vocab_size = 10000
embedding_dim = 16
max_length = 256

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length = max_length))
model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(AveragePooling1D(pool_size = 2))
# model.add(Bidirectional(LSTM(32, dropout = 0.5)))
model.add((LSTM(32, dropout = 0.5)))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

H = model.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_test, y_test), verbose=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 336} id="Siw557QhjpAk" outputId="dcdf01f2-89b6-4d00-9c4e-15dd01b705b5"
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.plot(H.history['accuracy'], label = 'Train')
plt.plot(H.history['val_accuracy'], label = 'Validation')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(H.history['loss'], label = 'Train')
plt.plot(H.history['val_loss'], label = 'Validation')
plt.legend()
plt.title('Loss')

plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="2yK6hbbGsbnF" outputId="b82001a6-93d4-43b7-eb80-3ad85f1f7888"
y_pred_proba = model.predict(X_test)
y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 916} id="zHiTEk5hsJOA" outputId="d2894645-eee9-4030-8ba9-b1d35e33da21"
# Classification Accuracy
print("Classification Accuracy:")
print('Loss and Accuracy on Training data:',model.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% colab={"base_uri": "https://localhost:8080/"} id="vcynOueVBIrh" outputId="ce6df702-df10-4ff8-8a9c-c438da0fb0ff"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.evaluate(X_train, y_train)
Test_Accuracy = model.evaluate(X_test, y_test)

# base_1 = []
base_1.append(['LSTM', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% [markdown] id="MegtHB5NC3fn"
# ### Logistic Regression

# %% colab={"base_uri": "https://localhost:8080/", "height": 881} id="nynw_f3VC_hv" outputId="8401966e-0c4a-4a83-c8f0-b8b1adf478af"
# Build the model
model = LogisticRegression()
# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification Accuracy
print("Classification Accuracy:")
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% id="AZ1mmARzC_eX"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.score(X_train, y_train)
Test_Accuracy = model.score(X_test, y_test)

# base_1 = []
base_1.append(['Logistic Regression', Train_Accuracy, Test_Accuracy, precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% [markdown] id="OIAfvIe9GXiW"
# ### KNN

# %% colab={"base_uri": "https://localhost:8080/", "height": 881} id="J15NHF81GXie" outputId="dd11137d-8e00-4a32-f624-60b57df67162"
# Build the model
model = KNeighborsClassifier()
# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification Accuracy
print("Classification Accuracy:")
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% id="VABeUdTgGXie"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.score(X_train, y_train)
Test_Accuracy = model.score(X_test, y_test)

# base_1 = []
base_1.append(['K Neighbors', Train_Accuracy, Test_Accuracy, precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% [markdown] id="E4ccm_KZGYa_"
# ### SVM

# %% colab={"base_uri": "https://localhost:8080/", "height": 881} id="EbhDsi5MGYbA" outputId="a3996529-ebb6-4ad3-bb5a-cb682bdd15f1"
# Build the model
model = SVC()
# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification Accuracy
print("Classification Accuracy:")
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% id="9ybNXLYRGYbA"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.score(X_train, y_train)
Test_Accuracy = model.score(X_test, y_test)

# base_1 = []
base_1.append(['SVM', Train_Accuracy, Test_Accuracy, precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% [markdown] id="AWF3wKcYGY3f"
# ### Multinomial NB

# %% colab={"base_uri": "https://localhost:8080/", "height": 881} id="hNYa-rAxGY3f" outputId="9d660645-35a5-44a9-8cdf-68d3ee88ef0e"
# Build the model
model = MultinomialNB()
# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification Accuracy
print("Classification Accuracy:")
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% id="q_ZZkWnYGY3g"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.score(X_train, y_train)
Test_Accuracy = model.score(X_test, y_test)

# base_1 = []
base_1.append(['Multinomial NB', Train_Accuracy, Test_Accuracy, precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% [markdown] id="PLpFGj6xGZUF"
# ### Decision Tree

# %% colab={"base_uri": "https://localhost:8080/", "height": 881} id="fW73ibRoGZUG" outputId="004a328d-44d3-49b0-c789-88ec5c2d71d9"
# Build the model
model = DecisionTreeClassifier()
# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification Accuracy
print("Classification Accuracy:")
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% id="6zxT2c8aGZUG"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.score(X_train, y_train)
Test_Accuracy = model.score(X_test, y_test)

# base_1 = []
base_1.append(['Decision Tree', Train_Accuracy, Test_Accuracy, precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% [markdown] id="FGnmlPBAGaFE"
# ### Random Forest

# %% colab={"base_uri": "https://localhost:8080/", "height": 881} id="DlVJjIRAGaFE" outputId="c3692685-e70c-442f-f1fb-1ae77878e4f9"
# Build the model
model = RandomForestClassifier()
# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification Accuracy
print("Classification Accuracy:")
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% id="ycD53gJRGaFE"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.score(X_train, y_train)
Test_Accuracy = model.score(X_test, y_test)

# base_1 = []
base_1.append(['Random Forest', Train_Accuracy, Test_Accuracy, precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% [markdown] id="XqwM3NatGadW"
# ### Ada Boost

# %% colab={"base_uri": "https://localhost:8080/", "height": 881} id="ApPWINWkGadW" outputId="45f650ed-8ee9-40c6-9569-e1320fe0e8cb"
# Build the model
model = AdaBoostClassifier()
# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification Accuracy
print("Classification Accuracy:")
print('Accuracy on Training data:',model.score(X_train, y_train))
print('Accuracy on Test data:',model.score(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% [markdown] id="Zb8j7VEjQXHn"
# ### Model Comparison

# %% colab={"base_uri": "https://localhost:8080/", "height": 394} id="B2867OtaGadX" outputId="4caa03ba-496d-40f3-8479-5b39d84f55bc"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.score(X_train, y_train)
Test_Accuracy = model.score(X_test, y_test)

# base_1 = []
base_1.append(['Ada Boost', Train_Accuracy, Test_Accuracy, precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)
model_comparison

# %% [markdown] id="MKMGlSyvTs-R"
# ### BERT

# %% colab={"base_uri": "https://localhost:8080/"} id="mT2CN7Dh2esc" outputId="9fde1332-a4ec-438d-ae8e-e174c3bdf655"
# Install Transformers library
# !pip install transformers

# %% colab={"base_uri": "https://localhost:8080/", "height": 234, "referenced_widgets": ["7ea3cbc9eb1b4a3e9afd03312d35dc55", "6561193a0dbf48ad90d67fa1e6e36fd7", "fe76d0a65d9946f4ac191277976cb6fc", "d0fe6c7d82344c36b0dc42e9c9780586", "5aaeb5efd2854f4f9a1d600a07fc38e0", "201e06cd64864c02bbe90551cd8643ad", "6b8387ac213843eaaccbbbfc5b97ba5d", "0146287c4dc542a3b740c590acc7cc79", "c342a83a5b4a4e29a5d80558a8c39709", "770539f8bc42450fb3733054b8689d8f", "1543d3afe15b4ab49fdb683ae0f0edb2", "c720cd1a464049239fed49528b02d593", "9ccf5080bb644441b97a036ff0d0a3d0", "29a5a3f09d534415abe63cfe072cf6c4", "946aab5f7bf346f1bcd2de1e449cfb19", "13786e94911b46459837c9bb03bccfa3", "f15c13341ef047c283cd95332c479887", "9370a01095f84027b239da7336838154", "eb800033377a4e22838185456d3f10c0", "ad4285570c734d5bb7fe94171ac146ef", "6fa621e93f6d424cb0e768adda9a6c25", "1de89960da684a1ea1ca2ee52995e782", "4ec48d8577a64634a13106a3d7bbac19", "03a38d8840ae4af5a944cf39efd9a02b", "814f22054d424e319ab366a8c5c55eb6", "2ae523d38f4f4a2cbe32775fbc96b04b", "84cc909da4b743598e1bc63c9c46f665", "f7c6e0dfaf914737948d9498dcd13c1e", "582221e3b39f41da93258511bee082f6", "1ffb249a3b974c7099d8626a8e9fa4f4", "4cd65f7ec6094faf945b32e309cbe88c", "a32c3f66be4e4fe0af48db72987ddcbc", "6588dbbed21e4c17837ff493e2da710c", "d0bd6505b59b45e68b6fcd01c4df5d4a", "216308163122405aa9d6a2408085509e", "5931fd4e436c46bcaf009ce309ba2a71", "072c0e94bb884764b93bb76b306a2220", "43599e25da36427892af1878675916d2", "dd79b9d21c294538a2bfb957756a4aa1", "e3ef21bcc8874a0d872826ed2d5c5c23", "242e1102a4204ea8aac38970a01c5a54", "1564cd5d4d9d49ddabc64aafe8f0c5a8", "b86ed0bfb4344dc8992030330ea35676", "135712cdf6bf42348bd3d047a4a7b660"]} id="Ej0b1jCJ2esd" outputId="e67be6b2-fc80-4bc9-8451-cb44a4926bf5"
# Load the BERT Classifier and Tokenizer alıng with Input modules
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# %% colab={"base_uri": "https://localhost:8080/"} id="qHk36Siu2ese" outputId="78fbec7c-48e8-44fb-e46f-62c9afc139ae"
# We have the main BERT model, a dropout layer to prevent overfitting, and finally a dense layer for classification task:
model.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="gnraXLYt2ese" outputId="2be365c3-cf8b-4c9d-f822-685aa72cba99"
# We have two pandas Dataframe objects waiting for us to convert them into suitable objects for the BERT model. 
# We will take advantage of the InputExample function that helps us to create sequences from our dataset. 
# The InputExample function can be called as follows:
InputExample(guid=None,
             text_a = "Hello, world",
             text_b = None,
             label = 1)


# %% [markdown] id="QgZg4Cx2ah0O"
# Now we will create two main functions:
# 1. convert_data_to_examples: This will accept our train and test datasets and convert each row into an InputExample object.
# 2. convert_examples_to_tf_dataset: This function will tokenize the InputExample objects, then create the required input format with the tokenized objects, finally, create an input dataset that we can feed to the model.

# %% id="Bq_Da6Np2esf"
def convert_data_to_examples(train, test, processed, label): 
  train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[processed], 
                                                          text_b = None,
                                                          label = x[label]), axis = 1)

  validation_InputExamples = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[processed], 
                                                          text_b = None,
                                                          label = x[label]), axis = 1)
  
  return train_InputExamples, validation_InputExamples

  train_InputExamples, validation_InputExamples = convert_data_to_examples(train, 
                                                                           test, 
                                                                           'processed', 
                                                                           'label')
  
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


processed = 'processed'
label = 'label'

# %% id="9cTYxeR42esg"
# Our dataset containing processed input sequences are ready to be fed to the model.
train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, processed, label)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

# %% colab={"base_uri": "https://localhost:8080/"} id="oOozxt2N2esg" outputId="815b85e2-0023-4278-f493-4afa5b173fc2"
# We will use Adam as our optimizer, CategoricalCrossentropy as our loss function, and SparseCategoricalAccuracy as our accuracy metric. 
# Fine-tuning the model for 2 epochs will give us around 90% accuracy, which is great.

# Training the model might take a while, so ensure you enabled the GPU acceleration from the Notebook Settings.

# %%time

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

H = model.fit(train_data, epochs=2, validation_data=validation_data)

# 50 min for maxlen = 128

# %% [markdown] id="8d7016b3"
# ## 6. Use the designed model to print the prediction on any one sample.

# %% id="z6CGWXGs2esg"
# Making Predictions
# I created a list of two reviews I created. The first one is a positive review, while the second one is clearly negative.
pred_sentences = ['This was an awesome movie. I watch it twice my time watching this beautiful movie if I have known it was this good',
                  'One of the worst movies of all time. I cannot believe I wasted two hours of my life for this movie']

# %% colab={"base_uri": "https://localhost:8080/"} id="nkR6EkwQ2esg" outputId="e976a5ac-67f3-4ad7-f447-240b6e85204a"
# We need to tokenize our reviews with our pre-trained BERT tokenizer. We will then feed these tokenized sequences to our model
# and run a final softmax layer to get the predictions. We can then use the argmax function to determine whether our sentiment 
# prediction for the review is positive or negative. Finally, we will print out the results with a simple for loop. 
# The following lines do all of these said operations:
tf_batch = tokenizer(pred_sentences, max_length=128, padding=True, truncation=True, return_tensors='tf')
tf_outputs = model(tf_batch)
tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
labels = ['Negative','Positive']
label = tf.argmax(tf_predictions, axis=1)
label = label.numpy()
for i in range(len(pred_sentences)):
  print(pred_sentences[i], ": \n", labels[label[i]])

# %% colab={"base_uri": "https://localhost:8080/"} id="fWAoBGEy2esg" outputId="babccd8a-f83e-4260-c371-dee5d740a47a"
# Using the BERT on 5 test samples
predict_set = test[0:5]
pred_sentences = list(predict_set['processed'])

tf_batch = tokenizer(pred_sentences, max_length=128, padding=True, truncation=True, return_tensors='tf')
tf_outputs = model(tf_batch)
tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
labels = ['Negative','Positive']
label = tf.argmax(tf_predictions, axis=1)
label = label.numpy()
for i in range(len(pred_sentences)):
  print(pred_sentences[i], ": \n", labels[label[i]])

# %% [markdown] id="atES6rFSSv0J"
# ## Conclusion

# %% [markdown] id="hj4shJGFSy8n"
# In this project, we have learned how to clean and prepare the text data to feed into various ML/DL Models.
#
# We have compared the performance of various ML/DL models with precision, recall, F1 and Accuracies (Train and Test).
#
# There are several ideas that we can try to improve the model performance:
#
# - We can change dimension of the embedding layer
# - Hyperparameter tuning of various models
# - Different vectorization mehtods can also be tested
# - Text cleaning can further improve the model performance
# - More advanced transformers can also be tried in this project

# %% [markdown] id="0dd7c2e4"
# # Part-B: Solution

# %% [markdown] id="e624389a"
# - **DOMAIN:** Social media analytics
# - **CONTEXT:** Past studies in Sarcasm Detection mostly make use of Twitter datasets collected using hashtag based supervision but such datasets are noisy in terms of labels and language. Furthermore, many tweets are replies to other tweets and detecting sarcasm in these requires the availability of contextual tweets.In this hands-on project, the goal is to build a model to detect whether a sentence is sarcastic or not, using Bidirectional LSTMs.
# - **DATA DESCRIPTION:** The dataset is collected from two news websites, theonion.com and huffingtonpost.com. This new dataset has the following advantages over the existing Twitter datasets:
#  - Since news headlines are written by professionals in a formal manner, there are no spelling mistakes and informal usage. This reduces the sparsity and also increases the chance of finding pre-trained embeddings.
#  - Furthermore, since the sole purpose of TheOnion is to publish sarcastic news, we get high-quality labels with much less noise as compared to Twitter datasets.
#  - Unlike tweets that reply to other tweets, the news headlines obtained are self-contained. This would help us in teasing apart the real sarcastic elements
#
# - Content: Each record consists of three attributes:
#
#  - is_sarcastic: 1 if the record is sarcastic otherwise 0
#  - headline: the headline of the news article
#  - article_link: link to the original news article. Useful in collecting supplementary data
#  - Reference: https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detecti
#
# - **PROJECT OBJECTIVE:** Build a sequential NLP classifier which can use input text parameters to determine the customer sentiments.

# %% id="EJ9x2TAkS-hD"
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
from sklearn.metrics import plot_precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, plot_roc_curve 

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

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import VotingClassifier

# from imblearn.over_sampling import RandomOverSampler
# from imblearn.over_sampling import SMOTENC, SMOTE, ADASYN
# from imblearn.under_sampling import RandomUnderSampler

import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
import pandas_profiling as pp

import gensim
import logging

# import cv2
# from google.colab.patches import cv2_imshow
# from glob import glob
# import itertools

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, GlobalMaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad

from tensorflow.keras.applications import MobileNetV2
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

# Set random_state
random_state = 42

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %% colab={"base_uri": "https://localhost:8080/"} id="SUu3mJIaS-hE" outputId="5f2e1eee-1ce5-4c08-be8f-65978c4e70a7"
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown] id="p822aEtTS-hE"
# ## 1. Read and explore the data

# %% colab={"base_uri": "https://localhost:8080/"} id="3MiG1b3WS-hE" outputId="e920592e-c930-46b5-81bd-cdaf4d22963e"
# Current working directory
# %cd "/content/drive/MyDrive/MGL/Project-NLP-2/"

# # List all the files in a directory
# for dirname, _, filenames in os.walk('path'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# %% colab={"base_uri": "https://localhost:8080/"} id="Oc4NtoSkS-hE" outputId="4a7b8343-68e6-4c11-8750-e45fb4cb7377"
# List files in the directory
# !ls

# %% id="dxgmxjZvS-hE"
# # Path of the data file
# path = 'Sarcasm_Headlines_Dataset_v2.json.zip'

# # Unzip files in the current directory

# with ZipFile (path,'r') as z:
#   z.extractall() 
# print("Training zip extraction done!")

# %% id="vP28RwRM1MuP"
# Import the dataset
# Creat dataframe from the json file
df = pd.read_json('Sarcasm_Headlines_Dataset_v2.json', lines=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="VluEspcE1MW5" outputId="b6bbcb98-db35-449b-a0c7-4ab90b591e84"
df.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 476} id="ovZb80Gv1MUn" outputId="4f487f2a-9e43-47cb-bf5c-0d86768ec594"
pd.set_option('display.max_colwidth', None)
df.info()
df.head()

# %% id="x4MOpaog1MOt"
# As the dataset is large; use a subset of the data. Let's Check what is working on the local machine.
# Can use 10,000/100,000 later
# df = pd.read_csv("blogtext.csv", nrows=1000) 
# df = df.sample(n=10000, random_state = 0)

# df.info()

# %% colab={"base_uri": "https://localhost:8080/"} id="XpkyO0Nq1MHk" outputId="44e85dac-a875-44e5-d00c-3a08f4f49065"
# Check for unique values: 1 = Sarcastic, 0 = Not Sarcastic
df.is_sarcastic.value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} id="hB7B3kTpGqxc" outputId="56d051f0-ebfb-46e2-c388-376439e5b121"
# Check for NaN values
df.isna().sum() 

# %% colab={"base_uri": "https://localhost:8080/", "height": 205} id="mwFhO3FPK30I" outputId="5ce92a6d-8767-4ce9-e231-74e43f91e8ca"
# Describe function generates descriptive statistics that summarize the central tendency, 
# dispersion and shape of a dataset’s distribution, excluding NaN values.

# This method tells us a lot of things about a dataset. One important thing is that 
# the describe() method deals only with numeric values. It doesn't work with any 
# categorical values. So if there are any categorical values in a column the describe() 
# method will ignore it and display summary for the other columns.

df.describe(include='all').transpose()

# %% id="bQWcEgToWWXc"
# Clear the matplotlib plotting backend
# %matplotlib inline
plt.close('all')

# %% colab={"base_uri": "https://localhost:8080/", "height": 459} id="yWDoWu015p_u" outputId="8ee61215-3f97-4f28-b4cb-5fcfb39413a4"
# Understand the 'sentiment' the target vector
f,axes=plt.subplots(1,2,figsize=(17,7))
df['is_sarcastic'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('is_sarcastic',data=df,ax=axes[1])
axes[0].set_title('Pie Chart for sarcasm')
axes[1].set_title('Bar Graph for sarcasm')
plt.show()

# %% [markdown] id="LsquxGMsh2-C"
# So, We can see that the dataset is balanced. Its good for a classification task.

# %% [markdown] id="W8cP5ddNS-hF"
# ## 2. Retain relevant columns

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="q9JeM73vzpLi" outputId="35a25752-76e1-49cd-effd-7d9801629d97"
df = df[['headline', 'is_sarcastic']]
df.head()

# %% [markdown] id="32foMnD9S-hF"
# ## 3. Get length of each sentence

# %% colab={"base_uri": "https://localhost:8080/", "height": 52} id="eGPvAO16_i43" outputId="896a6b34-c411-41f9-9a45-9537cda35fca"
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.express as px
from plotly.offline import init_notebook_mode
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import spacy

tqdm.pandas()
spacy_eng = spacy.load("en_core_web_sm")
nltk.download('stopwords')
lemm = WordNetLemmatizer()
init_notebook_mode(connected=True)
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (20,8)
plt.rcParams['font.size'] = 18

# %% colab={"base_uri": "https://localhost:8080/"} id="R_HN_DeqCXuo" outputId="0447efb7-2a4d-4dd3-e5a4-bb71d4960e8d"
nltk.download('all')

# %% id="1ce0cf5f"
# Text Cleaning:
# We will not remove numbers from the text data right away, lets further analyse if they contain any relevant information
# We can find the entity type of the tokens in the sentences using Named Entity Recognition (NER), this will help us identify
# the type and relevance of numbers in our text data

stop_words = stopwords.words('english')
stop_words.remove('not')

def text_cleaning(x):
    
    headline = re.sub('\s+\n+', ' ', x)
    headline = re.sub('[^a-zA-Z0-9]', ' ', x)
    headline = headline.lower()
    headline = headline.split()
    
    headline = [lemm.lemmatize(word, "v") for word in headline if not word in stop_words]
    headline = ' '.join(headline)
    
    return headline


# %% colab={"base_uri": "https://localhost:8080/"} id="YylPG8aoALqj" outputId="6c33c17f-c926-44eb-c49f-be41540c721c"
def get_entities(x):
    entity = []
    text = spacy_eng(x)
    for word in text.ents:
        entity.append(word.label_)
    return ",".join(entity)

df['entity'] = df['headline'].progress_apply(get_entities)

# %% colab={"base_uri": "https://localhost:8080/"} id="OQhPhkRNCIov" outputId="de1f46b8-966e-4a9e-d8b5-cf3444172806"
nltk.download('wordnet')

# %% colab={"base_uri": "https://localhost:8080/", "height": 537} id="KRcY3Mu2A_6t" outputId="cb5e005b-2928-4b17-dbe6-bf441961506d"
# Dataset with entity, clean_headline and sentence_length
df['clean_headline'] = df['headline'].apply(text_cleaning)

df['sentence_length'] = df['clean_headline'].apply(lambda x: len(x.split()))
df

# %% colab={"base_uri": "https://localhost:8080/", "height": 717} id="Dx2WnUfnA_3_" outputId="26292932-e663-4797-93b7-f770997807aa"
# Headline length distribution
# Check for outliers in headline column
# Generally the headlines shouldn't be more than 20-40 words
# Box Plot

fig = px.histogram(df, x="sentence_length",height=700, color='is_sarcastic', title="Headlines Length Distribution", marginal="box")
fig.show(renderer="colab")

# %% colab={"base_uri": "https://localhost:8080/"} id="H4Ay-i8DOqyZ" outputId="3da49d2e-872b-45a7-e3ee-879f21ba467c"
df[df['sentence_length']==107]['headline']

# %% id="b5n7qGDLOqt4"
df.drop(df[df['sentence_length'] == 107].index, inplace = True)
df.reset_index(inplace=True, drop=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 717} id="eXDRZg4aOqrJ" outputId="cd04f582-9d3c-486a-9c9b-a81a7c8450b1"
# Headline length distribution: Outliers Removed
# The headlines after the removal of outliers do not exceed the limit of 20-40 words
# They are mostly centered in the range of 5-10 words
fig = px.histogram(df, x="sentence_length",height=700, color='is_sarcastic', title="Headlines Length Distribution", marginal="box")
fig.show(renderer="colab")

# %% colab={"base_uri": "https://localhost:8080/", "height": 572} id="cfHtmF0GOqmK" outputId="ab6c24da-9370-42e0-c914-be5e0d025c2b"
# Filtering: Find Sentences that Contain Numbers
df['contains_number'] = df['clean_headline'].apply(lambda x: bool(re.search(r'\d+', x)))
df

# %% [markdown] id="MHef8hgqi8R8"
# Analysis of samples containing numbers of Time, Date or Cardinal Entity:
# - The numbers in a text data can have different implications
# - While the naive text preprocessing methods suggest that the numbers should be removed along with the special characters
# - The entity type of these numbers should be identified to get their exact implications

# %% colab={"base_uri": "https://localhost:8080/", "height": 407} id="ITOz0jL6A_yE" outputId="896d0e9f-aa5c-4a11-c55f-2639acd3b32d"
# Date Entity: Randome Samples
df[(df['contains_number']) & (df['sentence_length']<=5) & (df['entity']=='DATE')].sample(10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="tEJHUVo4kWMe" outputId="9d59342c-d489-4c93-d0f8-343bbc2e8a54"
# Time Entity: Randome Samples
df[(df['contains_number']) & (df['sentence_length']<=5) & (df['entity']=='TIME')].sample(10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 407} id="9vYnDzQ8kWKE" outputId="db0943a7-b35d-4492-afb0-8ee8c04a98c1"
# Cardinal Entity: Randome Samples
df[(df['contains_number']) & (df['sentence_length']<=5) & (df['entity']=='CARDINAL')].sample(10)

# %% [markdown] id="hoJ9R6SQlHNN"
# **Inference from NER:**
# - For some headlines, its important to retain the date, time and cardinal information
# - Special tokenization can be considered to retain the meaning of these numbers
# - Vocab size can be reduced further by removing these numbers
# - More research is required to improve the quality of vectorization and modeling performance
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 638} id="20eVf5uVkWHp" outputId="dc3ce4dd-7d9b-49d5-bd7b-7457a341e777"
# Wordcloud for text that is Not Sarcastic (LABEL - 0)
plt.figure(figsize = (20,20)) # Text that is Not Sarcastic
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.is_sarcastic == 0].headline))
plt.imshow(wc , interpolation = 'bilinear')

# %% colab={"base_uri": "https://localhost:8080/", "height": 638} id="knkIJMf9dTcM" outputId="ec31a270-79e1-448a-e5ab-529fbefa7d60"
# Wordcloud for text that is Sarcastic (LABEL - 1)
plt.figure(figsize = (20,20)) # Text that is Sarcastic
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.is_sarcastic == 1].headline))
plt.imshow(wc , interpolation = 'bilinear')

# %% [markdown] id="usoWJ9E3S-hH"
# ## 4. Define parameters

# %% [markdown] id="xv3w_BS9S-hH"
# ## 5. Get indices for words

# %% [markdown] id="e49c0b71"
# ## 6. Create features and labels

# %% [markdown] id="a9c15625"
# ## 7. Get vocabulary size

# %% [markdown] id="e8c30324"
# ## 8. Create a weight matrix using GloVe embeddings

# %% [markdown] id="4921a2f9"
# ## 9. Define and compile a Bidirectional LSTM model.
# Hint: Be analytical and experimental here in trying new approaches to design the best model.

# %% [markdown] id="77068c50"
# ## 10. Fit the model and check the validation accuracy

# %% [markdown] id="JSgaXKzAKOk_"
# **Considering the above 4, 5, 6, 7, 8, 9, 10 parts together in below code cells:**

# %% id="75Kl7xwKjiKV"
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Layer, Dense, Dropout, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.layers import LSTM, Bidirectional, SimpleRNN, GRU, Conv1D,  MultiHeadAttention, AveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# %% id="eza_OtVFgxD3"
X = df['clean_headline']
y = df['is_sarcastic']

# %% id="xctW0kjfgxBe"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# %% colab={"base_uri": "https://localhost:8080/"} id="FCNUlEj9gw_s" outputId="471aa46a-e442-4e30-a060-0050f461fef8"
# Tokenization
# Splitting sentences into words
# Finding the vocab size
# Important Parameters to consider
max_len = 20  
embedding_dim = 50     
oov_token = '00_V' 
padding_type = 'post'
trunc_type = 'post'  

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
print("Vocab Size: ",vocab_size)

# %% id="ejvIK20Hgw7p"
# Encoding of Inputs
# Converting the sentences to token followed by padded sequences in encoded format
# These are numeric encodings assigned to each word
train_sequences = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(train_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(test_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

# %% colab={"base_uri": "https://localhost:8080/"} id="M2yYCgWUgw5P" outputId="d869dba3-7449-47a6-c130-7873132d0a7d"
X_train[0]

# %% colab={"base_uri": "https://localhost:8080/"} id="ZIC6So3Xgw23" outputId="0b9b0f20-70bb-4050-dbb1-fe9a3a2e7317"
y_train[0]

# %% id="yf_tB--egwmp"
# # Path of the data file
# path = 'glove.6B.zip'

# # Unzip files in the current directory

# with ZipFile (path,'r') as z:
#   z.extractall() 
# print("Training zip extraction done!")

# %% id="imzomYmcgwj7"
# Embedding matrix with 50 dimensions
from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('glove.6B.50d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

vocab_size = len(tokenizer.word_index)+1

# Creating a embedding matrix for initial weights based on the precreated glove embedding

embedding_matrix = zeros((vocab_size, 50))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# %% [markdown] id="dnQW8DM5otis"
# ### ANN

# %% colab={"base_uri": "https://localhost:8080/"} id="WtROIfbHoe4Z" outputId="af5078c7-ea05-42c8-b372-9f2118a36655"
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length = max_len))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="Px9PZkMloe4e" outputId="0fe91aa2-e507-4103-a37b-4044109b5cd0"
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

H = model.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_test, y_test))

# %% colab={"base_uri": "https://localhost:8080/", "height": 350} id="IaRv9fz8oe4e" outputId="e3829dc2-080c-4598-90eb-53ffe14378cb"
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.plot(H.history['accuracy'], label = 'Train')
plt.plot(H.history['val_accuracy'], label = 'Validation')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(H.history['loss'], label = 'Train')
plt.plot(H.history['val_loss'], label = 'Validation')
plt.legend()
plt.title('Loss')

plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="1BBue9OCoe4e" outputId="537ec2f5-3a1d-4103-a316-210e8d48619d"
y_pred_proba = model.predict(X_test)
y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 939} id="97DlEK-doe4e" outputId="b33d6387-69f8-4ab6-fc90-8b8345aaa84c"
# Classification Accuracy
print("Classification Accuracy:")
print('Loss and Accuracy on Training data:',model.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% colab={"base_uri": "https://localhost:8080/"} id="uCo6sw3Doe4e" outputId="61915581-f2bc-43e5-e4b7-ec9db682b021"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.evaluate(X_train, y_train)
Test_Accuracy = model.evaluate(X_test, y_test)

base_1 = []
base_1.append(['ANN', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% [markdown] id="zRhe-AnsS-hJ"
# ### RNN

# %% colab={"base_uri": "https://localhost:8080/"} id="jBCM8jkXS-hJ" outputId="3cef01af-43d7-48fd-a054-83d12f220325"
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length = max_len))
model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(AveragePooling1D(pool_size = 2))
# model.add(Bidirectional(SimpleRNN(64, dropout = 0.5)))
model.add((SimpleRNN(64, dropout = 0.5)))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

H = model.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_test, y_test), verbose=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 350} id="wzYb34LaS-hJ" outputId="38de8896-911f-400c-c712-b0c6c0542290"
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.plot(H.history['accuracy'], label = 'Train')
plt.plot(H.history['val_accuracy'], label = 'Validation')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(H.history['loss'], label = 'Train')
plt.plot(H.history['val_loss'], label = 'Validation')
plt.legend()
plt.title('Loss')

plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="ABk6P9wMS-hJ" outputId="28bc783e-2ee4-4ce5-b95f-50bdde59e276"
y_pred_proba = model.predict(X_test)
y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 939} id="OEJyI2nCS-hJ" outputId="5308b44c-1aaa-4900-aaff-5ce6366a1e9f"
# Classification Accuracy
print("Classification Accuracy:")
print('Loss and Accuracy on Training data:',model.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% colab={"base_uri": "https://localhost:8080/"} id="fdbBA47pS-hJ" outputId="7a26e3d8-059c-49f6-812f-47e8d48bf868"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.evaluate(X_train, y_train)
Test_Accuracy = model.evaluate(X_test, y_test)

# base_1 = []
base_1.append(['RNN', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% [markdown] id="G1IED8LES-hJ"
# ### GRU

# %% colab={"base_uri": "https://localhost:8080/"} id="9Zj-ZAEbS-hJ" outputId="7b5b2fa3-7e2d-4883-8731-7721e34e511e"
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length = max_len))
model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(AveragePooling1D(pool_size = 2))
# model.add(Bidirectional(GRU(32, dropout = 0.5)))
model.add((GRU(64, dropout = 0.5)))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

H = model.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_test, y_test), verbose=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 350} id="1i_sXxpfS-hJ" outputId="3440c6e6-b269-45d1-d098-4116dab6e692"
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.plot(H.history['accuracy'], label = 'Train')
plt.plot(H.history['val_accuracy'], label = 'Validation')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(H.history['loss'], label = 'Train')
plt.plot(H.history['val_loss'], label = 'Validation')
plt.legend()
plt.title('Loss')

plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="6vM4d19NS-hJ" outputId="d4083403-12ba-404c-c4b2-69130b92ebc4"
y_pred_proba = model.predict(X_test)
y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 939} id="RQlwPUZRS-hJ" outputId="884f3d7e-2569-497d-e457-8acbc7ce8511"
# Classification Accuracy
print("Classification Accuracy:")
print('Loss and Accuracy on Training data:',model.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% colab={"base_uri": "https://localhost:8080/"} id="x3bIu2lOS-hK" outputId="e3485d14-eafe-4811-cceb-639a2cff6eec"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.evaluate(X_train, y_train)
Test_Accuracy = model.evaluate(X_test, y_test)

# base_1 = []
base_1.append(['GRU', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)

# %% [markdown] id="ppQpHJuxS-hK"
# ### LSTM

# %% colab={"base_uri": "https://localhost:8080/"} id="pkKQkl3IS-hR" outputId="f76d6b61-7dc3-43e8-bf5c-cab140bc0935"
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length = max_len))
model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(AveragePooling1D(pool_size = 2))
model.add(Bidirectional(LSTM(64, dropout = 0.5)))
# model.add((LSTM(32, dropout = 0.5)))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

H = model.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_test, y_test), verbose=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 350} id="ltymvt5cS-hS" outputId="0c9d13e0-cdd2-4772-ff3c-161dfc4046cd"
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.plot(H.history['accuracy'], label = 'Train')
plt.plot(H.history['val_accuracy'], label = 'Validation')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(H.history['loss'], label = 'Train')
plt.plot(H.history['val_loss'], label = 'Validation')
plt.legend()
plt.title('Loss')

plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="1a1Vhg4uS-hS" outputId="46481d79-b1ec-440f-d4e4-0c0004505c84"
y_pred_proba = model.predict(X_test)
y_pred = np.array([0 if proba < 0.5 else 1 for proba in y_pred_proba])

# %% colab={"base_uri": "https://localhost:8080/", "height": 939} id="De1RXhosS-hS" outputId="e48ce0e8-311f-4c01-9ebc-bec1af424678"
# Classification Accuracy
print("Classification Accuracy:")
print('Loss and Accuracy on Training data:',model.evaluate(X_train, y_train))
print('Loss and Accuracy on Test data:',model.evaluate(X_test, y_test))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test, y_pred))

# Confusion Matrix
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

# %% [markdown] id="-irqi2xhDn1a"
# ### Model Comparison

# %% colab={"base_uri": "https://localhost:8080/", "height": 210} id="_L8S_S5vS-hS" outputId="69e6a8f4-8b0c-4d5d-ee16-dfc216e20ab9"
# Model comparison
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')

Train_Accuracy = model.evaluate(X_train, y_train)
Test_Accuracy = model.evaluate(X_test, y_test)

# base_1 = []
base_1.append(['LSTM', Train_Accuracy[1], Test_Accuracy[1], precision, recall, f1])
model_comparison = pd.DataFrame(base_1,columns=['Model','Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
model_comparison.sort_values(by=['Recall','F1 Score'], inplace=True, ascending=False)
model_comparison

# %% [markdown] id="IRw3WwuXwfm3"
# ### BERT

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="CoJWagaVzn67" outputId="d973a135-34ba-44e2-a7ee-8666772cede7"
df1 = df[['clean_headline', 'is_sarcastic']]
df1.head()

# %% id="l3PwYNfvnfXs"
# Split the data for training and testing
# To be used in the transformers (BERT)
train, test = train_test_split(df1, test_size=0.5, random_state=0)

# %% id="IX9OqaDP05jZ"
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# %% colab={"base_uri": "https://localhost:8080/"} id="feQ1zcPPS-hS" outputId="ca39163d-c0b1-4455-ff01-a5e4b0f9afc2"
# Install Transformers library
# !pip install transformers

# %% colab={"base_uri": "https://localhost:8080/", "height": 234, "referenced_widgets": ["3926abcb43b34b39aa5a5c1312541df8", "a40d07623c3b408abf7314ebfccc93de", "087f0912d92b45ab81e6b430a9bbc32f", "a364879963e9475da81f8993b726304a", "42a7f22bb2ce4d5c967b4135605cce18", "3c9fbec2e3e04e0fb55460dda274528d", "62f7e717336145ffa57855307859419b", "5abf2bf5fb444341af8becc06d57d227", "b439b01bb2294018972b2d7aadc90562", "e87345033aa14ba3b44d429704933301", "0d42d135c2d24c04b136b5719213d3ee", "a0f18197493643299cdb75ec34a8f967", "9aead24eb827484d867a69bcc073e30f", "1e8bd276282849588c8a456228a98f57", "58e5210c46ad4c699ee9fca07628cfd1", "8615c300cc844eb596bda10257599f8e", "826eba962ea64cc1ae46829fbbe8bd54", "4946edc190af4ddeb38d8c9755030a8a", "fdb50168cd10435eb294e16c3189e10f", "bbdc672a416f4811890499786b71bc4a", "830dc24e5b7d415090f39f225d18d3f0", "4f0d212644444d09becca91e36694727", "d5ef499925e540ce9d132ef1632bd1b6", "ba91fb94ac9340a4acc8bd52dd3bc861", "02c31a67514c4d2f8037b2229fafd501", "e5cb5505074045c387c04457456e2a7a", "f5fdaa9966324830ae6f16f10c79a512", "65e07f5b8c8843b597f474acdb5391c1", "6485e28ac98847e09805466494101046", "c3a9c020e75a48d2ad51f37756cbf822", "210044e7085f4e55a3d0b948b234a9ee", "e32b974ef88e4195805faacaacf6219f", "31cf8d5cd4f04d7382acb4238047f4ac", "f8b7c6ce3d5246069bc30a4ba77b42ff", "3cf05eccbb74461d8bbf3d0adae513fa", "babda4ff837345a98cd0a09db2e20a49", "6b92f7725c004f76877a8f63d3fa295c", "71688c5cfc644c9bab34467f9fdef00a", "b163ce3f4034472f846fdc3a36562c29", "44ea612f602346409ad8a3562223247f", "c73630fcfc19493b9873a35541cc3c60", "9888b089495d45e0b4b40ab36c6f79e0", "08a545f84c7a41cebf6b9c7090acfbc7", "a9ca094056d34316b9b8380e04346e6e"]} id="Wwi8S9bQS-hT" outputId="f7c3cb77-5a6c-4608-bdfe-2bf9dc4ee41a"
# Load the BERT Classifier and Tokenizer alıng with Input modules
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# %% colab={"base_uri": "https://localhost:8080/"} id="33jyYXbXS-hT" outputId="5289b718-6363-4b47-d3ac-96392be735f3"
# We have the main BERT model, a dropout layer to prevent overfitting, and finally a dense layer for classification task:
model.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="GwxItw7FS-hT" outputId="a077d15a-d52f-48ed-c6ce-86d23d03c3a6"
# We have two pandas Dataframe objects waiting for us to convert them into suitable objects for the BERT model. 
# We will take advantage of the InputExample function that helps us to create sequences from our dataset. 
# The InputExample function can be called as follows:
InputExample(guid=None,
             text_a = "Hello, world",
             text_b = None,
             label = 1)


# %% [markdown] id="lABzvnMpS-hT"
# Now we will create two main functions:
# 1. convert_data_to_examples: This will accept our train and test datasets and convert each row into an InputExample object.
# 2. convert_examples_to_tf_dataset: This function will tokenize the InputExample objects, then create the required input format with the tokenized objects, finally, create an input dataset that we can feed to the model.

# %% id="NeZVr4rWS-hT"
def convert_data_to_examples(train, test, clean_headline, is_sarcastic): 
  train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[clean_headline], 
                                                          text_b = None,
                                                          label = x[is_sarcastic]), axis = 1)

  validation_InputExamples = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[clean_headline], 
                                                          text_b = None,
                                                          label = x[is_sarcastic]), axis = 1)
  
  return train_InputExamples, validation_InputExamples

  train_InputExamples, validation_InputExamples = convert_data_to_examples(train, 
                                                                           test, 
                                                                           'clean_headline', 
                                                                           'is_sarcastic')
  
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


clean_headline = 'clean_headline'
is_sarcastic = 'is_sarcastic'

# %% id="EzmzBGM5S-hT"
# Our dataset containing processed input sequences are ready to be fed to the model.
train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, clean_headline, is_sarcastic)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

# %% colab={"base_uri": "https://localhost:8080/"} id="o7YtPJ8_S-hT" outputId="8aacb894-d99f-4d74-9334-38e15d226664"
# We will use Adam as our optimizer, CategoricalCrossentropy as our loss function, and SparseCategoricalAccuracy as our accuracy metric. 
# Fine-tuning the model for 2 epochs will give us around 90% accuracy, which is great.

# Training the model might take a while, so ensure you enabled the GPU acceleration from the Notebook Settings.

# %%time

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

H = model.fit(train_data, epochs=2, validation_data=validation_data)

# 30 min for maxlen = 128

# %% [markdown] id="5o9X5EtAS-hT"
# ## 6. Use the designed model to print the prediction on any one sample.

# %% id="_hQx1QWcS-hU"
# Making Predictions
# I created a list of two reviews I created. The first one is a sarcastic review, while the second one is cnot sarcastic.
pred_sentences = ['What planet did you come from?',
                  'This is really a very beautiful pic']

# %% colab={"base_uri": "https://localhost:8080/"} id="a6uIgkRyS-hU" outputId="2b02a330-28fe-4525-a4ca-f652676107d6"
# We need to tokenize our reviews with our pre-trained BERT tokenizer. We will then feed these tokenized sequences to our model
# and run a final softmax layer to get the predictions. We can then use the argmax function to determine whether our sentiment 
# prediction for the review is positive or negative. Finally, we will print out the results with a simple for loop. 
# The following lines do all of these said operations:
tf_batch = tokenizer(pred_sentences, max_length=128, padding=True, truncation=True, return_tensors='tf')
tf_outputs = model(tf_batch)
tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
labels = ['0','1']
label = tf.argmax(tf_predictions, axis=1)
label = label.numpy()
for i in range(len(pred_sentences)):
  print(pred_sentences[i], ": \n", labels[label[i]])

# %% colab={"base_uri": "https://localhost:8080/"} id="GKfc3cK2S-hU" outputId="78cc24fd-3570-4845-c59d-3eb8d80bf207"
# Using the BERT on 5 test samples
predict_set = test[0:5]
pred_sentences = list(predict_set['clean_headline'])

tf_batch = tokenizer(pred_sentences, max_length=128, padding=True, truncation=True, return_tensors='tf')
tf_outputs = model(tf_batch)
tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
labels = ['0','1']
label = tf.argmax(tf_predictions, axis=1)
label = label.numpy()
for i in range(len(pred_sentences)):
  print(pred_sentences[i], ": \n", labels[label[i]])

# %% [markdown] id="JOjk-avPNVfW"
# ## Conclusion:

# %% [markdown] id="5s4MQfZxLchj"
# - In this notebook, We used text preprocessing to prepare the data and make 
# it compatible for vaious ML/DL models alongwith the required EDA.
# - Compared performances of various models like ANN, RNN, GRU, LSTM and BERT.
# - Using hyperparameter tuning, we can further improve the performance of various models.
# - Text cleaning by considering the NER aspects can further improve the model performance
# - More advanced transformers can also be tried in this project
#

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
