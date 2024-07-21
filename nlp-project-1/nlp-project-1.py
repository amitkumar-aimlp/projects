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
# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#00b3e5;">Contents: NLP Project - 1
# 1. [Part-A: Solution](#Part-A:-Solution)
# 2. [Part-B: Solution](#Part-B:-Solution)

# %% [markdown] id="5f36f6dd"
# # Part-A: Solution

# %% [markdown] id="dedd939b"
# - **DOMAIN:** Digital content management
# - **CONTEXT:** Classification is probably the most popular task that you would deal with in real life. Text in the form of blogs, posts, articles, etc. are written every second. It is a challenge to predict the information about the writer without knowing about him/her. We are going to create a classifier that predicts multiple features of the author of a given text. We have designed it as a Multi label classification problem.
# - **Data Description:** Over 600,000 posts from more than 19 thousand bloggers The Blog Authorship Corpus consists of the collected posts of 19,320 bloggers gathered from blogger.com in August 2004. The corpus incorporates a total of 681,288 posts and over 140 million words - or approximately 35 posts and 7250 words per person. Each blog is presented as a separate file, the name of which indicates a blogger id# and the blogger’s self-provided gender, age, industry, and astrological sign. (All are labelled for gender and age but for many, industry and/or sign is marked as unknown.) All bloggers included in the corpus fall into one of three age groups:
#  - 8240 "10s" blogs (ages 13-17),
#  - 8086 "20s" blogs (ages 23-27) and
#  - 2994 "30s" blogs (ages 33-47)
#  - For each age group, there is an equal number of male and female bloggers. Each blog in the corpus includes at least 200 occurrences of common English words. All formatting has been stripped with two exceptions. Individual posts within a single blogger are separated by the date of the following post and links within a post are denoted by the label url link.
# - **PROJECT OBJECTIVE:** To build a NLP classifier which can use input text parameters to determine the label/s of the blog. Specific to this case study, you can consider the text of the blog: ‘text’ feature as independent variable and ‘topic’ as dependent variable.

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

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import VotingClassifier

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

# %% colab={"base_uri": "https://localhost:8080/"} id="peP2nm91hndX" outputId="b69480e7-966b-4b1b-e183-4d99e1b00a0a"
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown] id="a3313f55"
# ## 1. Read and Analyse Dataset.

# %% colab={"base_uri": "https://localhost:8080/"} id="dzBl_FQExInw" outputId="0919f850-ddf5-4599-a4b0-29339a793598"
# Current working directory
# %cd "/content/drive/MyDrive/MGL/Project-NLP-1/"

# # List all the files in a directory
# for dirname, _, filenames in os.walk('path'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# %% colab={"base_uri": "https://localhost:8080/"} id="JR6VDtpLxKAU" outputId="e35be3b0-8310-41eb-be95-1e456fc799a7"
# List files in the directory
# !ls

# %% id="wWojkALIzpBw"
# # Path of the data file
# path = 'blogs.zip'

# # Unzip files in the current directory

# with ZipFile (path,'r') as z:
#   z.extractall() 
# print("Training zip extraction done!")

# %% [markdown] id="fb411bd2"
# ### 1A. Clearly write outcome of data analysis (Minimum 2 points)

# %% id="vP28RwRM1MuP"
# Import the dataset
dfa = pd.read_csv("blogtext.csv")

# %% colab={"base_uri": "https://localhost:8080/"} id="VluEspcE1MW5" outputId="5db4ca0c-c0be-4095-f320-9d88e4414647"
dfa.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 449} id="ovZb80Gv1MUn" outputId="6e75f26d-5685-44ff-fbf8-dc39f9417594"
dfa.info()
dfa.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="6kD9KOqT1MRj" outputId="b8c81279-bdf3-4770-c605-23dc6551fe0d"
dfa.tail()

# %% id="x4MOpaog1MOt"
# As the dataset is large; use a subset of the data. Let's Check what is working on the local machine.
# Can use 10,000/100,000 later
# dfa = pd.read_csv("blogtext.csv", nrows=1000) 
dfa = dfa.sample(n=60000, random_state = 0)

# %% colab={"base_uri": "https://localhost:8080/"} id="8ZFtGkEj1MLR" outputId="6b8b74aa-9e2c-4793-a57b-7c6491841c61"
dfa.info()

# %% colab={"base_uri": "https://localhost:8080/"} id="XpkyO0Nq1MHk" outputId="bd51d02c-cb25-4819-8b52-eeb03a82c2d0"
dfa.topic.value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} id="hB7B3kTpGqxc" outputId="f61230d2-5ac3-4b85-f7b1-7e4a1c64203b"
# Count of unique values
print('Unique values in the column gender are',dfa['gender'].nunique())
print('Unique values in the column age are',dfa['age'].nunique())
print('Unique values in the column topic are',dfa['topic'].nunique())
print('Unique values in the column sign are',dfa['sign'].nunique())

# %% colab={"base_uri": "https://localhost:8080/", "height": 269} id="mwFhO3FPK30I" outputId="ffd0a37c-4dae-480d-ceca-593324921673"
# Describe function generates descriptive statistics that summarize the central tendency, 
# dispersion and shape of a dataset’s distribution, excluding NaN values.

# This method tells us a lot of things about a dataset. One important thing is that 
# the describe() method deals only with numeric values. It doesn't work with any 
# categorical values. So if there are any categorical values in a column the describe() 
# method will ignore it and display summary for the other columns.

dfa.describe(include='all').transpose()

# %% id="yWDoWu015p_u"
# Clear the matplotlib plotting backend
# %matplotlib inline
plt.close('all')

# %% colab={"base_uri": "https://localhost:8080/", "height": 475} id="BdXz0EJ7ytD_" outputId="b94d4ae2-aabc-4631-c9b2-b9c7fb0e127b"
# Understand the 'age'
# Distribution and box plot
f, axes = plt.subplots(1, 2, figsize=(17,7))
sns.distplot(dfa['age'],  ax=axes[0],color='Green')
sns.boxplot(x = 'age', data=dfa,  orient='h' , ax=axes[1],color='Green')
axes[0].set_title('Distribution plot')
axes[1].set_title('Box plot')
plt.show()

# Outlier detection
q25,q75=np.percentile(dfa['age'],25),np.percentile(dfa['age'],75)
IQR=q75-q25
Threshold=IQR*1.5
lower,upper=q25-Threshold,q75+Threshold
Outliers=[i for i in dfa['age'] if i < lower or i > upper]
print('{} Total Number of outliers in age: {}'.format('\033[1m',len(Outliers)))

# %% colab={"base_uri": "https://localhost:8080/", "height": 458} id="ZqRjH3YHys5L" outputId="d4ff65d3-d476-41ef-f99f-17a975d40362"
# Understand the 'age'
f,axes=plt.subplots(1,2,figsize=(17,7))
dfa['age'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('age',data=dfa,ax=axes[1])
axes[0].set_title('Pie Chart for age')
axes[1].set_title('Bar Graph for age')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 458} id="msG1Tbcsysux" outputId="4c505be9-db5a-4942-c4a2-3d8ed5f660db"
# Understand the 'gender'
f,axes=plt.subplots(1,2,figsize=(17,7))
dfa['gender'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('gender',data=dfa,ax=axes[1])
axes[0].set_title('Pie Chart for gender')
axes[1].set_title('Bar Graph for gender')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 676} id="3tFtfvBi_nV4" outputId="e4818ce7-442b-48d4-85fe-9df4a5cb99f3"
# Understand the 'topic'
f,axes=plt.subplots(1,2,figsize=(40,20))
dfa['topic'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('topic',data=dfa,ax=axes[1])
axes[0].set_title('Pie Chart for topic')
axes[1].set_title('Bar Graph for topic')
plt.xticks(rotation=90)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 615} id="Hb12NvHj_nLs" outputId="6707259b-0193-4042-e186-794c796ad47f"
# Understand the 'sign'
f,axes=plt.subplots(1,2,figsize=(40,20))
dfa['sign'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
sns.countplot('sign',data=dfa,ax=axes[1])
axes[0].set_title('Pie Chart for sign')
axes[1].set_title('Bar Graph for sign')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 514} id="qNHf9SJ4YRLI" outputId="43b98c38-d69d-4d85-ffde-fa0748c7c96c"
fig, ax = plt.subplots(figsize=(20, 8))
sns.countplot(x="age", hue="gender", data=dfa)

# %% colab={"base_uri": "https://localhost:8080/", "height": 665} id="i37l62ddYVmU" outputId="6bc72596-82d2-4507-c80d-8ee3c409ccef"
fig, ax = plt.subplots(figsize=(20, 8))
sns.countplot(x="topic", hue="gender", data=dfa)
plt.xticks(rotation=90)

# %% [markdown] id="3qQmqVB8YiVc"
# Check some blogs

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="16nObbJGYdMt" outputId="25deb309-588c-4733-fc14-c9122f217474"
dfa['text'].iloc[1]

# %% colab={"base_uri": "https://localhost:8080/", "height": 53} id="FAGgjjxOYo2E" outputId="6f513afb-9c34-428a-e604-4f11adb8e41c"
dfa['text'].iloc[5]

# %% colab={"base_uri": "https://localhost:8080/", "height": 53} id="9rW9XXTNYqoc" outputId="9d526eb9-96d6-4133-d736-8eb0b49ab61c"
dfa['text'].iloc[10]


# %% [markdown] id="EJwuIGp3Fs8o"
# **There is imbalance in the target vector:**
#
# If the imbalanced data is not treated beforehand, then this will degrade the performance of the ML model. Most of the predictions will correspond to the majority class and treat the minority class of features as noise in the data and ignore them. This results in a high bias and low performance of the model.
#
# A widely adopted technique for dealing with highly unbalanced datasets is called re-sampling.
#
# **Two widely used re-sampling methods are:**
#
# - Under-sampling: It is the process where you randomly delete some of the observations from the majority class in order to match the numbers with the minority class.
# - Over-sampling: It is the process of generating synthetic data that tries to randomly generate a sample of the attributes from observations in the minority class
# - Here we can use oversampling because under-sampling may remove important information from the dataset

# %% [markdown] id="a0dc04f4"
# ### 1B. Clean the Structured Data
# - Missing value analysis and imputation.
# - Eliminate Non-English textual data.
# - Hint: Refer ‘langdetect’ library to detect language of the input text

# %% colab={"base_uri": "https://localhost:8080/", "height": 269} id="5toz1aXNmYPc" outputId="45b71ed5-2453-44bf-f2f3-7e0c2b0b76de"
# Percentage of missing values

# df.isnull().sum()
# df.isna().sum()

def missing_check(df):
    total = df.isnull().sum().sort_values(ascending=False)   # total number of null values
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)  # percentage of values that are null
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # putting the above two together
    return missing_data # return the dataframe
missing_check(dfa)

# %% colab={"base_uri": "https://localhost:8080/"} id="ETLceLS6mYG5" outputId="1e0a2457-fdcd-4a8d-8598-6fe35e3eae98"
# Chceck for na values
dfa.isna().sum()

# %% colab={"base_uri": "https://localhost:8080/"} id="00oYybvomX9f" outputId="7c7a3e73-ea47-416f-b9a7-fb35137396bd"
# To take a look at the duplication in the DataFrame as a whole, just call the duplicated() method on 
# the DataFrame. It outputs True if an entire row is identical to a previous row.
dfa.duplicated().sum()

# %% colab={"base_uri": "https://localhost:8080/"} id="hcivXPOOtT2Y" outputId="3378027a-a22c-420e-b6ed-e7947e2443b3"
# Count the number of non-duplicates
(~dfa.duplicated()).sum()

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="YUm5YHlYumH8" outputId="eda1a28d-828a-41e7-da2f-06fa509fb45c"
# Extract duplicate rows
dfa.loc[dfa.duplicated(), :]

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="VQN9HiZCwO-J" outputId="95d4b5be-8c61-4839-c004-86939ec7a68c"
# Dropping the duplicate rows
dfa = dfa.drop_duplicates()
dfa

# %% colab={"base_uri": "https://localhost:8080/"} id="FyxaSgYGJVWJ" outputId="1aaa38b8-7ff0-4dc3-add3-5ac2e8624f5c"
# !pip install langdetect

# %% id="w137XVU_MzP7"
import langdetect
from langdetect import detect

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="BBBci67bAI0P" outputId="098b3943-884a-48f8-ac71-7ae96163d9ec"
# Create function to detect other languages in the dataframe
df = pd.DataFrame({'text': ['This is written in English.', 'هذا مكتوب باللغة العربية', 'English is easy to learn', '']})

def det(x):
    try:
        lang = detect(x)
    except:
        lang = 'Other'
    return lang

df['lang'] = df['text'].apply(det)
df

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="wxXfnZ2FKrW9" outputId="90d48f0a-7b91-4dcf-857c-3c3fdcff03b0"
dfa['lang'] = dfa['text'].apply(det)
dfa

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="WzxV_ebODk49" outputId="c204686f-d72d-45eb-a9ae-06750e08a000"
dfa[dfa["lang"]=='Other']

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="jJYqQDqPQb21" outputId="99145cdb-68a0-4359-bc39-6bc3fd9ff103"
dfa = dfa[dfa["lang"]=='en']
dfa

# %% [markdown] id="5579308b"
# ## 2. Preprocess unstructured data to make it consumable for model training.

# %% [markdown] id="8c19ed0e"
# ### 2A. Eliminate All special Characters and Numbers.

# %% [markdown] id="eee0ed1e"
# ### 2B. Lowercase all textual data.

# %% [markdown] id="895e5f33"
# ### 2C. Remove all Stopwords.

# %% [markdown] id="944ee94e"
# ### 2D. Remove all extra white spaces.

# %% [markdown] id="rqVZfe5a4mu3"
# **Considering all the above 4 steps in below code:**

# %% colab={"base_uri": "https://localhost:8080/"} id="q9JeM73vzpLi" outputId="1f57124c-cf8d-45d1-ebb2-7637734b18f2"
# Use nltk for text pre-processing
import nltk

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

# Remove stopwords
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

# For stemming of the sentence in part 1 of the project
from nltk.stem.snowball import SnowballStemmer

# %% id="lWXGTR3azreJ"
# Preprocessing of the text
dfa['clean_data']=dfa['text'].apply(lambda x: re.sub(r'[^A-Za-z]+',' ',x))  # Keeping only english alphabets strings. removing numbers, the brackets, the full stops etc. 
dfa['clean_data']=dfa['clean_data'].apply(lambda x: re.sub(r'urlLink|urllink','',x))   # Remove all the places where the string urllink comes
dfa['clean_data']=dfa['clean_data'].apply(lambda x: re.sub(r'https?\S+','',x))   # Remove all the places where any url comes that starts with http or https
dfa['clean_data']=dfa['clean_data'].apply(lambda x: x.lower())   # Lowercase each word in the string
dfa['clean_data']=dfa['clean_data'].apply(lambda x: x.strip())   # Remove spaces
dfa['clean_data']=dfa['clean_data'].apply(lambda x: ' '.join([words for words in x.split() if words not in stopwords]))   # Removing stop words
dfa['clean_data']=dfa['clean_data'].apply(lambda x: re.sub(r'\b\w{1,2}\b','',x))    # Removing any word of length less than equal to 2
dfa['clean_data']=dfa['clean_data'].apply(lambda x: ' '.join(dict.fromkeys(x.split())))   # Removing duplicate words

# %% id="VLD1tbtbzrcp"
# Remove Non-English Words from Normalized text
words = set(nltk.corpus.words.words())
def remove_non_english_words(blog):
    return " ".join(w for w in nltk.wordpunct_tokenize(blog) if w.lower() in words or not w.isalpha())

dfa['clean_data'] = dfa['clean_data'].apply(remove_non_english_words)

# %% colab={"base_uri": "https://localhost:8080/", "height": 250} id="ixvIHFt9Bxgo" outputId="af50abec-562b-4d9c-bdaf-eeb0ad58ff4c"
dfa.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="5P3zSil8CuFb" outputId="51002884-c4ea-4d4a-b54f-c6e6696064f0"
nltk.download('all')

# %% id="Yub9rdx4zrZR"
# Lemmatizing the text
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

def lemmatize_text(text):
    return [lmtzr.lemmatize(w) for w in w_tokenizer.tokenize(text)]

dfa['clean_data'] = dfa['clean_data'].apply(lemmatize_text)

# %% id="6RHCxGNrzrWO"
# Stemming the text 
stemmer = SnowballStemmer("english")
dfa['clean_data'] = dfa['clean_data'].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.

# %% colab={"base_uri": "https://localhost:8080/", "height": 53} id="_rPnldZpzrTT" outputId="f3b860c5-5707-46fb-ff3d-361d49730199"
dfa['text'].iloc[10]

# %% colab={"base_uri": "https://localhost:8080/"} id="Zk-jjrP-zrPt" outputId="43a855f9-89cf-40e2-f51a-f7404eceb445"
dfa['clean_data'].iloc[10]

# %% colab={"base_uri": "https://localhost:8080/", "height": 250} id="Hlhkx2LczrLR" outputId="7b8fac23-18dc-4cb2-c0bb-e7c7f7922ae2"
dfa.head()

# %% [markdown] id="c891804a"
# ## 3. Build a base Classification model

# %% colab={"base_uri": "https://localhost:8080/"} id="j_-DqkAg5vpw" outputId="37d02e86-9ef2-492c-c79b-accdb04bbfab"
# Encode the Target Variable
le=LabelEncoder()

dfa['topic']=le.fit_transform(dfa['topic'])
dfa['topic']

# list(le.transform(['Accounting', 'Advertising']))
# list(le.inverse_transform([1, 2, 3]))
# list(le.classes_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="fTdeZmvgHQVu" outputId="5487eb54-9fb6-4f0c-9a9d-c4f2b6ae79c8"
# Dataframe for encoded labels
lab = le.classes_
lab_encoded = le.transform(lab)

# Dictionary of lists 
dict = {'topic': lab, 'topic_encoded': lab_encoded} 
    
df_lab_encoded = pd.DataFrame(dict)
df_lab_encoded

# %% [markdown] id="sW-LZLBWpitu"
# ### 3A. Create dependent and independent variables.
# - Hint: Treat ‘topic’ as a Target variable.

# %% id="Mle6qCwV0uw7"
X=dfa.clean_data
y=dfa.topic

# %% colab={"base_uri": "https://localhost:8080/"} id="WQpHqKMWXnCy" outputId="2bd92ee7-f78c-4fcb-a5e8-ee2524cc5c69"
X

# %% colab={"base_uri": "https://localhost:8080/"} id="3wu3lQS9Xm2-" outputId="fb29b62c-2eea-4a61-c49f-9542b7102084"
y

# %% [markdown] id="9BSF2-kXVsdh"
# ### 3B. Split data into train and test.

# %% id="Zqn4DFOw0ut4"
# Split X and y into training and test set in 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# %% colab={"base_uri": "https://localhost:8080/"} id="B7jCSEyV0urH" outputId="e76f8b08-a830-4473-d38a-af3427b1fa9b"
# Check the shape of train and test data
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %% [markdown] id="EtNMJVcopn23"
# ### 3C. Vectorize data using any one vectorizer.

# %% [markdown] id="TaDztVvW1LJM"
# - Create a Bag of Words using count vectorizer
# - Use ngram_range=(1, 2)
# - Vectorize training and testing features
# - Print the term-document matrix

# %% colab={"base_uri": "https://localhost:8080/"} id="mQ4CY8QuGdK4" outputId="a6011ef0-3460-404b-a647-1db63edfff4c"
# Instantiate the vectorizer
# Creating bag of words which include 1-grams and 2-gram

vect = CountVectorizer(ngram_range=(1,2), analyzer=lambda x: x)

# Learn training data vocabulary, then use it to create a document-term matrix
vect.fit(X)
X_train_transformed = vect.transform(X_train)
X_test_transformed = vect.transform(X_test)

print('The shape of the train data after vectorization is',X_train_transformed.shape)
print('The shape of the test data after vectorization is',X_test_transformed.shape)

# %% id="TaVJ7eOmhA9t"
# # Summarize the encoded vector
# print('Vocabulary :',vect.vocabulary_)
# print('\nShape of the vector: ',X_train_transformed.shape)
# print('\nType of vector: ',type(X_train_transformed))
# print('\nBelow are the sentences in vector form:')
# print(X_train_transformed.toarray())

# %% colab={"base_uri": "https://localhost:8080/"} id="gRCYjvdch_qe" outputId="55d182bc-055b-410d-c0a6-2c37afc79dae"
X_train_transformed[0]

# %% colab={"base_uri": "https://localhost:8080/"} id="eitig_qsjlC7" outputId="eac9424f-d568-4db3-ab3e-6d8d1836d42f"
print(X_train_transformed)

# %% [markdown] id="YiO013Dqpvvo"
# ### 3D. Build a base model for Supervised Learning - Classification.

# %% id="28Do-KcIkPNj"
# Build the Logistic Regression model
logit = LogisticRegression()

# Train the model
logit.fit(X_train_transformed, y_train)
logit_pred = logit.predict(X_test_transformed)

# %% [markdown] id="eEsu5LSwp1Aw"
# ### 3E. Clearly print Performance Metrics.
# - Hint: Accuracy, Precision, Recall, ROC-AUC

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="LOYsdC1fkO-Y" outputId="7396eefc-8f9b-4e92-9a86-069dafa7d960"
# Classification Accuracy
print('Accuracy on Training data:',logit.score(X_train_transformed, y_train))
print('Accuracy on Test data:',logit.score(X_test_transformed, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, logit_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, logit_pred, labels=list(range(0, 40)))
df_cm = pd.DataFrame(cm, index = [i for i in list(range(0, 40))],
                  columns = [i for i in list(range(0, 40))])
plt.figure(figsize = (20,18))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown] id="edGzeuxeCyOq"
# **Evaluation metrics allow us to estimate errors to determine how well our models are performing:**
#
# > Accuracy: ratio of correct predictions over total predictions.
#
# > Precision: how often the classifier is correct when it predicts positive.
#
# > Recall: how often the classifier is correct for all positive instances.
#
# > F-Score: single measurement to combine precision and recall.

# %% [markdown] id="e4db61f8"
# ## 4. Improve performance of Model

# %% [markdown] id="5667378a"
# ### 4A. Experiment with other vectorisers.

# %% [markdown] id="e8f74600"
# ### 4B. Build classifier Models using other algorithms than base model.

# %% [markdown] id="33737edf"
# ### 4C. Tune Parameters/Hyperparameters of the model/s.

# %% [markdown] id="d2ee8834"
# ### 4D. Clearly print Performance Metrics.
# - Hint: Accuracy, Precision, Recall, ROC-AUC

# %% [markdown] id="vbzUh5f2Cy7b"
# **Considering all the above 4 parts in below 6 cases: Except case-1, the code has been commented after testing because of high RAM issues and long training times.**

# %% [markdown] id="9W4p4cyEDEHG"
# ### Case-1: Using the CountVectorizer

# %% id="P2JjVV-EWDw5"
vect.fit(X)
X = vect.transform(X)

# %% colab={"base_uri": "https://localhost:8080/"} id="Ufwe5sgiasw7" outputId="9738eca6-55c7-4114-b300-9d3e6f254816"
X

# %% [markdown] id="Eq4Hjc1NZPpj"
# Select the best performing model

# %% id="xDNG1iddWDy6"
# Use K-Fold Cross Validation for model selection
# Define various classification models
LogisticRegression = LogisticRegression(n_jobs=1, C=1e5)
SGDClassifier = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
MultinomialNB = MultinomialNB()

# %% colab={"base_uri": "https://localhost:8080/", "height": 408} id="2TgS1qi8WDst" outputId="3f7fa89e-51c7-49b9-8dc4-55042951d85f"
# K Fold Cross Validation Scores

seed = 0

# Create models
models = []
models.append(('LogisticRegression', LogisticRegression))
models.append(('SGDClassifier', SGDClassifier))
models.append(('MultinomialNB', MultinomialNB))

# Evaluate each model in turn
results = []
names = []

# Use different metrics based on context
scoring = 'accuracy'
# scoring = 'precision'
# scoring = 'recall'
# scoring = 'f1'
for name, model in models:
	kfold = model_selection.KFold(n_splits=5,random_state=seed,shuffle=True)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
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

# %% [markdown] id="gHpIlM9gourP"
# Using the Artificial Neural Network (ANN)

# %% id="9oszPTPSWDhK"
# Encoding is used for categorical data
# We can consider 3,4,5,6,7,8 as categorical
# Use one hot encoding
y_train_dum = pd.get_dummies(y_train).to_numpy()
y_test_dum = pd.get_dummies(y_test).to_numpy()

# %% colab={"base_uri": "https://localhost:8080/"} id="1wZObqSGkt5q" outputId="03d617ca-e1ea-4b33-c85d-fadebfc6be39"
y_train_dum

# %% colab={"base_uri": "https://localhost:8080/"} id="P_HVp6G2kzZO" outputId="ee3a025a-e2b4-4363-bd1c-7c6ff21af2d2"
print(X_train_transformed.shape,X_test_transformed.shape)
print(y_train_dum.shape,y_test_dum.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="bgrZX3BlkzKc" outputId="d2c73d21-9672-49fd-9d27-80a0bed22db8"
n_inputs = X_train_transformed.shape[1]
n_outputs = y_train_dum.shape[1]

model_nn = Sequential()
model_nn.add(Dense(512, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
model_nn.add(BatchNormalization())

# The Hidden Layers :
model_nn.add(Dense(256, kernel_initializer='he_uniform', activation='relu'))
model_nn.add(BatchNormalization())
model_nn.add(Dense(128, kernel_initializer='he_uniform',activation='relu')) 
model_nn.add(BatchNormalization())

# the output layer
model_nn.add(Dense(n_outputs, activation='sigmoid'))
model_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics =['accuracy','categorical_crossentropy'])

# stop = EarlyStopping(monitor="val_loss", patience=3, min_delta=0.01)
model_nn.fit(X_train_transformed, y_train_dum, validation_data=(X_test_transformed,y_test_dum), verbose=1, epochs=20, batch_size = 64)
# model_nn.fit(X_train_transformed, y_train_dum, validation_data=(X_test_transformed,y_test_dum), verbose=1, epochs=8, batch_size = 64, callbacks=[stop])

# %% colab={"base_uri": "https://localhost:8080/"} id="80NzU-jQz0YV" outputId="cfd0e261-0fb5-4f15-be8d-f71aa905e8ac"
y_pred=model_nn.predict(X_test_transformed)
y_pred

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="nCaPcZcXzQUS" outputId="b17f578c-76c0-44e7-82ed-d1343000ed86"
# Classification Accuracy
print('Loss and Accuracy on Training data:',model_nn.evaluate(X_train_transformed, y_train_dum))
print('Loss and Accuracy on Test data:',model_nn.evaluate(X_test_transformed, y_test_dum))
print()

# Classification Report
print("Classification Report:\n",classification_report(y_test_dum.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test_dum.argmax(axis=1), y_pred.argmax(axis=1), labels=list(range(0, 40)))
df_cm = pd.DataFrame(cm, index = [i for i in list(range(0, 40))],
                  columns = [i for i in list(range(0, 40))])
plt.figure(figsize = (20,18))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown] id="KSijZVM-Z9RA"
# ### Case-2: Using the TfIdfVectorizer

# %% id="BSqNEaIIcXvA"
# from sklearn.feature_extraction.text import TfidfVectorizer

# %% id="KPp8k6Pba1Ym"
# # Instantiate the vectorizer
# # Creating bag of words which include 1-grams and 2-gram

# vect = TfidfVectorizer(ngram_range=(1,2), analyzer=lambda x: x)

# # Learn training data vocabulary, then use it to create a document-term matrix
# vect.fit(X_train)
# X_train_transformed = vect.transform(X_train).toarray()
# X_test_transformed = vect.transform(X_test).toarray()

# print('The shape of the train data after vectorization is',X_train_transformed.shape)
# print('The shape of the test data after vectorization is',X_test_transformed.shape)

# %% id="aGDofkzAbGsc"
# X=dfa.clean_data

# vect.fit(X)
# X = vect.transform(X)

# %% id="ANbzMNWLbHz8"
# X

# %% [markdown] id="m1-BsMDzZ9RA"
# Select the best performing model

# %% id="oiy97ffmeBKq"
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.naive_bayes import MultinomialNB

# %% id="U60f_d6CZ9RA"
# # Use K-Fold Cross Validation for model selection
# # Define various classification models
# LogisticRegression = LogisticRegression(n_jobs=1, C=1e5)
# SGDClassifier = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
# MultinomialNB = MultinomialNB()

# %% id="mINPfRYOZ9RB"
# # K Fold Cross Validation Scores

# seed = 0

# # Create models
# models = []
# models.append(('LogisticRegression', LogisticRegression))
# models.append(('SGDClassifier', SGDClassifier))
# models.append(('MultinomialNB', MultinomialNB))

# # Evaluate each model in turn
# results = []
# names = []

# # Use different metrics based on context
# scoring = 'accuracy'
# # scoring = 'precision'
# # scoring = 'recall'
# # scoring = 'f1'
# for name, model in models:
# 	kfold = model_selection.KFold(n_splits=5,random_state=seed,shuffle=True)
# 	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
# 	results.append(cv_results)
# 	names.append(name)
# 	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# 	print(msg)
    
# # Boxplot for algorithm comparison
# fig = plt.figure(figsize=(12,5))
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# %% [markdown] id="1XmZV5M7Z9RB"
# Using the Artificial Neural Network (ANN)

# %% id="LesBUCCcZ9RB"
# print(X_train_transformed.shape,X_test_transformed.shape)
# print(y_train_dum.shape,y_test_dum.shape)

# %% id="FMNYQrb5Z9RC"
# n_inputs = X_train_transformed.shape[1]
# n_outputs = y_train_dum.shape[1]

# model_nn = Sequential()
# model_nn.add(Dense(512, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
# model_nn.add(BatchNormalization())

# # The Hidden Layers :
# model_nn.add(Dense(256, kernel_initializer='he_uniform', activation='relu'))
# model_nn.add(BatchNormalization())
# model_nn.add(Dense(128, kernel_initializer='he_uniform',activation='relu')) 
# model_nn.add(BatchNormalization())

# # the output layer
# model_nn.add(Dense(n_outputs, activation='sigmoid'))
# model_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics =['accuracy','categorical_crossentropy'])

# # stop = EarlyStopping(monitor="val_loss", patience=3, min_delta=0.01)
# model_nn.fit(X_train_transformed, y_train_dum, validation_data=(X_test_transformed,y_test_dum), verbose=1, epochs=20, batch_size = 64)
# # model_nn.fit(X_train_transformed, y_train_dum, validation_data=(X_test_transformed,y_test_dum), verbose=1, epochs=8, batch_size = 64, callbacks=[stop])

# %% id="xJX3CRWaZ9RC"
# y_pred=model_nn.predict(X_test_transformed)
# y_pred

# %% id="w4bgjcizZ9RC"
# # Classification Accuracy
# print('Loss and Accuracy on Training data:',model_nn.evaluate(X_train_transformed, y_train_dum))
# print('Loss and Accuracy on Test data:',model_nn.evaluate(X_test_transformed, y_test_dum))
# print()

# # Classification Report
# print("Classification Report:\n",classification_report(y_test_dum.argmax(axis=1), y_pred.argmax(axis=1)))

# # Confusion Matrix
# print("Confusion Matrix Chart:")
# cm = confusion_matrix(y_test_dum.argmax(axis=1), y_pred.argmax(axis=1), labels=list(range(0, 40)))
# df_cm = pd.DataFrame(cm, index = [i for i in list(range(0, 40))],
#                   columns = [i for i in list(range(0, 40))])
# plt.figure(figsize = (20,18))
# sns.heatmap(df_cm, annot=True, fmt='g')
# plt.show()

# %% [markdown] id="L7WG23QUZwtM"
# ### Case-3: Using the HashingVectorizer

# %% id="9IK5WOsc7vlv"
# from sklearn.feature_extraction.text import TfidfVectorizer

# %% id="3Sig0F4y7vl1"
# # Instantiate the vectorizer
# # Creating bag of words which include 1-grams and 2-gram

# vect = TfidfVectorizer(ngram_range=(1,2), analyzer=lambda x: x)

# # Learn training data vocabulary, then use it to create a document-term matrix
# vect.fit(X_train)
# X_train_transformed = vect.transform(X_train).toarray()
# X_test_transformed = vect.transform(X_test).toarray()

# print('The shape of the train data after vectorization is',X_train_transformed.shape)
# print('The shape of the test data after vectorization is',X_test_transformed.shape)

# %% id="2HdTj2SE7vl1"
# X=dfa.clean_data

# vect.fit(X)
# X = vect.transform(X)

# %% id="Yi60vGs47vl1"
# X

# %% [markdown] id="BdFvhKYWZwtM"
# Select the best performing model

# %% id="4Smy6akxF_8u"
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.naive_bayes import MultinomialNB

# %% id="2QQTA-zuZwtM"
# # Use K-Fold Cross Validation for model selection
# # Define various classification models
# LogisticRegression = LogisticRegression(n_jobs=1, C=1e5)
# SGDClassifier = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
# MultinomialNB = MultinomialNB()

# %% id="NckIvo02ZwtN"
# # K Fold Cross Validation Scores

# seed = 0

# # Create models
# models = []
# models.append(('LogisticRegression', LogisticRegression))
# models.append(('SGDClassifier', SGDClassifier))
# models.append(('MultinomialNB', MultinomialNB))

# # Evaluate each model in turn
# results = []
# names = []

# # Use different metrics based on context
# scoring = 'accuracy'
# # scoring = 'precision'
# # scoring = 'recall'
# # scoring = 'f1'
# for name, model in models:
# 	kfold = model_selection.KFold(n_splits=5,random_state=seed,shuffle=True)
# 	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
# 	results.append(cv_results)
# 	names.append(name)
# 	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# 	print(msg)
    
# # Boxplot for algorithm comparison
# fig = plt.figure(figsize=(12,5))
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# %% [markdown] id="GeNKGPVNZwtN"
# Using the Artificial Neural Network (ANN)

# %% id="lSyY1448ZwtO"
# print(X_train_transformed.shape,X_test_transformed.shape)
# print(y_train_dum.shape,y_test_dum.shape)

# %% id="7YFm81dmZwtO"
# n_inputs = X_train_transformed.shape[1]
# n_outputs = y_train_dum.shape[1]

# model_nn = Sequential()
# model_nn.add(Dense(512, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
# model_nn.add(BatchNormalization())

# # The Hidden Layers :
# model_nn.add(Dense(256, kernel_initializer='he_uniform', activation='relu'))
# model_nn.add(BatchNormalization())
# model_nn.add(Dense(128, kernel_initializer='he_uniform',activation='relu')) 
# model_nn.add(BatchNormalization())

# # the output layer
# model_nn.add(Dense(n_outputs, activation='sigmoid'))
# model_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics =['accuracy','categorical_crossentropy'])

# # stop = EarlyStopping(monitor="val_loss", patience=3, min_delta=0.01)
# model_nn.fit(X_train_transformed, y_train_dum, validation_data=(X_test_transformed,y_test_dum), verbose=1, epochs=20, batch_size = 64)
# # model_nn.fit(X_train_transformed, y_train_dum, validation_data=(X_test_transformed,y_test_dum), verbose=1, epochs=8, batch_size = 64, callbacks=[stop])

# %% id="jxzHiyXqZwtO"
# y_pred=model_nn.predict(X_test_transformed)
# y_pred

# %% id="KWLggUX7ZwtO"
# # Classification Accuracy
# print('Loss and Accuracy on Training data:',model_nn.evaluate(X_train_transformed, y_train_dum))
# print('Loss and Accuracy on Test data:',model_nn.evaluate(X_test_transformed, y_test_dum))
# print()

# # Classification Report
# print("Classification Report:\n",classification_report(y_test_dum.argmax(axis=1), y_pred.argmax(axis=1)))

# # Confusion Matrix
# print("Confusion Matrix Chart:")
# cm = confusion_matrix(y_test_dum.argmax(axis=1), y_pred.argmax(axis=1), labels=list(range(0, 40)))
# df_cm = pd.DataFrame(cm, index = [i for i in list(range(0, 40))],
#                   columns = [i for i in list(range(0, 40))])
# plt.figure(figsize = (20,18))
# sns.heatmap(df_cm, annot=True, fmt='g')
# plt.show()

# %% [markdown] id="KWKnQCX0Qt4j"
# ### Case-4: Using the Word2Vec Model

# %% id="EqFdMvJwQyhy"
# # X_train_final_tokens = X_train.apply(nltk.word_tokenize)
# X_train_tokens_list = X_train.tolist()

# %% id="KbE1DiOAab9x"
# X_train_tokens_list

# %% id="z2hZmLsmQyfX"
# # X_test_final_tokens = X_test.apply(nltk.word_tokenize)
# X_test_tokens_list = X_test.tolist()

# %% id="nEf-C8Akpg63"
# # %%time

# from gensim.models import Word2Vec

# wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
# wv.init_sims(replace=True)

# %% id="Yi6DGFvXQyai"
# from itertools import islice
# list(islice(wv.vocab, 13030, 13050))

# %% id="zt36wzziQyXP"
# # The common way is to average the two word vectors. BOW based approaches which includes averaging.
# def word_averaging(wv, words):
#     all_words, mean = set(), []
    
#     for word in words:
#         if isinstance(word, np.ndarray):
#             mean.append(word)
#         elif word in wv.vocab:
#             mean.append(wv.syn0norm[wv.vocab[word].index])
#             all_words.add(wv.vocab[word].index)

#     if not mean:
#         logging.warning("cannot compute similarity with no input %s", words)
#         # FIXME: remove these examples in pre-processing
#         return np.zeros(wv.vector_size,)

#     mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
#     return mean

# def  word_averaging_list(wv, text_list):
#     return np.vstack([word_averaging(wv, post) for post in text_list ])

# %% id="RI808RBsQyVC"
# X_train_word_average = word_averaging_list(wv,X_train_tokens_list)
# X_test_word_average = word_averaging_list(wv,X_test_tokens_list)

# %% id="ZnAC-QLlQySN"
# # Build the Logistic Regression model
# logit = LogisticRegression()

# # Train the model
# logit.fit(X_train_word_average, y_train)
# logit_pred = logit.predict(X_test_word_average)

# %% id="b4ZElOlasQJ8"
# # Classification Accuracy
# print('Accuracy on Training data:',logit.score(X_train_word_average, y_train))
# print('Accuracy on Test data:',logit.score(X_test_word_average, y_test))

# # Classification Report
# print("Classification Report:\n",classification_report(y_test, logit_pred))

# # Confusion Matrix
# print("Confusion Matrix Chart:")
# cm = confusion_matrix(y_test, logit_pred, labels=list(range(0, 40)))
# df_cm = pd.DataFrame(cm, index = [i for i in list(range(0, 40))],
#                   columns = [i for i in list(range(0, 40))])
# plt.figure(figsize = (20,18))
# sns.heatmap(df_cm, annot=True, fmt='g')
# plt.show()

# %% [markdown] id="aoanDyLeIiBc"
# ### Case-5: Using the Glove Model

# %% id="vPjkTnrOInDx"
# import tensorflow
# from tensorflow import keras

# %% id="O65P6fcdInBi"
# # %%time

# glove_embeddings = {}
# with open("glove.840B.300d.txt") as f:
#     for line in f:
#         try:
#             line = line.split()
#             glove_embeddings[line[0]] = np.array(line[1:], dtype=np.float32)
#         except:
#             continue

# %% id="C_Hl2_4jIm_N"
# embeddings = glove_embeddings["the"]

# embeddings.shape, embeddings.dtype

# %% id="MSECLUOQIm9o"
# from keras.preprocessing.text import Tokenizer
# from keras_preprocessing.sequence import pad_sequences

# max_tokens = 100 ## Hyperparameter

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X)

# ## Vectorizing data to keep 50 words per sample.
# X_train_vect = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_tokens, padding="post", truncating="post", value=0.)
# X_test_vect  = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_tokens, padding="post", truncating="post", value=0.)

# print(X_train_vect[:3])

# X_train_vect.shape, X_test_vect.shape

# %% id="WNzMei0lIm8D"
# print("Vocab Size : {}".format(len(tokenizer.word_index)))

# %% id="PGG2T2SxIm5v"
# ## What is word 13

# print(tokenizer.index_word[13])

# ## How many times it comes in first text document??

# print(X_train[1]) ## 2 times

# %% id="XQ2xpwRyIm2e"
# # %%time

# embed_len = 300

# word_embeddings = np.zeros((len(tokenizer.index_word)+1, embed_len))

# for idx, word in tokenizer.index_word.items():
#     word_embeddings[idx] = glove_embeddings.get(word, np.zeros(embed_len))

# %% id="jwp13657ImzW"
# word_embeddings[1][:10]

# %% id="SLPMvMm8tkjS"
# # Define Network
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, Flatten

# model = Sequential([
#                     Embedding(input_dim=len(tokenizer.index_word)+1, output_dim=embed_len,
#                               input_length=max_tokens, trainable=False, weights=[word_embeddings]),
#                     Flatten(),
#                     Dense(128, activation="relu"),
#                     Dense(64, activation="relu"),
#                     Dense(len(dfa.topic), activation="softmax")
#                 ])

# model.summary()

# %% id="DmP_ty_FtkhY"
# model.weights[0][1][:10], word_embeddings[1][:10]

# %% id="x39Bziaztke0"
# # Compile Network
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# %% id="0Ugmu_Hvtkcf"
# # Train Network
# model.fit(X_train_vect, y_train, batch_size=32, epochs=8, validation_data=(X_test_vect, y_test))

# %% id="4mfeAqr9tkYs"
# y_pred = model.predict(X_test_vect).argmax(axis=-1)

# # Classification Accuracy
# print('Loss and Accuracy on Training data:',model.evaluate(X_train_vect, y_train))
# print('Loss and Accuracy on Test data:',model.evaluate(X_test_vect, y_test))
# print()

# # Classification Report
# print("Classification Report:\n",classification_report(y_test, y_pred))

# # Confusion Matrix
# print("Confusion Matrix Chart:")
# cm = confusion_matrix(y_test, y_pred, labels=list(range(0, 40)))
# df_cm = pd.DataFrame(cm, index = [i for i in list(range(0, 40))],
#                   columns = [i for i in list(range(0, 40))])
# plt.figure(figsize = (20,18))
# sns.heatmap(df_cm, annot=True, fmt='g')
# plt.show()

# %% [markdown] id="KeYtFE8RES0y"
# ### Case-6: Hyperparameter Tuning
# Using the Logistic Regression classifier as its giving a better performance.

# %% id="rMOMmAEOnrKO"
from sklearn.linear_model import LogisticRegression

# Build the Logistic Regression model
logit = LogisticRegression()

# Train the model
logit.fit(X_train_transformed, y_train)
logit_pred = logit.predict(X_test_transformed)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="VCFO4KjOnpVt" outputId="93a25cf7-808a-414b-e903-b43eedf747e4"
# Classification Accuracy
print('Accuracy on Training data:',logit.score(X_train_transformed, y_train))
print('Accuracy on Test data:',logit.score(X_test_transformed, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, logit_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, logit_pred, labels=list(range(0, 40)))
df_cm = pd.DataFrame(cm, index = [i for i in list(range(0, 40))],
                  columns = [i for i in list(range(0, 40))])
plt.figure(figsize = (20,18))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown] id="kw0SjhP82qJr"
# Not using all the hyperparameters due to long running times in RandomizedSearchCV:

# %% colab={"base_uri": "https://localhost:8080/"} id="Qdfsqq-TvxXy" outputId="65f9d681-a6c0-411e-93ff-0c0d2125dcf5"
# %%time
# Using the RandomizedSearchCV
# params =     {'penalty' : ['l1', 'l2'],
#               'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#               'solver' : ['lbfgs','liblinear','sag','saga'],
#               }

params =     {'penalty' : ['l1', 'l2'],
              'C' : [1, 10]
              }

gs = RandomizedSearchCV(estimator=logit, param_distributions=params, cv=5, verbose=10, random_state=0)
gs.fit(X_train_transformed, y_train)

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="BY0y8qoZvxMF" outputId="108119f7-7901-4083-e803-84df379719ac"
pd.DataFrame({'param': gs.cv_results_["params"], 
              'score mean': gs.cv_results_["mean_test_score"], 
              'score s.d.': gs.cv_results_["std_test_score"],
              'rank': gs.cv_results_["rank_test_score"]})

# %% colab={"base_uri": "https://localhost:8080/"} id="h7hYlVCcvw_j" outputId="5c0941e6-9bf9-4d65-a13c-583ce4782f7d"
# Print the best parameters
print('Best Parameters:', gs.best_params_, 'mean score: ', gs.best_score_, sep='\n')

# %% id="gXXo6RYL19Br"
# Build the Logistic Regression model
logit = LogisticRegression(penalty = 'l2', C = 1)

# Train the model
logit.fit(X_train_transformed, y_train)
logit_pred = logit.predict(X_test_transformed)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="qCTv6Q1K19Br" outputId="d2c1650d-f551-45f5-e06c-4e9e6ef379dd"
# Classification Accuracy
print('Accuracy on Training data:',logit.score(X_train_transformed, y_train))
print('Accuracy on Test data:',logit.score(X_test_transformed, y_test))

# Classification Report
print("Classification Report:\n",classification_report(y_test, logit_pred))

# Confusion Matrix
print("Confusion Matrix Chart:")
cm = confusion_matrix(y_test, logit_pred, labels=list(range(0, 40)))
df_cm = pd.DataFrame(cm, index = [i for i in list(range(0, 40))],
                  columns = [i for i in list(range(0, 40))])
plt.figure(figsize = (20,18))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# %% [markdown] id="p6aVj826EN72"
# From the above metrics; Its evident that Logistic regression with default parameters is giving a better performance.

# %% [markdown] id="fd09209e"
# ## 5. Share insights on relative performance comparison

# %% [markdown] id="b4f8ff41"
# ### 5A. Which vectorizer performed better? Probable reason?

# %% [markdown] id="Th7XE3hxP_7S"
# I found that CountVectorizer worked better than TfIdfVectorizer.
#
# Secondly, as we increase the ngram-range parameter of both CountVectorizer and TfIdfVectorizer, ideally accuracy of TfIdf improves over CountVectorizer. I would compare performance across folds (cross validation) to make sure countvectorizer consistently performs better.
#
# Count Vectors can be helpful in understanding the type of text by the frequency of words in it. But its major disadvantages are:
#
# - Its inability in identifying more important and less important words for analysis.
# - It will just consider words that are abundant in a corpus as the most statistically significant word.
# - It also doesn't identify the relationships between words such as linguistic similarity between words.
#

# %% [markdown] id="2c437bac"
# ### 5B. Which model outperformed? Probable reason?

# %% [markdown] id="x7DCesy-U_7x"
# As we are using a small data sample here the Logistic Regression is performng better than ANN.
#
# we need a good ratio of data points to parameters to get reliable estimates so the first criteria would be lots of data in order to estimate lots of parameters. If that's not true then we'd be estimating lots of parameters with little data per parameter and get a bunch of spurious results. Therefore depending upon the situation, the additional granularity of the Deep Neural Network would either represent a treasure trove of additional detail and value, or an error prone and misleading representation of the situation. 

# %% [markdown] id="78ea3ece"
# ### 5C. Which parameter/hyperparameter significantly helped to improve performance?Probable reason?

# %% [markdown] id="MXQft2Oo3TGx"
# The main hyperparameters we may tune in logistic regression are: solver, penalty, and regularization strength (sklearn documentation).
#
# 1. **Solver** is the algorithm to use in the optimization problem. The choices are {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’.
#
#  - lbfgs: relatively performs well compared to other methods and it saves a lot of memory, however, sometimes it may have issues with convergence.
#  - sag: faster than other solvers for large datasets, when both the number of samples and the number of features are large.
#  - saga: the solver of choice for sparse multinomial logistic regression and it’s also suitable for very large datasets.
#  - newton-cg: computationally expensive because of the Hessian Matrix.
#  - liblinear: recommended when you have a high dimension dataset - solving large-scale classification problems.
#
# 2. **Penalty (or regularization)** intends to reduce model generalization error, and is meant to disincentivize and regulate overfitting. Technique discourages learning a more complex model, so as to avoid the risk of overfitting. The choices are: {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’.
#
# 3. **C (or regularization strength)** must be a positive float. Regularization strength works with the penalty to regulate overfitting. Smaller values specify stronger regularization and high value tells the model to give high weight to the training data.
#
# 4. **Logistic regression offers other parameters like:** class_weight, dualbool (for sparse datasets when n_samples > n_features), max_iter (may improve convergence with higher iterations), and others. However, these provide less impact.

# %% [markdown] id="16484edd"
# ### 5D. According to you, which performance metric should be given most importance, why?

# %% [markdown] id="O_HHXjM3aUsd"
# Besides the ROC-AUC and Kohonen's kappa as the metircs for imbalanced dataset with multiclass classification, I'd also like to add a few metrics I've found useful for imbalanced data. They are both related to precision and recall. Because by averaging these you get a metric weighing TPs and both types of errors (FP and FN):
#
# - F1 score, which is the harmonic mean of precision and recall.
# - G-measure, which is the geometric mean of precision and recall. Compared to F1, I've found it a bit better for imbalanced data.
# - Jaccard index, which you can think of as the TP/(TP+FP+FN)
#
# Note: For imbalanced datasets, it is best to have your metrics be macro-averaged.

# %% [markdown] id="0dd7c2e4"
# # Part-B: Solution

# %% [markdown] id="e624389a"
# - **DOMAIN:** Customer support
# - **CONTEXT:** Great Learning has a an academic support department which receives numerous support requests every day throughout the year. Teams are spread across geographies and try to provide support round the year. Sometimes there are circumstances where due to heavy workload certain request resolutions are delayed, impacting company’s business. Some of the requests are very generic where a proper resolution procedure delivered to the user can solve the problem. Company is looking forward to design an automation which can interact with the user, understand the problem and display the resolution procedure if found as a generic request or redirect the request to an actual human support executive if the request is complex or not in it’s database.
# - **DATA DESCRIPTION:** A sample corpus is attached for your reference. Please enhance/add more data to the corpus using your linguistics skills.
# - **PROJECT OBJECTIVE:** Design a python based interactive semi - rule based chatbot which can do the following:

# %% [markdown] id="a3313f55"
# ## 1. Start chat session with greetings and ask what the user is looking for.

# %% [markdown] id="5579308b"
# ## 2. Accept dynamic text based questions from the user. Reply back with relevant answer from the designed corpus.

# %% [markdown] id="c891804a"
# ## 3. End the chat session only if the user requests to end else ask what the user is looking for. Loop continues till the user asks to end it.
# Hint: There are a lot of techniques using which one can clean and prepare the data which can be used to train a ML/DL classifier. Hence, it might require you to experiment, research, self learn and implement the above classifier. There might be many iterations between hand building the corpus and designing the best fit text classifier. As the quality and quantity of corpus increases the model’s performance i.e. ability to answer right questions also increases.
# Reference: https://www.mygreatlearning.com/blog/basics-of-building-an-artificial-intelligence-chatbot/

# %% [markdown] id="88a30183"
# ## 4. Evaluation: Evaluator will use linguistics to twist and turn sentences to ask questions on the topics described in DATA DESCRIPTION and check if the bot is giving relevant replies.

# %% [markdown] id="JKXDu5L8IXWD"
# ## Import packages

# %% [markdown] id="lQvRYxwaIKKx"
# This is a generic chatbot. Can be trained on pretty much any conversation as long as we have a correctly formatted JSON file.

# %% id="26856bb6"
# Import all the relevant libraries needed to complete the analysis, visualization, modeling and presentation
import pandas as pd
import numpy as np
import os

import json

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
sns.set_style('darkgrid')

from scipy import stats
from scipy.stats import zscore

# Using ML
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

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
import pandas_profiling as pp

import gensim
import logging

# Using tensorflow
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

from keras.utils.np_utils import to_categorical  
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

# Using torch
import torch
import torch.nn as nn
import random

from torch.utils.data import Dataset, DataLoader
from tkinter import *

import warnings
warnings.filterwarnings("ignore")

import random
from zipfile import ZipFile

# Set random_state
random_state = 42

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %% colab={"base_uri": "https://localhost:8080/"} id="Sdmaay16zYD5" outputId="77adb7f0-7160-4e14-f57e-0a1f2e856d24"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="p0GvFDjKw83z" outputId="c10e8cc0-4d0f-4544-c6af-cf99783c0364"
# Current working directory
# %cd "/content/drive/MyDrive/MGL/Project-NLP-1/"

# # List all the files in a directory
# for dirname, _, filenames in os.walk('path'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# %% colab={"base_uri": "https://localhost:8080/"} id="KrBsQMgcw80s" outputId="159b0256-45a2-440c-fd2a-2f425fc28437"
# List files in the directory
# !ls

# %% [markdown] id="EHvBFQBXIiGA"
# ## Quick EDA

# %% id="2TDP_65aw8r_"
# Creat dataframe from the json file
with open('GL Bot.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# %% colab={"base_uri": "https://localhost:8080/", "height": 441} id="LpTLjyr1w8pJ" outputId="41c9a7fc-191c-4ec3-88bb-60db32b9b453"
pd.set_option('display.max_colwidth', None)
df.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 657} id="5FByXy8aw8mQ" outputId="2ef6c56d-7ec6-4d4b-ca77-440eb81b0bd8"
df1 = df.intents.apply(pd.Series)
df1

# %% [markdown] id="20OipOSdJHu3"
# ## NLTK Packs

# %% colab={"base_uri": "https://localhost:8080/"} id="Zr_ytcVEtzD4" outputId="73c3612a-52c5-47de-c862-9e00ac0964a2"
# Download the relevant content from NLTK
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')

# %% [markdown] id="DNJIWrO5Kmf-"
# ## Load Data

# %% id="d3eahyOOfZAj"
# Load the data
with open("GL Bot.json", "r") as f:
    intents = json.load(f)

# %% [markdown] id="6ebeO6gELRbh"
# ## Data Preprocessing

# %% id="6AYszty6tAMx"
# Creating custom functions for tekenization. stemming, and BOW
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


# %% colab={"base_uri": "https://localhost:8080/"} id="DsSaUCPDtFxw" outputId="578407d0-e305-4d59-8107-f7ce74fe7174"
# Create xy pairs
all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents["intents"]:
    tag = intent["tag"]
    # add to tag list
    tags.append(tag)
    for pattern in intent["patterns"]:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))
print(xy)

# %% colab={"base_uri": "https://localhost:8080/"} id="1OLJtP4qtFvP" outputId="f8745079-686a-44c5-f0fa-669aa18597db"
# Separate all the tags & words into their separate lists
# stem and lower each word
ignore_words = ["?", ".", "!"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# %% [markdown] id="OmlpTbZyMD0U"
# ## Train and Test data

# %% colab={"base_uri": "https://localhost:8080/"} id="qH79d2qbtFtC" outputId="f99a5ccd-791a-4084-9bd4-458c13120234"
# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train)
print(y_train)


# %% [markdown] id="JAqnTU_DNUDk"
# ## NN Model

# %% id="p3kd3dc4tFrf"
# Creating our model. Here we have inherited a class from NN.Module because we will be customizing the model & its layers
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out


# %% id="vf2wpIa7tFos"
# We will use some Magic functions, write our class. You can read online about getitem and setitem magic funtions
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# %% colab={"base_uri": "https://localhost:8080/"} id="IYlzpFVttFlb" outputId="95404b3f-a7dc-469d-b99f-e319124c038d"
# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

# %% id="4k-0SQQYtFjS"
# We will now instantiate the model, loss and optimizer functions.
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %% [markdown] id="dSqr_jTcNcva"
# ## Model Training

# %% colab={"base_uri": "https://localhost:8080/"} id="MRUO0uIztFgY" outputId="bf9901b4-be5f-4a00-ca84-5f525098e4eb"
# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}

# %% id="rI0whnhXuPph"
# Saving the training model
FILE = "data.pth"
torch.save(data, FILE)

# %% colab={"base_uri": "https://localhost:8080/"} id="yFjOeECauPm3" outputId="432b786d-b50e-4f04-b738-49c13c792fc0"
# Loading the model and data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("GL Bot.json", "r") as json_data:
    intents = json.load(json_data)
FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# %% [markdown] id="osS30HoEN_qQ"
# ## Chatbot
# Our Model is Ready. As our training data was very limited, we can only chat about a handful of topics. You can train it on a bigger dataset to increase the chatbot’s generalization / knowledge.

# %% id="9vA6L0SXuPlG"
bot_name = "GLBot"
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
    return "Sorry, didn't get it..."


# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="IhN8sZ67uPhs" outputId="16e1df2c-6ef2-44c3-bc9e-ff34ec7ba21f"
# Test the function
get_response('hi')


# %% colab={"base_uri": "https://localhost:8080/"} id="PugvNEF1uPfI" outputId="7c51f3b7-6695-4a9c-f1e6-fe882ec67059"
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = get_response(inp)
        # results_index = numpy.argmax(results)
        # tag = labels[results_index]

        # for tg in data["intents"]:
        #     if tg['tag'] == tag:
        #         responses = tg['responses']

        print(results)
chat()

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
