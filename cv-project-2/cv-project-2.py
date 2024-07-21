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
# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#00b3e5;">Contents: Computer Vision Project - 2
# 1. [Part-A: Solution](#Part-A:-Solution)
# 2. [Part-B: Solution](#Part-B:-Solution)
# 3. [Part-C: Solution](#Part-C:-Solution)

# %% [markdown] id="5f36f6dd"
# # Part-A: Solution

# %% [markdown] id="dedd939b"
# - **DOMAIN:** Entertainment
# - **CONTEXT:** Company X owns a movie application and repository which caters movie streaming to millions of users who on subscription basis. Company wants to automate the process of cast and crew information in each scene from a movie such that when a user pauses on the movie and clicks on cast information button, the app will show details of the actor in the scene. Company has an in-house computer vision and multimedia experts who need to detect faces from screen shots from the movie scene.
# The data labelling is already done. Since there higher time complexity is involved in the process.
# - **DATA DESCRIPTION:** The dataset comprises of images and its mask for corresponding human face.
# - **PROJECT OBJECTIVE:** To build a face detection system.

# %% id="9b11c05c"
# Import all the relevant libraries required to complete the analysis, visualization, modeling and presentation
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, GlobalMaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K

from keras.utils.np_utils import to_categorical  
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings("ignore")

import random

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %% colab={"base_uri": "https://localhost:8080/"} id="peP2nm91hndX" outputId="35b3af61-3fd8-4e82-d848-614e87a875e3"
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown] id="a3313f55"
# ## 1. Import and Understand the data

# %% [markdown] id="fb411bd2"
# ### 1A. Import and read ‘images.npy’.

# %% id="ff8fc39f"
# Set the path to the dataset folder
path = "/content/drive/MyDrive/MGL/Project-CV-2/images.npy"

# Read npy file using numpy library .load method
data = np.load(path, allow_pickle=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="sdApxPo7UpI_" outputId="a55322ca-ab0e-440e-bb1a-98b75e8969ed"
# The data consists of 2 columns (Images, Masks)
data.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="VueDFz84Uo_r" outputId="a5dd79f6-98ca-4aaa-fb8e-9ec9e021e8ef"
# The shape of an image is accessed by image.shape. It returns a tuple of the number of rows, columns, and channels (if the image is colored).
data[0][0].shape

# %% colab={"base_uri": "https://localhost:8080/"} id="a6rkf4utUo3s" outputId="340bdf6b-d7e4-4acf-8863-806d48b683f2"
# We can see that the images are stored in the form of an array; Each element represents the pixel value of the image.
data[0][0]

# %% colab={"base_uri": "https://localhost:8080/"} id="0iQn726oUovK" outputId="bb8c207e-2777-426e-85ff-28c29084c7a1"
# The second column consists of information about label. It contains the size of the image (height and width), label, notes 
# which could be the description of the image, and points which contains the x and y coordinates of faces in the image.
data[0][1]

# %% colab={"base_uri": "https://localhost:8080/", "height": 350} id="S-3xX_24Uom2" outputId="a8e6558e-ea28-4df7-fb7f-67f33d7b0475"
# Sample Image
cv2_imshow(data[0][0])

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="BOhc2KGzVLW6" outputId="c75a4453-795f-45a5-f1d4-0ea12e9923ec"
# Let's check size of some sample images to determine if the images are of same size.
fig, ax = plt.subplots(10,3,figsize=(20,30))
row = 0
col = 0
index = 0
for i in range(30):
  ax[row][col].imshow(data[index][0], interpolation='nearest')
  index = index + 12
  col = col + 1
  if col > 2:
    row = row + 1
    col = 0
plt.show()

# %% [markdown] id="a0dc04f4"
# ### 1B. Split the data into Features(X) & labels(Y). Unify shape of all the images.
# - Imp Note: Replace all the pixels within masked area with 1.
# - Hint: X will comprise of array of image whereas Y will comprise of coordinates of the mask(human face). Observe: data[0], data[0][0], data[0][1].

# %% [markdown] id="ZZkUKVQK6p-o"
# **Image Preprocessing:**
#
# It is important to preprocess the images before we use them to train the model. We will resize all the images to equal width and height as 224. We'll create the features and target to train the model. The features are the images and the target is the coordinates of faces. In the next step, we will save all the preprocessed images in 'X' array of height and width as 224. We will first initialize the X array with zeroes. The (x,y) coordinates will be saved in 'masks' array. Following tasks would be performed:
#
# - Resize the images to equal width and height
# - Convert grayscale images (if any) to colored
# - Use preprocess_input module from tensorflow.keras.applications.mobilenet library

# %% colab={"base_uri": "https://localhost:8080/"} id="7545a1df" outputId="17178415-d079-4abf-d660-982a7d3de8fa"
from tensorflow.keras.applications.mobilenet import preprocess_input

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

HEIGHT_CELLS = 28
WIDTH_CELLS = 28

IMAGE_SIZE = 224

masks = np.zeros((int(data.shape[0]), IMAGE_HEIGHT, IMAGE_WIDTH))
X = np.zeros((int(data.shape[0]), IMAGE_HEIGHT, IMAGE_WIDTH, 3))

for index in range(data.shape[0]):
  img = data[index][0]
  img = cv2.resize(img, dsize=(IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
  """Assign all pixels in the first 3 channels only to the image, i.e., discard the alpha channel. 
  The alpha channel is a special channel that handles transparency. When an image has an alpha channel on it, 
  it means you can adjust the image's opacity levels and make bits translucent or totally see-through. 
  The alpha channel is instrumental when you want to remove the background from an image."""
  try:
    img = img[:,:,:3]
  except:
    print(f"Exception {index} Grayscale images with shape {img.shape}")
    # convert the grayscale image to color so that the number of channels are standardized to 3
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    continue
  X[index] = preprocess_input(np.array(img, dtype=np.float32))
  # Loop through the face co-ordinates and create mask out of it.
  for i in data[index][1]:
    x1 = int(i['points'][0]['x'] * IMAGE_WIDTH)
    x2 = int(i['points'][1]['x'] * IMAGE_WIDTH)
    y1 = int(i['points'][0]['y'] * IMAGE_HEIGHT)
    y2 = int(i['points'][1]['y'] * IMAGE_HEIGHT)
    # set all pixels within the mask co-ordinates to 1.
    masks[index][y1:y2, x1:x2] = 1
print(f"Shape of X is '{X.shape}' and the shape of mask is '{masks.shape}' ")

# %% [markdown] id="97580215"
# ### 1C. Split the data into train and test[400:9].

# %% colab={"base_uri": "https://localhost:8080/"} id="d7368177" outputId="54bb946a-3650-40f6-d54d-946e3e8e1159"
# We are not using the validation set here; The train set will be used to train the face detection model, the validation set will be used for evaluation during model training, 
# and the test set is used to evaluate the detected faces on the trained model.

X_train, X_test, y_train, y_test = train_test_split(X, masks, test_size=9, train_size=400, random_state=42)

print(f"Shape of X_train is '{X_train.shape}' and the shape of y_train is '{y_train.shape}'")
print(f"Shape of X_test is '{X_test.shape}' and the shape of y_test is '{y_test.shape}'")

# %% [markdown] id="827d518c"
# ### 1D. Select random image from the train data and display original image and masked image.

# %% colab={"base_uri": "https://localhost:8080/", "height": 528} id="c5613019" outputId="502bfc64-a26c-4bca-ee44-3e3d196f2e62"
fig = plt.figure(figsize=(15, 15))
a=1
b=5
c = 1
d = 1
for i in range(5):
  plt.subplot(a, b, c)
  # Show training images
  plt.imshow(X_train[i])  
  c = c + 1
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(15, 15))
for i in range(5):
   plt.subplot(a, b, d)
   # Show corresponding Mask
   plt.imshow(y_train[i])
   d = d + 1
plt.tight_layout()
plt.show()

# %% [markdown] id="5579308b"
# ## 2. Model building

# %% [markdown] id="8c19ed0e"
# ### 2A. Design a face mask detection model.
# - Hint: 1. Use MobileNet architecture for initial pre-trained non-trainable layers.
# - Hint: 2. Add appropriate Upsampling layers to imitate U-net architecture.

# %% [markdown] id="D0342Ho4CnM4"
# MobileNetV2 Model: Here, we will use transfer learning on the pre-trained MobileNetV2 model. Transfer learning is simply the process of using a pre-trained model that has been trained on a dataset for training and predicting on a new given dataset. MobileNet V2 model was developed at Google, pre-trained on the ImageNet dataset with 1.4M images and 1000 classes of web images. We will use this as our base model to train with our dataset to detect faces.
#
# Initializing the base model: The base model is the model that is pre-trained. We will create a base model using MobileNet V2. We will also initialize the base model with a matching input size as to the pre-processed image data we have which is 160×160. The base model will have the same weights from imagenet. We will exclude the top layers of the pre-trained model by specifying include_top=False which is ideal for feature extraction.

# %% id="4c72efef"
IMAGE_SIZE = 224
EPOCHS = 30 # Save time, Change later
BATCH = 8
LR = 1e-4


# %% id="EkkkWulsA-8I"
def model():
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")
    
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    f = [16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    
    model = Model(inputs, x)
    return model


# %% colab={"base_uri": "https://localhost:8080/"} id="BNcwHuhIDKou" outputId="7098aa4e-e202-4286-e995-bc28becd1679"
model = model()
model.summary()

# %% [markdown] id="eee0ed1e"
# ### 2B. Design your own Dice Coefficient and Loss function.

# %% [markdown] id="Xx3H-11fGOnx"
# **Dice Coefficient (F1 Score):**
#
# Simply put, the Dice Coefficient is 2 * the Area of Overlap divided by the total number of pixels in both images.
#
# The Dice coefficient is very similar to the IoU. They are positively correlated, meaning if one says model A is better than model B at segmenting an image, then the other will say the same. Like the IoU, they both range from 0 to 1, with 1 signifying the greatest similarity between predicted and truth.

# %% id="7a014062"
smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# %% id="ZbQ0YKFODMkT"
# Intersection-Over-Union (IoU, Jaccard Index)
# The Intersection-Over-Union (IoU), also known as the Jaccard Index, is one of the most commonly used metrics in semantic segmentation… 
# and for good reason. The IoU is a very straightforward metric that’s extremely effective.

# def iou_coef(y_true, y_pred, smooth=1):
#   intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
#   union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
#   iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
#   return iou

# %% [markdown] id="895e5f33"
# ### 2C. Train and tune the model as required.

# %% id="afbe6f2f"
opt = tf.keras.optimizers.Nadam(LR)
metrics = [dice_coef, Recall(), Precision()]
model.compile(loss=dice_loss, optimizer=opt, metrics=metrics)

# %% id="nZfu3uKxFMfj"
# The goal of a training is to minimize the loss. With this, the metric to be monitored would be 'loss', 
# and mode would be 'min'. A model.fit() training loop will check at end of every epoch whether the loss is no longer decreasing, 
# considering the min_delta and patience if applicable. Once it's found no longer decreasing, model.stop_training is marked True 
# and the training terminates.

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
]

# %% colab={"base_uri": "https://localhost:8080/"} id="xd8DaE_-FMPo" outputId="b936c7f9-5499-4c5a-8847-2732d35101f4"
train_steps = len(X_train)//BATCH
valid_steps = len(X_test)//BATCH

if len(X_train) % BATCH != 0:
    train_steps += 1
if len(X_test) % BATCH != 0:
    valid_steps += 1

history1 = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    callbacks=callbacks
)

# %% [markdown] id="JBd6-24LEk-g"
# The model is trained and the trained model has reduced the loss from 0.7120 to 0.3441, dice coefficient is improved to 0.6559, Recall is almost 94% and Precision is almost 78%.

# %% [markdown] id="944ee94e"
# ### 2D. Evaluate and share insights on performance of the model.

# %% colab={"base_uri": "https://localhost:8080/"} id="ab659401" outputId="6ca58788-7f91-4f04-8bfc-5df7b3f40a58"
# We are done with training the model. Now, we have to see how well the model has learnt the parameters 
# and how efficiently it can detect the faces.

# Let's test the model on test images and check the accuracy of the model.

test_steps = (len(X_test)//BATCH)
if len(X_test) % BATCH != 0:
    test_steps += 1

model.evaluate(X_test, y_test, steps=test_steps)

# %% [markdown] id="Xvhhci-dFXjc"
# Evaluation metrics allow us to estimate errors to determine how well our models are performing:
#
# > Accuracy: ratio of correct predictions over total predictions.
#
# > Precision: how often the classifier is correct when it predicts positive.
#
# > Recall: how often the classifier is correct for all positive instances.
#
# > F-Score: single measurement to combine precision and recall.
#
# > Dice Coefficient: Higher is better, Ranges from 0 to 1.

# %% [markdown] id="k2KZdDWNltYh"
# - As evident, we can see the Dice Coef. = 0.5364, Precision = 0.7480, and Recall = 0.6487. 
# - Model is not overfitting, we can use more complex models to improve the performance.
# - A decent performance is visible from the above metrics.

# %% colab={"base_uri": "https://localhost:8080/"} id="jrO4aqgxXu72" outputId="cde79fe8-8bd2-46e0-e71e-8c2aee8293ab"
# Predicting the model on test data
y_pred=model.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 312} id="NrChdy7FYONP" outputId="7c3b94aa-262f-483e-c995-81a0ac51e978"
# Capturing learning history per epoch
hist = pd.DataFrame(history1.history)
hist['epoch'] = history1.epoch

# Plotting validation loss at different epochs
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.legend(("train" , "valid") , loc = 0)
plt.title("Epochs Vs Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# %% colab={"base_uri": "https://localhost:8080/", "height": 312} id="XfqWosegflez" outputId="870920ef-0a55-49de-f823-a5f367d69125"
# Capturing learning history per epoch
hist = pd.DataFrame(history1.history)
hist['epoch'] = history1.epoch

# Plotting Dice Coef. at different epochs
plt.plot(hist['dice_coef'])
plt.plot(hist['val_dice_coef'])
plt.legend(("train" , "valid") , loc = 0)
plt.title("Epochs Vs Dice Coef.")
plt.xlabel("Epochs")
plt.ylabel("Dice Coef.")

# %% [markdown] id="c891804a"
# ## 3. Test the model predictions on the test image: ‘image with index 3 in the test data’ and visualise the predicted masks on the faces in the image.

# %% colab={"base_uri": "https://localhost:8080/"} id="eicONwJPK22L" outputId="abf9371f-9e39-4656-92a1-9701c2097578"
# Image with index 3
X_test[3]

# %% colab={"base_uri": "https://localhost:8080/"} id="pRrPJThHkYWw" outputId="1f1564b5-a72a-454e-a2ba-f646788454f2"
y_test[3]

# %% colab={"base_uri": "https://localhost:8080/"} id="3kZ_PHNckYNZ" outputId="b6fa6a56-9f4e-4396-d99e-7cd5f225e7b2"
# Predicted mask for the image
y_pred = model.predict(np.array([X_test[3]]))
y_pred

# %% id="iNV50-nfkYDf"
pred_mask = cv2.resize((1.0*(y_pred[0] > 0.5)), (IMAGE_WIDTH, IMAGE_HEIGHT))

# %% colab={"base_uri": "https://localhost:8080/", "height": 427} id="aBpXaog4I6Fn" outputId="8a4ab135-2a81-4365-a670-2ce4ba861210"
# Comparing the Original Image with predicted and real mask

# Original Image
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 3, 1)
imgplot = plt.imshow(X_test[3])
ax.set_title('Original Image')
# Predicted Mask
ax = fig.add_subplot(1, 3, 2)
imgplot = plt.imshow(pred_mask)
ax.set_title('Predicted Mask')
# Real Mask
ax = fig.add_subplot(1, 3, 3)
imgplot = plt.imshow(y_test[3])
ax.set_title('Real Mask')

# %% [markdown] id="OEgCHMtV0yu7"
# From the above graph; The model is able to detect one face in the image correctly.

# %% [markdown] id="5f36f6dd"
# # Part-B: Solution

# %% [markdown] id="dedd939b"
# Project description is same as in Part-A

# %% colab={"base_uri": "https://localhost:8080/"} id="Rq45JKHXDwOS" outputId="cce0cbd7-6ae7-49c9-e9c7-ce8c4f64fba4"
# Import all the relevant libraries required to complete the analysis, visualization, modeling and presentation
import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
sns.set_style('darkgrid')
# %matplotlib inline

import cv2
from google.colab.patches import cv2_imshow
from glob import glob
import itertools

import warnings
warnings.filterwarnings("ignore")

import sklearn, re, random

import matplotlib.gridspec as gridspec
from tqdm.notebook import tqdm

from zipfile import ZipFile

# Set random_state
random_state = 42

# Print versions
print(f'Pandas version: {pd.__version__}')
print(f'Numpy version: {np.__version__}')
print(f'CV version: {cv2.__version__}')

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %% colab={"base_uri": "https://localhost:8080/"} id="2vTZmr84EHL3" outputId="449b4a5f-bb6b-4b5f-e4ba-ebdb45c71c17"
# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="lg1eHt-AEHFx" outputId="c97048a7-b2ab-4c2e-9e57-b30a70fa525c"
# Current working directory
# %cd "/content/drive/MyDrive/MGL/Project-CV-2/"

# %% colab={"base_uri": "https://localhost:8080/"} id="E8Rmb398EUtS" outputId="4eb13774-2b7c-4f0c-b05c-799b0926ddc9"
# List files in the directory
# !ls

# %% [markdown] id="a3313f55"
# ## 1. Read/import images from folder ‘training_images’.

# %% id="kJyFjFLhEUq6"
# Path of the data file
path = '/content/drive/MyDrive/MGL/Project-CV-2/training_images-20211126T092819Z-001.zip'

# %% id="Ii8JpUiqEUoT"
# # Unzip files in the current directory

# with ZipFile (path,'r') as z:
#   z.extractall() 
# print("Training zip extraction done!")

# %% id="dFtIQYT3GgNN"
path = '/content/drive/MyDrive/MGL/Project-CV-2/training_images'

# %% colab={"base_uri": "https://localhost:8080/"} id="593a0486" outputId="6eafe119-248c-4efa-c82a-4023c1c6cf62"
# Creating an array of images from the training_images folder
data = []

for filename in os.listdir(path):
    if filename.endswith("jpg"): 
        print(filename)
        data.append(filename)

# Just another method
# # import the modules
# from os import listdir
 
# # get the path or directory
# folder_dir = "/content/drive/MyDrive/MGL/Project-CV-2/training_images"
# for images in os.listdir(folder_dir):
 
#     # check if the image ends with png or jpg or jpeg
#     if (images.endswith(".png") or images.endswith(".jpg")\
#         or images.endswith(".jpeg")):
#         # display
#         print(images)

# %% colab={"base_uri": "https://localhost:8080/"} id="3V0ZSZEWG7ol" outputId="03098bf3-1337-4cf3-a3e6-22f27995b00e"
# Total number of images
len(data)

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="fPKOE27DV143" outputId="1cc4d41b-9b7c-4943-8364-2515334014fc"
data[0]

# %% colab={"base_uri": "https://localhost:8080/"} id="gGnmCEgF3LkH" outputId="9d1f01db-1e93-48c2-df38-b5bed8958d71"
data[0:2]

# %% [markdown] id="5579308b"
# ## 2. Write a loop which will iterate through all the images in the ‘training_images’ folder and detect the faces present on all the images.
# - Hint: You can use ’haarcascade_frontalface_default.xml’ from internet to detect faces which is available open source.

# %% colab={"base_uri": "https://localhost:8080/", "height": 340} id="N-P5GfuRarvH" outputId="307bb821-f4fa-4fc3-d711-7119aeb7efc5"
# Loading the sample image
# Selecting an image from the dataset
test = 'training_images/real_00767.jpg'

# Using the cv2_imshow
from google.colab.patches import cv2_imshow
img1 = cv2.imread(test, cv2.IMREAD_COLOR)
# cv2_imshow(img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Using the imshow
plt.figure(figsize=(10, 5))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(img1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 340} id="x5mdeQfwark4" outputId="fbefff5e-b1ec-40ce-8d51-57ce56d022f7"
img2 = cv2.imread(test, cv2.IMREAD_GRAYSCALE)
# cv2_imshow(img2)

# Using the imshow
plt.figure(figsize=(10, 5))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img2)

# %% id="6f76d102"
# Loading Haar Cascade in OpenCV
# We can load any haar-cascade XML file using cv2.CascadeClassifier function.

face_detector=cv2.CascadeClassifier('/content/drive/MyDrive/MGL/Project-CV-2/haarcascade_frontalface_default.xml')
# eye_dectector = cv2.CascadeClassifier('haarcascade_eye.xml')

# results = face_detector.detectMultiScale(gray_img, scaleFactor=1.05,minNeighbors=5,minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
# Once cascade is loaded in OpenCV, we can call the detector function.
# results: It lists coordinates (x, y, w, h) of bounding boxes around the detected object.

# %% [markdown] id="NSQbGhyzi-fS"
# Parameters in detectMultiScale
#
# * scaleFactor – This tells how much the object’s size is reduced in each image.
# * minNeighbors – This parameter tells how many neighbours each rectangle candidate should consider.
# * minSize — This signifies the minimum possible size of an object to be detected. An object smaller than minSize would be ignored.
#
# Note : For detection, we must pass a gray_image , scaleFactor, and minNeighbors. Other parameters are optional.

# %% colab={"base_uri": "https://localhost:8080/", "height": 617} id="HWjUEhChm5oS" outputId="c42c5981-7a41-4734-f8c7-0bc578e9f7fa"
# Detecting the face on sample image

img = cv2.imread(test)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(gray, 1.05, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2_imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% colab={"base_uri": "https://localhost:8080/"} id="1fqLzsmuQKnx" outputId="82611036-5d6e-46fd-f875-a655c4101d35"
faces

# %% [markdown] id="Y3YbelNNA8n9"
# Looping through all the images in the ‘training_images’ folder and detect the faces present on all the images.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="FJWGgEPF2Xcf" outputId="acadafd4-0a95-4c38-c5e7-0709ca9395c6"
# %cd '/content/drive/MyDrive/MGL/Project-CV-2/training_images/'

# Lists to create the metadata for the images
X = []
Y = []
W = []
H = []
Total_Faces = []
Image_Name = []

# Referencing the above code for looping
for filename in data[0:1091]:
    img = plt.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.05, 5)
    if faces is ():
      print("No faces found")
    else:
      print(len(faces))
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
        f_name = filename
        X.append(faces[0][0])
        Y.append(faces[0][1])
        W.append(faces[0][2])
        H.append(faces[0][3])
        Total_Faces.append(len(faces))
        Image_Name.append(f_name)
    plt.imshow(img)
    plt.title(f_name)
    plt.show()

# Dictionary of lists 
dict = {'X': X, 'Y': Y, 'W': W, 'H': H, 'Total_Faces': Total_Faces, 'Image_Name': Image_Name} 
# Dataframe from above dictionary
df = pd.DataFrame(dict)

# %% [markdown] id="c891804a"
# ## 3. From the same loop above, extract metadata of the faces and write into a DataFrame.

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="UpT7GkcvHN7n" outputId="a5286027-1c6e-4c94-d03b-5173b9c6f0e8"
df_sorted = df.sort_values('Image_Name')
df_sorted

# %% [markdown] id="d08ee5d9"
# ## 4. Save the output Dataframe in .csv format.

# %% colab={"base_uri": "https://localhost:8080/"} id="138a5926" outputId="c11b4ea6-2f70-4ead-849f-1d92422b9bec"
df_sorted.to_csv(r'/content/drive/MyDrive/MGL/Project-CV-2/training_images.csv', header=True)

print(df_sorted)

# %% [markdown] id="0dd7c2e4"
# # Part-C: Solution

# %% [markdown] id="e624389a"
# - **DOMAIN:** Face Recognition
# - **CONTEXT:** Company X intends to build a face identification model to recognise human faces.
# - **DATA DESCRIPTION:** The dataset comprises of images and its mask where there is a human face.
# - **PROJECT OBJECTIVE:** Face Aligned Face Dataset from Pinterest. This dataset contains 10,770 images for 100 people. All images are taken from 'Pinterest' and aligned using dlib library. Some data samples: Check the sample image in project.

# %% colab={"base_uri": "https://localhost:8080/"} id="26856bb6" outputId="80809f5d-539e-4861-93d1-c6caf7f702ad"
# Import all the relevant libraries required to complete the analysis, visualization, modeling and presentation
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

from sklearn.decomposition import PCA
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

import cv2
from google.colab.patches import cv2_imshow
from glob import glob
import itertools

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, GlobalMaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K

from keras.utils.np_utils import to_categorical  
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings("ignore")

import sklearn, re, random

import matplotlib.gridspec as gridspec
from tqdm.notebook import tqdm

from zipfile import ZipFile

# Set random_state
random_state = 42

# Print versions
print(f'Pandas version: {pd.__version__}')
print(f'Numpy version: {np.__version__}')
print(f'Scikit-learn version: {sklearn.__version__}')
print(f'Tensorflow version: {tf.__version__}')
print(f'CV version: {cv2.__version__}')

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %% colab={"base_uri": "https://localhost:8080/"} id="qM2-0wcYgGXW" outputId="7800ff32-77f7-4324-8de3-33ee1c5b1e2f"
# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')

# %% [markdown] id="f0150bbf"
# ## 1. Unzip, read and Load data(‘PINS.zip’) into session.

# %% colab={"base_uri": "https://localhost:8080/"} id="N3IBdGFwNe6Z" outputId="8a63d364-95b3-42c9-93c2-dd6ab71133d3"
# Current working directory
# %cd "/content/drive/MyDrive/MGL/Project-CV-2/"

# %% colab={"base_uri": "https://localhost:8080/"} id="jsW7ordrNABf" outputId="deed9ef5-85e8-4379-e7d9-a506bb303923"
# List files in the directory
# !ls

# %% id="SQpbWohfKUuh"
# Path of the PINS data file
path = '/content/drive/MyDrive/MGL/Project-CV-2/PINS.zip'


# %% id="O3nm9VLO7ZqO"
# # Unzip files in the current directory

# with ZipFile (path,'r') as z:
#   z.extractall() 
# print("Training zip extraction done!")

# %% [markdown] id="5d281847"
# ## 2. Write function to create metadata of the image.

# %% [markdown] id="de7c9412"
# ## 3. Write a loop to iterate through each and every image and create metadata for all the images.

# %% [markdown] id="raZl2-iqk7BL"
# I am considering both the above requirements (2 and 3) in the below code:

# %% id="pv0Vrnu5CwnG"
# Define a function to load the images from the PINS folder and map each image with person id/URL

class IdentityMetadata():
    def __init__(self, base, name, file):
        # print(base, name, file)
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    
def load_metadata(path):
    metadata = []
    exts = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
                exts.append(ext)
    return np.array(metadata), exts

metadata, exts = load_metadata('PINS')
labels = np.array([meta.name for meta in metadata])

# %% colab={"base_uri": "https://localhost:8080/"} id="5eO5-LpKIPmL" outputId="d0a7edb3-60fc-4fe5-8129-cb187fa1526f"
metadata

# %% colab={"base_uri": "https://localhost:8080/"} id="_qWsWsLAIQpp" outputId="12944708-7929-47ea-a706-081340a4e980"
metadata.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="dUuTnUrSNxS7" outputId="a9b261d4-460a-4c8d-d426-42015e83e059"
labels


# %% id="lYRk5HI5IQgm"
# Define a function to load image from the metadata

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]


# %% colab={"base_uri": "https://localhost:8080/", "height": 286} id="wZKYs3sqSNq4" outputId="1924a872-1abc-42ee-eedc-2ebaca835305"
# Load one image using the function "load_image"
sample_image =load_image('/content/drive/MyDrive/MGL/Project-CV-2/PINS/pins_Amanda Crew/Amanda Crew2.jpg')

plt.imshow(sample_image)

# %% colab={"base_uri": "https://localhost:8080/", "height": 460} id="4tOzmDVoOLeJ" outputId="6c051474-07b9-4ba2-faa8-e41076f2b127"
# Random Imaage generator from the dataset
n = np.random.randint(1, len(metadata))
img_path = metadata[n].image_path()
img = load_image(img_path)

fig = plt.figure(figsize = (15, 7.2))
ax = fig.add_subplot(1, 1, 1)
title = labels[n].split('_')[1]
ax.set_title(title, fontsize = 20)
_ = plt.imshow(img)


# %% [markdown] id="69cf2f8e"
# ## 4. Generate Embeddings vectors on the each face in the dataset.
# - Hint: Use ‘vgg_face_weights.h5’

# %% [markdown] id="yauGGwmQn5iJ"
# ### 4.1 VGG Face model

# %% [markdown] id="avxCslo5MYnd"
# The VGGFace refers to a series of models developed for face recognition and demonstrated on benchmark computer vision datasets by members of the Visual Geometry Group (VGG) at the University of Oxford.
#
# A contribution of the paper was a description of how to develop a very large training dataset, required to train modern-convolutional-neural-network-based face recognition systems, to compete with the large datasets used to train models at Facebook and Google.
#
# This dataset is then used as the basis for developing deep CNNs for face recognition tasks such as face identification and verification. Specifically, models are trained on the very large dataset, then evaluated on benchmark face recognition datasets, demonstrating that the model is effective at generating generalized features from faces.
#
# They describe the process of training a face classifier first that uses a softmax activation function in the output layer to classify faces as people. This layer is then removed so that the output of the network is a vector feature representation of the face, called a face embedding. The model is then further trained, via fine-tuning, in order that the Euclidean distance between vectors generated for the same identity are made smaller and the vectors generated for different identities is made larger. This is achieved using a triplet loss function.
#
# [VGG Face Model: Deep Face Recognition with Keras](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/)

# %% id="SoOacX_KlUyT"
def vgg_face():	
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape = (224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation = 'relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2, 2), strides = (2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation = 'relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2, 2), strides = (2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation = 'relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation = 'relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2, 2), strides = (2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation = 'relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation = 'relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2, 2), strides =(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation = 'relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation = 'relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Convolution2D(4096, (7, 7), activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model


# %% [markdown] id="iuSI6E3ZoUH1"
# ### 4.2 Load the Model

# %% colab={"base_uri": "https://localhost:8080/"} id="5asUu487lUnh" outputId="c95be2b1-c6ff-44fb-b5c5-83f5b95860de"
# Load the model defined above
# Then load the given weights file named "vgg_face_weights.h5"

model = vgg_face()
model.load_weights('vgg_face_weights.h5')
print(model.summary())

# %% colab={"base_uri": "https://localhost:8080/"} id="EyTYCnUFlUiJ" outputId="da5838c5-4208-4015-ae9d-2a48996af8b2"
# Let's check first and second last layer of the model to understand the model structure
model.layers[0], model.layers[-2]

# %% [markdown] id="PGqOxVACojw3"
# ### 4.3 Use the vgg_face_descriptor 

# %% id="0ZPJcVd4lUej"
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

# %% colab={"base_uri": "https://localhost:8080/"} id="1_QfHjTZlUVp" outputId="f01d1073-b3c1-4a7a-efe7-ee9c6e70b565"
type(vgg_face_descriptor)

# %% colab={"base_uri": "https://localhost:8080/"} id="ZKbiXyYvo3IP" outputId="4c30fea5-85a4-4ef3-fadb-bf0e6a670978"
vgg_face_descriptor.inputs, vgg_face_descriptor.outputs

# %% [markdown] id="SRlSh4Cbrb3U"
# ### 4.4 Generate embeddings for each image in the dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="dVkGsw_xo2_S" outputId="4cf47f00-e615-44ad-bbca-96b30396b578"
# Get embedding vector for first image in the metadata using the pre-trained model

img_path = metadata[0].image_path()
img = load_image(img_path)

# Normalising pixel values from [0-255] to [0-1]: scale RGB values to interval [0, 1]
img = (img / 255.).astype(np.float32)

img = cv2.resize(img, dsize = (224, 224))
print(img.shape)

# Obtain embedding vector for an image
# Get the embedding vector for the above image using vgg_face_descriptor model and print the shape
embedding_vector = vgg_face_descriptor.predict(np.expand_dims(img, axis = 0))[0]
print(embedding_vector.shape)

# %% [markdown] id="QIzbvJM4Os62"
# **Generate embeddings for all the images:**
#
# - Write code to iterate through metadata and create embeddings for each image using vgg_face_descriptor.predict() and store in a list with name embeddings
#
# - If there is any error in reading any image in the dataset, fill the emebdding vector of that image with 2622-zeroes as the final embedding from the model is of length 2622.

# %% id="NQKMjpAFsM56"
# To save time; embeddings are saved in the drive first

# embeddings = []
# embeddings = np.zeros((metadata.shape[0], 2622))
# for i, meta in tqdm(enumerate(metadata)):
#   try:
#     image = load_image(str(meta))
#     image = (image/255.).astype(np.float32)
#     image = cv2.resize(image, (224, 224))
#     embeddings[i] = vgg_face_descriptor.predict(np.expand_dims(image, axis = 0))[0]
#   except:
#     embeddings[i] = np.zeros(2622)

# %% id="Mvx_Emj3VDjZ"
# Save the embeddings
# np.save('/content/drive/MyDrive/MGL/Project-CV-2/embeddings.npy', embeddings)

# %% id="G7NHgFYyVc2i"
# Load the embeddings
embeddings = np.load('/content/drive/MyDrive/MGL/Project-CV-2/embeddings.npy')

# %% colab={"base_uri": "https://localhost:8080/"} id="NzFOwMgWUXk6" outputId="65e6949a-3324-464a-867c-38af87f1fd00"
print("Shape of Embedding Vector: ",embeddings.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="fVLaXmpFUXd9" outputId="2b53cac7-2711-4d23-fa20-d34cd972b75c"
embeddings[0], embeddings[988], embeddings[988].shape

# %% colab={"base_uri": "https://localhost:8080/"} id="AUQxMiadUXVU" outputId="b1cb80ab-9847-4521-fc1b-ede1afbe04fe"
embeddings


# %% [markdown] id="178c3a6d"
# ## 5. Build distance metrics for identifying the distance between two similar and dissimilar images.

# %% [markdown] id="BIbHsIvNpnMD"
# **Vector Similarity:** We’ve represented input images as vectors. We will decide both pictures are same person or not based on comparing these vector representations. Now, we need to find the distance of these vectors. There are two common ways to find the distance of two vectors: cosine distance and euclidean distance. Cosine distance is equal to 1 minus cosine similarity. No matter which measurement we adapt, they all serve for finding similarities between vectors.
# ```
# # Cosine Distance
# def findCosineDistance(source_representation, test_representation):
# a = np.matmul(np.transpose(source_representation), test_representation)
# b = np.sum(np.multiply(source_representation, source_representation))
# c = np.sum(np.multiply(test_representation, test_representation))
# return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
# ```
# ```
# # Euclidean Distance
# def findEuclideanDistance(source_representation, test_representation):
# euclidean_distance = source_representation - test_representation
# euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
# euclidean_distance = np.sqrt(euclidean_distance)
# return euclidean_distance
# ```

# %% id="hyROACOMnrHb"
# Function to calculate distance between given 2 pairs of images
def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


# %% id="k6ZcXGj8nq8t"
# Plot images and get distance between the pair of images
def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance between {idx1} & {idx2}= {distance(embeddings[idx1], embeddings[idx2]):}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()));  


# %% colab={"base_uri": "https://localhost:8080/", "height": 881} id="neCujk5Hnqwh" outputId="008049bf-fbf4-4922-812e-40eb08b39fc3"
# Distance between two random similar and dissimilar images
show_pair(500, 501)
show_pair(500, 1600)

show_pair(1100, 1101)
show_pair(1100, 1300)

# %% [markdown] id="b86a336f"
# ## 6. Use PCA for dimensionality reduction.

# %% [markdown] id="zCtmyV2qnb7Z"
# ### 6.1 Create train and test sets
# - Create X_train, X_test and y_train, y_test
# - Use train_idx to seperate out training features and labels
# - Use test_idx to seperate out testing features and lab

# %% id="1F3rScywnPo8"
# Every 9th example goes in test data and rest go in train data
train_idx = np.arange(metadata.shape[0]) % 9 != 0     
test_idx = np.arange(metadata.shape[0]) % 9 == 0

# one half as train examples of 10 identities
X_train = embeddings[train_idx]
# another half as test examples of 10 identities
X_test = embeddings[test_idx]

targets = np.array([m.name for m in metadata])

#train labels
y_train = targets[train_idx]
#test labels
y_test = targets[test_idx]

# %% colab={"base_uri": "https://localhost:8080/"} id="f8JTimdtnPhc" outputId="ef260bf9-2fe4-4fa8-95bd-ecc20bb3e866"
print('X_train shape : ({0},{1})'.format(X_train.shape[0], X_train.shape[1]))
print('y_train shape : ({0},)'.format(y_train.shape[0]))
print('X_test shape : ({0},{1})'.format(X_test.shape[0], X_test.shape[1]))
print('y_test shape : ({0},)'.format(y_test.shape[0]))

# %% colab={"base_uri": "https://localhost:8080/"} id="mBGmno0jo3xX" outputId="f37f66ef-bed8-45ad-e974-e899ab642e25"
X_train

# %% colab={"base_uri": "https://localhost:8080/"} id="FEuv0H-Zo-Ec" outputId="195bcbc0-e1dc-40f9-f8f7-70b53b98c582"
y_train

# %% [markdown] id="6cfMbK5-onPl"
# ### 6.2 Encode the Labels

# %% id="p6Uplzd-nPeZ"
# Encode the labels
en = LabelEncoder()
y_train = en.fit_transform(y_train)
y_test = en.transform(y_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="gi0bBwlEnPbG" outputId="bc984c26-c958-446b-dcb1-4b526df9e75d"
y_train

# %% [markdown] id="Io3kmvemp9xN"
# ### 6.3 Standardize the feature values

# %% id="aIx2sz9DqC39"
# Standarize the features
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# %% [markdown] id="pv1g5dlWqXna"
# ### 6.4 Reduce dimensions using PCA

# %% colab={"base_uri": "https://localhost:8080/"} id="NxIWF8aVqcbp" outputId="01b650f7-393e-4679-bb24-92f13714fd1c"
# Covariance matrix
cov_matrix = np.cov(X_train_sc.T)

# Eigen values and vector
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# Cumulative variance explained
tot = sum(eig_vals)
var_exp = [(i /tot) * 100 for i in sorted(eig_vals, reverse = True)]
cum_var_exp = np.cumsum(var_exp)

print('Cumulative Variance Explained', cum_var_exp)

# %% colab={"base_uri": "https://localhost:8080/"} id="HE_IiHOnqrAF" outputId="35606db7-a463-4bec-eb67-e518d58ef80f"
# Get index where cumulative variance explained is > threshold
thres = 95
res = list(filter(lambda i: i > thres, cum_var_exp))[0]
index = (cum_var_exp.tolist().index(res))
print(f'Index of element just greater than {thres}: {str(index)}')

# %% colab={"base_uri": "https://localhost:8080/", "height": 585} id="ospPI14fqzuv" outputId="ba60c213-b4bc-4a35-b51a-b517d3460c55"
# Ploting 
plt.figure(figsize = (16, 8))
plt.bar(range(1, eig_vals.size + 1), var_exp, alpha = 0.5, align = 'center', label = 'Individual explained variance')
plt.step(range(1, eig_vals.size + 1), cum_var_exp, where = 'mid', label = 'Cumulative explained variance')
plt.axhline(y = thres, color = 'r', linestyle = '--')
plt.axvline(x = index, color = 'r', linestyle = '--')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 52} id="WPL450u4r5qM" outputId="d97f1192-a64d-4279-f020-b4d4e5bfcb92"
# Reducing the dimensions
pca = PCA(n_components = index, random_state = random_state, svd_solver = 'full', whiten = True)
pca.fit(X_train_sc)
X_train_pca = pca.transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)
display(X_train_pca.shape, X_test_pca.shape)

# %% [markdown] id="51a6cb22"
# ## 7. Build an SVM classifier in order to map each image to its right person.

# %% id="fYt8i2RoxvpO"
# Casual Hyperparameter tuning

# params_grid = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000], 'class_weight': ['balanced', None]}]

# svc = RandomizedSearchCV(SVC(random_state = random_state), params_grid, cv = 3, scoring = 'f1_macro', verbose = 50, n_jobs = -1)
# svc.fit(X_train_pca, y_train)

# print('Best estimator found by grid search:')
# print(svc.best_estimator_)

# Best estimator found by grid search:
# SVC(C = 10, gamma = 0.0001, kernel = 'rbf', class_weight = 'balanced')

# %% colab={"base_uri": "https://localhost:8080/"} id="dqa1s5Tsxvg6" outputId="07018499-5305-47f3-c84d-26833760c50f"
svc_pca = SVC(C = 10, gamma = 0.0001, kernel = 'rbf', class_weight = 'balanced', random_state = random_state)
svc_pca.fit(X_train_pca, y_train)
print('SVC accuracy for train set: {0:.3f}'.format(svc_pca.score(X_train_pca, y_train)))

# %% colab={"base_uri": "https://localhost:8080/"} id="1fLfcBcjxvX5" outputId="293a6fe9-b123-446e-a28a-db9af3b8d016"
# Predict
y_pred = svc_pca.predict(X_test_pca)

# Accuracy Score
print('Accuracy Score: {}'.format(accuracy_score(y_test, y_pred).round(3)))

# %% colab={"base_uri": "https://localhost:8080/"} id="8TcB7BC-xvOt" outputId="52d13be1-4d0b-4691-fbe9-fe7b5de2c694"
names = [name.split('_')[1].title().strip() for name in labels]

# Classification Report
print('Classification Report: \n{}'.format(classification_report(y_test, y_pred, target_names = np.unique(names))))

# %% [markdown] id="55b56cd9"
# ## 8. Import and display the the test images.
# - Hint: ‘Benedict Cumberbatch9.jpg’ and ‘Dwayne Johnson4.jpg’ are the test images.

# %% id="Ygj8Ur8xl3BW"
# Loading the images from the directory
Test_image_1=load_image('/content/drive/MyDrive/MGL/Project-CV-2/Dwayne Johnson4.jpg')
Test_image_2=load_image('/content/drive/MyDrive/MGL/Project-CV-2/Benedict Cumberbatch9.jpg')

# %% colab={"base_uri": "https://localhost:8080/", "height": 335} id="W4cxlSQS4zNo" outputId="9117367a-f8ed-4575-8b53-346d284bf3b0"
# Display the test images
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(Test_image_1)
ax.set_title('Dwayne Joshson4.jpg')

ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(Test_image_2)
ax.set_title('Benedict Cumberbatch9.jpg')

# %% [markdown] id="3749446e"
# ## 9. Use the trained SVM model to predict the face on both test images.

# %% colab={"base_uri": "https://localhost:8080/"} id="mM7VHnKql23I" outputId="fea7cae1-319a-4ac6-9143-68ceb3f1aea3"
Test_image_1 = (Test_image_1 / 255.).astype(np.float32)
Test_image_1 = cv2.resize(Test_image_1, dsize = (224,224))
Test_image_2 = (Test_image_2 / 255.).astype(np.float32)
Test_image_2 = cv2.resize(Test_image_2, dsize = (224,224))
print(Test_image_1.shape)
print(Test_image_2.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="dyM2yysgl2nU" outputId="5769dacd-89ef-425c-a647-1dd860dc44c8"
embeddings1 = np.zeros((1, 2622))
embeddings2 = np.zeros((1, 2622))

embedding_vector1 = vgg_face_descriptor.predict(np.expand_dims(Test_image_1, axis=0))[0]
embeddings1[0]= embedding_vector1
embedding_vector2 = vgg_face_descriptor.predict(np.expand_dims(Test_image_2, axis=0))[0]
embeddings2[0]= embedding_vector2
print(embedding_vector1.shape)
print(embedding_vector2.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="73HrdFTzl2d9" outputId="bbdb7efe-a3e3-41f4-f2c8-bdd2e5a52745"
first_image = embeddings1
second_image = embeddings2

X_test_1 = sc.transform(first_image)
X_test_2 = sc.transform(second_image)

X_test_pca_1 = pca.transform(X_test_1)
X_test_pca_2 = pca.transform(X_test_2)

X_test_pca_1.shape
X_test_pca_2.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 281} id="ycF8GCw8l2UF" outputId="2b0a14d6-0801-4f94-b948-1b88d0ae7fe7"
y_predict_1 = svc_pca.predict(X_test_pca_1)
y_predict_encoded_1 = en.inverse_transform(y_predict_1)

plt.imshow(Test_image_1)
plt.title(f'Identified as {y_predict_encoded_1}');

# %% colab={"base_uri": "https://localhost:8080/", "height": 281} id="TVN1wdTSovPq" outputId="db621383-ad7b-4010-d750-13c9b998cc33"
y_predict_2 = svc_pca.predict(X_test_pca_2)
y_predict_encoded_2 = en.inverse_transform(y_predict_2)

plt.imshow(Test_image_2)
plt.title(f'Identified as {y_predict_encoded_2}');


# %% [markdown] id="I7HxFYB2pv1x"
# ## 10. Testing the results from another approach

# %% id="CAtd_CN0pvGD"
# Identify the 10th image from the test set

def sample_img_plot(sample_idx):
  # Load image for sample_idx from test data
  sample_img = load_image(metadata[test_idx][sample_idx].image_path())
  # Get actual name
  actual_name = metadata[test_idx][sample_idx].name.split('_')[-1].title().strip()
  # Normalizing pixel values
  sample_img = (sample_img/255.).astype(np.float32)
  # Resize
  sample_img = cv2.resize(sample_img, (224, 224))

  # Obtain embedding vector for sample image
  embedding = vgg_face_descriptor.predict(np.expand_dims(sample_img, axis = 0))[0]
  # Scaled the vector and reshape
  embedding_scaled = sc.transform(embedding.reshape(1, -1))
  # Predict
  sample_pred = svc_pca.predict(pca.transform(embedding_scaled))
  # Transform back
  pred_name = en.inverse_transform(sample_pred)[0].split('_')[-1].title().strip()
  return sample_img, actual_name, pred_name


# %% colab={"base_uri": "https://localhost:8080/", "height": 470} id="ezsG7lGYqXMo" outputId="139365c0-d605-4632-f92e-ff346e6b19c5"
# Plot for 10th image in test set
sample_img, actual_name, pred_name = sample_img_plot(10)
fig = plt.figure(figsize = (15, 7.2))
plt.axis('off')
plt.imshow(sample_img)
plt.title(f"A: {actual_name} \n P: {pred_name}", color = 'green' if actual_name == pred_name else 'red')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Y7AcLQeaqa0w" outputId="90934469-f902-4b59-d4f6-1f36d15690be"
# Random 20 sample images from test set
plt.figure(figsize = (15, 15))
gs1 = gridspec.GridSpec(5, 4)
gs1.update(wspace = 0, hspace = 0.3) 

for i in range(20):
    ax1 = plt.subplot(gs1[i])
    plt.axis('on')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    
    sample_img, actual_name, pred_name = sample_img_plot(random.randint(1, 1197))
  
    plt.axis('off')
    plt.imshow(sample_img)
  
    plt.title(f"A: {actual_name} \n P: {pred_name}", color = 'green' if actual_name == pred_name else 'red')
plt.show()

# %% [markdown] id="3XQ4lhIysYRt"
# ## Conclusion:
# Project objective here was to recognize (aligned) faces from a dataset containing 10k+ images for 100 people using a pre-trained model on Face Recognition:
# * VGG Face model with pre-trained weights was used to generate embeddings for each image in the dataset.
# * Euclidean and consine distance was used to caclulate the similarity.
# * Because of large dimensions, PCA was used for dimension reduction after standardizing the features.
# * With an cumulative explained variance of 95%, 347 PCs were used.
# * Using SVM we predicted the labels for test dataset with an accuracy of more than 96%.
# * Finally, we compared the predicted and actual labels for given images as well as for 20 random images from the test dataset.

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
