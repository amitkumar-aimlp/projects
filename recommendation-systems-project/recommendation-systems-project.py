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
# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#00b3e5;">Recommendation Systems Project

# %% [markdown]
# # Solution

# %% [markdown]
# - **DOMAIN:** Smartphone, Electronics
# - **CONTEXT:** India is the second largest market globally for smartphones after China. About 134 million smartphones were sold across India in the year 2017 and is estimated to increase to about 442 million in 2022. India ranked second in the average time spent on mobile web by smartphone users across Asia Pacific. The combination of very high sales volumes and the average smartphone consumer behaviour has made India a very attractive market for foreign vendors. As per Consumer behaviour, 97% of consumers turn to a search engine when they are buying a product vs. 15% who turn to social media. If a seller succeeds to publish smartphones based on user’s behaviour/choice at the right place, there are 90% chances that user will enquire for the same. This Case Study is targeted to build a recommendation system based on individual consumer’s behaviour or choice.
# - **DATA DESCRIPTION:** 
#  - author : name of the person who gave the rating
#  - country : country the person who gave the rating belongs to
#  - data : date of the rating
#  - domain: website from which the rating was taken from
#  - extract: rating content
#  - language: language in which the rating was given
#  - product: name of the product/mobile phone for which the rating was given
#  - score: average rating for the phone
#  - score_max: highest rating given for the phone
#  - source: source from where the rating was taken
# - **PROJECT OBJECTIVE:** We will build a recommendation system using popularity based and collaborative filtering methods to recommend mobile phones to a user which are most popular and personalised respectively.

# %%
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

from surprise import Dataset,Reader
from surprise import NormalPredictor

from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import SVD, accuracy

from surprise import KNNWithMeans
from surprise import Prediction

import warnings
warnings.filterwarnings("ignore")

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %% [markdown]
# ## 1. Import the necessary libraries and read the provided CSVs as a data frame and perform the below steps.

# %%
# CSV File 1
dfa1=pd.read_csv('Dataset/phone_user_review_file_1.csv',encoding='latin-1')

# %%
dfa1.info()
dfa1.head()

# %%
# CSV File 2
dfa2=pd.read_csv('Dataset/phone_user_review_file_2.csv',encoding='latin-1')

# %%
dfa2.info()
dfa2.head()

# %%
# CSV File 3
dfa3=pd.read_csv('Dataset/phone_user_review_file_3.csv',encoding='latin-1')

# %%
dfa3.info()
dfa3.head()

# %%
# CSV File 4
dfa4=pd.read_csv('Dataset/phone_user_review_file_4.csv',encoding='latin-1')

# %%
dfa4.info()
dfa4.head()

# %%
# CSV File 5
dfa5=pd.read_csv('Dataset/phone_user_review_file_5.csv',encoding='latin-1')

# %%
dfa5.info()
dfa5.head()

# %%
# CSV File 6
dfa6=pd.read_csv('Dataset/phone_user_review_file_6.csv',encoding='latin-1')

# %%
dfa6.info()
dfa6.head()

# %% [markdown]
# ### 1A. Merge all the provided CSVs into one DataFrame.

# %%
# CSV File
dfa_list = [dfa1, dfa2, dfa3, dfa4, dfa5, dfa6]
dfa = pd.concat(dfa_list)

# %%
dfa.info()
dfa.head()

# %% [markdown]
# ### 1B. Explore, understand the Data and share at least 2 observations.

# %%
dfa.describe().T

# %% [markdown]
# **Observations:**
#
# - Feature score:
#  - Mean and median are not equal. Skewness is expected.
#  - Distribution is not normal because of big SD.
#  - 75 % of values are less than 10, and maximum value is 10.
# - Feature score_max:
#  - Mean and median are equal. Distribution seems to be normal.
#  - Range of values is small and constant.

# %% [markdown]
# ### 1C. Round off scores to the nearest integers.

# %%
dfa['score'] = round(dfa['score'])

# %%
dfa.info()
dfa.head()

# %% [markdown]
# ### 1D. Check for missing values. Impute the missing values, if any.

# %%
dfa.isna().sum()

# %%
# Replace missing values with mean
dfa['score_max'] = dfa['score_max'].fillna(dfa['score_max'].mean())

# %%
# Replace missing values with median
dfa['score'] = dfa['score'].fillna(dfa['score'].median())

# %%
# Drop the missing values
dfa.dropna(subset=['product'],inplace=True)

# %%
# Dataset after removing the missing values
dfa.isna().sum()

# %% [markdown]
# ### 1E. Check for duplicate values and remove them, if any.

# %%
duplicates = dfa[dfa.duplicated(subset=['phone_url','date','lang','country','extract','author','product'])]
duplicates

# %%
dfa.duplicated(subset=['phone_url','date','lang','country','extract','author','product']).sum()

# %%
dfa.drop_duplicates(subset=['phone_url','date','lang','country','extract','author','product'],keep='first',inplace=True)
dfa.duplicated(subset=['phone_url','date','lang','country','extract','author','product']).sum()

# %% [markdown]
# ### 1F. Keep only 1 Million data samples. Use random state=612.

# %%
dfa = dfa.sample(n=1000000, random_state=612)
dfa.shape

# %%
dfa.head()

# %% [markdown]
# ### 1G. Drop irrelevant features. Keep features like Author, Product, and Score.

# %%
dfa_final = dfa[['author','product','score']]

# %%
dfa_final.info()
dfa_final.head()

# %%
dfa_final.shape

# %% [markdown]
# ## 2. Answer the following questions.

# %% [markdown]
# ### 2A. Identify the most rated products.

# %%
dfa_final.groupby('product')['score'].count().sort_values(ascending=False).head()  

# %% [markdown]
# ### 2B. Identify the users with most number of reviews.

# %%
dfa_final.groupby('author')['score'].count().sort_values(ascending=False).head() 

# %% [markdown]
# ### 2C. Select the data with products having more than 50 ratings and users who have given more than 50 ratings. Report the shape of the final dataset.

# %%
dfa_final_products = pd.DataFrame(dfa_final.groupby('product').count())
filter1 = dfa_final_products['author'] > 50
filter2 = dfa_final_products['score'] > 50
dfa_final_products.where(filter1 & filter2, inplace=False).dropna().shape

# %%
dfa_final_products.where(filter1 & filter2, inplace=False).dropna().head(20)

# %% [markdown]
# ### 2D. Report your findings and inferences.

# %%
sns.countplot(dfa_final['score'])

# %% [markdown]
# **Observations:**
# * Most common scores are 8 and 10. Most of the users provided rating on higher end
# * Some higher scores can be seen for certain scores like 2, 4, 6, 8, 10. 
# * Note the pattern of even numbers for higher ratings.
# * Collaborative Filtering Model can be used to provide recommendations to the users.

# %% [markdown]
# ## 3. Build a popularity based model and recommend top 5 mobile phones.

# %%
ratings_mean_count = pd.DataFrame(dfa_final.groupby('product')['score'].mean()) 

# %%
ratings_mean_count['rating_counts'] = pd.DataFrame(dfa_final.groupby('product')['score'].count())  

# %%
ratings_mean_count.head()

# %%
ratings_mean_count['top 5 mobile phones'] = ratings_mean_count['rating_counts'] * ratings_mean_count['score']

# %%
# Top 5 mobile phones recommendation
ratings_mean_count.sort_values(by='top 5 mobile phones', ascending=False).head(5)

# %% [markdown]
# ## 4. Build a collaborative filtering model using SVD. You can use SVD from surprise or build it from scratch(Note: Incase you’re building it from scratch you can limit your data points to 5000 samples if you face memory issues). Build a collaborative filtering model using kNNWithMeans from surprise. You can try both user-based and item-based model.

# %%
reader = Reader(rating_scale=(1, 10))

# %%
dfa_final.shape

# %%
# Selecting a sample to manage the memory issues
dfa_final = dfa_final.sample(n=5000,random_state=0)

# %%
data = Dataset.load_from_df(dfa_final[['author','product','score']], reader = reader)

# %%
data.df

# %% [markdown]
# ### First Using SVD

# %%
# Using SVD
svd_model = SVD()

X, y = train_test_split(data, test_size=0.25, random_state=0)

# %%
svd_model.fit(X)

y_pred = svd_model.test(y)
accuracy.rmse(y_pred)

# %%
y_pred

# %%
test_pred_df = pd.DataFrame([[x.uid, x.iid, x.est] for x in y_pred])

# %%
test_pred_df.head()

# %% [markdown]
# ### Build a collaborative filtering model using kNNWithMeans from surprise.

# %%
# User Based Model
knn_model_user = KNNWithMeans(k=50, sim_options={'name': 'cosine', 'user_based': True})

knn_model_user.fit(X)

# %%
# Item Based Model
knn_model_item = KNNWithMeans(k=50, sim_options={'name': 'cosine', 'user_based': False})

knn_model_item.fit(X)

# %% [markdown]
# ## 5. Evaluate the collaborative model. Print RMSE value.

# %%
y_pred_user = knn_model_user.test(y)
y_pred_item = knn_model_item.test(y)

# %%
print("User-based Model : Accuracy RMSE)")
accuracy.rmse(y_pred_user)

# %%
print("Item-based Model : Accuracy RMSE)")
accuracy.rmse(y_pred_item)

# %%
# User based model test prediction output
test_pred_user_df = pd.DataFrame([[x.uid, x.iid, x.est] for x in y_pred_user])
test_pred_user_df.head(5)

# %%
# Item based model test prediction output
test_pred_item_df = pd.DataFrame([[x.uid, x.iid, x.est] for x in y_pred_item])
test_pred_item_df.head(5)

# %% [markdown]
# ## 6. Predict score (average rating) for test users.

# %%
# Predict score for one sample test user
knn_model_user.predict(uid="F Vossen",iid="Samsung Galaxy Note 3 zwart / 32 GB - Overzicht")

# %%
# Predicted value for all test users
y_pred_user

# %% [markdown]
# ## 7. Report your findings and inferences.

# %%
# Please refer the 2D above for details

# %% [markdown]
# ## 8. Try and recommend top 5 products for test users.

# %%
df_pred = pd.DataFrame(y_pred_user)


# %%
# Function to return recommended products based on user and number of recommendations, n
def recommendations(user, n):
    recommended_products = df_pred[df_pred['uid'] == user][['uid','iid','est']].sort_values('est', ascending=False).head(n)
    return recommended_products


# %%
# Recommend top 5 products
recommendations('Amazon Customer',5)

# %%
recommendations('Cliente Amazon',5)

# %%
recommendations('e-bit',5)

# %% [markdown]
# ## 9. Try other techniques (Example: cross validation) to get better results.

# %%
# Detect outliers and impute them as required.

# %%
dfa_final.boxplot()

# %%
Q1 = np.percentile(dfa_final['score'], 25, interpolation = 'midpoint') 
Q3 = np.percentile(dfa_final['score'], 75, interpolation = 'midpoint') 
Q1, Q3

# %%
IQR = Q3 - Q1
max_range = Q3 + (IQR * 1.5)
min_range = Q1 - (IQR * 1.5)
min_range, max_range

# %%
dfa_final[dfa_final['score'] > 13]

# %%
dfa_final[dfa_final['score'] < 5]

# %%
dfa_final_new = dfa_final.copy()
dfa_final_new.loc[dfa_final_new['score'] < 5.0, 'score'] = 5.0

# %%
dfa_final_new.boxplot()

# %%
dfa_final_new = dfa_final_new.sample(n=5000,random_state=42)

# %%
data_new = Dataset.load_from_df(dfa_final_new[['author','product','score']], reader = reader)

# %%
svd_model_new = SVD ()

X_train, y_test = train_test_split(data_new,test_size=0.25,random_state=42)

# %%
svd_model_new.fit(X_train)

# %%
y_pred_new = svd_model_new.test(y_test)
accuracy.rmse(y_pred_new)

# %%
# Using cross validation techniques

# %%
cross_validate(svd_model_new, data_new, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=1)

# %% [markdown]
# ## 10. In what business scenario you should use popularity based Recommendation Systems ?

# %% [markdown]
# It is a type of recommendation system which works on the principle of popularity and/or anything which is in trend. These systems check about the product or movie which are in trend or are most popular among the users and directly recommend those.
#
# - It does not suffer from cold start problems which means on day 1 of the business we can recommend products to various users based on different filters.
#
# - There is no need for the user's historical data.
#
# - Recommendations are not personalized. The system would recommend the same sort of products/movies which are based upon popularity to all users.
#
# Examples: Google News-News filtered by trending and most popular news; YouTube-Trending videos suggestions.

# %% [markdown]
# ## 11. In what business scenario you should use CF based Recommendation Systems ?

# %% [markdown]
# This is considered as one of the very smart recommendation systems that works on the similarity between different users and also items that are widely used as in e-commerce websites and online movie websites. It examines about the taste of similar users and does recommendations. 
#
# The similarity is not restricted to the taste of the users only there can be considerations of similarity between different items also. The system will give more efficient recommendations if we have a large volume of information about users and items.
#
# Main categories: User-based CF(neighborhood), Item-based CF(neighborhood), Latent matrix Factorization (SVD)
#
# Use cases: eCommerce Websites, music, new connection recommendations from Amazon, Last.fm, Spotify, LinkedIn, and Twitter

# %% [markdown]
# ## 12. What other possible methods can you think of which can further improve the recommendation for different users ?

# %% [markdown]
# **Other possible methods:**
# - Content Based Recommendation systems
# - Classification Model based
# - Association Rule Mining
#
# **Hybrid Approaches:** 
# Each recommendation engine has a central idea and solves the problem of predicting the unseen user-item rating through a unique approach. Each recommendation model has its own strengths and limitations, that is, each works better in specific data setting than others. Some recommenders are robust at handling the cold start problem, have model biases, and tend to overfit the training dataset. As in the case of the ensemble classifiers, hybrid recommenders combine the model output of multiple base recommenders into one hybrid recommender. 
#
# As long as the base models are independent, this approach limits the generalization error, improves the performance of the recommender, and overcomes the limitation of a single recommendation technique. The diversity of the base models can be imparted by selecting different modeling techniques like neighborhood BMF, content-based and supervised
# model-based recommenders.

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
# 17. [Surprise Library](https://surpriselib.com/)
# 18. [Surprise Documentation](https://surprise.readthedocs.io/en/stable/)
