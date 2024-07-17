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
# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#00b3e5;">Contents: Applied Statistics Project
# 1. [Part-A: Solution](#Part-A:-Solution)
# 2. [Part-B: Solution](#Part-B:-Solution)
# 3. [Part-C: Solution](#Part-C:-Solution)

# %% [markdown]
# # Part-A: Solution

# %%
# Import all the libraries needed to complete the analysis, visualization, modeling and presentation
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.patches as mpatches

# %% [markdown]
# ## 1. Please refer the table below to answer below questions:
# ![Q1.png](attachment:Q1.png)

# %% [markdown]
# **The Background:**
#
# - Contingency tables classify outcomes in rows and columns. Table cells at the intersections of rows and columns indicate frequencies of both events coinciding.
#
# - Consequently, to calculate joint probabilities in a contingency table, take each cell count and divide by the grand total.
#
# - In contingency tables, you can locate the marginal probabilities in the row and column totals. Statisticians refer to them as marginal probabilities because you find them in the margins of contingency tables!
#
# - Fortunately, using contingency tables to calculate conditional probabilities is straightforward. It’s merely a matter of dividing a cell value by a row or column total.
#
# Probability of A given B is written as P(A | B).
#
# $P(A\mid B) = \frac{P(A \quad and \quad B)}{P(B)}$
#
# where P(A and B) = Joint probability of A and B
#
# P(A) = Marginal probability of A
# P(B) = Marginal probability of B

# %% [markdown]
# ### 1A. Refer above table and find the joint probability of the people who planned to purchase and actually placed an order.

# %%
# The joint probability of the people who planned to purchase and actually placed an order
P1 = 400/2000
P1

# %% [markdown]
# ### 1B. Refer to the above table and find the joint probability of the people who planned to purchase and actually placed an order, given that people planned to purchase.

# %%
# The joint probability of the people who planned to purchase and actually placed an order, given that
# people planned to purchase.
P2 = 400/500
P2

# %% [markdown]
# ## 2. An electrical manufacturing company conducts quality checks at specified periods on the products it manufactures. Historically, the failure rate for the manufactured item is 5%. Suppose a random sample of 10 manufactured items is selected. Answer the following questions.

# %% [markdown]
# - Binomial Distribution
#
#     It is widely used probability distribution of a discrete random variable. 
#
#     Plays major role in quality control and quality assurance function. 
#
#     $P(X = x\mid n,\pi)$ = $\frac{n!}{x!(n - x)!}\pi^x (1 - \pi)^{n-x} $
#     * where P(X = x) is the probability of getting x successes in n trials
#     and $\pi$ is the probability of an event of interest
#
#
#     Some important functions in Python for Binomial distribution:
#
#     1) Probability mass function
#     scipy.stats.binom.pmf gives the probability mass function for the binomial distribution
#     binomial = scipy.stats.binom.pmf (k,n,p), 
#     where k is an array and takes values in {0, 1,..., n}
#     n and p are shape parameters for the binomial distribution
#     The output, binomial, gives probability of binomial distribution function in terms of array.
#
#     2) Cumulative Density function
#     cumbinomial = scipy.stats.binom.cdf(k,n,p) gives cumulative binomial distribution.
#     The output, cumbinomial, gives cumulative probability of binomial distribution function in terms of array.
#
#     3) Plot the binomial Density function
#     The function, matplotlib.pyplot.plot(k, binomial, ‘o-’) gives us plot of the binomial distribution function.

# %%
# The failure rate for the manufactured item is 5%
p = 0.05
q = 1 - p
n = 10
k=np.arange(0,8)
k

# %% [markdown]
# ### 2A. Probability that none of the items are defective?

# %%
# Considering k = 0
# NoneDefective = P(0)
NoneDefective = stats.binom.pmf(0,n,p)
NoneDefective

# %% [markdown]
# ### 2B. Probability that exactly one of the items is defective

# %%
# Considering k = 1
# OneDefectivee = P(1)
OneDefective = stats.binom.pmf(1,n,p)
OneDefective

# %% [markdown]
# ### 2C. Probability that two or fewer of the items are defective?

# %%
#Cosidering k =2
TwoDefective = stats.binom.pmf(2,n,p)
TwoDefective

# %%
# Two or fewer of the manufactured items are defective = P(0) + P(1) + P(2)
TwoOrFewerDefective = NoneDefective + OneDefective + TwoDefective
TwoOrFewerDefective

# %% [markdown]
# ### 2D. Probability that three or more of the items are defective

# %%
# Considering the sum of probabilites as 1

ThreeDefective = stats.binom.pmf(3,n,p)

ThreeOrMoreDefective = 1 - (NoneDefective + OneDefective + TwoDefective + ThreeDefective)
ThreeOrMoreDefective

# %% [markdown]
# ## 3. A car salesman sells on an average 3 cars per week.

# %% [markdown]
# - Poissson Distribution
#
#     This discrete distribution which also plays a major role in quality control. 
#
#     The Poisson distribution is a discrete probability distribution for the counts of events that occur randomly in a given 
#     interval of time or space. In such areas of opportunity, there can be more than one occurrence. In such situations, Poisson 
#     distribution can be used to compute probabilities.
#
#     Examples include number of defects per item, number of defects per transformer produced. 
#     Notes: Poisson Distribution helps to predict the arrival rate in a waiting line situation where a queue is formed and people wait to be served and the service rate is generally higher than the arrival rate.
#
#
# - Properties:
#
#     Mean μ = λ
#
#     Standard deviation σ = √ μ
#
#     The Poisson distribution is the limit of binomial distribution as n approaches ∞and p approaches 0
#
#     P(X = x) = $\frac{e^\lambda \lambda^x}{x!} $
#     where 
#     * P(x)              = Probability of x successes given an idea of  $\lambda$
#     * $\lambda$ = Average number of successes
#     * e                   = 2.71828 (based on natural logarithm)
#     * x                    = successes per unit which can take values 0,1,2,3,... $\infty$
#
#     Applications
#
#     Car Accidents
#
#     Number of deaths by horse kicking in Prussian Army (first application)
#
#     Birth defects and genetic mutation
#
#     Note
#
#     If there is a fixed number of observations, n, each of which is classified as an event of interest or not an event of interest,
#     use the binomial distribution.
#
#     If there is an area of opportunity, use the Poisson distribution.

# %%
# Average rate is denoted with lambda in the formula for poisson distribution 

rate = 3

# %%
# Using a numpy array to save different no. of successes ranging from 0 to 19 to construct a probability distribution

n=np.arange(0,20)

# %%
# Calculating the distribution and storing the distribution of probablitites in an array

PoissonPmf = stats.poisson.pmf(n,rate)

# %% [markdown]
# ### 3A. What is Probability that in a given week he will sell some cars?

# %%
# Assuming the sum of probabilites as 1

SellSomeCars = 1 - PoissonPmf[0]
SellSomeCars

# %% [markdown]
# ### 3B. What is Probability that in a given week he will sell 2 or more but less than 5 cars?

# %%
SellTwoToFive = PoissonPmf[2]+PoissonPmf[3]+PoissonPmf[4]
SellTwoToFive

# %% [markdown]
# ### 3C. Plot the poisson distribution function for cumulative probability of cars sold per-week vs number of cars sold per week (Plotting both the CDF and PMF here.)

# %%
# Now we can create an array with Poisson cumulative probability values:

from scipy.stats import poisson

cdf = poisson.cdf(n, mu=3)
cdf = np.round(cdf, 3)

print(cdf)

# %%
# If you want to print it in a nicer way with each 'n' value and the corresponding cumulative probability:

for val, prob in zip(n,cdf):
    print(f"n-value {val} has probability = {prob}")

# %%
# Using matplotlib library, we can easily plot the Poisson CDF using Python:

plt.plot(n, cdf, marker='o')
plt.title('Poisson: $\lambda$ = %i ' % rate)
plt.xlabel('Number of Cars Sold per Week')
plt.ylabel('Cumulative Probability of Number of Cars Sold per Week')

plt.show()

# %%
# Printing the probability distribution for different values of x

PoissonPmf

# %%
# Using matplotlib library, we can easily plot the Poisson PMF using Python:

plt.plot(n,PoissonPmf,'o-')
plt.title('Poisson: $\lambda$ = %i ' % rate)
plt.xlabel('Number of Cars Sold per Week')
plt.ylabel('Probability of Number of Cars Sold per Week')
plt.show()

# %% [markdown]
# ## 4. Accuracy in understanding orders for a speech based bot at a restaurant is important for the Company X which has designed, marketed and launched the product for a contactless delivery due to the COVID-19 pandemic. Recognition accuracy that measures the percentage of orders that are taken correctly is 86.8%. Suppose that you place an order with the bot and two friends of yours independently place orders with the same bot. Answer the following questions.

# %% [markdown]
# Because there are three orders and the probability of a correct order is 0.868.
# Using Binomial distribution equation,
#
#         P(X = 3|n =3, pi given = 0.868)
#         3!/ 3!(3-3)! * (0.868)^3 * (1-0.868)^3-3 = 0.6540
#
#         Likewise, calculate X= 0, X=2
#          (X = 0) = 0.0023
#          (X = 2) = 0.2984
#
#         Hence, P(X>=2) = P(X=2)+P(X=3) = 0.9524
#
#  - The probability that all the three orders are filled correctly is 0.6540, 65.4% 
#  - The probability that none of the orders are filled correctly is 0.0023, 0.23% 
#  - The probability that atleat two of the three are filled correctly is 0.9524, 95.24%

# %%
# Since there are only 2 events orders filled correctly and not filled correctly
# and the number of trials is 3, we can use Binomial distribution with following parameters:

p1 = 0.868
q1 = 1 - p1
n1 = 3

# %% [markdown]
# ### 4A. What is the probability that all three orders will be recognised correctly?

# %%
ThreeOrdersRecognisedCorrect = stats.binom.pmf(3,3,q1)
ThreeOrdersRecognisedCorrect

# %% [markdown]
# ### 4B. What is the probability that none of the three orders will be recognised correctly?

# %%
NoneThreeOrdersRecognisedCorrect = stats.binom.pmf(0,3,p1)
NoneThreeOrdersRecognisedCorrect

# %% [markdown]
# ### 4C. What is the probability that at least two of the three orders will be recognised correctly?

# %%
# We need to find the (1 - Probability of upto 1 Failure)

TwoOfThreeRecognisedCorrect = 1 - stats.binom.pmf(1,3,p1)
TwoOfThreeRecognisedCorrect

# %% [markdown]
# ## 5. Explain 1 real life industry scenario (other than the ones mentioned above) where you can use the concepts learnt in this module of Applied Statistics to get data driven business solution.

# %% [markdown]
# **We can use statistics to evaluate potential new versions of a children’s dry cereal. We can use taste tests to provide valuable statistical information on what customers want from a product. The four key factors that product developers may consider to enhance the taste of the cereal are the following:**
#
# 1. Ratio of wheat to corn in the cereal flake
# 2. Type of sweetener: sugar, honey, artificial or sugar free
# 3. Presence or absence of flavour in the cereal - Fruits, Vegetables, Spices
# 4. Cooking time - Short or Long
#
# We should design an experiment to determine what effects these four factors had on cereal taste. For example, one test cereal can be made with a specified ratio of wheat to corn, sugar as the sweetener, flavour bits, and a short cooking time; another test cereal can be made with a different ratio of wheat to corn and the other three factors the same, and so on. Groups of children then taste-test the cereals and state what they think about the taste of each.
#
# The Analysis of variance (ANOVA) is the statistical method we can use to study the data obtained from the taste tests. The results of the analysis may show the following:
#
# - The flake composition and sweetener type were highly influential in taste evaluation.
# - The flavour bits actually reduced the taste of the cereal.
# - The cooking time had no effect on the taste.
#
# **This information can be vital to identify the factors that would lead to the best-tasting cereal. The same information can be used by the marketing and manufacturing teams, and for a better product development strategy.**
#
# **Tools to be  used:** Python, R, Minitab, Excel, MS SQL Server
#
# References:
# - [ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance)
# - [BURKE MARKETING SERVICES](https://www.burke.com/)
#

# %% [markdown]
# # Part-B: Solution

# %% [markdown]
# - **DOMAIN:** Sports
# - **CONTEXT:** Company X manages the men's top professional basketball division of the American league system.
# The dataset contains information on all the teams that have participated in all the past tournaments. It has data
# about how many baskets each team scored, conceded, how many times they came within the first 2 positions,
# how many tournaments they have qualified, their best position in the past, etc.
# - **DATA DESCRIPTION:** Basketball.csv - The data set contains information on all the teams so far participated in
# all the past tournaments.
# - **DATA DICTIONARY:**
#  1. `Team`: Team’s name
#  2. `Tournament`: Number of played tournaments.
#  3. `Score`: Team’s score so far.
#  4. `PlayedGames`: Games played by the team so far.
#  5. `WonGames`: Games won by the team so far.
#  6. `DrawnGames`: Games drawn by the team so far.
#  7. `LostGames`: Games lost by the team so far.
#  8. `BasketScored`: Basket scored by the team so far.
#  9. `BasketGiven`: Basket scored against the team so far.
#  10. `TournamentChampion`: How many times the team was a champion of the tournaments so far.
#  11. `Runner-up`: How many times the team was a runners-up of the tournaments so far.
#  12. `TeamLaunch`: Year the team was launched on professional basketball.
#  13. `HighestPositionHeld`: Highest position held by the team amongst all the tournaments played.
# - **PROJECT OBJECTIVE:** Company’s management wants to invest on proposal on managing some of the best
# teams in the league. The analytics department has been assigned with a task of creating a report on the
# performance shown by the teams. Some of the older teams are already in contract with competitors. Hence
# Company X wants to understand which teams they can approach which will be a deal win for them.
# ___

# %% [markdown]
# ## Step-1: Read, Clean, and Prepare Dataset to be used for EDA

# %% [markdown]
# ### Import the Relevant Libraries

# %%
# Import all the libraries needed to complete the analysis, visualization, modeling and presentation
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.patches as mpatches

# Configure for any default setting of any library
# %matplotlib inline
sns.set(style='darkgrid', palette='deep', font='sans-serif', font_scale=1.3, color_codes=True)

# %% [markdown]
# **Some Comments about the Libraries:**
# - **``%matplotlib inline``** sets the backend of matplotlib to the 'inline' backend: With this backend, the output of plotting commands is displayed inline without needing to call plt.show() every time a data is plotted.
# - Set few of the Seaborn's asthetic parameters

# %% [markdown]
# ### Read the Dataset

# %%
basketball=pd.read_csv("Basketball.csv");

# %% [markdown]
# ###  Shape of the Dataset

# %%
#There are 61 Observations / Rows and 13 Attributes / Columns.

basketball.shape

# %% [markdown]
# ### Check Information about the Data/Data types of all Attributes

# %%
basketball.info()

# %% [markdown]
# ### Data Cleaning

# %%
# Check the head of the dataset

basketball.head()

# %%
# Check the tail of the dataset

basketball.tail()

# %%
# Check for null values
# No Null values in data

basketball.isna().sum()

# %%
# Check for duplicates in data
# We do not have duplicates

dupes = basketball.duplicated()
sum(dupes)

# %%
# Generate descriptive statistics.

# Descriptive statistics include those that summarize the central
# tendency, dispersion and shape of a
# dataset's distribution, excluding ``NaN`` values.

# Analyzes both numeric and object series, as well
# as ``DataFrame`` column sets of mixed data types. The output
# will vary depending on what is provided.

basketball.describe(include="all")

# %% [markdown]
# **Some Insights:**
#
# - Most of the values are showing NaN issue, so from the above dataset insights (Head, Tail, Descriptive Statistics), there would be some missing values.
#
# - We can see difference in the input data for TeamLaunch, TournamentChampion, and Runner-up columns. So lets inspect data in these 3 columns.
#
# - Also lets make the column Names consistent: Runner-up > RunnerUp

# %%
# Rename the columns for consistency

basketball.rename(columns = {'Runner-up': 'RunnerUp'}, inplace = True)

# %%
# Check for unique values in various columns

print("TournamentChampion",basketball['TournamentChampion'].unique())
print("RunnerUp",basketball['RunnerUp'].unique())
print("TeamLaunch",basketball['TeamLaunch'].unique())
print("Team",basketball['Team'].unique())
print("Score",basketball['Score'].unique())

# %% [markdown]
# **Some Observations:**
# - Every row in dataset represents a team, so we have 61 teams to analyze. 
# - For easy data handling and visualization, we can rename the Team as T in the 'Team' Column.
# - There is a special character in data i.e '_'. We should eliminate/replace this missing value.
# - For TeamLaunch year, we can consider just the initial year for easy data cleaning and analysis.
# - Data types of most of the columns are Objects; As values are in integers, it would be beneficial to do the type conversion here.
# - Team 61 seems to have no valid data: It participated in 1 tournament and got 9th position. No information about games or baskets. Lets drop it.

# %%
# Make a copy of the original dataset
basketballOriginal=basketball.copy(deep=True); 

# For TeamLaunch year, we can consider just the initial year for easy data cleaning and analysis.
basketball['TeamLaunch'] = basketballOriginal['TeamLaunch'].str.slice(0,4);

# We can fill Zero in place of missing value '-'. 
# Because if HighestPositionHeld is not equal to 1 or 2 then TournamentChampion, Runner-up fields are zero too.
basketball=basketball.replace('-',0);

# For easy data handling and visualization, we can rename the Team as T in the 'Team' Column.
basketball['Team']=basketball['Team'].str.replace('Team ','T');
columns = basketball.columns.drop('Team');

# Data types of most of the columns are Objects; 
# As values are in integers, it would be beneficial to do the type conversion here.
basketball[columns] = basketball[columns].apply(pd.to_numeric, errors='coerce')

# Team 61 seems to have no valid data: It participated in 1 tournament and got 9th position. 
# No information about games or baskets. We can safely drop it.
basketball.drop(60,inplace=True)

# %% [markdown]
# ### Final Dataset after Data Cleaning

# %%
# Check the head of the dataset after cleaning

basketball.head(5)

# %%
# Check the tail of the dataset after cleaning

basketball.tail(5)

# %%
basketball.describe(include="all")

# %% [markdown]
# ## Step-2: Univariate Analysis
#
# Univariate analysis refer to the analysis of a single variable. The main purpose of univariate analysis is to summarize and find patterns in the data. The key point is that there is only one variable involved in the analysis.

# %% [markdown]
# ### Basic Statistics

# %%
# Print the mean of each attribute. Ignore "Team as it is not a continuous variable"

print("basketball:",basketball.mean())

# %%
# Print the median values of the basketball.
# Observe that the values of mean and median are not the same for most of the attributes.

print("basketball:",basketball.median())

# %%
# Prints the mode of the attribute. The Attribute is unimodal

print(basketball['HighestPositionHeld'].mode())

# %%
# Prints the value below which 25% of the data lies

print("Data_quantile(25%):",basketball.quantile(q=0.25))

# %%
# Prints the value below which 50% of the data lies

print("Data_quantile(50%):",basketball.quantile(q=0.50))

# %%
# Prints the value below which 75% of the data lies

print("Data_quantile(75%):",basketball.quantile(q=0.75))

# %%
# The below output represents the IQR values for all the attributes

basketball.quantile(0.75) - basketball.quantile(0.25)

# %%
# Range: The difference between the highest value and lowest values for all individual attributes
# columns = basketball.columns.drop('Team')

print(basketball[columns].max() - basketball[columns].min())

# %%
# The below output says how much was the data dispersion

print(basketball.var())

# %%
# The below output says how much the data deviated from the mean.

print(basketball.std())

# %%
# Understand the Skewness of the data
# Positively skewed: Most frequent values are low and tail is towards the high values.

basketball.skew()

# %%
# Understand the Kurtosis of the data

basketball.kurtosis()

# %% [markdown]
# ### Histogram for checking the Distribution, Skewness

# %%
# Check for distribution, skewness

# This function combines the matplotlib hist function (with automatic calculation of a 
# good default bin size) with the seaborn kdeplot() and rugplot() functions. It can 
# also fit scipy.stats distributions and plot the estimated PDF over the data.

select = ['Tournament','Score','PlayedGames','WonGames','DrawnGames','LostGames',
          'BasketScored','BasketGiven','TournamentChampion','RunnerUp','TeamLaunch',
          'HighestPositionHeld']
plt.figure(figsize=(20,20))
index = 1
for col in basketball[select]:
    plt.subplot(4,3,index)
    sns.distplot(basketball[col], rug=True, kde=True,
                 rug_kws={"color": "r"},
                 kde_kws={"color": "k"},
                 hist_kws={"color": "c"})
    index += 1

# %% [markdown]
# ### Box Plot to understand the Distribution

# %%
# Draw a box plot to show distributions with respect to categories.

# A box plot (or box-and-whisker plot) shows the distribution of quantitative
# data in a way that facilitates comparisons between variables or across
# levels of a categorical variable. The box shows the quartiles of the
# dataset while the whiskers extend to show the rest of the distribution,
# except for points that are determined to be "outliers" using a method
# that is a function of the inter-quartile range.

fig, ax = plt.subplots(3, 4)

fig.set_figheight(5)
fig.set_figwidth(15)

sns.boxplot(x=basketball['PlayedGames'],ax=ax[0][0]);
sns.boxplot(x=basketball['WonGames'],ax=ax[0][1]);
sns.boxplot(x=basketball['TournamentChampion'],ax=ax[0][2]);
sns.boxplot(x=basketball['RunnerUp'],ax=ax[0][3]);

sns.boxplot(x=basketball['TeamLaunch'],ax=ax[1][0]);
sns.boxplot(x=basketball['HighestPositionHeld'],ax=ax[1][1]);
sns.boxplot(x=basketball['BasketScored'],ax=ax[1][2]);
sns.boxplot(x=basketball['BasketGiven'],ax=ax[1][3]);

sns.boxplot(x=basketball['Tournament'],ax=ax[2][0]);
sns.boxplot(x=basketball['Score'],ax=ax[2][1]);
sns.boxplot(x=basketball['DrawnGames'],ax=ax[2][2]);
sns.boxplot(x=basketball['LostGames'],ax=ax[2][3]);

fig.tight_layout() 
plt.show()

# %% [markdown]
# ### Understand the complete Dataset Distribution

# %%
# Consider the distibution of all the quantitative variables

plt.figure(figsize=(10,5))

columns = basketball.columns.drop('Team');
basketballColumns = basketball[columns];
sns.distplot(basketballColumns);
plt.show()

# %%
# Cumulative distribution for all the quantitative variables

plt.figure(figsize=(10,5))
sns.distplot(basketballColumns, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True));
plt.show()

# %% [markdown]
# **Important Insights:**
# - <font color='red'> Positively skewed data: Most frequent values are low and tail is towards the high values.</font>
# - Similar type of Box plots for Score, BasketScored, and WonGames. Also we have some outliers here.
# - TournamentChampion and RunnerUP have a zero width box as 75% data has value as Zero. Remaining are outliers.
# - Most of the values are scattered between the median and Upper Quartile which again shows that the data is positively skewed.
# - No outliers can be seen in PlayedGames, TeamLaunch, HighestPositionHeld, BasketGiven, Tournament, DrawnGames, and LostGames variables.

# %% [markdown]
# ### Explore the 'Score' Variable

# %%
# The following code plots a histrogram using the matplotlib package.
# The bins argument creates class intervals. In this case we are creating 50 such intervals

# In the above histogram, the first array is the frequency in each class and the second array 
# contains the edges of the class intervals. These arrays can be assigned to a variable and 
# used for further analysis.

plt.hist(basketball['Score'], bins=50)

# %%
# Lets plot a frequency polygon superimposed on a histogram using the seaborn package.
# Seaborn automatically creates class intervals. The number of bins can also be manually set.

sns.distplot(basketball['Score']) 

# %%
# Lets add an argument to plot only the frequency polygon

sns.distplot(basketball['Score'], hist=False)

# %%
# Lets plot a violin plot using the seaborn package.

# This distribution can also be visualised in another manner. 
# For this we can use the violin plot function from seaborn. 
# The violin plot shows a vertical mirror image of the distribution 
# along with the original distribution.

sns.violinplot(basketball['Score']) 

# %%
# Now let us have a closer look at the distribution by plotting a simple histogram with 10 bins.

plt.figure(figsize=(20,10)) # makes the plot wider
plt.hist(basketball['Score'], color='g') # plots a simple histogram
plt.axvline(basketball['Score'].mean(), color='r', linewidth=2, label='Mean')
plt.axvline(basketball['Score'].median(), color='b', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(basketball['Score'].mode()[0], color='k', linestyle='dashed', linewidth=2, label='Mode')
plt.legend()

# %%
# Create boxplot for column="Score"

# Make a box plot from DataFrame columns.

# Make a box-and-whisker plot from DataFrame columns, optionally grouped
# by some other columns. A box plot is a method for graphically depicting
# groups of numerical data through their quartiles.
# The box extends from the Q1 to Q3 quartile values of the data,
# with a line at the median (Q2). The whiskers extend from the edges
# of box to show the range of the data. By default, they extend no more than
# `1.5 * IQR (IQR = Q3 - Q1)` from the edges of the box, ending at the farthest
# data point within that interval. Outliers are plotted as separate dots.

basketball.boxplot(column="Score",return_type='axes',figsize=(8,8))

# sns.boxplot(x=basketball['Score'])

# %%
# Histogram for checking the Skewness

plt.figure(figsize=(10,5))

#convert pandas DataFrame object to numpy array and sort
h = np.asarray(basketball['Score'])
h = sorted(h)
 
#use the scipy stats module to fit a normal distirbution with same mean and standard deviation
fit = stats.norm.pdf(h, np.mean(h), np.std(h)) 
 
#plot both series on the histogram
plt.plot(h,fit,'-',linewidth = 2,label="Normal distribution with same mean and var")
plt.hist(h,density=True,bins = 100,label="Actual distribution")      
plt.legend()
plt.show()

# %% [markdown]
# **Important Insights:**
# - <font color='red'> We can see from the above chart that Mean > Median > Mode. This implies a Positive or Right Skewed Distribution.</font>
# - From the above figure we can see that the mean is represented by the Red line and the mode by the Black line . The median is represented by the Blue line.
# - Most of the observations are within the first bin out of the 10 bins i.e. 35 Teams from the total 60. Most of the score is between 0 to 500.
# - There are few teams whose score is more than 1500.
# - We don't have any team whose score is between 3500 to 4000.
# - There are very few teams which got more than 4000 score. We can consider them as outliers.
#

# %% [markdown]
# ### Explore the 'TeamLaunch' Variable

# %%
#Flexibly plot a univariate distribution of observations.

sns.distplot(basketball.TeamLaunch)

# %%
# We can categorize variable Teamlaunch based on Quantiles

# Quantile-based discretization function.

# Discretize variable into equal-sized buckets based on rank or based
# on sample quantiles. For example 1000 values for 10 quantiles would
# produce a Categorical object indicating quantile membership for each data point.

basketball['TeamLaunchCategory'] = (pd.qcut(basketball['TeamLaunch'], 4, 
                                              labels=['Very Old', 'Old', 'New', 'Very New']));

# %%
# Show the counts of observations in each categorical bin using bars.

# A count plot can be thought of as a histogram across a categorical, instead
# of quantitative, variable. The basic API and options are identical to those
# for :func:`barplot`, so you can compare counts across nested variables.

plt.figure(figsize=(25,8))
sns.countplot(x='TeamLaunch',hue='TeamLaunchCategory',data = basketball)
plt.title("Team Launch count along with Category",size=15)
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# ## Step-3: Multivariate Analysis
# Multivariate analysis is performed to understand interactions between different fields in the dataset (or) finding interactions between more than 2 variables.
#
# Examples: Pairplot, 3D scatter plot Etc.

# %% [markdown]
# ### Covariance

# %%
# Prints the covariance of each attribute against every other attribute

basketball.cov()

# %% [markdown]
# ### Correlation

# %%
# Prints the correlation coefficient between every pair of attributes

basketball.corr()

# %% [markdown]
# ### Heatmap

# %%
# Plot rectangular data as a color-encoded matrix.

# This is an Axes-level function and will draw the heatmap into the
# currently-active Axes if none is provided to the ``ax`` argument.  Part of
# this Axes space will be taken and used to plot a colormap, unless ``cbar``
# is False or a separate Axes is provided to ``cbar_ax``.

plt.figure(figsize=(10,5))
corrmat = basketball.corr(method='pearson')
sns.heatmap(corrmat, cmap="YlGnBu", fmt='.2f',annot=True)
plt.show();

# %% [markdown]
# ### Scatterplot - All Variables

# %%
# In the following plot scatter diagrams are plotted for all the numerical columns in the dataset. 
# A scatter plot is a visual representation of the degree of correlation between any two columns. 
# The pair plot function in seaborn makes it very easy to generate joint scatter plots for all the 
# columns in the data.

plt.figure(figsize=(10,5))
pairplot=sns.pairplot(basketball);
plt.show()

# %% [markdown]
# ### Scatterplot - Selected Variables

# %%
# Plots the scatter plot using two variables

sns.scatterplot(data=basketball, x="Tournament", y="Score", size="Score")

# %%
# Another way of looking at multivariate scatter plot is to use the hue option
# in the scatterplot() function in seaborn.

# basketball['TeamLaunchCategory'] = (pd.qcut(basketball['TeamLaunch'], 4, 
#                                     labels=['Very Old', 'Old', 'New', 'Very New']));

sns.scatterplot(data=basketball, x="Tournament", y="Score", hue="TeamLaunchCategory")

# %% [markdown]
# **Important Insights:**
# - <font color='red'>In Sumarry, We have a Highly Correlated dataset.</font>
# - Score, WonGames, BasketScored correlation value is 1. So it's a perfect positive correlation.
# - PlayedGames and Tournament correlation value is 1. So again it's a perfect positive correlation.
# - Other than TeamLaunch and HighestPositionHeld remaining all fields are positively correlated.
# - TeamLaunch is negatively correlated; For old teams given values are more and for new teams it's low.
# - PlayedGames and Drawn Games correlation is 0.99, It means most of the games are Drawn compared to wonGames and LostGames.
# - TournamentChampion and RunnerUp are the counts. Here we have 30% to 70% correlation with other 
# attributes.

# %% [markdown]
# ### Explore Tournament vs TournamentChampion vs Runnerup

# %%
SortedDf=basketball.sort_values('Tournament', ascending=False);
 
plt.figure(figsize=(40,10))

# set height of bar
bars1 = SortedDf.Tournament
bars2 = SortedDf.TournamentChampion
bars3 = SortedDf.RunnerUp

 # set width of bar
barWidth = 0.3

# Set position of bar on X axis
r1 = np.arange(len(bars1)) + barWidth
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Make the plot
plt.bar(r1, bars1, color='b', width=barWidth, edgecolor='white', label='Tournament')
plt.bar(r2, bars2, color='g', width=barWidth, edgecolor='white', label='TournamentChampion')
plt.bar(r3, bars3, color='y', width=barWidth, edgecolor='white', label='RunnerUp')

# Add xticks on the middle of the group bars
plt.xlabel('Teams', fontweight='bold')
plt.xticks([r + (barWidth) for r in range(len(bars1))], SortedDf.Team)
plt.title("Tournament vs TournamentChampion vs Runnerup ",size=35);

# Create legend & Show graphic
plt.legend()
plt.show()

# %% [markdown]
# ### Explore PlayedGames vs WonGames Vs DrawnGames vs LostGames

# %%
SortedDf=basketball.sort_values('PlayedGames',ascending=False);
  
plt.figure(figsize=(40,10))

# set height of bar
bars1 = SortedDf.PlayedGames
bars2 = SortedDf.WonGames
bars3 = SortedDf.DrawnGames
bars4 = SortedDf.LostGames

 # set width of bar
barWidth =  0.3

# Set position of bar on X axis
r1 = np.arange(len(bars1))+barWidth
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Make the plot
plt.bar(r1, bars1, color='b', width=barWidth, edgecolor='white', label='PlayedGames')
plt.bar(r2, bars2, color='g', width=barWidth, edgecolor='white', label='WonGames')
plt.bar(r3, bars3, color='y', width=barWidth, edgecolor='white', label='DrawnGames')
plt.bar(r4, bars4, color='m', width=barWidth, edgecolor='white', label='LostGames')

# Add xticks on the middle of the group bars
plt.xlabel('Teams', fontweight='bold')
plt.xticks([r + (3*barWidth) for r in range(len(bars1))], SortedDf.Team)
plt.title("PlayedGames vs WonGames Vs DrawnGames vs LostGames",size=35)

# Create legend & Show graphic
plt.legend()
plt.show()

# %% [markdown]
# ### Explore BasketScored vs BasketGiven

# %%
SortedDf=basketball.sort_values('BasketScored',ascending=False);
  
plt.figure(figsize=(40,10))

# set height of bar
bars1 = SortedDf.BasketScored
bars2 = SortedDf.BasketGiven

 # set width of bar
barWidth =  0.3

# Set position of bar on X axis
r1 = np.arange(len(bars1))+barWidth
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Make the plot
plt.bar(r1, bars1, color='g', width=barWidth, edgecolor='white', label='BasketScored')
plt.bar(r2, bars2, color='r', width=barWidth, edgecolor='white', label='BasketGiven')

# Add xticks on the middle of the group bars
plt.xlabel('Teams', fontweight='bold')
plt.xticks([r + (barWidth) for r in range(len(bars1))], SortedDf.Team)
plt.title("BasketScored vs BasketGiven",size=35)

# Create legend & Show graphic
plt.legend()
plt.show()

# %% [markdown]
# ### Explore some other Charts

# %%
# No of matches won/lost by teams

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
bb = basketball[['Team','WonGames']].sort_values(by="WonGames", ascending=False)
plot = plt.pie(bb['WonGames'], labels=list(bb['Team'][:15]) + [str()] * (len(bb)-15), explode=[0.015*x for x in range(len(bb))])
plt.title('# matches won')
plt.subplot(1,2,2)
bb = basketball[['Team','LostGames']].sort_values(by="LostGames", ascending=False)
plot = plt.pie(bb['LostGames'], labels=list(bb['Team'][:20]) + [str()] * (len(bb)-20), explode=[0.015*x for x in range(len(bb))])
plt.title('# matches lost')

# %%
# No of baskets scored/given by teams

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
bb = basketball[['Team','BasketScored']].sort_values(by="BasketScored", ascending=False)
plot = plt.pie(bb['BasketScored'], labels=list(bb['Team'][:15]) + [str()] * (len(bb)-15))
plt.title('# Baskets scored')
plt.subplot(1,2,2)
bb = basketball[['Team','BasketGiven']].sort_values(by="BasketGiven", ascending=False)
plot = plt.pie(bb['BasketGiven'], labels=list(bb['Team'][:20]) + [str()] * (len(bb)-20))
plt.title('# Baskets given')

# %%
# The most RunnerUp
plt.figure(figsize=(20,8))
bb = basketball[['Team','RunnerUp']].sort_values(by="RunnerUp", ascending=False).where(lambda x: x["RunnerUp"] > 0)
sns.barplot(bb['Team'], bb['RunnerUp'])
plt.xticks(rotation=90)

# %% [markdown]
# **Important Insights:**
# - <font color='red'> Teams that have enjoyed more active participation are from T1 to T8. The performance of T7 was quite low in comparison to other teams.</font>
# - Many teams participated in tournaments, but could not secure the first two positions.
# - Top 3 high performing teams are T1, T2, and T5.
# - T11 participated in less tournaments but it showed good results compared to many other teams.
# - T20 and T21 have participated in less tournaments but they stood in second places in some games.
# - T1, T2 and T5 played the most tournament matches but T1 and T2 happens to be the highest average scorer.
# - T3 played less games,but have kept their position up in the leader-board in comparison to T4 and T5
# - T1 seems to be a team with matured experience and game spirit.
# - T2 also exhibited fine and profound game skills as it appeared the most in the leader-board and runner-up rankings.

# %% [markdown]
# ## Step-4: Bivariate Analysis  
# Through bivariate analysis we try to analyze two variables simultaneously. As opposed to univariate analysis where we check the characteristics of a single variable, in bivariate analysis we try to determine if there is any relationship between two variables.
#
# **There are essentially 3 major scenarios that we will come accross when we perform bivariate analysis:**
# 1. Both variables of interest are qualitative
# 2. One variable is qualitative and the other is quantitative
# 3. Both variables are quantitative

# %% [markdown]
# ### Explore Team vs PlayedGames with TeamLaunch

# %%
SortedDf=basketball.sort_values('PlayedGames',ascending=False);
fig, ax = plt.subplots(figsize=(40,10))   # setting the figure size of the plot
ax.scatter(SortedDf['Team'], SortedDf['PlayedGames'])  # scatter plot
ax.set_xlabel('Team ', fontsize=20)
ax.set_ylabel('Played Games', fontsize=20)

team=np.array(SortedDf.Team);
basket=np.array(SortedDf.PlayedGames);
for i, txt in enumerate(SortedDf.TeamLaunch):
    plt.annotate(txt, (team[i], basket[i]))
plt.title("Team vs PlayedGames annotated with TeamLaunch",size=35);
plt.show()

# %% [markdown]
# ### Explore PlayedGames across TeamLaunchCategory

# %%
# basketball['TeamLaunchCategory'] = (pd.qcut(basketball['TeamLaunch'], 4, 
#                                     labels=['Very Old', 'Old', 'New', 'Very New']));

GamesPlayed=basketball.sort_values('PlayedGames',ascending=False).groupby('TeamLaunchCategory')['PlayedGames'].sum().reset_index();

plt.figure(figsize=(15,5))
sns.barplot(x ='TeamLaunchCategory', y='PlayedGames' ,data = GamesPlayed)
plt.title("PlayedGames across Team launch category",size=15)
plt.show()

# %% [markdown]
# ### Explore Tournament vs Team

# %%
plt.figure(figsize=(30,10))
ax = sns.barplot(x="Team", y="Tournament", data=basketball)

# %% [markdown]
# ### Explore Score Vs Teams

# %%
# Scores of the teams
plt.figure(figsize=(20,15))
ax = sns.barplot(x="Score", y="Team", data=basketball, orient='h')

# %% [markdown]
# ### Explore Score Vs Teams with TeamLaunchCategory

# %%
plt.figure(figsize=(40,10))
sns.barplot(data=basketball,x='Team',y='Score',hue='TeamLaunchCategory')
plt.show()

# %% [markdown]
# ## Step-5: Performance Matrix

# %% [markdown]
# ### Correlation Matrix

# %%
#Plot correlation matrix using pandas, and select values in upper triangle
#https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas

CorMatrix = basketball.corr().abs()
Upper = CorMatrix.where(np.triu(np.ones(CorMatrix.shape),k=1).astype(np.bool))

Upper

# %%
PerformanceMatrix=basketball[['Team','TeamLaunch','TeamLaunchCategory','PlayedGames','HighestPositionHeld']]

PerformanceMatrix['Win']=round((basketball.WonGames/ basketball.PlayedGames)*100,2)
PerformanceMatrix['Drawn']=round((basketball.DrawnGames/ basketball.PlayedGames)*100,2)
PerformanceMatrix['Lost']=round((basketball.LostGames/ basketball.PlayedGames)*100,2)
PerformanceMatrix['TChampionship']=round((basketball.TournamentChampion/ basketball.Tournament)*100,2)
PerformanceMatrix['TRunnerUp']=round((basketball.RunnerUp/ basketball.Tournament)*100,2)

PerformanceMatrix.describe()

# %%
fig, ax = plt.subplots(1, 3)

fig.set_figheight(5)
fig.set_figwidth(15)

sns.distplot(PerformanceMatrix['Win'], ax = ax[0])
sns.distplot(PerformanceMatrix['Lost'], ax = ax[1])
sns.distplot(PerformanceMatrix['Drawn'], ax = ax[2])

ax[0].set_title('Win %')
ax[1].set_title('Lost %')
ax[2].set_title('Drawn %')
plt.show()

# %%
fig, ax = plt.subplots(2, 3)

fig.set_figheight(5)
fig.set_figwidth(15)

sns.boxplot(x=PerformanceMatrix['Win'],ax=ax[0][0]);
sns.boxplot(x=PerformanceMatrix['Lost'],ax=ax[0][1]);
sns.boxplot(x=PerformanceMatrix['Drawn'],ax=ax[0][2]);
sns.boxplot(x=PerformanceMatrix['TChampionship'],ax=ax[1][0]);
sns.boxplot(x=PerformanceMatrix['TRunnerUp'],ax=ax[1][1]);
sns.boxplot(x=PerformanceMatrix['TeamLaunch'],ax=ax[1][2]);


fig.tight_layout() 
plt.show()

# %% [markdown]
# ### Finding Outliers using Z-Score

# %%
# Get the z-score of every value with respect to their columns

PmSelect = PerformanceMatrix[['Win','Drawn','Lost']];
z = np.abs(stats.zscore(PmSelect))

threshold = 3
np.where(z > threshold)

# %%
PerformanceMatrix.iloc[[0,1,45,59]]

# %% [markdown]
# **Important Insights:**
# - Ouliers due to high Win % and low Lost %: T1 and T2
# - Ouliers due to low Win % and high Lost %: T46 and T60

# %% [markdown]
# ## Step-6: Recommended Teams for Company X based on all the above Analysis

# %% [markdown]
# ### Explore Win% Vs Drawn% Vs Lost%

# %%
SortedDf=PerformanceMatrix.sort_values('Win',ascending=False);
 
plt.figure(figsize=(35,10));
# set height of bar
bars1 = SortedDf.Win
bars2 = SortedDf.Drawn
bars3 = SortedDf.Lost
 # set width of bar
barWidth =  0.2
# Set position of bar on X axis
r1 = np.arange(len(bars1))+barWidth
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
# Make the plotax
plt.bar(r1, bars1, color='g', width=barWidth, edgecolor='white', label='Win%')
plt.bar(r2, bars2, color='y', width=barWidth, edgecolor='white', label='Drawn%')
plt.bar(r3, bars3, color='r', width=barWidth, edgecolor='white', label='Lost%')

# Add xticks on the middle of the group bars
plt.xlabel('Teams', fontweight='bold')
plt.xticks([r + (3*barWidth) for r in range(len(bars1))], SortedDf.Team)
plt.title("Win% Vs Drawn% Vs Lost%",size=35)
    
# Create legend & Show graphic
plt.legend()
plt.show()

# %% [markdown]
# ### Explore PlayedGames vs Teams vs Team Launch

# %%
SortedDf=(PerformanceMatrix.sort_values(['TeamLaunch'],ascending=[True]))

plt.figure(figsize=(30,5))
sns.pointplot(x='Team',y='PlayedGames',hue='TeamLaunchCategory' ,data=SortedDf);
plt.title("Played games across teams in the order of team launch ")
plt.show()

# %% [markdown]
# ### Performance Report of Teams in Playing Games

# %%
SortedDf=(PerformanceMatrix.sort_values(['TeamLaunch'],ascending=[True]))
fig, ax =plt.subplots(6,figsize=(30, 30), sharey=True)

sns.pointplot(x='Team',y='Win',hue='TeamLaunchCategory' ,data=SortedDf,ax=ax[0]);
sns.pointplot(x='Team',y='Drawn',hue='TeamLaunchCategory' ,data=SortedDf,ax=ax[1]);
sns.pointplot(x='Team',y='Lost',hue='TeamLaunchCategory' ,data=SortedDf,ax=ax[2]);
sns.pointplot(x='Team',y='TChampionship',hue='TeamLaunchCategory' ,data=SortedDf,ax=ax[3]);
sns.pointplot(x='Team',y='TRunnerUp',hue='TeamLaunchCategory' ,data=SortedDf,ax=ax[4]);
sns.pointplot(x='Team',y='HighestPositionHeld',hue='TeamLaunchCategory' ,data=SortedDf,ax=ax[5]);


fig.tight_layout()
plt.subplots_adjust(hspace=0.3)
ax[0].set_title("Performance report of teams in playing games\n Win%",size='30')
ax[1].set_title("Drawn %",size='30')
ax[2].set_title("Lost %",size='30')
ax[3].set_title("Tournment Championship %",size='30')
ax[4].set_title("Tournment RunnerUp  %",size='30')
ax[5].set_title("HighestPositionHeld",size='30')

# %% [markdown]
# ### Analyze Team Launch Categories

# %%
GroupLostMean=PerformanceMatrix.groupby(['TeamLaunchCategory'])['Lost'].mean().reset_index();
GroupDrawnMean=PerformanceMatrix.groupby(['TeamLaunchCategory'])['Drawn'].mean().reset_index();
GroupWinMean=PerformanceMatrix.groupby(['TeamLaunchCategory'])['Win'].mean().reset_index();

fig, ax =plt.subplots((3),figsize=(10, 10), sharey=True)

sns.pointplot(x=GroupWinMean.TeamLaunchCategory,y=GroupWinMean.Win,data=GroupWinMean,ax=ax[0]);
sns.pointplot(x=GroupDrawnMean.TeamLaunchCategory,y=GroupDrawnMean.Drawn,data=GroupDrawnMean,ax=ax[1]);
sns.pointplot(x=GroupLostMean.TeamLaunchCategory,y=GroupLostMean.Lost,data=GroupLostMean,ax=ax[2]);

fig.tight_layout()
plt.subplots_adjust(hspace=0.4)
ax[0].set_title("Analyze Team launch categories \n Mean Win%",size='20')
ax[1].set_title("Mean Drawn %",size='20')
ax[2].set_title("Mean Lost %",size='20')
plt.show()

# %% [markdown]
# **Important Insights:**
#     
# - Av. drawn is high for very new teams comapred to old teams; With some more practise this drawn % can be converted to more successes.
# - Av. Lost is low for very new teams comapred to other teams; Very new teams are better compared to old/new teams.
# - Av. win % is high for very old teams and gradully decreasing for new teams.
# - More detailed information can be inferred from the "Team vs PlayedGames Annotated with TeamLaunch" and other Charts.

# %% [markdown]
# ### Top 10 teams in the given list with Hightest Win %

# %%
PerformanceMatrix.sort_values(['Win'],ascending=[False]).head(10)

# %% [markdown]
# ### Top 10 winning teams excluding very old teams

# %%
PerformanceMatrix[PerformanceMatrix.TeamLaunchCategory != 'Very Old'].sort_values(['Win'],ascending=[False]).head(10)

# %% [markdown]
# ### Top Teams with High Performance

# %%
basketball[(basketball.PlayedGames==basketball.PlayedGames.max())]

# %% [markdown]
# ### Best Performance Team in the Dataset

# %%
PerformanceMatrix[(PerformanceMatrix.Win==PerformanceMatrix.Win.max()) & (PerformanceMatrix.Lost==PerformanceMatrix.Lost.min())]

# %%
# Performance order of teams is T1>T2>T5
# Above teams are Outliers, So we look for other max target teams

basketball[(basketball.PlayedGames==basketball.PlayedGames.nlargest(4)[3])]

# %% [markdown]
# ### Teams with Low Performance

# %%
(basketball.sort_values(['PlayedGames'],ascending=[True])).head(3)

# %% [markdown]
# ### Teams with high rank in position

# %%
(basketball.sort_values(['HighestPositionHeld'],ascending=[False])).head(1)

# %% [markdown]
# ### Old Teams with less performance

# %%
sorted=PerformanceMatrix.loc[PerformanceMatrix['TeamLaunchCategory']=='Very Old'].sort_values(['Win'],ascending=[True])
sorted.head(5)

# %% [markdown]
# ### Team with Most Drawn games

# %%
PerformanceMatrix.sort_values(['Drawn'],ascending=[False]).head(1)

# %% [markdown]
# ### Old teams with low targets

# %%
sorted=(basketball.sort_values(['PlayedGames'],ascending=[True]))

sorted[(sorted.TeamLaunch==sorted.TeamLaunch.min())].head(2)

# %% [markdown]
# ### Recommended Teams for company X based on all the above analysis

# %%
TeamsFilters=['T21','T39','T46','T19','T20','T11','T9','T5','T3','T4','T6','T14','T7','T8','T10','T18']

BetterTeams=pd.DataFrame(columns=PerformanceMatrix.columns);

for i in TeamsFilters:
       if i in PerformanceMatrix.Team.values:
            BetterTeams = pd.concat([BetterTeams,PerformanceMatrix[PerformanceMatrix.Team==i]], ignore_index=True)
                 
BetterTeams.sort_values('Win',ascending=False)

# %% [markdown]
# ### Recommended Teams for company X based on all the above analysis (Excluding the Very Old Teams)

# %%
BetterTeams[BetterTeams.TeamLaunchCategory!='Very Old']

# %% [markdown]
# ## Step-7: Improvements or suggestions to the association management on quality, quantity, variety, velocity, veracity etc. on the data points collected by the association to perform a better data analysis in future.

# %% [markdown]
# **Please find below suggestions for data point collection and other relevant guidelines:**
#
# 1. Volume: We can add some more teams for better data understanding. To increase the prediction power of the dataset, we can add some other information like player information, demographics and tournament location Etc. Database systems can move from the traditional to the more advanced big data systems.
#
# 2. Velocity: If we consider the time value of the data, It seems to be outdated and of little use; Particularly if the Big Data project is to serve any real-time or near real-time business needs. In such context  we should re-define data quality metrics so that they are relevant as well as feasible in the real-time context.
#
# 3. Variety: For better insights and modeling projects in AI and ML, we can add several other data types like structured, semi-structured, and unstructured coming in from different data sources relevant to the basketball.
#
# 4. Veracity: We have incomplete team information. For example in Team 61: This Team don't have any information about Score, PlayedGames Etc.. It has HighestPoistionHeld as 1. Accuracy of data collection should be improved. Besides data inaccuracies, Veracity also includes data consistency (defined by the statistical reliability of data) and data trustworthiness (based on data origin, data collection and processing methods, security infrastructure, etc.). These data quality issues in turn impact data integrity and data accountability.
#
# 5. Value: The Value characteristic connects directly to the end purpose and the business use cases. We can harness the power of Big Data for many diverse business pursuits, and those pursuits are the real drivers of how data quality is defined, measured, and improved. Data Science is already playing a pivotal role in sports analytics.
#
# 6. Based on a strong understanding of the business use cases and the Big Data architecture, we can design and implement an optimal layer of data governance strategy to further improve the data quality with data definitions, metadata requirements, data ownership, data flow diagrams, etc.
#
# 7. We can add more attributes to the dataset. More relevant attributes will help us to analyze teams accurately like Canceled Games, Basket Ratio, Winning Ratio, Win/Loss Percentage
#
# 8. Simplest ML models can add more value to the present EDA use case.
#
# 9. Big data and data science together allow us to see both the forest and the trees (Micro and Macro perspectives).
#
# 10. Visualization, Dashboarding, and Interactivity makes the data more useful to the general public. We can use the API and deploy it on the cloud to serve our purpose in this context.

# %% [markdown]
# ## References:
#
# 1. [Towards Data Science. Sports Analytics](https://towardsdatascience.com/sports-analytics/home)
# 2. [Kaggle. Kaggle Code](https://www.kaggle.com/code)
# 3. [KdNuggets](https://www.kdnuggets.com/)
# 4. [AnalyticsVidhya](https://www.analyticsvidhya.com/blog/)
# 5. [Wikipedia. Basketball](https://en.wikipedia.org/wiki/Basketball)
# 6. [Wikipedia. Sports Analytics](https://en.wikipedia.org/wiki/Sports_analytics)
# 7. [Wikipedia. National Basketball Association](https://en.wikipedia.org/wiki/National_Basketball_Association)
# 8. Zuccolotto, Manisera, & Sandri. "Basketball Data Science: With Applications in R." Chapman & Hall/CRC Data Science Series, 2020. Print.
# 9. Baker, & Shea. "Basketball Analytics: Objective and Efficient Strategies for Understanding How Teams Win, 2013." Print.
# 10. Shea. "Basketball Analytics: Spatial Tracking." Kindle.
# 11. Oliver, & Alamar. "Sports Analytics: A Guide for Coaches, Managers, and Other Decision Makers, 2013." Print.
# 12. [Numpy](https://numpy.org/)
# 13. [Pandas](https://pandas.pydata.org/)
# 14. [SciPy](https://scipy.org/)
# 15. [MatplotLib](https://matplotlib.org/)
# 16. [Seaborn](https://seaborn.pydata.org/)
# 17. [Python](https://www.python.org/)
# 18. [Plotly](https://plotly.com/)
# 19. [Bokeh](https://docs.bokeh.org/en/latest/)
# 20. [RStudio](https://www.rstudio.com/)
# 21. [MiniTab](https://www.minitab.com/en-us/)
# 22. [Anaconda](https://www.anaconda.com/)

# %% [markdown]
# # Part-C: Solution

# %% [markdown]
# **DOMAIN:** Startup ecosystem
#
# **CONTEXT:** Company X is a EU online publisher focusing on the startups industry. The company specifically reports on the business related to technology news, analysis of emerging trends and profiling of new tech businesses and products. Their event i.e. Startup Battlefield is the world’s pre-eminent startup competition. Startup Battlefield features 15-30 top early stage startups pitching top judges in front of a vast live audience,
# present in person and online.
#
# **DATA DESCRIPTION:** CompanyX_EU.csv - Each row in the dataset is a Start-up company and the columns describe the company.
#
# **DATA DICTIONARY:**
# 1. Startup: Name of the company
# 2. Product: Actual product
# 3. Funding: Funds raised by the company in USD
# 4. Event: The event the company participated in
# 5. Result: Described by Contestant, Finalist, Audience choice, Winner or Runner up
# 6. OperatingState: Current status of the company, Operating ,Closed, Acquired or IPO
#
# *Dataset has been downloaded from the internet. All the credit for the dataset goes to the original creator of the data.
#
# **PROJECT OBJECTIVE:** Analyse the data of the various companies from the given dataset and perform the tasks that are specified in the below steps. Draw insights from the various attributes that are present in the dataset, plot distributions, state hypotheses and draw conclusions from the dataset.

# %% [markdown]
# ## 1. Read the CSV file

# %%
compx=pd.read_csv("Compx.csv");

# %%
compx.head()

# %%
compx.tail()

# %%
compx.shape

# %% [markdown]
# ## 2. Data Exploration

# %% [markdown]
# ### 2A. Check the datatypes of each attribute.

# %%
compx.info()


# %% [markdown]
# ### 2B. Check for null values in the attributes.

# %%
def missing_check(df):
    total = df.isnull().sum().sort_values(ascending=False)   # total number of null values
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)  # percentage of values that are null
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # putting the above two together
    return missing_data # return the dataframe
missing_check(compx)

# %% [markdown]
# ## 3. Data preprocessing & visualisation:

# %% [markdown]
# ### 3A. Drop the null values.

# %%
compx1 = compx.dropna(subset=['Funding', 'Product'])

# %%
compx1

# %%
compx1.info()

# %% [markdown]
# ### 3B. Convert the ‘Funding’ features to a numerical value.

# %%
compx1.loc[:,'Funds_in_million'] = compx1['Funding'].apply(lambda x: float(x[1:-1])/1000 if x[-1] == 'K' 
                                                           else (float(x[1:-1])*1000 if x[-1] == 'B' 
                                                           else float(x[1:-1])))

# %%
compx1

# %% [markdown]
# ### 3C. Plot box plot for funds in million.

# %%
plt.figure(figsize=(20,5))
ax = sns.boxplot(x=compx1["Funds_in_million"])

# %%
# Adding swarmplot to boxplot for better visualization

plt.figure(figsize=(20,8))
ax = sns.boxplot(x=compx1["Funds_in_million"], whis=np.inf)
ax = sns.swarmplot(x=compx1["Funds_in_million"], color=".2")

# We can also use jitter for better visualization
# ax = sns.stripplot(x=compx1["Funds_in_million"], color=".2", jitter=0.3)

# %% [markdown]
# ### 3D. Check the number of outliers greater than the upper fence.

# %%
# Calculate the IQR
cols = ['Funds_in_million']
Q1 = compx1[cols].quantile(0.25)
Q3 = compx1[cols].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

# %%
Outliers = np.where(compx1[cols] > (Q3 + 1.5 * IQR))
Outliers

# %%
# Rows without outliers
NonOutliers = compx1[cols][~((compx1[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
NonOutliers

# %%
NonOutliers.shape, compx1.shape

# %%
NumberOfOutliers = compx1.shape[0] - NonOutliers.shape[0]
NumberOfOutliers

# %% [markdown]
# ### 3E. Check frequency of the OperatingState features classes.

# %%
compx1["OperatingState"].value_counts()

# %%
plt.figure(figsize=(10,5))
ax = sns.countplot(x="OperatingState", data=compx1)

# %%
compx1.shape[0]

# %% [markdown]
# ## 4. Statistical Analysis

# %% [markdown]
# ### 4A. Is there any significant difference between Funds raised by companies that are still operating vs companies that closed down?

# %%
plt.figure(figsize=(10,5))
sns.barplot(x ='OperatingState', y='Funds_in_million' ,data = compx1)
plt.title("Operating State vs Funds in Million",size=15)
plt.show()

# %%
#Understand the distribution of 'Funds_in_million' variable
plt.figure(figsize=(10,5))
ax = sns.histplot(compx1['Funds_in_million'], log_scale=True, kde=True)

# %%
# Companies Operating vs Companies Closed

CompOperating = compx1[['Funds_in_million','OperatingState']][compx1['OperatingState']=='Operating']
CompClosed = compx1[['Funds_in_million','OperatingState']][compx1['OperatingState']=='Closed']

# %%
CompOperating.head()

# %%
CompOperating.describe()

# %%
plt.figure(figsize=(10,5))
ax = sns.histplot(CompOperating['Funds_in_million'], log_scale=True, kde=True)

# %%
CompClosed.head()

# %%
CompClosed.describe()

# %%
plt.figure(figsize=(10,5))
ax = sns.histplot(CompClosed['Funds_in_million'], log_scale=True, kde=True)

# %% [markdown]
# **Important Insights:**
# - From the graphical and descriptive statistics it can be concluded that there is significant difference between the Funds raised by the companies that are operating compared to the companies that are closed. 
#
# - More would be validated using the Hypothesis Testing.

# %% [markdown]
# ### 4B. Write the null hypothesis and alternative hypothesis.

# %% [markdown]
# **The two hypotheses for this particular two sample t-test are as follows:**
#
# - µ1  = Mean funding for companies that are Operating
# - µ2 = Mean funding for companies that are Closed
# - Null Hypothesis, H0: µ1 = µ2 (the two population means are equal)
# - Alternate Hypothesis, HA: µ1 ≠ µ2 (the two population means are not equal)

# %% [markdown]
# ### 4C. Test for significance and conclusion

# %% [markdown]
# **Assumptions:**
# - Observations in two groups have an approximately normal distribution (Shapiro-Wilks Test)
#
# - Homogeneity of variances (variances are equal between treatment groups) (Levene or Bartlett Test); It's different in our case here.
#
# - The two groups are sampled independently from each other from the same population
#
# Note: Two sample t-test is relatively robust to the assumption of normality and homogeneity of variances when sample size is large (n ≥ 30) and there are equal number of samples (n1 = n2) in both groups.
#
# If the sample size small and does not follow the normal distribution, We should use non-parametric Mann-Whitney U test (Wilcoxon rank sum test).
#
# **Conduct a two sample t-test:**
#
# Next, we’ll use the ttest_ind() function from the scipy.stats library to conduct a two sample t-test, which uses the following syntax:
#
# ttest_ind(a, b, equal_var=True)
#
# where:
#
# - a: an array of sample observations for group 1
# - b: an array of sample observations for group 2
# - equal_var: if True, perform a standard independent 2 sample t-test that assumes equal population variances. If False, perform Welch’s t-test, which does not assume equal population variances. This is True by default.
#
# **Before we perform the test, we need to decide if we’ll assume the two populations have equal variances or not. As a rule of thumb, we can assume the populations have equal variances if the ratio of the larger sample variance to the smaller sample variance is less than 4:1.**

# %%
# Find variance for each group
var1, var2 = np.var(CompOperating['Funds_in_million']), np.var(CompClosed['Funds_in_million'])
print(var1, var2)

# %%
VarianceRatio = var1/var2
VarianceRatio

# %%
# Lets use the Unequal Variance Case

stats.ttest_ind(CompOperating['Funds_in_million'], CompClosed['Funds_in_million'], equal_var=False)

# %% [markdown]
# **Intepretation:**
#
# 1. Because the p-value of our test (0.00789) is less than alpha = 0.05, we reject the null hypothesis of the test. 
#
# 2. We do have sufficient evidence to say that there is a significant difference between the funds raised by the companies that are operating vs the companies that are closed.

# %% [markdown]
# ### 4D. Make a copy of the original data frame.

# %%
# Make a copy of the original dataset
compxOriginal=compx.copy(deep=True);

# %%
compxOriginal.head()

# %%
compxOriginal.info()

# %% [markdown]
# ### 4E. Check frequency distribution of Result variables.

# %%
compxOriginal["Result"].value_counts()

# %%
plt.figure(figsize=(10,5))
ax = sns.countplot(x="Result", data=compxOriginal)

# %% [markdown]
# ### 4F. Calculate percentage of winners that are still operating and percentage of contestants that are still operating

# %%
# Companies Operating vs Companies Closed

winnerOp = compxOriginal[['Result','OperatingState']][(compxOriginal['OperatingState']=='Operating') 
                                                    & (compxOriginal['Result']=='Winner')]
contestantOp = compxOriginal[['Result','OperatingState']][(compxOriginal['OperatingState']=='Operating') 
                                                    & (compxOriginal['Result']=='Contestant')]

# %%
winnerOp.head()

# %%
winnerOp.describe()

# %%
contestantOp.head()

# %%
contestantOp.describe()

# %%
# Operating winner and contestant

winnerContestantOp = 19 + 332
winnerContestantOp

# %%
# Using the data from above analysis

winnerOpPercent, contestantOpPercent = 19/351, 332/351
print(winnerOpPercent, contestantOpPercent)

# %% [markdown]
# ### 4G. Write your hypothesis comparing the proportion of companies that are operating between winners and contestants:

# %% [markdown]
# **The two hypotheses for this particular two sample z-test are as follows:**
#
# - P1 = Proportion of companies that are operating and winners
# - P2 = Proportion of companies that are operating and contestants
# - Null Hypothesis, H0: P1 = P2 (the two population proportions are equal)
# - Alternate Hypothesis, HA: P1 ≠ P2 (the two population proportions are not equal)

# %% [markdown]
# ### 4H. Test for significance and conclusion

# %%
# Assuming independent samples, large sample sizes, and the hypothesized population proportion difference as zero.

from statsmodels.stats.proportion import proportions_ztest
count = np.array([19, 332])
nobs = np.array([351, 351])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval), '{0:0.3f}'.format(stat))

# %% [markdown]
# **Intepretation:**
#
# 1. Because the p-value of our test (0.000) is less than alpha = 0.05, we reject the null hypothesis of the test. 
#
# 2. We do have sufficient evidence to say that there is a significant difference between the proportion of operating companies in two classes like winners and contestants.

# %% [markdown]
# ### 4I. Select only the Event that has ‘disrupt’ keyword from 2013 onwards.

# %%
# Usng the str, contains functions, We can solve it

CompDisrupt2013 = compxOriginal[compxOriginal['Event'].str.contains('Disrupt') & compxOriginal['Event'].str.contains('2013')]
CompDisrupt2013
