<img src="https://raw.githubusercontent.com/RPI-DATA/website/master/static/images/rpilogo.png" alt="RPI LOGO">

# Panel Data vs Time Series Analysis
- - - - - - - -- - - - - - - -- - - - - - - - - - - 
A **_time series_** is a series of data points indexed (or listed or graphed) in time order. Time series analysis pertains to methods extracting meaningful statistics from time series data. This is commonly used for forecasting and other models.

A **_panel dataset_** are multi-dimensional data involving measurements for the same firm, entity, region, or person over time. 

This is an example of a panel dataset, measuring multiple stores over time.  You can make it into a simple time series by selecting one store. 

# Learning Objectives
- - - - - - - - - - - - - -- - - - - - - - - - -
1. Understand the uses of Time Series Analysis
2. Understand the pros and cons of various TSA methods, including differentiating between linear and non-linear methods.
3. Apply the facebook prophet model and analyze the results on given rossman store data.

# Sections
- - - -- - - - - - - - - - - - - - -- - - - -
1. [ Problem Description](#scrollTo=O28Uk6v0KxS-)
2. Exploratory Data Analysis
  1. [Training Data](#scrollTo=bce8x1WuKxTF)
  2. [Store Data](#scrollTo=qCUy3qWVKxTj)
3. [Moving Average Model](#scrollTo=drEtObHEKxT-)
4. [Facebook Prophet Model](#scrollTo=MHCK3E7gKxUe)
5. [Conclusion](#scrollTo=xqUvAJGnKxU-)
6. [References](#scrollTo=vwv9NZDZKxU-)

# Problem Description/Description of Data
- - - - -- -- - - - -   - - - - 
We will use the rossman store sales database for this notebook. Following is the description of Data from the website:

"Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied."

# Library Imports
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from math import sqrt
from sklearn.metrics import mean_squared_error
import fbprophet

# matplotlib parameters
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

%config InlineBackend.figure_format = 'retina' 
%matplotlib inline

# Data Reading
train = pd.read_csv('https://raw.githubusercontent.com/RPI-DATA/tutorials-intro/master/rossmann-store-sales/rossmann-store-sales/train.csv', parse_dates = True, low_memory = False, index_col = 'Date')
store = pd.read_csv('https://raw.githubusercontent.com/RPI-DATA/tutorials-intro/master/rossmann-store-sales/rossmann-store-sales/store.csv', low_memory = False)

# Exploratory Data Analysis (Train)
- - - - - - - - - --  - - - - - - -  - - - - - - - - --  - - -- - - -
We start by seeing what our data conists of. We want to see which variables are continuous vs which are categorical. After exploring some of the data, we see that we can create a feature. Number of sales divided by customers could give us a good metric to measure average sales per customer. We can also make an assumption that if we have missing values in this column that we have 0 customers. Since customers drive sales, we elect to remove all of these values.

Notice the order in which the data is listed. It is ordered from most recent date to oldest date. This may cause a problem when we look to develop our model.

train.head()

train.head()

train.shape

After that we will use the amazing `.describe()` function which can provide us most of the statistic elements. 

train.describe()

We will check all the missing elements over here.

missing = train.isnull().sum()
missing.sort_values(ascending=False)

Next, we create a new metric to see average sales per customer.

train['SalesPerCustomer'] = train['Sales']/train['Customers']
train['SalesPerCustomer'].head()

We are going to Check if there are any missing values with our new metric and drop them. Either the customers or the sales should be zero to give us a null SalesPerCustomer. 

missing = train.isnull().sum()
missing.sort_values(ascending=False)

train.dropna().head()

# Exploratory Data Analysis (Store Data)
- - - - - -- - - - - - - - - - - - - -  - - - - - - - - - -
We do the same as we did for our training set. Exploring the data, we see that there are only 3 missing values in CompetitionDistance. Because this is such a small amount, we elect to replace these with the mean of the column. The other missing values are all dependent on Promo2. Because these missing values are because Promo2 is equal to 0, we can replace these nulls with 0.

store.head()

store.shape

store.isnull().sum()

Since there are only 3 missing values from this, we fill with the average from the column

store['CompetitionDistance'].fillna(store['CompetitionDistance'].mean(), inplace = True)

The rows that do not have any Promo2 we can fill the rest of the values with 0

store.fillna(0, inplace = True)

store.head()

Join the data together using an inner join so only the data that is in both data set is joined

train = train.merge(right=store, on='Store', how='left')

# Moving-Average Model (Naive Model)
- - - - - - - - - - - - - - -
We are going to be using a moving average model for the stock prediction of GM for our baseline model. The moving average model will take the average of different "windows" of time to come up with its forecast

We reload the data because now we have a sense of how we want to maniplate it for our model. After doing the same data manipulation as before, we start to look at the trend of our sales. 


train = pd.read_csv('https://raw.githubusercontent.com/RPI-DATA/tutorials-intro/master/rossmann-store-sales/rossmann-store-sales/train.csv', parse_dates = True, low_memory = False, index_col = 'Date')
train = train.sort_index(ascending = True)

train['SalesPerCustomer'] = train['Sales']/train['Customers']
train['SalesPerCustomer'].head()

train = train.dropna()

Here, we are simply graphing the sales that we have. As you can see, there are a tremendous amount of sales to the point where our graph just looks like a blue shading. However, we can get a sense of how our sales are distributed.

plt.plot(train.index, train['Sales'])
plt.title('Rossmann Sales')
plt.ylabel('Price ($)');
plt.show()

To clean up our graph, we want to form a new column which only accounts for the year of the sales. 

train['Year'] = train.index.year

# Take Dates from index and move to Date column 
train.reset_index(level=0, inplace = True)
train['sales'] = 0


Split the data into a train and test set. We use an 80/20 split. Then, we look to start are modein. 

test_store stands for the forecasting part. 

train_store=train[0:675472] 
test_store=train[675472:]

train_store.Date = pd.to_datetime(train_store.Date, format="%Y-%m-%d")
train_store.index = train_store.Date
test_store.Date = pd.to_datetime(test_store.Date, format="%Y-%m-%d")
test_store.index = test_store.Date

train_store = train_store.resample('D').mean()
train_store = train_store.interpolate(method='linear')

test_store = test_store.resample('D').mean()
test_store = test_store.interpolate(method='linear')

train_store.Sales.plot(figsize=(25,10), title='daily sales', fontsize=25)
test_store.Sales.plot()

y_hat_avg_moving = test_store.copy()
y_hat_avg_moving['moving_avg_forcast'] = train_store.Sales.rolling(90).mean().iloc[-1]
plt.figure(figsize=(25,10))
plt.plot(train_store['Sales'], label='Train')
plt.plot(test_store['Sales'], label='Test')
plt.plot(y_hat_avg_moving['moving_avg_forcast'], label='Moving Forecast')
plt.legend(loc='best')
plt.title('Moving Average Forecast')

rms_avg_rolling = sqrt(mean_squared_error(test_store.Sales, y_hat_avg_moving.moving_avg_forcast))
print('ROLLING AVERAGE',rms_avg_rolling)

The rolling average for our model is 1,915.88. This prediction seems to be very consistent in hitting the average of the future sales. This naive model definitely looks like a solid model, however, it is not the best one. 

# Facebook Prophet Model
- - - - - - - -  - - - - -- - - -
The Facebook Prophet package is designed to analyze time series data with daily observations, which can display patterns on different time scales. Prophet is optimized for business tasks with the following characteristics:

hourly, daily, or weekly observations with at least a few months (preferably a year) of history
strong multiple “human-scale” seasonalities: day of week and time of year
important holidays that occur at irregular intervals that are known in advance (e.g. the Super Bowl)
a reasonable number of missing observations or large outliers
historical trend changes, for instance due to product launches or logging changes
trends that are non-linear growth curves, where a trend hits a natural limit or saturates
inspired by https://research.fb.com/prophet-forecasting-at-scale/

According to the "facebook research" website, there is four main component inside the facebook prophet model.

** A piecewise linear or logistic growth trend. 

** A yearly seasonal component modeled using Fourier series. 

** A weekly seasonal component using dummy variables.

** A user-provided list of important holidays.

The method of combing different models into one makes the facebook prophet model much more precise and flexible.

First of all, we will dealt with the data as same as the naive model

store = pd.read_csv('https://raw.githubusercontent.com/RPI-DATA/tutorials-intro/master/rossmann-store-sales/rossmann-store-sales/store.csv', low_memory = False)



train['SalesPerCustomer'] = train['Sales']/train['Customers']
train['SalesPerCustomer'].head()

train = train.dropna()

sales = train[train.Store == 1].loc[:, ['Date', 'Sales']]

# reverse to the order: from 2013 to 2015
sales = sales.sort_index(ascending = False)

sales['Date'] = pd.DatetimeIndex(sales['Date'])
sales.dtypes

sales = sales.rename(columns = {'Date': 'ds',
                                'Sales': 'y'})

sales.head()

Now, despite of the naive model, we will apply our sales set to the face book model by using the function `fbprophet`. `fbprophet.Prophet` can change the value of "changepoint_prior_scale" to 0.05 to achieve a better fit or to 0.15 to control how sensitive the trend is.

sales_prophet = fbprophet.Prophet(changepoint_prior_scale=0.05, daily_seasonality=True)
sales_prophet.fit(sales)

We will figure out the best forecasting by changing the value of changepoints.

If we find that our model is is fitting too closely to our training data (overfitting), our data will not be able to generalize new data.

If our model is not fitting closely enough to our training data (underfitting), our data has too much bias.

Underfitting: increase changepoint to allow more flexibility Overfitting: decrease changepoint to limit flexibili

# Try 4 different changepoints
for changepoint in [0.001, 0.05, 0.1, 0.5]:
    model = fbprophet.Prophet(daily_seasonality=False, changepoint_prior_scale=changepoint)
    model.fit(sales)
    
    future = model.make_future_dataframe(periods=365, freq='D')
    future = model.predict(future)
    
    sales[changepoint] = future['yhat']

We can now create the plot under all of the situation.

# Create the plot
plt.figure(figsize=(10, 8))

# Actual observations
plt.plot(sales['ds'], sales['y'], 'ko', label = 'Observations')
colors = {0.001: 'b', 0.05: 'r', 0.1: 'grey', 0.5: 'gold'}

# Plot each of the changepoint predictions
for changepoint in [0.001, 0.05, 0.1, 0.5]:
    plt.plot(sales['ds'], sales[changepoint], color = colors[changepoint], label = '%.3f prior scale' % changepoint)
    
plt.legend(prop={'size': 14})
plt.xlabel('Date'); plt.ylabel('Rossmann Sales'); plt.title('Rossmann Effect of Changepoint Prior Scale');

Predictions for 6 Weeks
In order to make forecasts, we need to create a future dataframe. We need to specify the amount of future periods to predict and the frequency of our prediction.

Periods: 6 Weeks 

Frequency: Daily

# Make a future dataframe for 6weeks
sales_forecast = sales_prophet.make_future_dataframe(periods=6*7, freq='D')
# Make predictions
sales_forecast = sales_prophet.predict(sales_forecast)

sales_prophet.plot(sales_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('Rossmann Sales');

sales_prophet.changepoints[:10]

We have listed out the most significant changepoints in our data. This is representing when the time series growth rate significantly changes.

# Conclusion
- -- - - - - - -  - - 
In this notebook, we made 2 different math model for the rossmann store sales dataset to forecast the future sales. Moving-average model brings us a basic understand of how the math model works, while facebook prophet model calculates the best solid result. Those math model will give us both of the rolling average and test model. 

# References
- - - -  - -  - - - - - - - - -
The dataset is the rossmann store sales dataset from kaggle:
https://www.kaggle.com/c/rossmann-store-sales

Facebook Prophet Documentation:
https://research.fb.com/prophet-forecasting-at-scale/



# Contributers

* nickespo21
* Linghao Dong
* Jose Figueroa