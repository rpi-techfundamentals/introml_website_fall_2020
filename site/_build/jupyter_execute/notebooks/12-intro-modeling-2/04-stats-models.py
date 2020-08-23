[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Regression with Stats-Models </h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

## Scikit-learn vs Stats-Models
- Scikit-learn provides framework which enables a similar api (way of interacting with codebase) for many different types of machine learning (i.e., predictive) models.
- Stats-Models provices a clear set of results for statistical analsyses (understanding relationships) common to scientific (i.e., explanitory) models

#Get a sample dataset
df = sm.datasets.get_rdataset("Guerry", "HistData").data

## About the Data
- See [link](https://cran.r-project.org/web/packages/HistData/HistData.pdf).
- Andre-Michel Guerry (1833) was the first to systematically collect and analyze social data on such things as crime, literacy and suicide with the view to determining social laws and the relations among these variables.

df.columns

df

## Predicting Gambling
- `Lottery` Per capita wager on Royal Lottery. Ranked ratio of the proceeds bet on the royal lottery to population— Average for the years 1822-1826.  (Compte rendus par le ministre des finances)
- `Literacy` Percent Read & Write: Percent of military conscripts who can read and write. 
- `Pop1831` Population in 1831, taken from Angeville (1836), Essai sur la Statis-
tique de la Population francais, in 1000s.

#Notice this is an R style of Analsysis
results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=df).fit()

print(results.summary())

Key Metrics
- [Link](http://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLSResults.html) for summary of results.
- 

## Challenge: Compare Results
- Explore another predictor of `Lottery`, was it significant in influecing. 
- Add a train test split to the above analysis and assess overfitting. 
- Use scikit learn regression to replicate the above analysis.  
- Then use Stats-models to replicate Boston Housing example below. 


!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/boston_test.csv && wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/boston_train.csv

train = pd.read_csv("boston_train.csv")
test = pd.read_csv("boston_test.csv")

train.head()

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

### Data preprocessing: 
We're not going to do anything fancy here: 
 
- First I'll transform the skewed numeric features by taking log(feature + 1) - this will make the features more normal    
- Create Dummy variables for the categorical features    
- Replace the numeric missing values (NaN's) with the mean of their respective columns

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

X_train.shape

### Models

Now we are going to use regularized linear regression models from the scikit learn module. I'm going to try both l_1(Lasso) and l_2(Ridge) regularization. I'll also define a function that returns the cross-validation rmse error so we can evaluate our models and pick the best tuning par