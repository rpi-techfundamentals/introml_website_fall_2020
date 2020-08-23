[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Boston Housing - Feature Selection and Importance</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

## Overview
- Getting the Data
- Reviewing Data
- Modeling 
- Model Evaluation
- Using Model
- Storing Model


## Getting Data 
- Available in the [sklearn package](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) as a Bunch object (dictionary).
- From FAQ: ["Don’t make a bunch object! They are not part of the scikit-learn API. Bunch objects are just a way to package some numpy arrays. As a scikit-learn user you only ever need numpy arrays to feed your model with data."](http://scikit-learn.org/stable/faq.html)
- Available in the UCI data repository. 
- Better to convert to Pandas dataframe. 

#From sklearn tutorial.
from sklearn.datasets import load_boston
boston = load_boston()
print( "Type of boston dataset:", type(boston))


#A bunch is you remember is a dictionary based dataset.  Dictionaries are addressed by keys. 
#Let's look at the keys. 
print(boston.keys())


#DESCR sounds like it could be useful. Let's print the description.
print(boston['DESCR'])

# Let's change the data to a Panda's Dataframe
import pandas as pd
boston_df = pd.DataFrame(boston['data'] )
boston_df.head()

#Now add the column names.
boston_df.columns = boston['feature_names']
boston_df.head()

#Add the target as PRICE. 
boston_df['PRICE']= boston['target']
boston_df.head()

## What type of data are there?
- First let's focus on the dependent variable, as the nature of the DV is critical to selection of model. 
- *Median value of owner-occupied homes in $1000's* is the Dependent Variable  (continuous variable).
- It is relevant to look at the distribution of the dependent variable, so let's do that first.
- Here there is a normal distribution for the most part, with some at the top end of the distribution we could explore later.

## Preparing to Model 
- It is common to separate `y` as the dependent variable and `X` as the matrix of independent variables.
- Here we are using `train_test_split` to split the test and train.
- This creates 4 subsets, with IV and DV separted: `X_train, X_test, y_train, y_test`
 


#This will throw and error at import if haven't upgraded. 
# from sklearn.cross_validation  import train_test_split  
from sklearn.model_selection  import train_test_split
#y is the dependent variable.
y = boston_df['PRICE']
#As we know, iloc is used to slice the array by index number. Here this is the matrix of 
#independent variables.
X = boston_df.iloc[:,0:13]

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


## Modeling 
- First import the package: `from sklearn.linear_model import LinearRegression`
- Then create the model object.  
- Then fit the data. 
- This creates a trained model (an object) of class regression.
- The variety of methods and attributes available for regression are shown [here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit( X_train, y_train )


## Evaluating the Model Results
- You have fit a model. 
- You can now store this model, save the object to disk, or evaluate it with different outcomes. 
- Trained regression objects have coefficients (`coef_`) and intercepts (`intercept_`) as attributes. 
- R-Squared is determined from the `score` method of the regression object.
- For Regression, we are going to use the coefficient of determination as our way of evaluating the results, [also referred to as R-Squared](https://en.wikipedia.org/wiki/Coefficient_of_determination)

print('labels\n',X.columns)
print('Coefficients: \n', lm.coef_)
print('Intercept: \n', lm.intercept_)
print('R2 for Train)', lm.score( X_train, y_train ))
print('R2 for Test (cross validation)', lm.score(X_test, y_test))

#Alternately, we can show the results in a dataframe using the zip command.
pd.DataFrame( list(zip(X.columns, lm.coef_)),
            columns=['features', 'estimatedCoeffs'])

# L1 Regularized Regression
By increasing the alpha, we can zero in on the variables which are more important in the analysis. 



from sklearn import linear_model
reg = linear_model.Ridge(alpha=5000)
reg.fit(X_train, y_train ) 
print('R2 for Train)', reg.score( X_train, y_train ))
print('R2 for Test (cross validation)', reg.score(X_test, y_test))

#Alternately, we can show the results in a dataframe using the zip command.
pd.DataFrame( list(zip(X.columns, reg.coef_)),
            columns=['features', 'estimatedCoeffs'])

## Feature Importance With Random Forrest Regression


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(random_state=99)
forest.fit(X_train, y_train) 
importances = forest.feature_importances_

print('R2 for Train)', forest.score( X_train, y_train ))
print('R2 for Test (cross validation)', forest.score(X_test, y_test))

## Feature Selection
"SelectFromModel is a meta-transformer that can be used along with any estimator that has a coef_ or feature_importances_ attribute after fitting. The features are considered unimportant and removed, if the corresponding coef_ or feature_importances_ values are below the provided threshold parameter. Apart from specifying the threshold numerically, there are built-in heuristics for finding a threshold using a string argument."





from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(forest, prefit=True, max_features=3)
feature_idx = model.get_support()
feature_names = X.columns[feature_idx]
X_NEW = model.transform(X)
pd.DataFrame(X_NEW, columns= feature_names)


# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_NEW, y, test_size=0.3, random_state=0)
lm = LinearRegression()
lm.fit( X_train, y_train )
print('R2 for Train)', lm.score( X_train, y_train ))
print('R2 for Test (cross validation)', lm.score(X_test, y_test))



from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit( X_train, y_train )

from sklearn.metrics import r2_score
r2_train_reg = r2_score(y_train, lm.predict(X_train))
r2_test_reg = r2_score(y_test, lm.predict(X_test))
print(r2_train_reg,r2_test_reg  )


Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016.
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.
