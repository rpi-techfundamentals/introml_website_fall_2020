[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1> What's Cooking  in Python</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

# What's Cooking in Python


This was adopted from. 
https://www.kaggle.com/manuelatadvice/whats-cooking/noname/code

#This imports a bunch of packages.  
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import json
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn import grid_search




#If you import the codes locally, this seems to cause some issues.  
import json
from urllib.request import urlopen

urltrain= 'https://raw.githubusercontent.com/RPI-Analytics/MGMT6963-2015/master/data/whatscooking/whatscookingtrain.json'
urltest = 'https://raw.githubusercontent.com/RPI-Analytics/MGMT6963-2015/master/data/whatscooking/whatscookingtest.json'


train = pd.read_json(urlopen(urltrain))
test = pd.read_json(urlopen(urltest))

#First we want to see the most popular cuisine for the naive model. 
train.groupby('cuisine').size()

#Here we write the most popular selection.  This is the baseline by which we will judge other models. 
test['cuisine']='italian'

#THis is a much more simple version that selects out the columns ID and cuisinte
submission=test[['id' ,  'cuisine' ]]
#This is a more complex method I showed that gives same.
#submission=pd.DataFrame(test.ix[:,['id' ,  'cuisine' ]])

#This outputs the file.
submission.to_csv("1_cookingSubmission.csv",index=False)
from google.colab import files
files.download('1_cookingSubmission.csv')


#So it seems there is some data we need to use the NLTK leemmatizer.  
stemmer = WordNetLemmatizer()
nltk.download('wordnet')

 

train

#We see this in a Python Solution. 
train['ingredients_clean_string1'] = [','.join(z).strip() for z in train['ingredients']] 

#We also know that we can do something similar though a Lambda function. 
strip = lambda x: ' , '.join(x).strip() 
#Finally, we call the function for name
train['ingredients_clean_string2'] = train['ingredients'].map(strip)

#Now that we used the lambda function, we can reuse this for the test dataset. 
test['ingredients_clean_string1'] = test['ingredients'].map(strip)
 


#We see this in one of the solutions.  We can reconstruct it in a way that makes it abit easier to follow, but I found when doing that it took forever.  

#To interpret this, read from right to left. 
train['ingredients_string1'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['ingredients']]       
test['ingredients_string1'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in test['ingredients']]       




train['ingredients_string1']

ingredients = train['ingredients'].apply(lambda x:','.join(x))
ingredients

#Now we will create a corpus.
corpustr = train['ingredients_string1']
corpusts = test['ingredients_string1']
corpustr

#http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#You could develop an understanding based on each.  
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
vectorizerts = TfidfVectorizer(stop_words='english')

#Note that this doesn't work with the #todense option.  
tfidftr=vectorizertr.fit_transform(corpustr)
predictors_tr = tfidftr

#Note that this doesn't work with the #todense option.  This creates a matrix of predictors from the corpus. 
tfidfts=vectorizertr.transform(corpusts)
predictors_ts= tfidfts

#This is target variable.  
targets_tr = train['cuisine']


## Logistic Regression and Regularization.

- Regularlization can help us with the large matrix by adding a penalty for each parameter. 
- Finding out how much regularization via grid search (search across hyperparameters.)

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

```C : float, default: 1.0
Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.```

#Logistic Regression. 
parameters = {'C':[1, 10]}
#clf = LinearSVC()
clf = LogisticRegression()



predictors_tr

from sklearn.model_selection import GridSearchCV
#This uses that associated paramters to search a grid space. 
classifier = GridSearchCV(clf, parameters)
classifier=classifier.fit(predictors_tr,targets_tr)



#This predicts the outcome for the test set. 
predictions=classifier.predict(predictors_ts)

#This adds it to the resulting dataframe. 
test['cuisine'] = predictions

#This creates the submision dataframe
submission2=test[['id' ,  'cuisine' ]]

#This outputs the file.
submission2.to_csv("2_logisticSubmission.csv",index=False)
from google.colab import files
files.download('2_logisticSubmission.csv')

from sklearn.ensemble import RandomForestClassifier 



# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 10)



# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(predictors_tr,targets_tr)



# Take the same decision trees and run it on the test data
predictions = forest.predict(predictors_ts)

#This adds it to the resulting dataframe. 
test['cuisine'] = predictions

#This creates the submision dataframe
submission3=test[['id' ,  'cuisine' ]]
submission3.to_csv("3_random_submission.csv",index=False)

from google.colab import files
files.download('3_random_submission.csv')

ingredients = train['ingredients'].apply(lambda x:','.join(x))
ingredients
train

