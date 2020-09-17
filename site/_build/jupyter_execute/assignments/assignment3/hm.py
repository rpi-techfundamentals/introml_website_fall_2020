# Assignment 3

![](https://github.com/rpi-techfundamentals/hm-01-starter/blob/master/notsaved.png?raw=1)

**Save your working file in Google drive so that all changes will be saved as you work. MAKE SURE that your final version is saved to GitHub.** 

Before you turn this in, make sure everything runs as expected. First, restart the kernel (in the menu, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All).  You can speak with others regarding the assignment but all work must be your own. 


## This is a 30 point assignment.

**You may find it useful to go through the notebooks from the course materials when doing these exercises.**

**If you attempt to fake passing the tests you will receive a 0 on the assignment and it will be considered an ethical violation.**


files = "https://github.com/rpi-techfundamentals/hm-03-starter/raw/master/files.zip"
!pip install otter-grader && wget $files && unzip -o files.zip

#Run this. It initiates autograding. 
import otter
grader = otter.Notebook()

## Exercises - For and If and Functions

(1). Create a function `list_step` that accepts 3 variables (`start`, `stop`, `step`).  The function returns a list starting at `start`, ending at `stop`, and with a step of `step`.

For example:

`list_step(5, 19, 2)`

Should return: 

`[5, 7, 9, 11, 13, 15, 17]`


#Answer question 1 here.


list1= list_step(5, 19, 2) #include this code
grader.check('q01')

(2). Create a function `list_divisible` that accepts 3 variables (`start`, `stop`, `divisible`).  Use a for loop to create a list of all numbers from `start` to `stop` which are divisible by `divisible`.

For example:

`list_divisible(1, 50, 7)`

Should return: 

`[7, 14, 21, 28, 35, 42, 49]`

#Answer question 2 here.


list2= list_divisible(1, 50, 7) #include this code
grader.check('q02')


(3). Create a function `list_divisible_not` that accepts 4 variables (`start`, `stop`, `divisible`, `not_divisible`).  Use a for loop to create a list of all numbers from `start` to `stop` which are divisible by `divisible` but not divisible by `not_divisible`.

For example:

`list_divisible_not(1, 100, 4, 3)`

Should return: 

`[4, 8, 16, 20, 28, 32, 40, 44, 52, 56, 64, 68, 76, 80, 88, 92]`


#Answer question 3 here.


list3= list_divisible_not(1, 100, 4, 3) #include this code
grader.check('q03')

## Exercises-Titanic

The following exercises will use the titanic data from Kaggle.  I've included it in the input folder just like Kaggle. 

import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# Let's input them into a Pandas DataFrame
train = pd.read_csv("./input/train.csv")
test  = pd.read_csv("./input/test.csv")

(4) What is the key difference between the train and the test?

man4="""

"""

(5) Create a new column `family` that is equal to the `SibSp` * `Parch` for both the train and the test dataframes.  DON'T use a `for` loop. 


#Answer 

grader.check('q05')


(6). While we can submit our answer to Kaggle to see how it will perform, we can also utilize our training data to assess accuracy. Accuracy is the percentage of predictions made correctly-i.e., the percentage of people in which our prediction regarding their survival is correct. In other words, accuracy = (#correct predictions)/(Total # of predictions). Create a function `generate_accuracy` which accepts two Pandas series objects (`predicted`, `actual`) and returns the accuracy.  

For example, when a and b are two different Pandas Series: 
`generate_accuracy(predicted, actual)`

For the sample data below, the data should retun `50.0` (i.e., a percentage).



#Example DATA
import pandas as pd
ex = [{'predicted': 1, 'actual': 1},
         {'predicted': 1,  'actual': 0},
         {'predicted': 0,  'actual': 1},
         {'predicted': 0,  'actual': 0} ]

df = pd.DataFrame(ex)
df

#Answer



grader.check('q06')

(7) Create a column `PredEveryoneDies` which is equal to 0 for everyone in both training and testing datasets. 

#Answer

grader.check('q07')

(8) Find the accuracy of `PredEveryoneDies` in predicting `Survived` using the function `generate_accuracy` that you created earlier and assign it to the `AccEveryoneDies` variable. 

# Answer


grader.check('q08')

(9) In both the training and testing datasets, create the column `PredGender` that is 1 -- if the person is a woman and 0 -- if the person is a man. (This is based on the "women and children first" law of shipwrecks).  Then set `AccGender` to the accuracy of the `PredGender` in the Train dataset.


#Answer


grader.check('q09')

(10). Create a `generate_submission` function that accepts a DataFrame, a target column, and a filename (`df`, `target`, `filename`) and writes out the submission file with just the `passengerID` and the `Survived` columns, where the `Survived` column is equal to the target column.

For example:
`submitdie = generate_submission(train, 'PredEveryoneDies', 'submiteveryonedies.csv')`

Should return a dataframe with just `passengerID` and the `Survived` column.  

**Make sure your submission file prediction for Survived is an integer and not at float. If you submit a float it may not work.**


#Answer


grader.check('q10')

(11).  To use the [women and children first](https://en.wikipedia.org/wiki/Women_and_children_first) protocol, we will need to use the age field. This has some missing values. We are going to replace null values in the train and test set with the median value for each.  


For this particular question: 

Set the variables `AgeMissingTrain` and `AgeMissingTest` using the count of the number of missing values in the train and test sets, respectively.

Set the variable `AgeMedianTrain` and `AgeMedianTest` using the median age of the train and test sets, respectively. 

#Answer


grader.check('q11')

(12) For rows in which the age value is  missing, set the age to the appropriate median value for the train/test set. 

#Answer


grader.check('q12')


(13). In our initial calculation of the `PredGender` column, we made our prediction based on whether the individual was male or female.  In accordance to the [women and children first](https://en.wikipedia.org/wiki/Women_and_children_first) protocol, we hypothesize that our model could be improved by including whether the individual was a child in addition to gender. We also have a question, what age to use to determine "child"? (People weren't likely to check for IDs.)  We will check 2 ages...<13 and <18 (somewhat arbitrary but have to start somewhere) and see which yields a better accuracy. <br>

<br> 

Specifically, create 2 predictions as follows:

`train['PredGenderAge13']` should be the prediction incorporating both Gender (women survive) and Age (Children Age<13 survived while Age>=13 died)  <br>
`train['PredGenderAge18']` should be the prediction incorporating both Gender (women survive) and Age (Children Age<18 survived while Age>=18 died)  <br>

*The analysis assumes that you have addressed missing values in the earlier step and you should do it for both the train and test dataframes*

#Answer


grader.check('q13')

(14). Calculate the accuracy for your new predictions.  Use `PredGenderAge13` in the training set to calculate `AccGenderAge13` (you can use your function again!) and `PredGenderAge18` to calcuate `AccGenderAge18`. 

#Answer


grader.check('q14')

(15). You should find that the accuracy is higher when using 13 as a designation for a child than 18. What does this tell you about the role of age in surviving a shipwreck? 


man15="""
Answer here.
"""

(16) Create a prediction file for the "women and children first" model in using the test dataset and upload it to Kaggle. Go through the process of uploading it to Kaggle. Put your Kaggle username so we can verify your prediction occued. 

**Make sure your submission file prediction is an integer and not at float. If you submit a float it may not work.**

kaggle_username="***your kaggle username****"
man15="https://www.kaggle.com/"+kaggle_username+"/competitions"
#This gives a link to your competition page
man15