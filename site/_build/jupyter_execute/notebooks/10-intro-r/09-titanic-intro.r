[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://www.analyticsdojo.com)
<center><h1>Introduction to R - Titanic Baseline </h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

## Running Code using Kaggle Notebooks
- Kaggle utilizes Docker to create a fully functional environment for hosting competitions in data science.
- You could download/run kaggle/python docker image from [GitHub](https://github.com/kaggle/docker-python) and run it as an alternative to the standard Jupyter Stack for Data Science we have been using.
- Kaggle has created an incredible resource for learning analytics.  You can view a number of *toy* examples that can be used to understand data science and also compete in real problems faced by top companies. 

train <- read.csv('../../input/train.csv', stringsAsFactors = F)
test  <- read.csv('../../input/test.csv', stringsAsFactors = F)

## `train` and `test` set on Kaggle
- The `train` file contains a wide variety of information that might be useful in understanding whether they survived or not. It also includes a record as to whether they survived or not.
- The `test` file contains all of the columns of the first file except whether they survived. Our goal is to predict whether the individuals survived.

head(train)

head(test)

## Baseline Model: No Survivors
- The Titanic problem is one of classification, and often the simplest baseline of all 0/1 is an appropriate baseline.
- Even if you aren't familiar with the history of the tragedy, by checking out the [Wikipedia Page](https://en.wikipedia.org/wiki/RMS_Titanic) we can quickly see that the majority of people (68%) died.
- As a result, our baseline model will be for no survivors.

test["Survived"] <- 0

submission <- test[,c("PassengerId", "Survived")]

head(submission)

# Write the solution to file
write.csv(submission, file = 'nosurvivors.csv', row.names = F)

## The First Rule of Shipwrecks
- You may have seen it in a movie or read it in a novel, but [women and children first](https://en.wikipedia.org/wiki/Women_and_children_first) has at it's roots something that could provide our first model.
- Now let's recode the `Survived` column based on whether was a man or a woman.  
- We are using conditionals to *select* rows of interest (for example, where test['Sex'] == 'male') and recoding appropriate columns.

#Here we can code it as Survived, but if we do so we will overwrite our other prediction. 
#Instead, let's code it as PredGender

test[test$Sex == "male", "PredGender"] <- 0
test[test$Sex == "female", "PredGender"] <- 1

submission = test[,c("PassengerId", "PredGender")]
#This will Rename the survived column
names(submission)[2] <- "Survived"
head(submission)

names(submission)[2]<-"new"
submission

write.csv(submission, file = 'womensurvive.csv', row.names = F)