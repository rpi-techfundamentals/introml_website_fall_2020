## Introduction to Python Exercises

Before you turn this problem in, make sure everything runs as expected. First, restart the kernel (in the menubar, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All).  You can speak with others regarding the assignment but all work must be your own. 


### This is a 30 point assignment graded from answers to questions and automated tests that should be run at the bottom. Be sure to clearly label all of your answers and commit final tests at the end. If you attempt to fake passing the tests you will receive a 0 on the assignment and it will be considered an ethical violation. (Note, not all questions have tests).


!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/train.csv
!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/test.csv

NAME = "Jason Kuruzovich"
COLLABORATORS = "Alyssa Hacker"  #You can speak with others regarding the assignment, but all typed work must be your own.

 
**If you attempt to fake passing the tests you will receive a 0 on the assignment and it will be considered an ethical violation.**

## Exercises - For and If.

(1). Write a for loop which create a list called `fivetoten` of all numbers from 5 to 10 (inclusive).


#Answer question 1 here.


(2). Write a program which uses a for loop and if statements to create a list called `divby7` of all numbers from 1-50 that are divisible by 7.
Hint: 14 is divisible by 7 if 14%7==0.

#Answer question 2 here.

(3). Write a program which uses a for loop and if statements create a list `divby7not5` of all numbers which are divisible by 7 but are not a multiple of 5, between 10000 and 10100 (both included). 


#Answer question 3 here.

## Exercises - Functions

(4). Create a function `divby2` that accepts a list and returns all values from that list that are divisible by 2.  For example, passing the list  `numbers = [3, 12, 91, 33, 21, 34, 54, 34, 34, 54]` should return a list. Generate a new list `divby2` that includes only numbers that are divisible by 2. 

#Define your function for question (4) here. 

#Execute this code to assign divby2 to the correct values. 
numbers = [3, 12, 91, 33, 21, 34, 54, 34, 34, 54]
divby2=divby2(numbers)
print(divby2)



(5) Create an external module for your `divby2` function called `myutilities.py`.  Import myutilities as mu, such that that following runs.



#After importing this should work. 
#Add code to re-import the module using the example in class
import myutilities as mu
divby2mod=mu.divby2(numbers)

## Exercises-Titanic

The following exercises will use the titanic data from Kaggle.  I've included it in the input folder just like Kaggle. 

import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# Let's input them into a Pandas DataFrame
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")



(6). While we can submit our answer to Kaggle to see how it will perform, we can also utilize our test data to assess accuracy. Accuracy is the percentage of predictions made correctly-i.e., the percentage of people in which our prediction regarding their survival. <br>Create columns in the training dataset `PredEveryoneDies` and `PredGender` with the same predictions which were included in the example notebook (06-intro-kaggle-baseline in the materials repository).   
  



#Answer

### YOU CAN DO THIS!
- For the next question we have to combine a few bits of data. For the first time we will be connecting multiple operations.   
- First, you should find out the mathematical definition for accuracy. OK, here is a [link](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification). 
- Next, consider how you would program accuracy? 
- Next, try it with some sample data where you know the answer. I've created that below.
- Next, hand calculate the accuracy for the sample data.
- Next, programatically find the accuracy for the sample data.  How might you do this using the tools that you have?  (a) create a new column where for the Training set `Survived==PredEveryoneDies` it equals 1 if true and 0 if false. (b) Sum the number of 1s and (c)  divide by the total number of records in the training set.  
- Finally, turn your calculations of accuracy into a function so you can reuse it in the next step. 

import pandas as pd
example_data = [{'predicted': 1, 'actual': 1},
         {'predicted': 1,  'actual': 0},
         {'predicted': 0,  'actual': 1},
         {'predicted': 0,  'actual': 0}]
#What should 
df = pd.DataFrame(example_data)
df

(7) Use your function to create varaibles `AccEveryoneDies` and `AccGender`.  `AccEveryoneDies` should be the accuracy of the EveryoneDies model in the Train dataset. Similarly, `AccGender` is the a accuracy of the Gender (women survive) model in the Train dataset. 




#Answer question 7 here.

(8). Create a `generate_submission` function that accepts a DataFrame, a target column, and a filename and writes out the submission file with just the `passengerID` and the `Survived` columns, where the survived column is equal to the target column. It should then return a DataFrame with the `passengerID` and the `Survived` columns.

Executeing the following:
`submitdie = generate_submission(train, 'PredEveryoneDies', 'submiteveryonedies.csv')`

Should return a dataframe with just `passengerID` and the `Survived` column.  

Create a prediction file for the "PredEveryoneDies" model using the test dataset and upload it to Kaggle.  Put that screenshot in this repository of what happens. Sometimes Kaggle won't give you a score if the exact same file has been submitted by someone else.  Don't worry about that. 

The syntax for including a markdown picture is shown below. 

```
![](myscreenshot.png)
```

You will have to change the cell type to a markdown cell below.  You can see more [here](https://www.youtube.com/watch?v=xlD8FIM5biA).




def generate_submission...
...

submitdie = generate_submission(test, 'PredEveryoneDies', 'submiteveryonedies.csv')

#Change this to a markdown cell and insert your picture of you
#Kaggle upload here.  Use this code: 


`![](myscreenshot.png)`





(9). In our initial calculation of the `PredGender` column, we made our prediction based on whether the individual was male or female.  In accordance to the [women and children first](https://en.wikipedia.org/wiki/Women_and_children_first) protocol, we hypothesize that our model could be improved by including whether the individual was a child in addition to gender. We also have a question, what age to use to determine "child"? (People weren't likely to check for IDs.)  We will check 2 ages...<13 and <18 (somewhat arbitrary but have to start somewhere) and see which yields a better accuracy. <br>

*After* coding survival based on gender, update your recommendation to prediction in the training dataset survival based on *age.* In other words, your model should predict that a male child would survive. If you first code for age and then code by gender, the prediction will be off. <br> 

Specifically:

`train['PredGenderAge13']` should be the prediction incorporating both Gender and whether Age < 13 (i.e., <13 survived while >=13 died)  <br>
`train['PredGenderAge18']` should be the prediction incorporating both Gender and whether Age < 18 (i.e., <18 survived while >=18 died).


#Complete requirements for #9 here.

(10). Calculate the accuracy for your new predictions.  Use `PredGenderAge13` in the training set to calculate `AccGenderAge13` (you can use your function again!) and `PredGenderAge18` to calcuate `AccGenderAge18`. 

#complete #10 here.

(11). You should find that the `AccGenderAge13` is better (has a higher accuracy) than `AccGenderAge18`. Create a new column `child` in the `test` and `train` DataFrames that is 1 if `Age < 13` and `0` otherwise. This is a feature. We will talk more about features next time.

#complete #11 here.

(12). Create a prediction file for the "women and children first" model in using the test dataset and upload it to Kaggle. Go through the process of uploading it to Kaggle. Put that screenshot in this repository of what happens. Sometimes Kaggle won't give you a score if the exact same file has been submitted by someone else.  Don't worry about that.  The syntax for including a markdown picture is shown below.  

```
![](myscreenshot.png)
```

You will have to change the cell type to a markdown cell below.  

#Change this to a markdown cell and insert your picture of you
#Kaggle upload here.

### (13) How would you compare the final "women and children" first model with the initial baseline mode ("everyone died)" ?  
Include change in accruacy and how you would interpret this intial analysis.  

#complete #13 here.

## Final Tests
- These final tests will confirm you did exercises correctly. 
- First you need to install the ipython_unittest package 
- Then you load the extensions.
- Then you run the tests.  
- Try to work through the exercises, inspecting your own results.  Then confirm with the tests.

!pip install ipython_unittest 

%load_ext ipython_unittest
#This runs tests against your b array.  If you complete the assingment correctly, you will pass the tests.

### Run the cells below before submission. 

%%unittest_main
class TestExercise3(unittest.TestCase):
    def test_forif1(self):
        self.assertTrue(fivetoten == [5,6,7,8,9,10])
    def test_forif2(self):
        self.assertTrue(divby7 == [7,14,21,28,35,42,49])
    def test_forif(self):
        self.assertTrue(divby7not5 == [10003, 10017, 10024, 10031, 10038, 10052, 10059, 10066, 10073, 10087, 10094])
    def test_functions(self):
        self.assertTrue(divby2 == [12, 34, 54, 34, 34, 54])
    def test_functions(self):
        self.assertTrue(divby2mod == [12, 34, 54, 34, 34, 54])
    def test_titanic1(self):
        self.assertAlmostEqual(AccEveryoneDies, 61.6161616162)
    def test_titanic2(self):
        self.assertAlmostEqual(AccGender, 78.6756453423)
    def test_titanic3(self):
        self.assertAlmostEqual(train['PredEveryoneDies'].mean(), 0.0)
    def test_titanic4(self):
        self.assertAlmostEqual(train['PredGender'].mean(), 0.35241301908)
    def test_titanic5(self):
        self.assertTrue(['PassengerId', 'Survived']==list(pd.read_csv('submiteveryonedies.csv').columns.values))
    def test_titanic6(self):
        self.assertAlmostEqual(train['PredGenderAge13'].mean(), 0.393939393939)
    def test_titanic7(self):
        self.assertAlmostEqual(train['PredGenderAge18'].mean(), 0.417508417508)
    def test_titanic8(self):
        self.assertAlmostEqual(AccGenderAge13, 79.2368125701)
    def test_titanic9(self):
        self.assertAlmostEqual(AccGenderAge18, 77.3288439955)
    def test_titanic10(self):
        self.assertTrue(train['child'].sum()==69)
    def test_titanic11(self):
        self.assertTrue(test['child'].sum()==25)

#This is a collection of all of the tests from the exercises above. 