# Assignment 2

## ![](https://github.com/rpi-techfundamentals/hm-01-starter/blob/master/notsaved.png?raw=1) Before you start working on this assignment please click File -> Save a Copy in Drive. 

Before you turn this problem in, make sure everything runs as expected. First, restart the kernel (in the menubar, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All).  You can speak with others regarding the assignment but all work must be your own. 

### This is a 30 point assignment.  

**You may find it useful to go through the notebooks from the course materials when doing these exercises.**

**If you attempt to fake passing the tests you will receive a 0 on the assignment and it will be considered an ethical violation.**

files = "https://github.com/rpi-techfundamentals/introml_website_fall_2020/raw/master/files/assignment2.zip" 
!pip install otter-grader && wget $files && unzip -o assignment2.zip

#Run this. It initiates autograding. 
import otter
grader = otter.Notebook()

## Exercise-Packages

This creates an Numpy array. Numpy is a common package that we will use to work with arrays. You can read more about Numpy [here](http://www.numpy.org/). 

```
a = np.array([2,3,4])
print(a)
```

To get this to work, you will have to make sure that the numpy(np) package is installed. 

(1) Verify that Numpy is installed. How did you know?  
Describe how you would you install it if it wasn't installed?


man1="""
Enter your answer here.
"""

(2) Fix the cell below so that `a` is a `numpy` array.

#Fix this code of q2. 
a = [5,6,7,8]
print(a, type(a))


grader.check('q02')

(3) Create a numpy array `b` with the values `12, 13, 14, 15`.



#<insert q3 code here>



grader.check('q03')

## Exercise - Operations on Variables

(4) Describe what happens when you multiply an integer times a boolean? 
What is the resulting type? Provide examples.


#You must assign your answer to q4_answer. 
man4="""
Enter your answer here.
"""

(5) Describe happens when you try to multiply an integer value times a null?

man5="""
Enter your answer here.
"""


(6) Take 5 to the power of 4 and assign it to a variable `c`. Then transform the variable `c` to a type `float`. 

#<insert q6 code here>

grader.check('q06')

## Exercise-Lists
Hint: [This link is useful.](https://docs.python.org/3/tutorial/datastructures.html#more-on-lists) as is the process of tab completion (using tab to find available methods of an object).

(7) Create a list `elist1` with the following values `1,2,3,4,5`.<br>


#<insert q7 code here>


grader.check('q07')

(8) Create a new list `elist2` by first creating a copy of `elist1` and then reversing the order.

*HINT, remember there is a specific function to copy a list.* 


#<insert q8 code here>


grader.check('q08')

(9) Create a new list `elist3` by first creating a copy of `elist1` and then adding `7, 8, 9` to the end. (Hint: Search for a different function if appending doesn't work.) 

#<insert q9 code here>


grader.check('q09')

(10) Create a new list `elist4` by first creating a copy of `elist3` and then insert `6` between `5` and `7`.

#<insert q10 code here>


grader.check('q10')

## Exercise-Sets/Dictionary

This [link to documentation on sets](https://docs.python.org/3/tutorial/datastructures.html#sets) may be useful.

(11) Create a set `eset1` with the following values (1,2,3,4,5).


#<insert q11 code here>


grader.check('q11')


(12) Create a new set `eset2` the following values (1,3,6).



#<insert q12 code here>


grader.check('q12')

(13) Create a new set `eset3` that is `eset1-eset2`.



#<insert q13 code here>

grader.check('q13')

(14) Create a new set `eset4` that is the union of `eset1+eset2`.


#<insert q14 code here>

grader.check('q14')


(15) Create a new set `eset5` that includes values that are in both `eset1` and `eset2` (intersection).



#<insert q15 code here>

grader.check('q15')

(16) Create a new dict `edict1` with the following keys and associated values: st1=45; st2=32; st3=40; st4=31.

*Hint: There is a good section on dictionaries [here](https://docs.python.org/3/tutorial/datastructures.html#dictionaries).


#<insert q16 code here>


grader.check('q16')


(17) Create a new variable `key1` where the value is equal to the value of dictionary edict1 with key `st3`.

#<insert q17 code here>

grader.check('q17')

# Exercise-Numpy Array

(18) Create a new numpy array `nparray1` that is 3x3 and all the number 3 (should be integer type).



#<insert q18 code here>


grader.check('q18')


(19) Create a new variable `nparray1sum` that sums all of column 0.


#<insert q19 code here>

grader.check('q19')

(20) Create a new variable `nparray1mean` that takes the average of column 0.


#<insert q20 code here>


grader.check('q20')

(21) Create a new numpy array `nparray2` that selects only column 1 of `nparray1` (all rows).


#<insert q21 code here>


grader.check('q21')

(22) Create a new numpy array `nparray3` that is equal to `nparray1` times `2` (you should not alter `nparray1`).


#<insert q22 code here>


grader.check('q22')

(23) Create a new numpy array nparray4 that is a verticle stack of `nparray1` and `nparray3`.

#<insert q23 code here>


grader.check('q23')

## Exercise-Pandas

For these you will need to import the iris dataset. You should find the file `iris.csv` in the main directory.  

While we showed 2 ways of importing a csv, you should use the `read_csv` method of Pandas to load the csv into a dataframe called `df`. 


#Load iris.csv into a Pandas dataframe df here.
#Check out the first few rows with the head command. 


(24) Create a variable `df_rows` that includes the number of rows in the `df` dataframe.

#<insert q24 code here>

grader.check('q24')

(25) Create a new dataframe `df_train` that includes the first half of the `df` dataframe. Create a new dataframe `df_test` that includes the second half. 

#<insert q25 code here>


grader.check('q25')

(26) Create a new Pandas Series `sepal_length` from the `sepal_length` column of the df dataframe.

#<insert q26 code here>


grader.check('q26')


(27) Using, the Iris dataset, find the mean of the `sepal_length` series in our sample and assign it to the `sepal_length_mean` variable. You should round the result to 3 digits after the decimal. 

```
#Round example
a=99.9999999999999
#For example, the following will round a to 2 digits.
b = round(a,2)   

```

#<insert q27 code here>


grader.check('q27')


## MAKE SURE THAT THIS ENTIRE NOTEBOOK RUNS WITHOUT ERRORS. TO TEST THIS DO RUNTIME --> RESTART AND RUN ALL

It should run without errors.  


### Click File -> Download .ipynb to download the assignment.  Then Upload it to  the LMS.

