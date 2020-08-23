[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - Functions </h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

## Overview
- Why functions?
- Predefined functions
- Custom functions
- Exercises

## R is a Functional Language

- Operations are carried out with functions. Functions take objects as inputs and return objects as outputs. 
- An analysis can be considered a pipeline of function calls, with output from a function used later in a subsequent operation as input to another function.
- Functions themselves are objects. 

## Why Functions?
- Code reuse. 
- Abstract away complexity. 
- Simple, efficient robust code.
- Specific functional programming languages like Lisp & Haskell built around *functional programming*, which enforces great practices.
- Read more about functional programming in Python [here](http://www.ibm.com/developerworks/library/l-prog/).

## Predefined Functions
- We have used predefined functions in earlier exercises
- R has predefined functions for embedded data structures like vectors and data frames.
- See [here](http://www.statmethods.net/management/functions.html) for a list of R functions.


#Simple Rounding Functions
a<-3.14
a<-round(a)
a

#Function to create 5 random numbers and assign to vector. 
random <- rnorm(5)
random

## Using Functions

- Functions generally take arguments, some of which are often optional
- To get information about a function you know exists, use `help` or `?`, e.g., `?lm`. For information on a general topic, use `apropos` or `??`

#Functions generally take arguments, some of which are often optional:
random <- rnorm(5)
median(random)
median(random, na.rm = TRUE)
help(lm)
?lm

?log


## Custom Functions in R
- The fuction in R doesn't demend on white space, but it functions just like it did in Python. 
- It returns a value using the `return` command, which should be the last command in the function. 



#This defines a function called "addTwo."

a = 1000000
addTwo <- function(a, b){
c<-a+b
return(c)
}

d<-addTwo(5,10)
d


#we can use the short form that pu
square<-function(x) x^2
square(1:5)
addtwo<-function(a,b) a+b
addtwo(5,9)

## Functional Programming

- Functions are, like everything else in R, object.
- Functions can passed around just like any other value.
- That means we can do really cool things, like pass a function to a function. 





print(addTwo)
class(addTwo)



#Functions are objects and can be assigned another value.
addTwob<-addTwo
print(addTwob)
str(addTwob)
d<-addTwob(5,10)
d







Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016.
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.
Adopted from [Berkley R Bootcamp](https://github.com/berkeley-scf/r-bootcamp-2016). 