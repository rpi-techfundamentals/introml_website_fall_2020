[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - Datastructures</h1></center>
<center><h3><a href = "http://rpi.analyticsdojo.com">rpi.analyticsdojo.com</a></h3></center>



# Overview
Common to R and Python
- Vectors
- Opearations on Numeric and String Variables
- Lists




## Vectors in R
- The most basic form of an R object is a vector. 
- In fact, individual (scalar) values (variables) are vectors of length one.
- An R vector is a single set of values in a particular order of the **same type**. 
- We can concatenate values into a vector with c(): `ages<-c(18,19,18,23)`
- Comparable Python objects include Panda Series and single dimensional numpy array. 
- While Python arrays start at 0, R arrays start at index position 1. 


ages<-c(18,19,18,23)
ages
ages[1]
ages[2:4]



## Vectors Type in R
- Items in a vector must be of the same type. 
- *Character.* These are the clear character vectors. (Typically use quotes to add to these vectors.)
- *Numeric.* Numbers in a set. Note there is not a different type.
- *Boolean.* TRUE or FALSE values in a set.
- *Factor.* A situation in which there is a select set of options. Things such as states or zip codes. These are typically things which are related to dummy variables, a topic we will discuss later.
- Determine the data type by using the `str` command: `str(teachers)`

names<-c("Sally", "Jason", "Bob", "Susy") #Text
female<-c(TRUE, FALSE, FALSE, TRUE)  #While Python uses True and False, R uses TRUE and FALSE.
teachers<-c("Smith", "Johnson", "Johnson", "Smith")
teachers.f<-factor(teachers)
grades<-c(20, 15, 13, 19) #25 points possible
gradesdec<-c(20.32, 15.32, 13.12, 19.32) #25 points possible

str(names)
str(female)
str(teachers)  
str(teachers.f) 
str(grades)    #Note that the grades and gradesdec are both numeric.
str(gradesdec) #Note that the grades and gradesdec are both numeric.


## Strings in R
- Lot's of different types of operations we can perform on Strings. 

chars <- c('hi', 'hallo', "mother's", 'father\'s', "He said, \'hi\'" )
length(chars)
nchar(chars)
paste("bill", "clinton", sep = " ")  # paste together a set of strings
paste(chars, collapse = ' ')  # paste together things from a vector

strlist<-strsplit("This is the Analytics Dojo", split = " ") #This taks a string ant splits to a list
strlist
substring(chars, 2, 3) #this takes the 2nd-3rd character from the sentance above 
chars2 <- chars
substring(chars2, 2, 3) <- "ZZ"  #this takes the 2nd-3rd character from the sentance above 
chars2

## Factors in R 
- A factor is a special data type in R used for categorical data. In some cases it works like magic and in others it is incredibly frustrating.



class(teachers.f) # What order are the factors in?
levels(teachers.f)  # note alternate way to get the variable
summary(teachers.f) #gives the count for each level. 


## Creating Vectors in R
- Concatenate fields to a vector: `nums <- c(1.1, 3, -5.7)`
- Generate random values from normal distribution with `devs <- rnorm(5)`
- idevs <- sample(ints, 100, replace = TRUE)


# numeric vector
nums <- c(1.1, 3, -5.7)
devs <- rnorm(5)
devs

# integer vector
ints <- c(1L, 5L, -3L) # force storage as integer not decimal number
# "L" is for "long integer" (historical)

idevs <- sample(ints, 100, replace = TRUE)

# character vector
chars <- c("hi", "hallo", "mother's", "father\'s", 
   "She said", "hi", "He said, \'hi\'" )
chars
cat(chars, sep = "\n")

# logical vector
bools <- c(TRUE, FALSE, TRUE)
bools

## Variable Type 
- In R when we write `b = 30` this means the value of `30` is assigned to the `b` object. 
- R is a [dynamically typed](https://pythonconquerstheuniverse.wordpress.com/2009/10/03/static-vs-dynamic-typing-of-programming-languages/).
- Unlike some languages, we don"t have to declare the type of a variable before using it. 
- Variable type can also change with the reassignment of a variable. 
- We can query the class a value using the `class` function.
- The `str` function gives additional details for complex objects like dataframes.



a <- 1L
print (c("The value of a is ", a))
print (c("The value of a is ", a), quote=FALSE)
class(a)
str(a)

a <- 2.5
print (c("Now the value of a is ", a),quote=FALSE)
class(a)
str(a)

a <- "hello there"
print (c("Now the value of a is ", a ),quote=FALSE)
class(a)
str(a)

## Converting Values Between Types

- We can convert values between different types.
- To convert to string use the `as.character` function.
- To convert to numeric use the `as.integer` function.
- To convert to an integer use the `as.integer` function.
- To convert to a boolean use the `as.logical` function.


#This is a way of specifying a long integer.
a <- 1L
a
class(a)
str(a)
a<-as.character(a)
a
class(a)
str(a)
a<-as.numeric(a)
a
class(a)
str(a)
a<-as.logical(a)
a
class(a)
str(a)


## Quotes 
- Double Quotes are preferred in R, though both will work as long as they aren't mixed. 

#Double Quotes are preferred in R, though both will work as long as they aren't mixed. 
a <- "hello"
class(a)
str(a)
a <- 'hello'
class(a)
str(a)

# Null Values

- Since it was designed by statisticians, R handles missing values very well relative to other languages.
- `NA` is a missing value


#Notice nothing is printed.
a<-NA
a
vec <- rnorm(12)    #This creates a vector with randomly distributed values
vec[c(3, 5)] <- NA  #This sets values 3 and 5 as NA
vec                 #This prints the Vector
sum(vec)            #What is the Sum of a vector that has NA?  
sum(vec, na.rm = TRUE)   #This Sums the vector with the NA removed. 
is.na(vec)          #This returns a vector of whether specific values are equal to NA.

## Logical/Boolean Vectors
- Here we can see that summing and averaging boolean vectors treats `TRUE=1 & FALSE=0`

answers <- c(TRUE, TRUE, FALSE, FALSE)
update <- c(TRUE, FALSE, TRUE, FALSE)

# Here we see that True coul
sum(answers)
mean(answers)
total<-answers + update
total
class(total)

## R Calculations
- R can act as a basic calculator.

2 + 2 # add numbers
2 * pi # multiply by a constant
7 + runif(1) # add a random number
3^4 # powers
sqrt(4^4) # functions
log(10)
log(100, base = 10)
23 %/% 2 
23 %% 2

# scientific notation
5000000000 * 1000
5e9 * 1e3

## Operations on Vectors
- R can be used as a basic calculator.
- We can do calculations on vectors easily. 
- Direct operations are much faster easier than looping.


#vals <- rnorm(10)
#squared2vals <- vals^2
#sum_squared2vals <- sum(chi2vals)
#ount_squared2vals<-length(squared2vals)  
#vals
#squared2vals
#sum_df1000
#count_squared2vals


## R is a Functional Language

- Operations are carried out with functions. Functions take objects as inputs and return objects as outputs. 
- An analysis can be considered a pipeline of function calls, with output from a function used later in a subsequent operation as input to another function.
- Functions themselves are objects. 
- We can get help on functions with help(lm) or ?lm

vals <- rnorm(10)
median(vals)
class(median)
median(vals, na.rm = TRUE)
mean(vals, na.rm = TRUE)
help(lm)
?lm
?log



## Matrix 
- Multiple column vector
- Matrix is useful for linear algebra
- Matrix must be **all of the same type**
- Could relatively easily do regression calculations using undlerlying matrix

#This is setup a matrix(vector, nrow,ncol)
mat <- matrix(rnorm(12), nrow = 3, ncol = 4)
mat
# This is setup a matrix(vector, rows)
A <- matrix(1:12, 3)
B <- matrix(1:12, 4)
C <- matrix(seq(4,36, by = 4), 3)
A
B
C

## Slicing Vectors and Matrixs
- Can use matrix[rows,columns] with specificating of row/column index, name, range. 


vec <- rnorm(12)
mat <- matrix(vec, 4, 3)
rownames(mat) <- letters[1:4] #This assigns a row name
vec
mat



#Slicing Vector
vec[c(3, 5, 8:10)] # This gives position 3, 5, and 8-10

# matrix[rows,columns]  leaving blank means all columns/rows
mat[c('a', 'd'), ]
mat[c(1,4), ]
mat[c(1,4), 1:2]
mat[c(1,4), c(1,3)] #Notice when providing a list we surround with c
mat[, 1:2]          #When providing a range we use a colon.

## Lists
- Collections of disparate or complicated objects.
- Can be of multiple different types. 
- Here we assign individual values with the list with `=`.
- Slice the list with the index position or the name. 

#myList <- list(stuff = 3, mat = matrix(1:4, nrow = 2), moreStuff = c('china', 'japan'), list(5, 'bear'))
myList<-list(stuff=3,mat = matrix(1:4, nrow = 2),vector=c(1,2,3,4),morestuff=c("Albany","New York", "San Francisco"))
myList

#
myList['stuff']
myList[2]
myList[2:3]
myList[c(1,4)]


## CREDITS

Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.

Adopted from the [Berkley R Bootcamp](https://github.com/berkeley-scf/r-bootcamp-2016).



