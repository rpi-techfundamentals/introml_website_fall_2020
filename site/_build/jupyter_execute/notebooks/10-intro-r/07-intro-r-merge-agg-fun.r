[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - Merging and Aggregating Data</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>


## Overview
- Merging Dataframes 
- Aggregating Dataframes
- Advanced Functions


## Merging Data Frame with Vector
- Can combine vector with data frame in multiple ways. 
- `data.frame(a,b)` where a & b can be vectors, matrices, or data frames. 

#Below is the sample data we will be creating 2 dataframes  
key=(1:10)

#Here we are passing the row names and column names as a list. 
m<- data.frame(matrix(rnorm(40, mean=20, sd=5), nrow=10, ncol=4, dimnames=list((1:10),c("a","b","c","d"))))
m2<- data.frame(matrix(rnorm(40, mean=1000, sd=5), nrow=10, ncol=4, dimnames=list((1:10),c("e","f","g","h"))))

#This is one way of combining a vector with a dataframe. 
df<-  data.frame(key,m)
df2<- data.frame(key,m2)

#This is another way way of combining a vector with a dataframe. 
dfb<-  cbind(key,m)
df2b<- cbind(key,m2)

df
df2
dfb
df2b


## Merging Columns of Data Frame with another Data Frame
- Can combine data frame in multiple ways. 
- `merge(a,b,by="key")` where a & b are dataframes with the same keys.
- `cbind(a,b)` where a & b are dataframes with the same number of rows.

# This manages the merge by an associated key.
df3 <- merge(df,df2,by="key")
# This just does a "column bind" 
df4<- cbind(df,df2)
df5<- data.frame(df,df2)
df3
df4
df5

## Merging Rows of Data Frame with another Data Frame
- `rbind(a,b)` combines rows of data frames of a and b.
- `rbind(a,b, make.row.names=FALSE)` this will reset the index.

#Here we can combine rows with rbind. 
df5<-df
#The make Row
df6<-rbind(df,df5)
df6
df7<-rbind(df,df5, make.row.names=FALSE)
df7

df7

## `aggregate` and `by`
- Aggregation is a very important function.
- Can have variables/analyses that happen at different levels.
- `by(x, by, FUN)` provides similar functionality.

iris=read.csv(file="../../input/iris.csv", header=TRUE,sep=",")
head(iris)

iris<-read.csv(file="../../input/iris.csv", header=TRUE,sep=",")

#Aggregate by Species  aggregate(x, by, FUN, ...)
iris.agg<-aggregate(iris[,1:4], by=list("species" = iris$species), mean)
print(iris.agg)

#Notice this gives us the same output but structured differently. 
by(iris[, 1:4], iris$species, colMeans)

## `apply`(plus `lapply`/`sapply`/`tapply`/`rapply`)
- `apply` - Applying a function to **an array or matrix**, return a vector or array or list of values. `apply(X, MARGIN, FUN, ...)`
- [`lapply`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/lapply.html) - Apply a function to **each element of a list or vector**, return a **list**. 
- [`sapply`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/lapply.html) - A user-friendly version if `lapply`. Apply a function to **each element of a list or vector**, return a **vector**.
- `tapply` - Apply a function to **subsets of a vector** (and the subsets are defined by some other vector, usually a factor), return a **vector**. 
- `rapply` - Apply a function to **each element of a nested list structure, recursively,** return a list.
- Some functions aren't vectorized, or you may want to use a function on every row or column of a matrix/data frame, every element of a list, etc.
- For more info see this [tutorial](https://nsaunders.wordpress.com/2010/08/20/a-brief-introduction-to-apply-in-r/)


## `apply`
- `apply` - Applying a function to **an array or matrix**, return a vector or array or list of values. `apply(X, MARGIN, FUN, ...)`
- If you are using a data frame the data types must all be the same. 
- `apply(X, MARGIN, FUN, ...) where X is an array or matrix. 
- `MARGIN` is a vector giving the where function should be applied. E.g., for a matrix 1 indicates rows, 2 indicates columns, c(1, 2) indicates rows and columns.
- `FUN` is any function.  

iris<-read.csv(file="../../input/iris.csv", header=TRUE,sep=",")
iris$sum<-apply(iris[1:4], 1, sum) #This provides a sum across  for each row. 
iris$mean<-apply(iris[1:4], 1, mean)#This provides a mean across collumns for each row. 
head(iris)
apply(iris[1:4], 2, mean)

## `lapply` & `sapply`
- [`lapply`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/lapply.html) - Apply a function to **each element of a list or vector**, return a **list**.
- `lapply(X, FUN, ...)`
- [`sapply`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/lapply.html) - A user-friendly version if `lapply`. Apply a function to **each element of a list or vector**, return a **vector**.
- `sapply(X, FUN, ...)`

# create a list with 2 elements
sample <- list("count" = 1:5, "numbers" =5:10)

# sum each and return as a list. 
sample.sum<-lapply(sample, sum)

class(sample.sum)
print(c(sample.sum, sample.sum["numbers"],sample.sum["count"]))


# create a list with 2 elements
sample <- list("count" = 1:5, "numbers" =5:10)

# sum each and return as a list. 
sample.sum<-sapply(sample, sum)

class(sample.sum)
print(c(sample.sum, sample.sum["numbers"],sample.sum["count"],sample.sum[["count"]]))

#Note the differenece between #sample.sum[["count"]] and sample.sum["count"]

# We can also utilize simple 
square<-function(x) x^2
square(1:5)

# We can use our own function here.     
sapply(1:10, square)

#We can also specify the function directly in sapply.
sapply(1:10, function(x) x^2)


## `tapply` 
- `tapply` - Apply a function to subsets of a vector (and the subsets are defined by some other vector, usually a factor), return a vector.
- Can do something similar to aggregate. 

#Tapply example
#tapply(X, INDEX, FUN, …) 
#X = a vector, INDEX = list of one or more factor, FUN = Function or operation that needs to be applied. 
iris<-read.csv(file="../../input/iris.csv", header=TRUE,sep=",")
iris.sepal_length.agg<-tapply(iris$sepal_length, iris$species, mean)
print(iris.sepal_length.agg)



## CREDITS


Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016.
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.
This work is adopted from the Berkley R Bootcamp.  

