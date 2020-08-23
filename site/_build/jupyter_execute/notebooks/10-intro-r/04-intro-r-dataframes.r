[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - DataFrames</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>



## Introduction to R DataFrames
- Data frames are combinations of vectors of the same length, but can be of different types.
- It is a special type of list.  
- Data frames are what is used for standard rectangular (record by field) datasets, similar to a spreadsheet
- Data frames are a functionality that both sets R aside from some languages (e.g., Matlab) and provides functionality similar to some statistical packages (e.g., Stata, SAS) and Python's Pandas Packages.


frame=read.csv(file="../../input/iris.csv", header=TRUE,sep=",")
class(frame)
head(frame) #The first few rows.
tail(frame) #The last few rows.
str(frame) #The Structure.



dim(frame) #Results in rows x columns
nrow(frame)  #The number of Rows
names(frame) #Provides the names
length(frame) #The number of columns
summary(frame) #Provides summary statistics.
is.matrix(frame) #Yields False because it has different types.  
is.list(frame) #Yields True
class(frame$sepal_length)
class(frame$species)
levels(frame$species)

frame[c("species","sepal_width")]

frame['petals']<-0
frame$petals2<-0
head(frame)

mean.sepalLenth.setosa<-mean(frame[,'sepal_length'])

## Slicing a Dataframe by Column
- Remember the syntax of `df[rows,columns]` 
- Using `dataframe$column` provides one way of selecting a column. 
- We can also specify the index position: `dataframe[,columnIndex]`
- We can also specify the column name: `dataframe[,columnsName]`

sepal_length1<-frame$sepal_length #Using Dollar Sign and the column name.
sepal_length2<- frame[,1]  #Using the Index Location
sepal_length3<- frame[,'sepal_length']
sepal_length4<- frame[,c('sepal_length','sepal_width')]

sepal_length1[1:5]  #Print the first 5  
sepal_length2[1:5]
sepal_length3[1:5]


## Selecting Rows
- We can select rows from a dataframe using index position: `dataframe[rowIndex,columnIndex]`. 
- Use `c(row1, row2, row3)` to select out specific rows. 

frame2<-frame[1:20,]   
frame3<-frame[c(1,5,6),] #This selects out specific rows
nrow(frame2)
nrow(frame3)
frame3

## Conditional Statements and Dataframes with Subset
- We can select subsets of a dataframe by putting an equality in the row or subset. 
- Subset is also a dataframe. 
- Can optionally select columns with the `select = c(col1, col2)`

setosa.df <- subset(frame, species == 'setosa')

head(setosa.df)
class(setosa.df)
nrow(setosa.df)
mean.sepalLenth.setosa<-mean(setosa.df$sepal_length) #This creates a new vector
mean.sepalLenth.setosa
setosa.df.highseptalLength <- subset(setosa.df, sepal_length > mean.sepalLenth.setosa)
nrow(setosa.df.highseptalLength)
head(setosa.df.highseptalLength)
setosa.dfhighseptalLength2 <- subset(setosa.df, sepal_length > mean.sepalLenth.setosa, select = c(sepal_length, species))
head(setosa.dfhighseptalLength2)

## Subsetting Rows Using Indices
- Just like pandas, we are using conditional statements to specify specific rows. 
- See [here](http://www.ats.ucla.edu/stat/r/faq/subset_R.htm) for good coverage and examples. 

setosa.df <- frame[frame$species == "setosa",]
head(setosa.df)
class(setosa.df)
nrow(setosa.df)
mean.sepalLenth.setosa<-mean(setosa.df$sepal_length) #This creates a new vector
mean.sepalLenth.setosa
setosa.df.highseptalLength <- setosa.df[setosa.df$sepal_length > mean.sepalLenth.setosa,]
nrow(setosa.df.highseptalLength)
head(setosa.df.highseptalLength)

specific.df <- frame[frame$sepal_length %in% c(5.1,5.8),]
head(specific.df)


## Basics

1. Load the Titanic train.csv data into an R data frame.
2. Calculate the number of rows in the data frame.
3. Calcuated general descriptive statistics for the data frame.
4. Slice the data frame into 2 parts, selecting the first half of the rows. 
5. Select just the columns passangerID and whether they survivied or not. 

## CREDITS

Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.
Adopted from [Berkley R Bootcamp](https://github.com/berkeley-scf/r-bootcamp-2016).


