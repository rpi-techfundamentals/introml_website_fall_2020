[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - Test Local Jupyter Notebook</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

## Test Notebook

The goal of this notebook is to simply test the R environment and to show how to interact with a local data file. Let's first check the R version. 


version

## Reading a Local CSV File 
- The `read.csv` command can accepted a variety of delimited files.
- Relative references are indicated with `..` to indcate going up a directory.
- Windows: - use either \ or / to indicate directories
- setwd('C:\\Users\\Your_username\\Desktop\\r-bootcamp-2016')
- setwd('..\\r-bootcamp-2016')

# This will load the local iris.csv file into an R dataframe.  We will work with these a lot in the future.
#This is refered to a relative reference.  This is Relative to the current working directory. 
frame=read.csv(file="../../input/iris.csv", header=TRUE, sep=",")

# This will print out the dataframe.
frame

## Writing data out from R

Here you have a number of options. 

1) You can write out R objects to an R Data file, as we've seen, using `save()` and `save.image()`.
2) You can use `write.csv()` and `write.table()` to write data frames/matrices to flat text files with delimiters such as comma and tab.
3) You can use `write()` to write out matrices in a simple flat text format.
4) You can use `cat()` to write to a file, while controlling the formatting to a fine degree.
5) You can write out in the various file formats mentioned on the previous slide

#Writing Dataframe to a file. 
write.csv(frame, file= "iris2.csv")

#Kaggle won't want the rownames. 
write.csv(frame, file = "iris3.csv",row.names=FALSE)

setwd('/Users/jasonkuruzovich/githubdesktop/0_class/techfundamentals-spring2018-materials/classes/')

## The Working Directory
- To read and write from R, you need to have a firm grasp of where in the computer's filesystem you are reading and writing from.
- It is common to set the working directory, and then just list the specific file without a path.
- Windows: - use either \\ or / to indicate directories
- `setwd('C:\\Users\\Your_username\\Desktop\\r-bootcamp-2016')`
- `setwd('..\\r-bootcamp-2016')`

#Whhile this example is Docker, should work similarly for Mac or Windows Based Machines. 
setwd("/home/jovyan/techfundamentals-fall2017-materials/classes/05-intro-r")
getwd()  # what directory will R look in?
setwd("/home/jovyan/techfundamentals-fall2017-materials/classes/input") # change the working directory
getwd() 
setwd("/home/jovyan/techfundamentals-fall2017-materials/classes/05-intro-r")
setwd('../input') # This is an alternate way of moving to data.
getwd()

#Notice path isn't listed. We don't have to list the path if we have set the working directory

frame=read.csv(file="iris.csv", header=TRUE,sep=",")

## Exercises

### Basics

1) Make sure you are able to install packages from CRAN. E.g., try to install *lmtest*.

2) Figure out what your current working directory is.

### Using the ideas

3) Put the *data/iris.csv* file in some other directory. Use `setwd()` to set your working directory to be that directory. Read the file in using `read.csv()`.  Now use `setwd()` to point to a different directory. Write the data frame out to a file without any row names and without quotes on the character strings.


### CREDITS

Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016.
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.
Work adopted from [Berkley R Bootcamp](https://github.com/berkeley-scf/r-bootcamp-2016).