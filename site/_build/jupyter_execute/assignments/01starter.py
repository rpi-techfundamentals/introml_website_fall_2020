## Homework 01 - Instructions

![](https://github.com/rpi-techfundamentals/hm-01-starter/blob/master/notsaved.png?raw=1)


Before you turn this problem in, make sure everything runs as expected. First, restart the kernel (in the menubar, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All).  You can speak with others regarding the assignment but all work must be your own. 


### This is a 30 point assignment graded from answers to questions and automated tests that should be run at the bottom. Be sure to clearly label all of your answers and commit final tests at the end.  


## BEFORE YOU BEGIN

Please work through each of these notebooks, which will give you some understanding of the Google Colab environment.

### Working with Notebooks in Colaboratory
- [Overview of Colaboratory](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)
- [Guide to Markdown](https://colab.research.google.com/notebooks/markdown_guide.ipynb)
- [Importing libraries and installing dependencies](https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb)
- [Saving and loading notebooks in GitHub](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)


### Working with Data
Some of this is a bit more advanced, but at this point just make sure you know where the code is for how to upload and download a file. 

- [Loading data: Drive, Sheets, and Google Cloud Storage](https://colab.research.google.com/notebooks/io.ipynb) 


## Run these Cells
This will setup the automated testing environment on Colab

files = "https://github.com/rpi-techfundamentals/hm-01-starter/raw/master/files.zip" 
!pip install otter-grader && wget $files && unzip -o files.zip

#Run this. It initiates autograding. 
import otter
grader = otter.Notebook()

**Question 1.** 

In the next cell, assign the value for `x` too 150.  Set the value for `y` to 13 times `x` , and set the value for `z` to `y` divided by `x` squared.



#enter your answer here



#Run this cell to see if you passed the test.  If it fails, you probably didn't run the above required code. 
grader.check('q01')

**Question 2.** 

Packages are really important compontent of most programming languages. 

In the overview, you learnned about tab completion as a way to explore python objects.  This can be really useful.  Let's use it to find the formula for the the factorial of 15. Assign the results to the variable `m`.

#we have to first import the math function to use tab completion. 
import math  

#Assign the result to the variable m.
m = math.
m

#Run this cell to see if you passed the test.  If it fails, you probably didn't run the above required code. 
grader.check('q02')

**Question 3.** 

Markdown is a useful aspect of Jupyter Notebooks. 
- [Guide to Markdown](https://colab.research.google.com/notebooks/markdown_guide.ipynb)

Use what you learned about markdown to adjust the text below to create the following:
![](https://github.com/rpi-techfundamentals/hm-01-starter/blob/master/md_result.png?raw=1)


Header
For the above header, make it an h1 tag using markdown. 

Sub-Header
For the above sub-header, make it an h5 tag using markdown. 

Bold

Italics

https://tw.rpi.edu//images/rpi-logo-red.jpg  
(Embed this image)


**Question 4.** 

### Installing Packages 
Python packages are an important part of data science and critical to leveraging the broader Python ecosystem.

You typically have two options when installing a package. You can install it with [Conda](http://anaconda.org) or [pip](https://pypi.org).

The `!` in a jupyter notebook means that the line is being processed on the `commmand line` and not by the Python interpreter. 

```
!conda install -c conda-forge <packagename>

!pip install <packagename>

```

If you try to import something and get an error, it is usally a tell that you need to install a package. 



### Install the `fastparquet` Package to be able to work with Parquet Files
- CSV (comma delimited  files are great for humans to read and understand.  
- For "big data" though, it isn't a great long term storage option (inefficient/slow).
- Parquet is a type columnar storage format.  It makes dealing with lots of columns fast. 
- [fastparquet](https://fastparquet.readthedocs.io) is a Python package for dealing with Parquet files. 
- Apache Spark also natively reads Parquet Files. 
- Look [here](https://pypi.org/project/fastparquet/) for instructions on installing the fastparquet package. 

#Install package for fastparquet here.

#This gets a parque file we can use to test. 
!wget https://github.com/rpi-techfundamentals/hm-01-starter/raw/master/Demographic_Statistics_By_Zip_Code.parq

#Run this to try to load the name.parq. It won't work unless you downloaded the file and installed the package. 
from fastparquet import ParquetFile  #This imports the package. 
pf = ParquetFile('Demographic_Statistics_By_Zip_Code.parq')
dfparq = pf.to_pandas()   #This changes the Parquet File object to a pandas dataframe. We will learn more about that later. 
dfparq  #Just listing the value prints it out. 

#Run this cell to see if you passed the test.  
grader.check('q04')

**Question 5.** 

### Importing CSV into a Pandas Dataframe
- Comma delimited files are a common way of transmitting data. 
- Data for different columns is separated by a comma.
- It is possible to open a csv in different ways, but Pandas is the easiest.  
- Data structured like CSV's are extremely common and known as tabular data.
- Pandas will give access to many useful methods for working with data.  
- `pandas` is often imported as the abbreviated `pd`.

### Download CSV from Web to Colab using `wget`
Click this link [https://data.cityofnewyork.us/api/views/kku6-nxdu/rows.csv?accessType=DOWNLOAD](https://data.cityofnewyork.us/api/views/kku6-nxdu/rows.csv?accessType=DOWNLOAD)
to download the file to your local computer. 

To get the file onto the colab computational environment, the easiest way is to use wget.  If you have something on google drive, saving 

`!wget -O <filename you want to save to> <link to file>`

For example:
`!wget -O foo.csv https://data.us/122/?accessType=DOWNLOAD`

You can read more about wget [here](https://www.electrictoolbox.com/wget-save-different-filename/). 

**Note that this is not a strategy for data that needs to remain secured, as anyone with access to your notebook would also gain access to the file.** 




#Insert your wget command here



# This will load the local name.csv file into a Pandas dataframe.  We will work with these a lot in the future.
import pandas as pd # This line imports the pandas package. 
dfcsv = pd.read_csv('Demographic_Statistics_By_Zip_Code.csv') 
dfcsv

#Run this cell to see if you passed the test.  
grader.check('q05')

## Enter your information
This is required to make sure that you get credit for the assignment.

## Homework Rubric
The following Rubric will be used to grade homwork. 
4 pts. x 6 questions. 
6 points for commit to github and 0 errors.
30 points total. 



### MAKE SURE THAT THIS ENTIRE NOTEBOOK RUNS WITHOUT ERRORS. TO TEST THIS DO RUNTIME --> RESTART AND RUN ALL

It should complete till the end. 



This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.