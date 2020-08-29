# Assignment 1

![](https://github.com/rpi-techfundamentals/hm-01-starter/blob/master/notsaved.png?raw=1)


Before you turn this problem in, make sure everything runs as expected. First, restart the kernel (in the menubar, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All).  You can speak with others regarding the assignment but all work must be your own. 


### This is a 30 point assignment.  


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

files = "https://github.com/rpi-techfundamentals/introml_website_fall_2020/raw/master/files/assignment1.zip" 
!pip install otter-grader && wget $files && unzip -o files.zip

#Run this. It initiates autograding. 
import otter
grader = otter.Notebook()

**Question 1.** 

In the next cell:

a. Assign the value for `x` to 150  

b. Set the value for `y` to 13 times `x` 

c. Set the value for `z` to `y` divided by `x` squared.



#enter your answer here



#Run this cell to see if you passed the test.  If it fails, you probably didn't run the above required code. 
grader.check('q01')

**Question 2.** 

Packages are really important compontent of most programming languages. 

In the overview, you learnned about tab completion as a way to explore python objects.  This can be really useful.  Let's use it to find the formula for the the factorial of 15. Assign the results to the variable `m`.

#we have to first import the math function to use tab completion. 
import math  

#Assign the result to the variable m.  Press tab after the period to show available functions
m = math.
m

#Run this cell to see if you passed the test.  If it fails, you probably didn't run the above required code. 
grader.check('q02')

**Question 3.** 

Markdown is a useful aspect of Jupyter Notebooks. 
- [Guide to Markdown](https://colab.research.google.com/notebooks/markdown_guide.ipynb)

Use what you learned about markdown to adjust the text below to create the following:
![](https://github.com/rpi-techfundamentals/hm-01-starter/blob/master/md_result.png?raw=1)

**Double click on cell below to open it for markdown editing. There is no test for this question.**


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

#Install package for fastparquet here.   Please comment it out after installing. 


#Run this to try to load the name.parq. It won't work unless you downloaded the file and installed the package. 
from fastparquet import ParquetFile  #This imports the package. 
import pandas as pd  #pandas is usually imported as pd
pf = ParquetFile('./data/Demographic_Statistics_By_Zip_Code.parq')
dfparq = pf.to_pandas()   #This changes the Parquet File object to a pandas dataframe. We will learn more about that later. 
dfparq.head()  #Just listing the value prints it out. 

## Show All Columns in a Pandas Dataframe

Notice there is a `...` which indicates you are only seeing some of the columns, and the output has been truncated. 

Read [this article](https://towardsdatascience.com/how-to-show-all-columns-rows-of-a-pandas-dataframe-c49d4507fcf) and find how to show all the columns of a pandas dataframe.

#Set the display options to show all columns. 
 

#This will print out the notebook. 
dfparq.head() 

#View the dataframe and set the following values to the numbers you see for row 0. Don't put in quotes.  
#COUNT PARTICIPANTS
row_0_count_participants=    #enter what you see above.
row_0_count_hispanic_latino=  #enter what you see above. 

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
- You can also get help by using a `?` after the method call. For example, to find the doc string for the read csv function you could execute:

`pd.read_csv?`  or 

`help(pd.read_csv)`




# Adjust the code below so that you load only the first 100 rows of a dataframe and assign to df_smalle
df_small=
df_small.shape

### Get CSVs from the Web/Github.  

You can also get a CSV directly from a web url. 

View this file in your web browser. You won't be able to load this into pandas. 
[https://github.com/rpi-techfundamentals/introml_website_fall_2020/blob/master/files/webfile.csv](https://github.com/rpi-techfundamentals/introml_website_fall_2020/blob/master/files/webfile.csv)

To get the link you can load, you need to click on the `raw` button. That should lead to this url:

`https://raw.githubusercontent.com/rpi-techfundamentals/introml_website_fall_2020/master/files/webfile.csv`


# Load the web url and set it equal to df_web
df_web = 
df_web.head()

#Run this cell to see if you passed the test.  
grader.check('q05')


## MAKE SURE THAT THIS ENTIRE NOTEBOOK RUNS WITHOUT ERRORS. TO TEST THIS DO RUNTIME --> RESTART AND RUN ALL

It should run without errors.  



This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.