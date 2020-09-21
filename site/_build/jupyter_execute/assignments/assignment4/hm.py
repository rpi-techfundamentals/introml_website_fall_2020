## Homework 04 - Instructions

![](https://github.com/rpi-techfundamentals/hm-01-starter/blob/master/notsaved.png?raw=1)

**WARNING!!!  If you see this icon on the top of your COLAB sesssion, your work is not saved automatically.**


**When you are working on homeworks, make sure that you save often. You may find it easier to save intermident copies in Google drive. If you save your working file in Google drive all changes will be saved as you work. MAKE SURE that your final version is saved to GitHub.** 

Before you turn this problem in, make sure everything runs as expected. First, restart the kernel (in the menubar, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All).  You can speak with others regarding the assignment but all work must be your own. 


### This is a 30 point assignment graded from answers to questions and automated tests that should be run at the bottom. Be sure to clearly label all of your answers and commit final tests at the end.  


files = "https://github.com/rpi-techfundamentals/introml_website_fall_2020/raw/master/files/assignment4.zip" 
!pip install otter-grader && wget $files && unzip -o assignment4.zip

#Run this. It initiates autograding. 
import otter
grader = otter.Notebook()

## Exercise - Seaborn


#Import and load some libraries
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
titanic = sns.load_dataset("titanic")
import grading_object_detection as god


(S1) Create the following figure using the Titanic dataset (included with Seaborn). Output the file to hm4-1.png in the current directory.  
![](https://github.com/rpi-techfundamentals/hm-04-starter/raw/master/fig/hm4-1.png)



#Answer



#Test
grader.check('q01')

(S2) Create the following figure using the Titanic dataset (included with Seaborn). Output the plot to hm4-2.png in the current directory.  
![](https://github.com/rpi-techfundamentals/hm-04-starter/raw/master/fig/hm4-2.png)

#Answer


grader.check('q02')

(S3) Create the following figure using the Titanic dataset (included with Seaborn). Output the plot to hm4-3.png in the current directory.  
![](https://github.com/rpi-techfundamentals/hm-04-starter/raw/master/fig/hm4-3.png)

#Answer


grader.check('q03')

#Please provide an interpretation for the graph provided in question 3. 

man4="""



"""

(S5) Create the following figure using the Titanic dataset (included with Seaborn). Output the plot to hm4-5.png in the current directory.  
![](https://github.com/rpi-techfundamentals/hm-04-starter/raw/master/fig/hm4-5.png)

#Answer

grader.check('q05')

## Exercise-Beautiful Soup
Imagine you would like to write a simple python script that pulls in data from https://webrobots.io/kickstarter-datasets/.   

First go to the above website.

![](https://github.com/rpi-techfundamentals/hm-04-starter/raw/master/fig/crawler.png)


To do that, we would have to get all the links.  Here is some starter code that will download the url into a soup object.

```
#This will pull the data
from bs4 import BeautifulSoup
import requests
response = requests.get("https://webrobots.io/kickstarter-datasets/")
html_doc = response.text
soup = BeautifulSoup(html_doc, 'html.parser')
print(soup.prettify())
 
```




#This will pull the data
from bs4 import BeautifulSoup
import requests
response = requests.get("https://webrobots.io/kickstarter-datasets/")
html_doc = response.text
soup = BeautifulSoup(html_doc, 'html.parser')
print(soup.prettify())
 

(bs1) Look through the html and find the links to the files that we want.  Set the variable `file_url` to be equal to the url of the files that we want to get

#update base url up to but not including the first /
file_url= "https://**************"


grader.check('bs1')

(bs2) Before we try to get the links, lets make sure we can parse the web page. Set the value of the variable `page_title` equal to the contents inside the page `<title>` tag. The variable `page_title` should not include the title tag (i.e., the `<title>`). 

#answer


grader.check('bs2')

(bs3) We talked about both regular expressions and beautiful soup.  Here is an example of how you could combine them to match a pattern. 


```
import re
soup.find("a", href=re.compile(file_url), href=True)
```

Further process the data so that you get a list equal to all of the links to the json files `json_links` and a list equal to all the csv files `csv_links`.  

![](https://github.com/rpi-techfundamentals/hm-04-starter/raw/master/fig/links.png)



#Answer 


grader.check('bs3')

### Regular Expressions

Regular expressions is a useful way of parsing through string data. 

(RE1) Please complete a function `blind_text` that accepts a text string and does the the following transformations. 

- Change `text` to lower case.
- Remove all emails and substitute them with '--'
- Remove all digits and replace with *. 
- Split the lines and put them into a list called `textlines`.

Given the text string:

```
my_text="""The test score is 85 with the email john@rpi.edu for MGMT33223.
The test score is 83 with the email jim@rpi.edu for MGMT33223."""
```

The resulting list should be:
```
['the test score is ** with the email -- for mgmt*****.',
 'the test score is ** with the email -- for mgmt*****.']
```




my_text="""The test score is 85 with the email john@rpi.edu for MGMT33223.
The test score is 83 with the email jim@rpi.edu for MGMT33223."""

#Create your function here. 

#Call the function
result = blind_text(my_text)
print(result)  


#Tests
grader.check('re1')


## MAKE SURE THAT THIS ENTIRE NOTEBOOK RUNS WITHOUT ERRORS. TO TEST THIS DO RUNTIME --> RESTART AND RUN ALL

It should run without errors.  


### Click File -> Download .ipynb to download the assignment.  Then Upload it to  the LMS.

