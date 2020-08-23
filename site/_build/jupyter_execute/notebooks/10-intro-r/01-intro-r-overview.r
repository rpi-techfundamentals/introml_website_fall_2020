[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - Overview and Packages</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>



# “The best thing about R is that it was developed by statisticians. The worst thing about R is that… it was developed by statisticians.”  
##                                         - Bo Cowgill, Google

## Overview
- Language Features and Use Cases  
- R 
- R Studio
- R and Packages

## What is R?

- R is an Open Source (and freely available) environment for statistical computing and graphics.
- It is a full-featured programming language, in particular a scripting language (with similarities to Matlab and Python).
- It can be run interactively or as a batch/background job.
- R is being actively developed with ongoing updates/new releases.
- R has a variety of built-in as well as community-provided packages that extend its functionality with code and data; see CRAN for the thousands of add-on packages.
- It is freely-available and modifiable (Available for Windows, Mac OS X, and Linux).

## Modes of Using R

- From the command line in a Linux/Mac terminal window
- Using the Windows/Mac GUIs
- Using the RStudio GUI, an 'integrated development environment'
- Running an R script in the background on a Linux/Mac machine (Windows?)
- RStudio

## Why R?

- R is widely used (statisticians, scientists, social scientists) and has the widest statistical functionality of any software
- Users add functionality via packages all the time
- R is free and available on all major platforms
- As a scripting language, R is very powerful, flexible, and easy to use
- As a scripting language, R allows for reproducibility and automating tasks
- As a language, R can do essentially anything
- Wide usage helps to improve quality and reduce bugs
- R can interact with other software, databases, the operating system, the web, etc.
- R is built on C and can call user-written and external C code and packages (in particular, see the *Rcpp* R package)


## Why Not R?

- Other software is better than R at various tasks
    
i.e., [Python](http://imgs.xkcd.com/comics/python.png) is very good for text manipulation, interacting with the operating system, and as a glue for tying together various applications/software in a workflow-* R can be much slower than compiled languages (but is often quite fast with good coding practices!)
- R's packages are only as good as the person who wrote them; no explicit quality control
- R is a sprawling and unstandardized ecosystem
- Google has a [recommended style](https://google.github.io/styleguide/Rguide.xml) guide that should be taken into account. 

## R and Jupyter
- R commands can be executed in a Jupyter Notebook just by play at the end of a cell.
- Blocks of cells or even the entire notebook can be executed by clicking on the *Cell* above.
- The Kernal is responsible for interpreting the code, and the current kernal is listed on the top right of the notebook. 
- While Jupyter started as a Python project, there are now a variety of Kernals for different programming languages including R, Scala, and SAS. 
- Read more about Jupyter in the documentation [here](http://jupyter.readthedocs.io/en/latest/).
- If a variable isn't assigned it will be provided as output. 


test<-5
test

## R and RStudio
- Powerful IDE for R
- Integrated usage of git
- Integrated GUI based package management
- Integrated GIT
- Solid enterprise infrastructure, owned by Microsoft


### R and Conda
- Conda can quickly and easily install R. We love conda! 
- This is a rather long install, so may do it from command line. 
`!conda install r-essentials`

#conda install -y r-essentials

## R as calculator

# R as a calculator
2 * pi # multiply by a constant
7 + runif(1) # add a random number
3^4 # powers
sqrt(4^4) # functions
log(10)
log(100, base = 10)
23 %/% 2 
23 %% 2

## Assignment of Values
- Don't used the equal sign.  
- Do use `<-`


#Assignment DON"T use =
val <- 3
val
print(val)

Val <- 7 # case-sensitive!
print(c(val, Val))


# This will gnerate numbers from 1 to 6. 
mySeq <- 1:6
mySeq

#Notice how this worrks like arrange in Python.
myOtherSeq <- seq(1.1, 11.1, by = 2)
myOtherSeq
length(myOtherSeq)

fours <- rep(4, 6)
fours

# This is a comment: here is an example of non-numeric data
depts <- c('espm', 'pmb', 'stats')
depts


## R and Packages (R's killer app)
- There are a tremendous number of packages available which extend the core capabilities of the R language.
-  "Currently, the CRAN package repository features 9233 available packages." (from https://cran.r-project.org/web/packages/)
- [Packages on CRAN](https://cran.r-project.org/web/packages/). Also see the [CRAN Task Views](https://cran.r-project.org/web/views/). 
- Packages may be source or compiled binary files.
- Installing source packages which contain C/C++/Fortran code requires that compilers and related tools be installed. 
- Binary packages are platform-specific (Windows/Mac). This can cause problems in a large class and is a great reason to work in a standardized environment like the cloud or on Docker/VM.
- If you want to sound like an R expert, make sure to call them *packages* and not *libraries*. A *library* is the location in the directory structure where the packages are installed/stored.


## Using R packages
1. Install the package on your machine
2. Load the package

## Installing Packages in RStudio 
- Only needs to be done one time on machine. 
- To install a package, in RStudio, just do `Packages->Install Packages`.
- Option to specify the source repository: `install.packages('chron', repos='http://cran.us.r-project.org')`
- Option to install multiple packages: `install.packages(c("pkg1", "pkg2"))`
- You can install dependencies with: `install.packages("chron", dependencies = TRUE)`
- If binary files aren't available for your OS, install from source: `install.packages(path_to_file, repos = NULL, type="source")`
- R installs are not done from the terminal (no `!`).
    

## Install Packages with Conda
This will only work from the Anaconda Prompt or terminal. 

`!conda install -y -c r r-yaml`

## Loading Packages 
- R packages must be imported before using. 
- Packages only have to be imported once in a notebook (not in every cell). 
- The `library(chron)` would import the chron package, provided it has beeen installed. 
- See all packages loaded with `search()`
- search()    # see packages currently loaded

search() # see packages currently loaded
library("yaml") # Load the package chron
search()    # see packages currently loaded


## CREDITS


Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016.
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.
This work is adopted from the Berkley R Bootcamp.  