[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - Conditional Statements and Loops </h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

## Overview
- What are conditional statements? Why do we need them?
- If statements in R
- Why, Why not Loops?
- Loops in R

# What are conditional statements? Why do we need them?

## `if` Statements
- Enables logical branching and recoding of data.
- BUT, `if statements` can result in long code branches, repeated code.
- Best to keep if statements short.

## Conditional Statements
- `if` statemenet enable logic. 
- `else` gives what to do if other conditions are not met.
- The else if function is achieved through nesting a if statement within an else function. 

#How long did the homework take? 
#Imagine this is the hours for each assignment. 
hours<- c(1,3,4,3)
#This is the experience of the individual, which can be high, medium, or low.
experience<-'low'
#experience<- 'high'

#toy = 'old'
if(experience=='high'){
exp.hours <- hours/2       
} else {
  if(experience=='low'){
     exp.hours <- hours * 2    
  } else {
    exp.hours <- hours      
  }
}
#Notice how this adjusted 
print(exp.hours)


## R Logit and Conditions
- `<`     less than
- `<=`    less than or equal to
- `>`     greater than
- `>=`    greater than or equal to
- `==`    exactly equal to
- `!=`    not equal to
- `!x`    This corresponsds to *not x.* 
- `x & y`  This cooresponds to `and`. *(This does and element by element comparson.)*
- `x | y`  This cooresponds to `or`. *(This does and element by element comparson.)*

#simple
x<-FALSE
y<-FALSE
if (!x){
print("X is False")
}else{
print("X is True")
}




x<-TRUE
y<-TRUE
if((x==TRUE)|(y==TRUE)){
print("Either X or Y is True")
}

if((x==TRUE)&(y==TRUE)){
print("X and Y are both True")
}

## Conditionals and `ifelse`
- `ifelse(*conditional*, True, False)` can be used to recode variables. 
- `ifelse` can be nested.
- Use the cut function as an alternative for more than 3 categroies. 
- This can be really useful when 

# create 2 age categories 
age<-c(18,15, 25,30)
agecat <- ifelse(age > 18, c("adult"), c("child"))
agecat

df=read.csv(file="../../input/iris.csv", header=TRUE,sep=",")

#Let's say we want to categorize sepalLenth as short/long or short/medium/long.
sl.med<-median(df$sepal_length)
sl.sd<-sd(df$sepal_length)
sl.max<-max(df$sepal_length)

df$agecat2 <- ifelse(df$sepal_length > sl.med, c("long"), c("short"))
df$agecat3 <- ifelse(df$sepal_length > (sl.med+sl.sd), c("long"), 
        ifelse(df$sepal_length < (sl.med-sl.sd), c("short"), c("medium")))


#This sets the different cuts for the categories. 
cuts<-c(0,sl.med-sl.sd,sl.med+sl.sd,sl.max)
cutlabels<-c("short", "medium", "long") 

df$agecat3altcut<-cut(df$sepal_length, breaks=cuts, labels=cutlabels)
df[,c(1,6,7,8)]

# Why, Why Not Loops?

## Why, Why Not Loops?
- Iterate over arrays or lists easily. 
- BUT, in many cases for loops don't scale well and are slower than alternate methods involving functions. 
- BUT, don't worry about prematurely optimizing code.
- Often if you are doing a loop, there is a function that is faster.  You might not care for small data applications.
- Here is a basic example of `For`/`While` loop.

sum<-0
avgs <- numeric (8)
for (i in 1:8){
    print (i)
    sum<-sum+i  
}
print(sum)
for (i in 1:8) print (i)


for (i in 1:8) print (i)

## While Loop
- Performs a loop while a conditional is TRUE.
- Doesn't auto-increment.

#This produces the same.
i<-1
sum<-0
x<-TRUE
while (x) {
  print (i)
  sum<-sum+i
  i<-i+1
  if (i>8){x<-FALSE}
}  
print(sum)

## For Loops can be Nested

#Nexting Example,
x=c(0,1,2)
y=c("a","b","c")
#Nested for loops
for (a in x){
    for (b in y){
        print(c(a,b), quote = FALSE)
}}

Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016.
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.


