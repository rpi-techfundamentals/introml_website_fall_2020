[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to Python - Datastructures</h1></center>
<center><h3><a href = 'http://introml.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>



# Introduction Datastructures (Varibles, Lists, Dictionaries, and Sets)
Common to R and Python
- Variables
- Opearations on Numeric and String Variables
- Lists

Python Only
- Dictionaries
- Sets



## Variables
- Single value
- Strings, Integer, Floats and boolean are the most common types of variables.
- Remember, under the covers they are all objects.
- Multiple variables can be output with the `print()` statement. 
- `\t` can be used to add a tab while `\n` can input a new line.

a = '#pythonrules' # string
b = 30              # integer
c = True            # boolean

#This prints (1) only the variables, (2) with labels, (3) including tabs, and (4) with new lines.
print('1:', a,  b, c)
print('2:','String:', a, 'Integer:', b, 'Boolean:', c)
print('3:','String:', a, '\tInteger:', b, '\tBoolean:', c)
print('4a:','String:', a, '\n4b: Integer:', b, '\n4c: Boolean:', c)
print(a+str(b))

## Variable Type (continued)
- In Python when we write `b = 30` this means the value of `30` is assigned to the `b` object. 
- Python is a [dynamically typed](https://pythonconquerstheuniverse.wordpress.com/2009/10/03/static-vs-dynamic-typing-of-programming-languages/).
- Unlike some languages, we don't have to declare the type of a variable before using it. 
- Variable type can also change with the reassignment of a variable. 



a = 1
print ('The value of a is ', a,  'and type ', type(a) )

a = 2.5
print ('Now the value of a is ', a,  'and type ', type(a) )

a = 'hello there'
print ('Now the value of a is ', a,  'and of type ', type(a) )

## Variable Type (continued)

- _Variables_ themselves do not have a fixed type.
- It is only the values that they refer to that have an associated _type_.
- This means that the type referred to by a variable can change as more statements are interpreted.
- If we combine types incorrectly we get an error. 

#We can't add 5 to a 
b = 'string variable'
c=b+5
c

## The `type` Function

- We can query the type of a value using the `type` function.
- Variables can be reassigned to a different type. 
- There are integer, floating point, and complex number [numeric types](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex).
- Boolean is a special type of integer.

a = 1
type(a)

a = 'hello'
type(a)

a=2.5
type(a)

a=True
type(a)

## Converting Values Between Types

- We can convert values between different types.
- To convert to string use the `str()` function.
- To convert to floating-point use the `float()` function.
- To convert to an integer use the `int()` function.
- To convert to a boolean use the `bool()` function.


a = 1
print(a, type(a))

a = str(a)
print (a, type(a))

a = float(a)
print (a, type(a))

a = int(a)
print (a, type(a))

## Converting Values Between Types (Continued)
- To convert to a boolean use the `bool()` function.
- `bool` can work with a String type that is `True` or `False`
- `bool` can work with an integer type that is `1` for `True` or `0` for `False`

b = 'True'
print (b, type(b))

b = bool(b)
print (b, type(b))

c = 1
c= bool(c)
print (c, type(c))

d = 0
d= bool(d)
print (d, type(d))

## Null Values

- Sometimes we represent "no data" or "not applicable".  
- In Python we use the special value `None`.
- This corresponds to `NA` in R of `Null` in Java/SQL.
- When we print the value `None` is printed. 
- If we enter the variable, no result is printed out.


a = None
print(a)


#Notice nothing is printed.
a

## Operations on Numeric Variables
- Python can be used as a basic calculator.
- Check out this associated [tutorial](https://docs.python.org/3/tutorial/introduction.html#using-python-as-a-calculator).

print('Addition:', 53 + 5)
print('Multiplication:', 53 * 5)
print('Subtraction:', 53 - 5)
print('Division', 53 / 5 )
print('Floor Division (discards the fractional part)', 53 // 5 )
print('Floor Division (returns the remainder)', 53 % 5 )
print('Exponents:', 5 ** 2 )

## Operations on String Variables
- Just as we can do numeric operations, we can also do operations on strings.
- Concatentate Strings 
- A *backslash* is used as an escape variable.
- More info on this [tutorial](https://docs.python.org/3/tutorial/introduction.html#using-python-as-a-calculator).

a='Start'
b='End'
tab='\t'
newline='\n'
c='can\'t'  #Note that we have to use the Escape character '\' to inclue a apostrophe '  in the key.
cb="can't"
continueline = 'This is the first line. \
This is the second line, but we have included a line continuation character: \\'
#Note that to print the continueline character we have to list 2 (\\)
#Note that to print the continueline character we have to list 2 (\\)

contin2= """
This is the second line, but we have included a line continuation character: 
#Note that to print the continueline character we have to list 2 
#Note that to print the continueline character we have to list 2


"""



print('Concatenation:', a+b )
print('Tab:', a+tab+b )
print('Newline:', a+newline+b )
print('Apostrophe:', c )
print('Apostrophe:', cb )
print('Continue line:', continueline )
print(contin2)

## Calling Functions on Variables

- We can call functions in a conventional way using round brackets
- Python has a wide variety of [built in functions](https://docs.python.org/3/library/functions.html),

a=abs(-98.45)
print('abs() takes the absolute value:', a )
a=round(a)
print('round() rounds to nearest integer:', a )
character=chr(a)
print('chr(98) returns the string representing a character whose Unicode code point is associated with the integer:',character) 

## Exercise - Operations on Variables

1. What happens when you multiply a number times a boolean? What is the resulting type? 
2. What happens when you try to multiply an integer value times a null?
3. Take 5 to the power of 4. 



## Lists
- Lists can be used to contain a sequence of values of any type. 
- You can do operations on lists.
- The list values start at 0 and that the first value of a list can be printed using `a[0]`
- Lists can be *sliced* or *indexed* using the start and end value `a[start:end]`
- Lists are *mutable datastructures*, meaning that they can be changed (added to).

#Set the value of the list
a = [1, 2, 'three', 'four', 5.0]

print('Print the entire array:', a)
print('Print the first value:', a[0])
print('Print the first three value:', a[0:3])
print('Print from second value till end  of list:', a[2:])
print('Print the last value of a list:', a[-1])
print('Print up till the 2nd to last value:', a[:-2]) 
type(a)

## Lists
- Lists can be nested, where there are lists of lists.
- The elements of a nested list is specified after the first list when slicing `c[0][0]`

a = [1, 2, 'three', 'four', 5.0]
b = [6, 'seven', 8, 'nine']
c = [a, b]

print('This is a list with 2 lists in it:', c)
print('This is the first list:', c[0])
print('This is the first element of the second list:', c[1][0])

## Lists
- Lists can added to with the `append` method or your can directly assign location in list.
- You can identify the length of a list with `len(a)`
- [More fuctions on lists](https://docs.python.org/3/tutorial/datastructures.html) include `pop()` `insert()` etc.
- If you set a `lista = listb` this list will not be a copy but instead be the same list, where if you modify one it will modify both.
- To create a copy of a list, you can use `lista=listb[:]`

b = [6, 'seven', 8, 'nine']
b.append(10)
print('We added 10 to b:', b)
print('the length of b is now:', len(b))

b[len(b):] = ['Eleven',12]
print('We added 11 to b:', b)



## List

- If you set a lista = listb this list will not be a copy but instead be the same list, where if you modify one it will modify both.
- To create a copy of a list, you can use `lista=listb[:]` or `lista=listb.copy()`

listb=[1,2,3,4]
listb1=[1,2,3,4]
listb2=[1,2,3,4]
#This assigns one variable to another, linking them
lista=listb
#This creates a copy
lista1=listb1[:]
lista2=listb2.copy() # This does the same thing.
#This deletes the third item in the array.
lista.pop(3)
lista1.pop(3)
lista2.pop(3)
#Notice how when we pop lista, listb is also impacted.
print(lista, listb)
#Notice how when we use a copy, listb1 is not impacted. 
print(lista1, listb1)
print(lista2, listb2)

## Exercise-Lists
Hint: [This list of functions on lists is useful.](https://docs.python.org/3/tutorial/datastructures.html)

1. Create a list `elists1` with the following values (1,2,3,4,5).
2. Create a new list `elists2` by first creating a copy of `elist1` and then reversing the order.
3. Create a new list `elists3` by first creating a copy of `elist1` and then adding 7 8 9 to the end. *(Hint: Search for a different function if appending doesn't work.)*
4. Create a new list `elists4` by first creating a copy of `elist3` and then insert 6 between 5 and 7. 




## Sets

- Lists can contain duplicate values.
- A set, in contrast, contains no duplicates.
- Sets can be created from lists using the `set()` function.
- Alternatively we can write a set literal using the `{` and `}` brackets.




#This creates a set from a list. 
X = set([1, 2, 3, 3, 4])

print(X, type(X))

X = {1, 2, 3, 4, 4}
print(X, type(X))

## Sets are Mutable

- Sets are mutable like lists (meaning we can change them)
- Duplicates are automatically removed

X = {1, 2, 3, 4}
X.add(0)
X.add(5)
print(X)
X.add(5)
print(X)

## Sets are Unordered

- Sets do not have an order.
- Therefore we cannot index or slice them.



X[0]

## Operations on Sets

- Union: $X \cup Y$ combines two sets


X = {1, 2, 3, 4}
Y = {4, 5, 6}
X.union(Y)

## Operations on Sets
- Intersection: $X \cap Y$:

X = {1, 2, 3, 4}
Y = {3, 4, 5}
X.intersection(Y)

## Operations on Sets
- Difference $X - Y$:


X = {1, 2, 3, 4}
Y = {3, 4, 5}
X - Y

## Dictionaries
- You can think of dictionaries as arrays that help you assocaite a `key` with a `value`.
- Dictionaries can be specified with `{key: value, key: value}`
- Dictionaries can be specified with dict([('key', value), ('key', value)])
- Key's and values can be either string or numeric. 
- Dictionaries are mutable, (can be changed) `adict['g'] = 41`

adict1 = {'a' : 0, 'b' : 1, 'c' : 2}
adict2 = dict([(1, 'a'), (2, 'b'), (3, 'c')])
print(adict1,adict2, '\n', type(adict1),type(adict2), '\n',adict1['b'],adict2[2])

adict2['g']=1234

adict2['g']

## Exercise-Sets/Dictionary

1. Create a set `eset1` with the following values (1,2,3,4,5).
2. Create a new set `eset2` the following values (1,3,6).
3. Create a new set `eset3` that is `eset1-eset2`.
4. Create a new set `eset4` that is the union of `eset1+eset2`.
5. Create a new set `eset5` that includes values that are in both `eset1` and `eset2` (intersection).
6. Create a new dict `edict1` with the following keys and associated values: st1=45; st2=32; st3=40; st4=31.
7. Create a new variable edict2 from edict 1 where the key is st3.
 




## CREDITS


Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2020
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.

This work has been adopted from the [origional version](https://github.com/phelps-sg/python-bigdata):
Copyright [Steve Phelps](http://sphelps.net) 2014



