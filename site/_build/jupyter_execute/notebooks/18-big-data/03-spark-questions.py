## Spark

Before you turn this problem in, make sure everything runs as expected. First, restart the kernel (in the menubar, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All).  You can speak with others regarding the assignment but all work must be your own. 


### This is a  20 point assignment graded from answers to questions. 


#![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# **Word Count Lab: Building a word count application**
#### This lab will build on the techniques covered in the Spark tutorial to develop a simple word count application.  The volume of unstructured text in existence is growing dramatically, and Spark is an excellent tool for analyzing this type of data.  
#### *Part 1:* Creating a base RDD and pair RDDs
#### *Part 2:* Counting with pair RDDs
#### *Part 3:* Finding unique words and a mean value

#### Note that, for reference, you can look up the details of the relevant methods in [Spark's Python API](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD)

#Don't need to do this if running on databricks.
import pyspark
#sc = pyspark.SparkContext('local[*]')

### ** Part 1: Creating a base RDD and pair RDDs **

#### In this part of the lab, we will explore creating a base RDD with `parallelize` and using pair RDDs to count words.

#### ** (1a) Create a base RDD **
#### We'll start by generating a base RDD by using a Python list and the `sc.parallelize` method.  Then we'll print out the type of the base RDD.

wordsList = ['cat', 'elephant', 'rat', 'rat', 'cat']
wordsRDD = sc.parallelize(wordsList, 4)
# Print out the type of wordsRDD
print (type(wordsRDD))

#### What does it mean that the list is an RDD? What special operations does this enable and how is it different from a dataset? 
#


#### ** (1b) Pluralize and test **
#### Let's use a `map()` transformation to add the letter 's' to each string in the base RDD we just created. We'll define a Python function that returns the word with an 's' at the end of the word.  Please replace `<FILL IN>` with your solution.  If you have trouble, the next cell has the solution.  After you have defined `makePlural` you can run the third cell which contains a test.  If you implementation is correct it will print `1 test passed`.
#### This is the general form that exercises will take, except that no example solution will be provided.  Exercises will include an explanation of what is expected, followed by code cells where one cell will have one or more `<FILL IN>` sections.  The cell that needs to be modified will have `# TODO: Replace <FILL IN> with appropriate code` on its first line.  Once the `<FILL IN>` sections are updated and the code is run, the test cell can then be run to verify the correctness of your solution.  The last code cell before the next markdown section will contain the tests.

# TODO: Replace <FILL IN> with appropriate code
def makePlural(word):
    """Adds an 's' to `word`.

    Note:
        This is a simple function that only adds an 's'.  No attempt is made to follow proper
        pluralization rules.

    Args:
        word (str): A string.

    Returns:
        str: A string with 's' added to it.
    """
    return <FILL IN>

makePlural('cat')

#### ** (1c) Apply `makePlural` to the base RDD **
#### Now pass each item in the base RDD into a [map()](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.map) transformation that applies the `makePlural()` function to each element. And then call the [collect()](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.collect) action to see the transformed RDD.

# TODO: Replace <FILL IN> with appropriate code
pluralRDD = wordsRDD.map(<FILL IN>)
print(pluralRDD.collect())

#### ** (1d) Pass a `lambda` function to `map` **
#### Let's create the same RDD using a `lambda` function.
Let's remember that a lampda function 

lambda x: x + 1

# TODO: Replace <FILL IN> with appropriate code
pluralLambdaRDD = wordsRDD.map(<FILL IN>)
print(pluralLambdaRDD.collect())

#### ** (1e) Length of each word **
#### Now use `map()` and a `lambda` function to return the number of characters in each word.  You can do this with the length function. We'll `collect` this result directly into a variable.

# TODO: Replace <FILL IN> with appropriate code
pluralLengths = (pluralRDD
                 .map(lambda w: <FILL IN>)
                 .collect())
print(pluralLengths)

#### ** (1f) Pair RDDs **
#### The next step in writing our word counting program is to create a new type of RDD, called a pair RDD. A pair RDD is an RDD where each element is a pair tuple `(k, v)` where `k` is the key and `v` is the value. In this example, we will create a pair consisting of `('<word>', 1)` for each word element in the RDD.
#### We can create the pair RDD using the `map()` transformation with a `lambda()` function to create a new RDD.

# TODO: Replace <FILL IN> with appropriate code
wordPairs = wordsRDD.map(<FILL IN> w: (w, 1))
print(wordPairs.collect())

### ** Part 2: Counting with pair RDDs **

#### Now, let's count the number of times a particular word appears in the RDD. There are multiple ways to perform the counting, but some are much less efficient than others.
#### A naive approach would be to `collect()` all of the elements and count them in the driver program. While this approach could work for small datasets, we want an approach that will work for any size dataset including terabyte- or petabyte-sized datasets. In addition, performing all of the work in the driver program is slower than performing it in parallel in the workers. For these reasons, we will use data parallel operations.

#### ** (2a) `groupByKey()` approach **
#### An approach you might first consider (we'll see shortly that there are better ways) is based on using the [groupByKey()](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.groupByKey) transformation. As the name implies, the `groupByKey()` transformation groups all the elements of the RDD with the same key into a single list in one of the partitions. There are two problems with using `groupByKey()`:
  + #### The operation requires a lot of data movement to move all the values into the appropriate partitions.
  + #### The lists can be very large. Consider a word count of English Wikipedia: the lists for common words (e.g., the, a, etc.) would be huge and could exhaust the available memory in a worker.
 
#### Use `groupByKey()` to generate a pair RDD of type `('word', iterator)`.

# TODO: Replace <FILL IN> with appropriate code
# Note that groupByKey requires no parameters
<FILL IN> = wordPairs.groupByKey()
for key, value in wordsGrouped.collect():
    print('{0}: {1}'.format(key, list(value)))


#### ** (2b) Use `groupByKey()` to obtain the counts **
#### Using the `groupByKey()` transformation creates an RDD containing 3 elements, each of which is a pair of a word and a Python iterator.
#### Now sum the iterator using a `map()` transformation.  The result should be a pair RDD consisting of (word, count) pairs.

# TODO: Replace <FILL IN> with appropriate code
wordCountsGrouped = wordsGrouped.map(<FILL IN> kv: (kv[0], len(kv[1])))
print(wordCountsGrouped.collect())

#### ** (2c) Counting using `reduceByKey` **
#### A better approach is to start from the pair RDD and then use the [reduceByKey()](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.reduceByKey) transformation to create a new pair RDD. The `reduceByKey()` transformation gathers together pairs that have the same key and applies the function provided to two values at a time, iteratively reducing all of the values to a single value. `reduceByKey()` operates by applying the function first within each partition on a per-key basis and then across the partitions, allowing it to scale efficiently to large datasets.

# TODO: Replace <FILL IN> with appropriate code
# Note that reduceByKey takes in a function that accepts two values and returns a single value
wordCounts = wordPairs.reduceByKey(lambda a,b : a + b)
print(wordCounts.collect())

#### ** (2d) All together **
#### The expert version of the code performs the `map()` to pair RDD, `reduceByKey()` transformation, and `collect` in one statement.

# TODO: Replace <FILL IN> with appropriate code
wordCountsCollected = (wordsRDD
                       .map(lambda w: (w,1))
                       .reduceByKey(lambda a, b : a + b)
                       .collect())
print(<FILL IN>)

### ** Part 3: Finding unique words and a mean value **


#See Unique words. 
uniqueWords =  wordsRDD.distinct().count()
print (uniqueWords)

#### ** (3a) Mean using `reduce` **
#### Find the mean number of words per unique word in `wordCounts`.
#### Use a `reduce()` action to sum the counts in `wordCounts` and then divide by the number of unique words.  First `map()` the pair RDD `wordCounts`, which consists of (key, value) pairs, to an RDD of values.

# TODO: Replace <FILL IN> with appropriate code
from operator import add
totalCount = (wordCounts
              .map(lambda kv: kv[0])
              .count())
average = totalCount / float(uniqueWords)
print (totalCount)
print (round(average, 2))

### ** Part 4: Apply word count to a file **

#### In this section we will finish developing our word count application.  We'll have to build the `wordCount` function, deal with real world problems like capitalization and punctuation, load in our data source, and compute the word count on the new data.

#### ** (4a) `wordCount` function **
#### First, define a function for word counting.  You should reuse the techniques that have been covered in earlier parts of this lab.  This function should take in an RDD that is a list of words like `wordsRDD` and return a pair RDD that has all of the words and their associated counts.

# TODO: Replace <FILL IN> with appropriate code
def wordCount(wordListRDD):
    """Creates a pair RDD with word counts from an RDD of words.

    Args:
        wordListRDD (RDD of str): An RDD consisting of words.

    Returns:
        RDD of (str, int): An RDD consisting of (word, count) tuples.
    """
    return wordListRDD.map(lambda w: (w, 1)).reduceByKey(lambda a, b: a + b)
  
print (wordCount(wordsRDD).collect())

