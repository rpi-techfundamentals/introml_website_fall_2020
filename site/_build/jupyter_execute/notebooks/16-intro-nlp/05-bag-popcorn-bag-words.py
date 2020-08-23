[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1> Bag of Words</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

This is adopted from: [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words)
[https://github.com/wendykan/DeepLearningMovies](https://github.com/wendykan/DeepLearningMovies)


import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

!wget https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/input/labeledTrainData.tsv
!wget https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/input/unlabeledTrainData.tsv
!wget https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/input/testData.tsv

train = pd.read_csv('labeledTrainData.tsv', header=0, \
                    delimiter="\t", quoting=3)
unlabeled_train= pd.read_csv('unlabeledTrainData.tsv', header=0, \
                    delimiter="\t", quoting=3)
test = pd.read_csv('testData.tsv', header=0, \
                    delimiter="\t", quoting=3)

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

print(train.columns.values, test.columns.values)

train.head()

print('The train shape is: ', train.shape)
print('The train shape is: ', test.shape)

print('The first review is:')
print(train["review"][0])


# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup             

# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0], "html.parser" )  
print(example1.get_text())

import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print (letters_only)

lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words


# Enter Download then stopwords.
nltk.download('stopwords')

print (stopwords.words("english"))

# Remove stop words from "words"
words = [w for w in words if not w in stopwords.words("english")]
print (words)

#Now we are going to do our first class
class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review,"html.parser"  ).get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(words)

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( KaggleWord2VecUtility.review_to_wordlist( raw_sentence, remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences



clean_review_word = KaggleWord2VecUtility.review_to_wordlist \
        ( train["review"][0], True )
clean_review_sentence = KaggleWord2VecUtility.review_to_wordlist \
        ( train["review"][0], True )


# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size





print ("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
for i in range( 0, len(train["review"])):
    if( (i+1)%1000 == 0 ):
        print ("Review %d of %d\n" % ( i+1, num_reviews )  )  
    clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))

clean_train_reviews[0:5]

print ("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 



train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()

print ("Training the random forest (this may take a while)...")

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)
  

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )

# Create an empty list and append the clean reviews one by one
clean_test_reviews = []

print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,len(test["review"])):
    clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))



# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()


# Use the random forest to make sentiment label predictions
print ("Predicting test labels...\n")
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv('Bag_of_Words_model.csv', index=False, quoting=3)
print ("Wrote results to Bag_of_Words_model.csv")

### Word2Vec


#!pip install gensim

import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

"In lexical analysis, tokenization is the process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens. The list of tokens becomes input for further processing such as parsing or text mining." -[Wikipedia](https://en.wikipedia.org/wiki/Tokenization_(lexical_analysis)

Punkt is a specific tokenizer.
[http://www.nltk.org/_modules/nltk/tokenize/punkt.html](http://www.nltk.org/_modules/nltk/tokenize/punkt.html) 

# download punkt
nltk.download('punkt')

#What is a tokenizer 
# http://www.nltk.org/_modules/nltk/tokenize/punkt.html 
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# ****** Split the labeled and unlabeled training sets into clean sentences
# Note this will take a while and produce some warnings. 
sentences = []  # Initialize an empty list of sentences

print ("Parsing sentences from training set")
for review in train["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)


    print ("Parsing sentences from unlabeled set")
    for review in unlabeled_train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

# ****** Define functions to create average word vectors
#

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
    # Print a status message every 1000th review
       if counter%1000. == 0.:
            print ("Review %d of %d" % (counter, len(reviews)))
               # Call the function (defined above) that makes average feature vectors
            reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
            # Increment the counter
            counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print ("Training Word2Vec model...")
model = Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling, seed=1)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)



# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)

model.doesnt_match("man woman child kitchen".split())



model.doesnt_match("france england germany soccer".split())


model.most_similar("soccer")

model.most_similar("man")

model["computer"]

model.most_similar("car")