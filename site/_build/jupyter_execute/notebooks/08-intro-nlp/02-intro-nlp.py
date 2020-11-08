# Natural Language Toolkit

## Introduction to Natural Language Processing

In this workbook, at a high-level we will learn about text tokenization; text normalization such as lowercasing, stemming; part-of-speech tagging; Named entity recognition; Sentiment analysis; Topic modeling; Word embeddings





####PLEASE EXECUTE THESE COMMANDS BEFORE PROCEEDING####

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

#Tokenization -- Text into word tokens; Paragraphs into sentences;
from nltk.tokenize import sent_tokenize 
  
text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."
sent_tokenize(text) 



import nltk.data 
  
german_tokenizer = nltk.data.load('tokenizers/punkt/PY3/german.pickle') 
  
text = 'Wie geht es Ihnen? Mir geht es gut.'
german_tokenizer.tokenize(text) 


from nltk.tokenize import word_tokenize 
  
text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."
word_tokenize(text) 



from nltk.tokenize import TreebankWordTokenizer 
  
tokenizer = TreebankWordTokenizer() 
tokenizer.tokenize(text) 


### n-grams vs tokens

##### n-grams are contiguous sequences of n-items in a sentence. N can be 1, 2 or any other positive integers, although usually we do not consider very large N because those n-grams rarely appears in many different places.

##### Tokens do not have any conditions on contiguity

#Using pure python

import re

def generate_ngrams(text, n):
    # Convert to lowercases
    text = text.lower()
    
    # Replace all none alphanumeric characters with spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in text.split(" ") if token != ""]
    
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."
print(text)
generate_ngrams(text, n=2)

#Using NLTK import ngrams

import re
from nltk.util import ngrams

text = text.lower()
text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
tokens = [token for token in text.split(" ") if token != ""]
output = list(ngrams(tokens, 3))
print(output)

#Text Normalization

#Lowercasing
text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."
lowert = text.lower()
uppert = text.upper()

print(lowert)
print(uppert)


#Text Normalization
#stemming
#Porter stemmer is a famous stemming approach

from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
   
ps = PorterStemmer() 
  
# choose some words to be stemmed 
words = ["hike", "hikes", "hiked", "hiking", "hikers", "hiker"] 
  
for w in words: 
    print(w, " : ", ps.stem(w)) 



from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
import re
   
ps = PorterStemmer() 
text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."
print(text)


#Tokenize and stem the words
text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
tokens = [token for token in text.split(" ") if token != ""]

i=0
while i<len(tokens):
  tokens[i]=ps.stem(tokens[i])
  i=i+1

#merge all the tokens to form a long text sequence 
text2 = ' '.join(tokens) 

print(text2)

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize 
import re
   
ss = SnowballStemmer("english")
text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."
print(text)


#Tokenize and stem the words
text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
tokens = [token for token in text.split(" ") if token != ""]

i=0
while i<len(tokens):
  tokens[i]=ss.stem(tokens[i])
  i=i+1

#merge all the tokens to form a long text sequence 
text2 = ' '.join(tokens) 

print(text2)

#Stopwords removal 

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."

stop_words = set(stopwords.words('english')) 
word_tokens = word_tokenize(text) 
  
filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 
  
print(word_tokens) 
print(filtered_sentence) 

text2 = ' '.join(filtered_sentence)

#Part-of-Speech tagging

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = 'GitHub is a development platform inspired by the way you work. From open source to business, you can host and review code, manage projects, and build software alongside 40 million developers.'

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

sent = preprocess(text)
print(sent)


#Named entity recognition

#spaCy is an NLP Framework -- easy to use and having ability to use neural networks

import en_core_web_sm
nlp = en_core_web_sm.load()

text = 'GitHub is a development platform inspired by the way you work. From open source to business, you can host and review code, manage projects, and build software alongside 40 million developers.'

doc = nlp(text)
print(doc.ents)
print([(X.text, X.label_) for X in doc.ents])

#Sentiment analysis

#Topic modeling

#Word embeddings


#Class exercise

#### 1. Read a file from its URL 
#### 2. Extract the text and tokenize it meaningfully into words. 
#### 3. Print the entire text combined after tokenization. 
#### 4. Perform stemming using both porter and snowball stemmers. Which one works the best? Why? 
#### 5. Remove stopwords
#### 6. Identify the top-10 unigrams based on their frequency.



#Load the file first
!wget https://www.dropbox.com/s/o8lxi6yrezmt5em/reviews.txt
