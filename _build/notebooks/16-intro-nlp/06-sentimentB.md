---
redirect_from:
  - "/notebooks/16-intro-nlp/06-sentimentb"
interact_link: content/notebooks/16-intro-nlp/06-sentimentB.ipynb
kernel_name: python3
has_widgets: false
title: 'Sentiment - Package'
prev_page:
  url: /notebooks/16-intro-nlp/07-fastai-imdb.html
  title: 'FAST.ai NLP'
next_page:
  url: /notebooks/16-intro-nlp/08-intro2.html
  title: 'Overview of NLP V2'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


## Lecture-21: Introduction to Natural Language Processing

###Sentiment Analysis

#### What is sentiment analysis?

Take a few movie reviews as examples (taken from Prof. Jurafsky's lecture notes): 

1. unbelievably disappointing 
2. Full of zany characters and richly applied satire, and some great plot twists
3. This is the greatest screwball comedy ever filmed
4. It was pathetic. The worst part about it was the boxing scenes. 


*Positive*: 2, 3
*Negative*: 1, 4


![alt text](https://www.dropbox.com/s/zmowjdhfodh9na5/danlecture.tif)
Google shopping; Bing shopping; Twitter sentiment about airline customer service

#### Sentiment analysis is the detection of **attitudes** “enduring, affectively colored beliefs, dispositions towards objects or persons”


1. **Holder (source)** of attitude
2. **Target (aspect)** of attitude
3. **Type** of attitude
    * From a set of types
        - Like, love, hate, value, desire, etc.
    * Or (more commonly) simple weighted polarity: 
        - positive, negative, neutral, together with strength
4. **Text** containing the attitude
    * Sentence or entire document






<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# We will use vader sentiment analysis here considering short text phrases
!pip install vaderSentiment
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Requirement already satisfied: vaderSentiment in /usr/local/lib/python3.6/dist-packages (3.2.1)
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```def measure_sentiment(textval):
  sentObj = SentimentIntensityAnalyzer() 
  sentimentvals = sentObj.polarity_scores(textval)
  print(sentimentvals)
  if sentimentvals['compound']>=0.5: 
    return("Positive")
  elif sentimentvals['compound']<= -0.5: 
    return("Negative")
  else:
    return("Neutral")

text1 = "I love the beautiful weather today. It is absolutely pleasant."
text2 = "Unbelievably disappointing"
text3 = "Full of zany characters and richly applied satire, and some great plot twists"
text4 = "This is the greatest screwball comedy ever filmed"
text5 = "This is the greatest screwball comedy ever filmed"
text6 = "It was pathetic. The worst part about it was the boxing scenes."

#print(measure_sentiment(text1))
#print(measure_sentiment(text2))
#print(measure_sentiment(text3))
#print(measure_sentiment(text4))
#print(measure_sentiment(text5))
print(measure_sentiment(text6))



```
</div>

</div>



### Topic Modeling -- Latent Dirichlet Allocation

#### Topic model is a type of statistical model to discover the abstract latent topics present in a given set of documents.

#### Topic modeling allows us to discover the latent semantic structures in a text corpus through learning probabilistic distributions over words present in the document.

#### It is a generative statistical model that allows different classes of observations to be explained by groups of unobserved data similar to clustering.
**It assumes that documents are probability distributions over topics and topics are probability distributions over words.**  

#### Latent Dirichlet Allocation (LDA) was proposed by Blei et al. in 2003 LDA assumes that the document is a mixture of topics where each topic is a mixture of words assigned to a topic where the topic distribution is assumed to have a dirichlet prior.

#### We consider the Python package “gensim”to perform topic modeling over the online reviews in our notebook.

![alt text](https://www.dropbox.com/s/wk87yos1jmjnm26/lda.jpeg)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```
#Load the file first
!wget https://www.dropbox.com/s/o8lxi6yrezmt5em/reviews.txt


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
--2019-11-07 22:03:45--  https://www.dropbox.com/s/o8lxi6yrezmt5em/reviews.txt
Resolving www.dropbox.com (www.dropbox.com)... 162.125.65.1, 2620:100:6021:1::a27d:4101
Connecting to www.dropbox.com (www.dropbox.com)|162.125.65.1|:443... connected.
HTTP request sent, awaiting response... 301 Moved Permanently
Location: /s/raw/o8lxi6yrezmt5em/reviews.txt [following]
--2019-11-07 22:03:45--  https://www.dropbox.com/s/raw/o8lxi6yrezmt5em/reviews.txt
Reusing existing connection to www.dropbox.com:443.
HTTP request sent, awaiting response... 302 Found
Location: https://uca49837b6228b70be8d77e61bc1.dl.dropboxusercontent.com/cd/0/inline/Ar_TmvTj6aB8uANEqIJKbxZu2qjWL2AIrR9DGtrYalyog06i9GD2Hv6zuVGLnpHoj7Tp-SDZUq1NmgtzS1w9p-RfSoXlIdmrOad1piGku8eWddl-nWPXPcD6-6dTI-0tF4g/file# [following]
--2019-11-07 22:03:46--  https://uca49837b6228b70be8d77e61bc1.dl.dropboxusercontent.com/cd/0/inline/Ar_TmvTj6aB8uANEqIJKbxZu2qjWL2AIrR9DGtrYalyog06i9GD2Hv6zuVGLnpHoj7Tp-SDZUq1NmgtzS1w9p-RfSoXlIdmrOad1piGku8eWddl-nWPXPcD6-6dTI-0tF4g/file
Resolving uca49837b6228b70be8d77e61bc1.dl.dropboxusercontent.com (uca49837b6228b70be8d77e61bc1.dl.dropboxusercontent.com)... 162.125.65.6, 2620:100:6021:6::a27d:4106
Connecting to uca49837b6228b70be8d77e61bc1.dl.dropboxusercontent.com (uca49837b6228b70be8d77e61bc1.dl.dropboxusercontent.com)|162.125.65.6|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 3851 (3.8K) [text/plain]
Saving to: ‘reviews.txt’

reviews.txt         100%[===================>]   3.76K  --.-KB/s    in 0s      

2019-11-07 22:03:46 (388 MB/s) - ‘reviews.txt’ saved [3851/3851]

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import nltk
from nltk.corpus import stopwords 
#from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     /root/nltk_data...
[nltk_data]   Unzipping taggers/averaged_perceptron_tagger.zip.
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
True
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

f=open('reviews.txt')
text = f.read()

stop_words = stopwords.words('english')
sentences=sent_tokenize(text)

data_words = list(sent_to_words(sentences))
data_words_nostops = remove_stopwords(data_words)

dictionary = corpora.Dictionary(data_words_nostops)
corpus = [dictionary.doc2bow(text) for text in data_words_nostops]

NUM_TOPICS = 2
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
print("ldamodel is built")
#ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=6)
for topic in topics:
    print(topic)


```
</div>

</div>



### Word Embeddings




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```model = gensim.models.Word2Vec(data_words_nostops, min_count=1)
#print(model.most_similar("fish", topn=10))

print(model.most_similar("bar", topn=10))


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[('actual', 0.24399515986442566), ('blue', 0.23981201648712158), ('burgers', 0.23935973644256592), ('fish', 0.22530747950077057), ('hole', 0.2182062566280365), ('bit', 0.20433926582336426), ('may', 0.20417353510856628), ('little', 0.1970674693584442), ('refills', 0.19307652115821838), ('sign', 0.19287416338920593)]
```
</div>
</div>
</div>



### Bag of words model and TF-IDF computations

##### tf-idf stands for Term frequency-inverse document frequency. The tf-idf weight is a weight often used in information retrieval and text mining. Variations of the tf-idf weighting scheme are often used by search engines in scoring and ranking a document’s relevance given a query.

##### This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus (data-set).

**STEP-1: Normalized Term Frequency (tf)** -- tf(t, d) = N(t, d) / ||D||
wherein, ||D|| = Total number of term in the document

tf(t, d) = term frequency for a term t in document d.

N(t, d)  = number of times a term t occurs in document d

**STEP-2: Inverse Document Frequency (idf)** -- 
idf(t) = N/ df(t) = N/N(t)

idf(t) = log(N/ df(t))

idf(pizza) = log(Total Number Of Documents / Number Of Documents with term pizza in it)

**STEP-3: tf-idf Scoring** 

tf-idf(t, d) = tf(t, d)* idf(t, d)

Example:

Consider a document containing 100 words wherein the word kitty appears 3 times. The term frequency (i.e., tf) for kitty is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word kitty appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.


Doc1: I love delicious pizza
Doc2: Pizza is delicious
Doc3: Kitties love me



## Class exercise

Data files we use for this exercise are here: https://www.dropbox.com/s/cvafrg25ljde5gr/Lecture21_exercise_1.txt?dl=0

https://www.dropbox.com/s/9lqnclea9bs9cdv/lecture21_exercise_2.txt?dl=0

### 1. Read the 2nd file and preprocess it by removing non-alphanumeric characters

### 2. Perform sentiment analysis, topic modeling and identify the most commonly co-occurring words with "religion", "politics", "guns"

### 3. Build a bag of words model and compute TF-IDF the query "pizza".

