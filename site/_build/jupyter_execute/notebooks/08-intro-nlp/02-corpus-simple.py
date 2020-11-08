[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to Text Mining in Python</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

# Introduction to Text Mining in Python


These exercises were adapted from Mining the Social Web, 2nd Edition [See origional here](https://github.com/ptwobrussell/Mining-the-Social-Web-2nd-Edition/) 
Simplified BSD License that governs its use.


### Key Terms for Text Mining
- A collection of documents –  corpus
- Document – a piece of text 
- Terms/tokens – a word in a document
- Entity – Some type of person, place, or organization


 corpus = { 
 'a' : "Mr. Green killed Colonel Mustard in the study with the candlestick. \
Mr. Green is not a very nice fellow.",
 'b' : "Professor Plum has a green plant in his study.",
 'c' : "Miss Scarlett watered Professor Plum's green plant while he was away \
from his office last week."
}

#This will separate the documents (sentences) into terms/tokins/words.
terms = {
 'a' : [ i.lower() for i in corpus['a'].split() ],
 'b' : [ i.lower() for i in corpus['b'].split() ],
 'c' : [ i.lower() for i in corpus['c'].split() ]
 }
terms

### Term Frequency
- A very common factor is to determine how frequently a word or term occurs with a document. 
- This is how early web search engines worked. (Not very well).
- A common basic standarization method is to control for the number of words in the document.


from math import log

#This is our terms we would like to use.
QUERY_TERMS = ['mr.', 'green']

#This calculates the term frequency normalized by the length.
def tf(term, doc, normalize):
    doc = doc.lower().split()
    if normalize:
        return doc.count(term.lower()) / float(len(doc))
    else:
        return doc.count(term.lower()) / 1.0
 

#This prints the basic documents. We can see that Mr. Green is in the first document.
for (k, v) in sorted(corpus.items()):
    print (k, ':', v)
print('\n')
    

# Score queries by calculating cumulative tf (normalized and unnormalized).
query_scores = {'a': 0, 'b': 0, 'c': 0}

#This starts the search for each query
for term in [t.lower() for t in QUERY_TERMS]:
    #This starts the search for each document in the corpus
    for doc in sorted(corpus):
        print ('TF(%s): %s' % (doc, term), tf(term, corpus[doc], True))
        
print('\n')   #Let's skip a line.     
print ("This does the same thing but unnormalized.")
for term in [t.lower() for t in QUERY_TERMS]:
    #This starts the search for each document in the corpus
    for doc in sorted(corpus):
        print ('TF(%s): %s' % (doc, term), tf(term, corpus[doc], False))

### TF-IDF
- TF-IDF incorporates the inverse document frequency in the analysis.  This type of factor would limit the impact of *frequent words* that would show up in a large number of documents.  
- The tf-idf calc involves multiplying against a tf value less than 0, so it's necessary to return a value greater than 1 for consistent scoring. (Multiplying two values less than 1 returns a value less than each of them.)

def idf(term, corpus):
    
    num_texts_with_term = len([True for text in corpus if term.lower()
                              in text.lower().split()])
    try:
        return 1.0 + log(float(len(corpus)) / num_texts_with_term)
    except ZeroDivisionError:
        return 1.0

    
for term in [t.lower() for t in QUERY_TERMS]:
        print ('IDF: %s' % (term, ), idf(term, corpus.values()))
        





#TF-IDF Just multiplies the two together
def tf_idf(term, doc, corpus):
    return tf(term, doc, True) * idf(term, corpus)

query_scores = {'a': 0, 'b': 0, 'c': 0}
for term in [t.lower() for t in QUERY_TERMS]:
    for doc in sorted(corpus):
        print ('TF(%s): %s' % (doc, term), tf(term, corpus[doc], True))
    print ('IDF: %s' % (term, ), idf(term, corpus.values()))
    print('\n')

    for doc in sorted(corpus):
        score = tf_idf(term, corpus[doc], corpus.values())
        print ('TF-IDF(%s): %s' % (doc, term), score)
        query_scores[doc] += score
        print('\n')

print ("Overall TF-IDF scores for query '%s'" % (' '.join(QUERY_TERMS), ))
for (doc, score) in sorted(query_scores.items()):
    print (doc, score)



