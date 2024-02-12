# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 08:56:58 2023

@author: Vaibhav Bhorkade
"""
import re
sentence="sharad twitted ,Wittnessing 70th republic day India from Rajpath,\new Delhi,Mesomorizing performance by Indian Army"
re.sub(r"([^\s\w]|_)+",' ',sentence).split()
# extracting n-grams
# n-gram can be extracted using three techniques
# 1.custom defined function
# 2.nltk
# 3. TextBlob

# extracting n-grams using custom defined function

import re
def n_gram_extractor(input_str,n):
    tokens=re.sub(r"([^\s\w]|_)+",' ',input_str).split()
    for i in range(len(tokens)-n+1):
        print(tokens[i:i+n])
n_gram_extractor("The cute little boy is playing with kitten", 2)
n_gram_extractor("The cute little boy is playing with kitten", 3)

#################################################################

from nltk import ngrams
# extraction n-grams with nltk
list(ngrams("The cute little boy is playing with kitten".split(), 2))
list(ngrams("The cute little boy is playing with kitten".split(), 3))
#################################################################
from textblob import TextBlob
blob=TextBlob("The cute little boy is playing with kitten.")
blob.ngrams(n=2)
blob.ngrams(n=3)

#################################################################
# Tokenization using Keras
sentence
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(sentence)

# Tokenization using TextBlob
from textblob import TextBlob
blob=TextBlob(sentence)
blob.words

# Tweet Tokenizer
from nltk.tokenize import TweetTokenizer
tweet_tokenize=TweetTokenizer()
tweet_tokenize.tokenize(sentence)

# Multi-Word_Expression
from nltk.tokenize import MWETokenizer
sentence
mwe_tokenizer=MWETokenizer([('republic','day')])
mwe_tokenizer.tokenize(sentence.split())
mwe_tokenizer.tokenize(sentence.replace('!', ' ').split())

################################################################
# Regular expression Tokenizer
from nltk.tokenize import RegexpTokenizer
reg_tokenizer=RegexpTokenizer("\w+|\$[\d\.]+|\s+")
reg_tokenizer.tokenize(sentence)

# White space tokenizer
from nltk.tokenize import WhitespaceTokenizer
wh_tokenizer=WhitespaceTokenizer()
wh_tokenizer.tokenize(sentence)

###############################################################

from nltk.tokenize import WordPunctTokenizer
wp_tokenizer=WordPunctTokenizer()
wp_tokenizer.tokenize(sentence)

##############################################################
# Removing ing of word
sentence6="I love playing cricket. Cricket players practices hard in their inning"
from nltk.stem import RegexpStemmer
regex_stemmer=RegexpStemmer('ing$')
' '.join(regex_stemmer.stem(wd)for wd in sentence6.split())

############################################################
sentence7="Before eating , it would be nice to sanitize your hand"
from nltk.stem.porter import PorterStemmer
ps_stemmer=PorterStemmer()
words=sentence7.split()
" ".join([ps_stemmer.stem(wd)for wd in words])

#############################################################
# Lemmatization - finds original form of word in dictionary
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download("wordnet")
lemmatizer=WordNetLemmatizer()
sentence8='The codes executed today are for better than what we execute generally'
words=word_tokenize(sentence8)
" ".join(lemmatizer.lemmatize(word) for word in words)

################################################################
# Singularize and pluaralization
from textblob import TextBlob
sentence9=TextBlob("She sells seashells on the seashore")
words=sentence9.words
# we want to make word[2] i.e seashells in singular form
sentence9.words[2].singularize()
# we want word 5 i.e seashore in plural form
sentence9.words[5].pluralize() 

################################################################
# Language translation from spanish to english
from textblob import TextBlob
en_blob=TextBlob(u'my bien')
en_blob.translate(from_lang='es',to='en')
# es: spanish en:English

###############################################################
# custom stopwords removel
from nltk import word_tokenize
sentence9="She sells seashells on the seashore"
custom_stop_word_list=['she','on','the','am','is']
words=word_tokenize(sentence9)
" ".join([word for word in words if word.lower() not in custom_stop_word_list])
# select words which are not in defined list
##############################################################
# Extracting general features from raw text
# number of words
# detect presence of wh word
# popularity
# subjectivity
# language identification
import pandas as pd
df=pd.DataFrame([['The vaccine for covid-19 will be announced on 1 st august'],['Do you know how much expections the world population is having from this research?'],['The risk of virus will come to an end on 31st July']])
df.columns=['text']
df
# Now let us measure the number of words
from textblob import TextBlob
df['number_of_words']=df['text'].apply(lambda x:len(TextBlob(x).words))
df['number_of_words']

##############################################################
# Detect presence of words wh
wh_words=set(['why','who','which','what','where','when','how'])
df['is_wh_words_present']=df['text'].apply(lambda x:True if len(set(TextBlob(str(x)).words).intersection(wh_words))>0 else False)
df['is_wh_words_present']

#############################################################
# Polarity of the sentence
df['polarity']=df['text'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']
sentence10="I like this example very much"
pol=TextBlob(sentence10).sentiment.polarity
pol
sentence10="This is fantastic example but i would have prefer another one"
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10="This is my personal opinion that it was helpuful"
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10="I do not like , It is bad"
pol=TextBlob(sentence10).sentiment.polarity
pol
# -0.6999999999999998
##############################################################
# Subjectivity of the dataframe df and check whether there is 
df['subjectivity']=df['text'].apply(lambda x:TextBlob(str(x)).sentiment.subjectivity)
df['subjectivity']

##############################################################
# To find language of the sentence , this part of code will get http error
df['language']=df['text'].apply(lambda x:TextBlob(str(x)).detect_language())

###############################################################

