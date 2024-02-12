# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:25:13 2023

@author: Vaibhav Bhorkade
"""
# Text Mininng
sentence="we are learning Textmining from sanjivani AI"
# if we want to know position of learning 
sentence.index("learning")
# It will show learning is at position 7
# This is going to show charachter position from 0 in cludi
######################################################
# we want to know position TextMining word
sentence.split().index("Textmining")
# it will split the words in list and count the position 
# if you want to see the list select sentence.split() and
# it will show at 3
######################################################
# Suppose we want print any word in reverse order
sentence.split()[2][::-1]
# [start:end end:-1(start)] will start from -1,-2,-3,-4 till the end
# learning will be printed as gninrael
sentence.split()[3][::-1]
#####################################################

# Suppose we want to print first and and last word of the sentence
words=sentence.split()
first_word=words[0]
first_word
last_word=words[-1]
last_word
# now we want to calculate the first and last word
concat_word=first_word+" "+last_word
concat_word
#####################################################
# we want to print even words from the sentence
[words[i] for i in range(len(words)) if i%2==0]

####################################################
sentence
# now we want to display only AI
sentence[-3:]
# it will start from -3,-2,-1

###################################################
# Suppose we want entire sentence in reverse order
sentence[::-1]
# 'IA inavijnas morf gninimtxeT gninrael era ew'

# Suppose we want to select each word and print in reversed order
words
print(" ".join(word[::-1] for word in words))
# ew era gninrael gninimtxeT morf inavijnas IA
###################################################

# Tokenization
# nltk - natural language toolkit
import nltk
nltk.download('punkt')
from nltk import word_tokenize
words=word_tokenize("I am reading NLP Fundamentals")
print(words)

##################################################
# parts of speech(PoS) tagging
nltk.download("averaged_perceptron_tagger")
nltk.pos_tag(words)
# It is going mention parts of speech

#################################################
# Stop words from NLTK library
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words=stopwords.words("English")
# You can verify 179 stopwords in variable explorer
print(stop_words)
sentence1="I am learning nlp: It is one the most popular library in python"
# First we will tokenize the sentence
sentence_words=word_tokenize(sentence1)
print(sentence_words)
# Now let us filter the sentence1 using stop_words
sentence_no_stops=" ".join([words for words in sentence_words if words not in stop_words])
print(sentence_no_stops)
sentence1
# you can notice that am,is,of, the most , popular,in are missing
###############################################################
# Suppose we want to replace words in string
sentence2="I visited MY from IND on 14-02-19"
normalized_sentence=sentence2.replace("MY","Malaysia").replace("IND","INDIA")
normalized_sentence=normalized_sentence.replace("-19", "-2020")
print(normalized_sentence)
###############################################################
# Suppose we want auto correction in the sentence
from autocorrect import Speller
# declare the function Speller defined for english
spell=Speller(lang="en")
spell("Engilish")
##############################################################
# Suppose we want to correct whole sentence
sentence3="Ntural lanagage processin deals withh the aart of extracting sentiiiments"
# let us first tokenize this sentence
sentence3=word_tokenize(sentence3)
corrected_sentence=" ".join([spell(word) for word in sentence3])
print(corrected_sentence)
##############################################################
# Stemming
stemmer=nltk.stem.PorterStemmer()
stemmer.stem("programming")
stemmer.stem("programmed")
stemmer.stem("jumping")
stemmer.stem("Jumped")

#############################################################
# Lematizer
# lematizer looks into dictionary words
nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
lemmatizer.lemmatize("programed")
lemmatizer.lemmatize("programs")
lemmatizer.lemmatize("battling")
lemmatizer.lemmatize("amazing")

############################################################
nltk.download('averaged_perceptron_tagger')
# Chuking (shallow Parsing)Identifying named entities
nltk.download("maxent_ne_chunker")
nltk.download("words")
sentence4="We are learning NLP in python by SanjivaniAI  based in India"
# first we will tokenize
words=word_tokenize(sentence4)
words=nltk.pos_tag(words)
i=nltk.ne_chunk(words,binary=True)
[a for a in i if len(a)==1]

############################################################
# sentence tokenization
from nltk.tokenize import sent_tokenize
sent=sent_tokenize("we are learning NLP in python. Delivered by SanjivaniAI. ")
sent
# ['we are learning NLP in python.', 'Delivered by SanjivaniAI.']
############################################################
from nltk.wsd import lesk
sentence1="keep your savings in the bank"
print(lesk(word_tokenize(sentence1),'bank'))
# output Synset("saving_bank.n.02")
sentence2="It is so risky to drive over the banks of river"
print(lesk(word_tokenize(sentence2),'bank'))
## Synset("bank.v.07")
# Synset('bank.v.07') a slope in the turn of a road or track;
# the sentence is higher than the inside in order to reduce the "bank" as 
# multiple meanings. if you want to find exact meaning executing 
# folling code
# The defination for "bank" can be seen here
from nltk.corpus import wordnet as wn
for ss in wn.synsets('bank'):print(ss,ss.definition())


'''
CC coordinating conjunction
CD cardinal digit
DT determiner
FW foreign word
IN preposition/subordinating conjunction
JJ adjective ‘big’
JJR adjective, comparative ‘bigger’
JJS adjective, superlative ‘biggest’
LS list marker 1)
MD modal could, will
NN noun, singular ‘desk’
NNS noun plural ‘desks’
NNP proper noun, singular ‘Harrison’
NNPS proper noun, plural ‘Americans’
PDT predeterminer ‘all the kids’
POS possessive ending parent’s
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO, to go ‘to’ the store.
UH interjection, errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
'''
