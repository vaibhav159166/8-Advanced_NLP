# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 09:16:26 2023

@author: Vaibhav Bhorkade
"""
# Bag of words
# This BoW converts unstructured data to structured form
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
corpus=['At least seven idian pharma companies are working to develop vaccine against the corona virus.','The deadly virus that has already infected more than 14 million globally','Bharat Biotech is the among the domastic pharma firm working on the corona virus vaccine in India']
bag_of_word_model=CountVectorizer()
print(bag_of_word_model.fit_transform(corpus).todense())
bag_of_word_df=pd.DataFrame(bag_of_word_model.fit_transform(corpus).todense())
# This will create dataframe
bag_of_word_df.columns=sorted(bag_of_word_model.vocabulary_)
bag_of_word_df.head()

############################################################
# bag of words model small
bag_of_word_model_small=CountVectorizer(max_features=5)
bag_of_word_df_small=pd.DataFrame(bag_of_word_model_small.fit_transform(corpus).todense())
bag_of_word_df_small.columns=sorted(bag_of_word_model_small.vocabulary_)
bag_of_word_df.head()
