# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:15:44 2023

@author: Vaibhav Bhorkade
"""
# how to use TFIDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer    
corpus=['The mouse had a tiny little mouse','The cat saw the mouse','The cat catch the mouse','The end of mouse story']
# Step1 initialize count vector
cv=CountVectorizer()
# To count the total no.of TF
word_count_vector=cv.fit_transform(corpus)
word_count_vector.shape
# Now next step is to apply IDF
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
# This matrix is in raw matrix from, let us convert it in dataframe
df_idf=pd.DataFrame(tfidf_transformer.idf_,index=cv.get_feature_names_out(),columns=["idf_weights"])
# sort ascending
df_idf.sort_values(by=['idf_weights'])

###########################################################

from sklearn.feature_extraction.text import TfidfVectorizer
corpus=[
        "Thor eating pizza, Loki is eating pizza, Ironman ate pizza already",
        "Apple is announcing new iphone tomorrow",
        "Tesla is announcing new model -3 tomarrow",
        'Google is announcing new pixel-6 tomarrow',
        "Microsoft is anouncing new surface tomarrow",
        "Amazon is announcing new eco-dot tomarrow",
        "I am eating biryani and you are eating grapes"
        ]
# Lets create the vectorizer and fit the corous and transform them according
v=TfidfVectorizer()
v.fit(corpus)
transform_output=v.transform(corpus)
# lets print the vocabulary

print(v.vocabulary_)
# Lets print the idf of each word

all_feature_names=v.get_feature_names_out()

for word in all_feature_names:
    # lets get the index in the cocabulary
    index=v.vocabulary_.get(word)
    # get the score
    idf_score=v.idf_[index]
    print(f"{word}:{idf_score}")

################################################

import pandas as pd

#read the data into a pandas Dataframe
df=pd.read_csv("Ecommerce_data.csv")
print(df.shape)
df.head()
#check the distribution of labels 
df['label'].value_counts()
#Add  the new column which gives a unique number to each of these label

df['label_num']=df['label'].map({
    'Household':0,
    'Books':1,
    'Electronics':2,
    'Clothing & Accessories':3})

#check the result
df.head(5)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(
    df.Text,
    df.label_num,
    test_size=0.2, #20 % sample will go to test dataset
    random_state=2022,
    stratify=df.label_num # it is used to distribute eqully
    )

print("Shape of X_train:", X_train.shape)
print("Shape of X_test: ", X_test.shape)
y_train.value_counts()
y_test.value_counts()
#######
#Apply to classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Create a pipeline object
clf=Pipeline([
    ('vectorizer_tfidf',TfidfVectorizer()),
    ('KNN',KNeighborsClassifier())
    ])

# 2. fit with X_train and y_train
clf.fit(X_train,y_train)

# 3. get the predictions for X_test and store it in y_pred 
y_pred=clf.predict(X_test)

# 4. print the classification report
print(classification_report(y_test, y_pred))
