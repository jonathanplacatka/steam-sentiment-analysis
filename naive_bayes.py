import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset.csv')

#convert review text to unicode
review_text = df['review_text'].astype('U').values

#split into train and test data (75% train, 25% test by default)
review_train, review_test, score_train, score_test = train_test_split(review_text, df['review_score'].values)

#convert text to TF-IDF representation, then fit NB classifier
pipe = make_pipeline(TfidfVectorizer(), MultinomialNB())
pipe.fit(review_train, score_train)

#predict labels for test data
labels = pipe.predict(review_test)

#print some results (will need to add plotting)
for i in range(10000, 10025):
    print("review text: {}..., actual score {}, predicted score {}".format(review_test[i][:20], score_test[i], labels[i]))

#manually testing some strings
print("result:", pipe.predict(["good"]))
print("result:", pipe.predict(["don't buy this game it's bad"]))
print("result:", pipe.predict(["worst game ive ever played"]))
print("result:", pipe.predict(["amazing game 10/10"]))
print("result:", pipe.predict(["terrible game"]))
print("result:", pipe.predict(["broken game"]))



