# Getting Started

# Setting Up

import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Global Parameters
stop_words = set(stopwords.words('english'))


# Loading the Dataset

def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset


def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset


# Pre-processing Tweets

def preprocess_tweet_text(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if w not in stop_words]

    # ps = PorterStemmer()
    # stemmed_words = [ps.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]

    return " ".join(lemma_words)


def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector


def int_to_string(sentiment):
    if sentiment == 0:
        return "Negative"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Positive"


# Bringing Everything Together

# Load dataset
dataset = load_dataset("training.csv", ['target', 't_id', 'created_at', 'query', 'user', 'text'])
# Remove unwanted columns from dataset
n_dataset = remove_unwanted_cols(dataset, ['t_id', 'created_at', 'query', 'user'])
# Preprocess data
dataset.text = dataset['text'].apply(preprocess_tweet_text)
# Split dataset into Train, Test

# Same tf vector will be used for Testing sentiments on unseen trending data
tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
x = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
y = np.array(dataset.iloc[:, 0]).ravel()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# Training Naive Bayes model
NB_model = MultinomialNB()
NB_model.fit(x_train, y_train)
y_predict_nb = NB_model.predict(x_test)
print(accuracy_score(y_test, y_predict_nb))
# 0.768521875 without implementing stemming or lemmatization
# 0.762571875 with implementing stemming
# 0.7682 with implementing lemmatization
# 0.762315625 with implementing stemming and lemmatization

# Training Logistics Regression model
LR_model = LogisticRegression(solver='lbfgs', max_iter=500)
LR_model.fit(x_train, y_train)
y_predict_lr = LR_model.predict(x_test)
print(accuracy_score(y_test, y_predict_lr))
# 0.788175 without implementing stemming or lemmatization
# 0.782321875 with implementing stemming
# 0.78804375 with implementing lemmatization
# 0.78208125 with implementing stemming and lemmatization

# Testing on Real-time Feeds
