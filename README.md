# Twitter-Sentiment-Analysis

This project aims to analyze the sentiment of tweets on Twitter using Python and various libraries including Pandas, NumPy, Natural Language Toolkit (NLTK), and Scikit-learn.

## Features

- Sentiment analysis: The project uses sentiment analysis techniques to categorize the opinions expressed in tweets with a three-point ordinal scale consisting of Negative, Neutral, and Positive sentiments.

- Preprocessing techniques: The tweets are preprocessed to normalize the text and improve the accuracy of sentiment analysis. The following techniques are applied:

  - Letter casing: The text is converted to a consistent letter case (e.g., lowercase) to remove inconsistencies.
  - Noise removal: Unwanted characters, such as URLs or special symbols, are removed from the tweets.
  - Stopword removal: Commonly used words without significant meaning (e.g., "the", "is", "and") are eliminated from the tweets.
  - Stemming and lemmatization: Words are reduced to their base or root form to reduce variation in the text.

## Models and Accuracy

Two models are implemented for sentiment prediction:

- Naïve Bayes model: The Naïve Bayes algorithm is used for prediction, achieving an accuracy of nearly 76%.

- Logistics Regression model: The Logistics Regression algorithm is used for prediction, achieving an accuracy of nearly 79%.
