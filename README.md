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

## Usage

To run the project and perform sentiment analysis on tweets:

1. Install the required libraries: Pandas, NumPy, NLTK, and Scikit-learn.

2. Open the Python script file, `Twitter-Sentiment-Analysis.py`, in your preferred Python IDE or text editor.

3. Ensure that the dataset containing the tweets is accessible. Adjust the file path or provide the necessary input for loading the dataset.

4. Execute the script to perform the sentiment analysis. The tweets will be processed, and the sentiment prediction will be displayed or stored as desired.

## Contributing

Contributions to this project are welcome! If you have any suggestions, improvements, or additional features to enhance the sentiment analysis of tweets, feel free to submit a pull request. Let's work together to improve the understanding of sentiment on Twitter.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code for personal or commercial purposes.
