import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    # -----------------------
    # Normalization
    # -----------------------

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\S+@\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text)

    # -----------------------
    # Tokenization
    # -----------------------

    tokens = word_tokenize(text)

    # -----------------------
    # Remove punctuation
    # -----------------------

    tokens = [
        token
        for token in tokens
        if token not in string.punctuation
    ]

    # -----------------------
    # Stopwords
    # -----------------------

    tokens = [
        token
        for token in tokens
        if token not in stop_words
    ]

    # -----------------------
    # Stemming
    # -----------------------

    stemmed_tokens = [
        stemmer.stem(token)
        for token in tokens
    ]

    # -----------------------
    # Lemmatization
    # -----------------------

    lemmatized_tokens = [
        lemmatizer.lemmatize(token)
        for token in stemmed_tokens
    ]

    return lemmatized_tokens