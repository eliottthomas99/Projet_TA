# CHANTRE Honorine  CHAH2807
# THOMAS Eliott THOE2303

import string
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from wordcloud import WordCloud


def compute_null_values(dataframe, name_dataframe):
    """
    Function that allow to compute null values and display the result
    :param dataframe: a dataframe containing data
    :param name_dataframe: a string with the name of the dataframe
    :display: the result of missing data
     """
    null = dataframe.isnull().sum().sort_values(ascending=False)  # missing values
    total = dataframe.shape[0]  # total number of rows in the full dataset
    percent_missing = (dataframe.isnull().sum()/total).sort_values(ascending=False)

    missing_data = pd.concat([null, percent_missing], axis=1, keys=['Total missing', 'Percent missing'])

    missing_data.reset_index(inplace=True)
    missing_data = missing_data.rename(columns={"index": " column name"})

    print(f"Null Values in each column {name_dataframe} data :\n", missing_data)


def tranforme_number_of_class(x):
    """
    Function that allows to transforme the number of classes 5 to 3
    :param x: the sentiment
    :return "positive", "negative", "neutral": the name of the new class
     """
    if x == "Extremely Positive" or x == "Positive" or x == 1:
        return "positive"
    elif x == "Extremely Negative" or x == "Negative" or x == -1:
        return "negative"
    else:
        return "neutral"


def display_length_tweet(dataframe, name_dataframe):
    """
    Function that allow to display a histogram of the length of each tweet for each sentiment
    :param dataframe: a dataframe containing data
    :param name_dataframe: a string with the name of the dataframe
    :display: a histogram of the length of each tweet for each sentiment
     """
    fig = make_subplots(1, 3)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    tweet_len = dataframe[dataframe['sentiment'] == "positive"]['text'].str.len()
    ax1.hist(tweet_len, color='#17C37B')
    ax1.set_title('Positive Sentiments')

    tweet_len = dataframe[dataframe['sentiment'] == "negative"]['text'].str.len()
    ax2.hist(tweet_len, color='#F92969')
    ax2.set_title('Negative Sentiments')

    tweet_len = dataframe[dataframe['sentiment'] == "neutral"]['text'].str.len()
    ax3.hist(tweet_len, color='#FACA0C')
    ax3.set_title('Neutral Sentiments')

    fig.suptitle(f'Characters in tweets for each sentiment in the {name_dataframe} data')
    plt.savefig(f"out/characters_in_tweets_for_each_sentiment_{name_dataframe}_data.png")
    plt.show()


def display_numbers_words_for_each_tweet(dataframe, name_dataframe):
    """
    Function that allows to display a histogram of the number of words in a tweet for each sentiment
    :param dataframe: a dataframe containing data
    :param name_dataframe: a string with the name of the dataframe
    :display: a histogram of the number of words in a tweet for each sentiment
     """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    tweet_len = dataframe[dataframe['sentiment'] == "positive"]['text'].str.split().map(lambda x: len(x))
    ax1.hist(tweet_len, color='#17C37B')
    ax1.set_title('Positive Sentiments')

    tweet_len = dataframe[dataframe['sentiment'] == "negative"]['text'].str.split().map(lambda x: len(x))
    ax2.hist(tweet_len, color='#F92969')
    ax2.set_title('Negative Sentiments')

    tweet_len = dataframe[dataframe['sentiment'] == "neutral"]['text'].str.split().map(lambda x: len(x))
    ax3.hist(tweet_len, color='#FACA0C')
    ax3.set_title('Neutral Sentiments')

    fig.suptitle(f'Words in a tweet for each sentiment in the {name_dataframe} data')
    plt.savefig(f"out/words_in_tweets_for_each_sentiment_{name_dataframe}_data.png")
    plt.show()


def display_mean_length_words(dataframe, name_dataframe):
    """
    Function that allows to display a histogram of the mean length of words in a tweet for each sentiment
    :param dataframe: a dataframe containing data
    :param name_dataframe: a string with the name of the dataframe
    :display: a histogram of the mean length of words in a tweet for each sentiment
     """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    word = dataframe[dataframe['sentiment'] == "positive"]['text'].str.split().apply(lambda x: [len(i) for i in x])
    ax1.hist(word.map(lambda x: np.mean(x)), color='#17C37B')
    ax1.set_title('Positive')

    word = dataframe[dataframe['sentiment'] == "negative"]['text'].str.split().apply(lambda x: [len(i) for i in x])
    ax2.hist(word.map(lambda x: np.mean(x)), color='#F92969')
    ax2.set_title('Negative')

    word = dataframe[dataframe['sentiment'] == "neutral"]['text'].str.split().apply(lambda x: [len(i) for i in x])
    ax3.hist(word.map(lambda x: np.mean(x)), color='#FACA0C')
    ax3.set_title('Neutral')

    fig.suptitle(f'Mean word length in each tweet for each class in the {name_dataframe} data')
    fig.savefig(f'out/mean_word_length_in_each_tweet_for_each_class_{name_dataframe}_data.png')


def create_corpus(target, dataframe):
    """
    Function that allows to create a corpus with a certain sentiment
    :param target: a string with the name of sentiment
    :param dataframe: a dataframe containing data
    :return corpus: a list with the corpus
     """
    corpus = []
    for x in dataframe[dataframe['sentiment'] == target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


def display_the_use_punctuation_for_each_class(dataframe, sentiment, name_dataframe):
    """
    Function that allows to display a histogram of the use punctuation for each sentiment
    :param dataframe: a dataframe containing data
    :param name_dataframe: a string with the name of the dataframe
    :display: a histogram of the use punctuation for each sentiment
     """
    plt.figure(figsize=(10, 5))
    corpus = create_corpus(sentiment, dataframe)
    dic = defaultdict(int)

    special = string.punctuation
    for i in (corpus):
        if i in special:
            dic[i] += 1
    x, y = zip(*dic.items())

    if sentiment == "positive" or sentiment == 1:
        plt.title(f"Total number of uses of punctuation for all tweets for positive sentiment in the {name_dataframe} data")
        plt.bar(x, y, color='#17C37B')
        plt.savefig(f"out/number_punctuation_positive_{name_dataframe}_data.png")
    elif sentiment == "negative" or sentiment == -1:
        plt.title(f"Total number of uses of punctuation for all tweets for negative sentiment in the {name_dataframe} data")
        plt.bar(x, y, color='#F92969')
        plt.savefig(f"out/number_punctuation_negative_{name_dataframe}_data.png")
    else:
        plt.title(f"Total number of uses of punctuation for all tweets for neutral sentiment in the {name_dataframe} data")
        plt.bar(x, y, color='#FACA0C')
        plt.savefig(f"out/number_punctuation_neutral_{name_dataframe}_data.png")
    plt.show()


def get_frequencies(words):
    """
    Function that allows to get the frequencie of a list of words for each word
    :param words: a list of words
    :return: a dictionary with words and their frequencies
     """
    frequencies = {}
    for word in words.split(' '):
        if word in frequencies:
            frequencies[word] += 1
        else:
            frequencies[word] = 1
    return {key: value for key, value in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}


def word_cloud(frequencies, title):
    """
    Function that allows to display a word cloud with the most frequent words
    :param frequencies: a dictionary with words and their frequencies
    :param title: a string with the title of the word cloud
    :display: a word cloud with the most frequent words
     """
    wordcloud = WordCloud(width=1000, height=800, background_color='white').generate_from_frequencies(frequencies)
    plt.figure(figsize=(14, 8))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.title(title, fontsize=30, fontweight='bold')
    plt.savefig("out/"+title+".png")
    plt.show()
