# CHANTRE Honorine  CHAH2807
# THOMAS Eliott THOE2303

import pandas as pd
import preprocessor as p
import spacy
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


def preprocess_tweet(row):
    """
    Function that allows to preprocess tweet
    :param row: a string containing the orignal tweet
    :return text: the tweet preprocessed
     """
    text = row['OriginalTweet']
    new_text = ' '
    final_text = ''

    # We delete some punctuation for each row
    for i in range(len(text)):
        if text[i] not in [
                            '-', '.', 'Ã', '±', 'ã',
                            '¼', 'â', '»', '«', '§',
                            '$', "'", '(', ')', '+',
                            ',', '=', '^', '`', '|', '~']:
            new_text += text[i]

    # We clean the tweet, delete : URLs, Hashtags, Mentions, Reserved words (RT, FAV), Emojis and Smileys
    new_text = p.clean(new_text)    

    # We delete some common words for each row
    for word in new_text.split(' '):
        if word not in ['and', 'are']:
            final_text += word + ' '
    final_text = final_text[:-1]

    return final_text


def give_number_to_class(row, original_class):
    """
    Function that allows to give number to class instead of sentiment
    :param row: a string containing the orignal sentiment
    :param original_class: a boolean wich its true if we want 5 classes or false if we want 3 classes
    :return -1, 0, 1 or -2, -1, 0, 1, 2: the number of the class
    """
    sent = row['Sentiment']

    if not original_class:
        if sent == 'Extremely Negative' or sent == 'Negative':
            return -1
        elif sent == 'Neutral':
            return 0
        else:
            return 1

    else:
        if sent == 'Extremely Negative':
            return -2
        elif sent == 'Negative':
            return -1
        elif sent == 'Neutral':
            return 0
        elif sent == 'Positive':
            return 1
        else:
            return 2


def lemmatisation_spacy(text, nlp):
    """
    lemmatising with spacy
    :param text: a string containing the orignal tweet
    :param nlp: a spacy object
    :return out: the tweet lemmatised
    """
    doc = nlp(text)
    out = ""
    for token in doc:
        lemme = token.lemma_
        out += lemme+" "
    out = out[:-1]

    return out


def lemmatisation_nltk(text):
    """
    lemmatising with nltk
    :param text: a string containing the orignal tweet
    :return out: the tweet lemmatised
    """

    lemmatizer = WordNetLemmatizer()

    out = ""
    text = text.split()
    for word in text:
        lemme = lemmatizer.lemmatize(word)
        out += lemme+" "
    out = out[:-1]

    return out


def prepare_dataframe(file_name, original_class, lemmatising=None):
    """
    Function that allows to prepare the two dataframe for models
    :param file_name: a string containing the name of the file
    :param original_class: a boolean witch is true if we want 5 classes or false if we want 3 classes
    :param lemmatising: a boolean witch is true if we want lemmatising or false if we don't want
    :return X_df: a dataframe with the preprocessed text tweet
    :return Y_df: a dataframe with the preprocessed sentiment_number tweet
    """

    # We save data of the csv file in a dataframe
    data_df = pd.read_csv(file_name, sep=',', encoding='latin')

    # We drop the column Location
    data_df = data_df.drop(['Location'], axis=1)
    # We drop the missing values
    data_df.dropna(inplace=True)
    # We drop the duplicates
    data_df.drop_duplicates()

    # We apply the preprocess_tweet function to the dataframe
    data_df['OriginalTweet'] = data_df.apply(preprocess_tweet, axis=1)

    # We compute the text len for each tweet
    text_len = []
    for text in data_df['OriginalTweet']:
        tweet_len = len(text.split())
        text_len.append(tweet_len)
    data_df['text_len'] = text_len

    # We apply the give_number_to_class function to the dataframe
    data_df['Sentiment_Number'] = data_df.apply(lambda x: give_number_to_class(x, original_class), axis=1)
    # We only keep the tweet with lenght > 4 characters
    data_df = data_df[data_df['text_len'] > 4].reset_index()

    X_df = data_df['OriginalTweet']
    y_df = data_df['Sentiment_Number']

    # Lemmatisation
    # if lemmatising is different than spacy and nltk there is no lemmatisation:
    if lemmatising == 'spacy':
        nlp = spacy.load('en_core_web_sm')
        tqdm.pandas()
        X_df = X_df.progress_apply(lemmatisation_spacy, args=(nlp,))

    elif lemmatising == 'nltk':
        tqdm.pandas()
        X_df = X_df.progress_apply(lemmatisation_nltk)

    return X_df, y_df
