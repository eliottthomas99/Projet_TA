# CHANTRE Honorine  CHAH2807
# THOMAS Eliott THOE2303

import pandas as pd
import preprocessor as p


def preprocess_tweet(row):
    """
    Function that allows to clean the tweet, delete : URLs, Hashtags, Mentions, Reserved words (RT, FAV), Emojis and Smileys
    :return text: the tweet cleaned
     """
    text = row['OriginalTweet']
    text = p.clean(text)

    return text


def give_number_to_class(row):
    """
    Function that allows to give number to class instead of sentiment
    :return -1, 0, 1: the number of the class
    """
    sent = row['Sentiment']
    if sent == 'Extremely Negative' or sent == 'Negative':
        return -1
    elif sent == 'Neutral':
        return 0
    else:
        return 1


def prepare_dataframe(file_name):
    """
    Function that allows to prepare the two dataframe for models
    :return X_df: a dataframe with the preprocessed text tweet
    :return Y_df: a dataframe with the preprocessed sentiment_number tweet
    """

    # We save data of the csv file in a dataframe
    data_df = pd.read_csv(file_name, sep=',', encoding='latin')

    # We drop the column Location
    data_df = data_df.drop(['Location'], axis=1)
    # We drop the missing values
    data_df.dropna()
    # We drop the duplicates
    data_df.drop_duplicates()

    # We apply the preprocess_tweet function to the dataframe
    data_df['OriginalTweet'] = data_df.apply(preprocess_tweet, axis=1)
    # We apply the give_number_to_class function to the dataframe
    data_df['Sentiment_Number'] = data_df.apply(give_number_to_class, axis=1)

    X_df = data_df['OriginalTweet']
    Y_df = data_df['Sentiment_Number']

    return X_df, Y_df
