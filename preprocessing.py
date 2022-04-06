# CHANTRE Honorine  CHAH2807
# THOMAS Eliott THOE2303

import pandas as pd
import preprocessor as p


def preprocess_tweet(row):
    """
    Function that allows to clean the tweet, delete : URLs, Hashtags, Mentions, Reserved words (RT, FAV), Emojis and Smileys
    :param row: a string containing the orignal tweet
    :return text: the tweet cleaned
     """
    text = row['OriginalTweet']
    listToStr = ' '
    
    # We delete the hyphen of each row
    for i in range(len(text)):
        if text[i] != '-':
            listToStr += text[i]            
    
    listToStr = p.clean(listToStr)
    
    return listToStr


def give_number_to_class(row, original_class):
    """
    Function that allows to give number to class instead of sentiment
    :param row: a string containing the orignal sentiment
    :param original_class: a boolean wich its true if we want 5 classes or false if we want 3 classes
    :return -1, 0, 1 or -2, -1, 0, 1, 2: the number of the class
    """
    sent = row['Sentiment']
    
    if original_class == False:
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
        else :
            return 2


def prepare_dataframe(file_name, original_class, lemmatising=False):
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
    data_df.dropna()
    # We drop the duplicates
    data_df.drop_duplicates()

    # We apply the preprocess_tweet function to the dataframe
    data_df['OriginalTweet'] = data_df.apply(preprocess_tweet, axis=1)
    # We apply the give_number_to_class function to the dataframe
    data_df['Sentiment_Number'] = data_df.apply(lambda x: give_number_to_class(x, original_class), axis=1)
    
    X_df = data_df['OriginalTweet']
    Y_df = data_df['Sentiment_Number']
    
    # Lemmatisation

    if lemmatising:
        nlp = spacy.load('en_core_web_sm')

        tqdm.pandas()
        X_df = X_df.progress_apply(lemmatisation , args=(nlp,)) 

    return X_df, Y_df
