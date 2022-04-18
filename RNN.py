# CHANTRE Honorine  CHAH2807
# THOMAS Eliott THOE2303


import numpy as np
import plotly.express as px
import tensorflow as tf
import tensorflow.keras.layers as L
from sklearn import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# hyper parameters

EPOCHS = 2
BATCH_SIZE = 32
embedding_dim = 16
units = 256
dp = 0.4
n_neurons = 64


class RNN():

    def __init__(self, X_train, y_train, y_test,
                 embedding_dim=embedding_dim, units=units,
                 dropout=dp, n_neurons=n_neurons, optimize=False
                 ):
        """
        RNN class constructor.
        :param X_train: The training data.
        :param y_train: The training labels.
        :param y_test: The test labels.
        :param embedding_dim: The embedding dimension.
        :param units: The number of units.
        :param dropout: The dropout rate.
        :param n_neurons: The number of neurons.
        :param optimize: If True, the hyperparameters will be optimized.        
        """

        if not optimize:
            print("##### CREATING RNN #####")

        self.X_train = X_train  # X_train is a list of strings
        self.y_train = y_train
        self.y_test = y_test

        self.history = None  # history of the training

        self.tokenizer = Tokenizer()  # tokenizer to tokenize the strings
        self.tokenizer.fit_on_texts(X_train)  # fit the tokenizer on the training data
        self.X = self.tokenizer.texts_to_sequences(X_train)
        self.vocab_size = len(self.tokenizer.word_index)+1  # the number of words in the vocabulary

        if not optimize:
            print("Vocabulary size: {}".format(self.vocab_size))
            print("\nExample:\n") 
            print("Sentence:\n{}".format(self.X_train[6]))
            print("\nAfter tokenizing :\n{}".format(self.X[6]))

        self.X = pad_sequences(self.X, padding='post')  # padding = 'post' means that the sentence will be padded at the end of the sequence
        if not optimize:
            print("\nAfter padding :\n{}".format(self.X[6]))  # the sequence will be padded with 0 at the end
        self.pad_len = self.X[0].shape[0]  # the length of the longest tweet

        tf.keras.backend.clear_session()  # clear the session

        self.model = tf.keras.Sequential([
            L.Embedding(self.vocab_size, embedding_dim, input_length=self.X.shape[1]),  # input_length = self.X.shape[1] means that the input will be of shape (batch_size, input_length)
            L.Bidirectional(L.LSTM(units, return_sequences=True)),  # return_sequences = True means that the output will be a sequence of vectors
            L.GlobalMaxPool1D(),  # GlobalMaxPool1D() is a layer that takes the maximum value of the output of the previous layer across the sequence dimension.
            L.Dropout(dropout),  # dropout to avoid overfitting
            L.Dense(n_neurons, activation="relu"),  # n_neurons neurons in the hidden layer
            L.Dropout(dropout),  # Dropout is a way to prevent overfitting
            L.Dense(3)  # 3 classes
        ])

        self.model.compile(
                            loss="mean_squared_error",
                            optimizer='adam', metrics=['accuracy']
                            )
        if not optimize:
            self.model.summary() 
            print("##### RNN CREATED #####") 
            print("##### Preparing the training data #####")

        ohe = preprocessing.OneHotEncoder()  # one hot encoder to transform the labels into vectors

        self.y_train = ohe.fit_transform(np.array(self.y_train).reshape(-1, 1)).toarray()  # fit the encoder on the training data and transform the labels into vectors
        self.y_test = ohe.fit_transform(np.array(self.y_test).reshape(-1, 1)).toarray()  # fit the encoder on the test data and transform the labels into vectors
        if not optimize:
            print("##### training data prepared #####")

    def train(self, epochs=EPOCHS, batch_size=BATCH_SIZE, optimize=False):
        """
        Train the model.
        :param epochs: The number of epochs.
        :param batch_size: The batch size.
        :param optimize: If True, the hyperparameters will be optimized.
        :return: The history of the training.
        """
        if not optimize:
            print("##### TRAINING #####")
        self.history = self.model.fit(self.X, self.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)  # validation_split = 0.1 means that 10% of the data will be used for validation
        if not optimize:
            print("##### TRAINING COMPLETED #####")
        return self.history.history['val_accuracy'][-1]  # return the accuracy of the model on the validation set at the end of the training

    def print_history(self):
        """
        Plot the history of the training.
        :display: The history of the training.
        """
        print("##### PRINTING HISTORY #####")
        fig = px.line(  # plot the accuracy
            self.history.history, y=['accuracy', 'val_accuracy'],
            labels={'index': 'epoch', 'value': 'accuracy'}
        )

        fig.show()

        fig = px.line(  # plot the loss
            self.history.history, y=['loss', 'val_loss'],
            labels={'index': 'epoch', 'value': 'loss'}
        )

        fig.show()

        print("##### PRINTING HISTORY COMPLETED #####")

    def test(self, X_test):
        """
        Test the model.
        :param X_test: The test data.
        :return: The loss and accuracy of the model on the test data.
        """
        print("##### TESTING #####")
        X_test = self.tokenizer.texts_to_sequences(X_test)  # tokenize the test data
        X_test = pad_sequences(X_test, padding='post', maxlen=self.pad_len)  # we have to precise the maxlen because the test max size is not the same length as the training max size

        loss, acc = self.model.evaluate(X_test, self.y_test, verbose=1)  # evaluate the model on the test data

        print('Test loss: {}'.format(loss))
        print('Test Accuracy: {}'.format(acc))
        print("##### TESTING COMPLETED #####")
        return loss, acc
