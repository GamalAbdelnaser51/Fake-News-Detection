import keras.models as model
from preprossingText import cleanup
import numpy as np
import warnings
warnings.simplefilter("ignore")
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import Counter
import os
import getEmbeddings2
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt


#function to run for prediction
def detecting_fake_news(t):
    text=cleanup(t)
    text=preproces(text)
    print(text)
    filename = '../SavedModels/model.h5'

    load_model = model.load_model(filename)

    prediction = load_model.predict(text)
    prob = load_model.predict_proba(text)
    print("The truth probability score is ",prob[0][0])
    if (prob[0][0]>=0.5):
        print("REAL")
    else:
        print("FAKE")

def preproces(text):
    top_words = 5000

    cnt = Counter()

    x_train = []
    x_train.append(text.split())
    for word in x_train[-1]:
        cnt[word] += 1

        # Storing most common words
    most_common = cnt.most_common(top_words + 1)
    word_bank = {}
    id_num = 1
    for word, freq in most_common:
        word_bank[word] = id_num
        id_num += 1

    # Encode the sentences
    for news in x_train:
        i = 0
        while i < len(news):
            if news[i] in word_bank:
                news[i] = word_bank[news[i]]
                i += 1
            else:
                del news[i]

    # Delete the short news
    i = 0
    while i < len(x_train):
        if len(x_train[i]) > 10:
            i += 1
        else:
            del x_train[i]

    # Truncate and pad input sequences
    max_review_length = 50
    return sequence.pad_sequences(x_train, maxlen=max_review_length)


if __name__ == '__main__':
    for i in range(10):
        text = input("Please enter the news text you want to verify: ")
        print("You entered: " + str(text))
        detecting_fake_news(text)
