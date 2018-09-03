'''Author: Rongjie Wang'''

from __future__ import print_function
from keras.callbacks import LambdaCallback, Callback
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import model_from_json
import numpy as np
import random
import sys
import io
import math
import keras
from keras.models import load_model
from keras import backend as K
from itertools import product
from Bio import SeqIO
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
np.random.seed(1337) # for reproducibility
train_path = './train.fasta'
valid_path = './valid.fasta'
test_path = './test.fasta'


chars = "ACGT"
print('total chars:', len(chars))
print('chars:', chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 64
step = 1
batch_size = 64
epochs = 10
input_dim = len(chars)
def read_fasta(data_path):
    records = list(SeqIO.parse(data_path, "fasta"))
    text = ""
    for record in records:
        text += str(record.seq)
    return text

def read_data(data_path):
    text = read_fasta(data_path)
    for i in range(0, len(text) - maxlen, step):
        sentence = text[i: i + maxlen]
        next_char = text[i + maxlen]
        yield sentence, next_char
def vectorization(sentences, next_chars):
    x = np.zeros((batch_size, maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((batch_size, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return x, y
def get_batch(stream):
    sentences = []
    next_chars = []
    for sentence, next_char in stream:
        sentences.append(sentence)
        next_chars.append(next_char)
        if len(sentences) == batch_size:
            data_tuple = vectorization(sentences,next_chars)
            yield data_tuple
            sentences = []
            next_chars = []

def my_kernel_initializer(shape, dtype=None):
    x = np.zeros(shape, dtype=np.bool)
    for i, c in enumerate(product('ACGT', repeat=5)):
        kmer=c*3
        for t, char in enumerate(kmer):
            x[t,char_indices[char],i] = 1
    return x

def loadModel():
    #model.load_weights('my_model_weights.h5')
    #json and create model
    json_file = open('./model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./model/model.h5")
    print("Loaded model from disk")
    return model


def model_patern():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     #kernel_initializer=my_kernel_initializer,
                     #trainable=False,
                     #padding='same',
                     #activation=None,
                     #use_bias=False,
                     #bias_initializer= keras.initializers.Constant(value=-7),
                     strides=1))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.1))
    model.add(LSTM(256,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def model_CNN_LSTM():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.1))
    model.add(LSTM(256,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def model_CNN():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(Conv1D(filters=320,
                     kernel_size=6,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=3,strides=3))
    model.add(Dropout(0.1))

    model.add(Conv1D(filters=480,
                     kernel_size=4,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=2,strides=2))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=960,
                     kernel_size=4,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
def model_LSTM():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(input_dim, activation='softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def saveModel(epoch):
    # serialize model to JSON
    model_json = model.to_json()
    with open("./model/model.json", "w") as json_file:
        json_file.write(model_json)
    #serialize weights to HDF5
    name="./model/model_"+str(epoch)+".h5"
    model.save_weights(name)
    print("Saved model to disk")
    return

model = model_CNN_LSTM()
def on_epoch_end(epoch):
    # Function invoked at end of each epoch. Prints generated text.
    print('----- Testing entorpy after Epoch: %d' % epoch)
    entropy = 0
    batch_num = 0
    for i, batch in enumerate(get_batch(read_data(valid_path))):
        _input = batch[0]
        _labels = batch[1]
        x=model.test_on_batch(_input,_labels)
        entropy += x
        batch_num = i
    return entropy/batch_num*math.log(math.e, 2)

entropy = []
for epoch in range(epochs):
    print("this is epoch: ", epoch)
    for i, batch in enumerate(get_batch(read_data(train_path))):
        _input = batch[0]
        _labels = batch[1]
        x=model.train_on_batch(_input,_labels)
        if(i%100==0):
            print(epoch,'\t', x*math.log(math.e,2))
    saveModel(epoch)
    testEntropy = on_epoch_end(epoch)
    print(testEntropy)
    entropy.append(testEntropy)
print(entropy)





