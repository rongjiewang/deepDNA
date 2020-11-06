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
train_path = './data/train.fasta'
valid_path = './data/valid.fasta'
test_path = './data/test.fasta'


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
def readFasta(filename):
    reads = []
    with open(filename, "rU") as handle:
        for record in SeqIO.parse(handle, "fasta") :
                reads.append(record)
    return reads
def read_fasta(data_path):
    records = list(SeqIO.parse(data_path, "fasta"))
    text = ""
    for record in records:
        text += str(record.seq)
    return text

def read_data(text):
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
# cut the text in semi-redundant sequences of maxlen characters


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
    json_file = open('./model/model_0_1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./model/model_0_1.h5")
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    print("Loaded model from disk")
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

model = loadModel()
def on_test(text):
    # Function invoked at end of each epoch. Prints generated text.
    entropy = 0
    batch_num = 0
    for i, batch in enumerate(get_batch(read_data(text))):
        _input = batch[0]
        _labels = batch[1]
        x=model.test_on_batch(_input,_labels)
        entropy += x
        batch_num = i
    return entropy/batch_num*math.log(math.e, 2)

entropy = []
for i, record in enumerate(readFasta(test_path)):        
    testEntropy = on_test(record.seq)
    print(i,'\t',record.name,'\t',len(record.seq),'\t', testEntropy)
    entropy.append(testEntropy)
print("Total average is:")    
print(np.mean(entropy))





