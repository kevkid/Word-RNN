# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:11:20 2016

@author: kevin
"""

import random
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# load up our text
text = open('/path/to/input.txt', 'r').read()
unNeededChar = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+", ":", ";", '"']
for char in unNeededChar:
    text = text.replace(char, "")

fullWords = text.split()
words = list(set(text.split()))

# extract all (unique) characters
# these are our "categories" or "labels"
#chars = list(set(text))

# set a fixed vector size
# so we look at specific windows of characters
max_len = 20

step = 3
inputs = []
outputs = []
for i in range(0, len(fullWords) - max_len, step):
    inputs.append(fullWords[i:i+max_len])
    outputs.append(fullWords[i+max_len])
del fullWords#We dont need it
word_labels = {wd:i for i, wd in enumerate(words)}
labels_words = {i:wd for i, wd in enumerate(words)}


#Here we must cut the array in to parts, if we dont cut it we will run in to
#some memory error.
#Depending on the amount of memory you have you may have to cut it more than
#two times.
data_cut = 3

X = np.zeros((len(inputs)/data_cut, max_len, len(words)), dtype=np.bool)
y = np.zeros((len(inputs)/data_cut, len(words)), dtype=np.bool)


# set the appropriate indices to 1 in each one-hot vector
#for cuts in range(0,data_cut):#number of cuts
#    cutLocation = len(inputs)/data_cut*cuts#location of cut
#    start = (cutLocation)
#    end = (len(inputs)/data_cut) * (cuts + 1)
#    ins = (inputs[start:end])#first half
#    del X; del y
#    X = np.zeros((len(inputs)/data_cut, max_len, len(words)), dtype=np.bool)
#    y = np.zeros((len(inputs)/data_cut, len(words)), dtype=np.bool)
#    for i, example in enumerate(ins):
#        for t, word in enumerate(example):
#            X[i, t, word_labels[word]] = 1
#        y[i, word_labels[outputs[i]]] = 1


#Now we’ll define our RNN. Keras makes this trivial:

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(max_len, len(words))))
model.add(Dropout(0.5))
model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(len(words)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
# more epochs is usually better, but training can be very slow if not on a GPU
epochs = 100
model.load_weights("rapgodWeights.h5")
#model.fit(X, y, batch_size=64, nb_epoch=epochs)
#model.save_weights("rapgodWeights", overwrite=True)


##################Test area

def generate(temperature=0.35, seed=None, predicate=lambda x: len(x) < 100, max_length=max_len):
#    if seed is None and len(seed) < max_length:
#        raise Exception('Seed text must be at least {} chars long'.format(max_length))
#    # if no seed text is specified, randomly select a chunk of text
#    else:
#        start_idx = random.randint(0, len(text) - max_length - 1)
#        seed = text[start_idx:start_idx + max_length]
    sentence = seed.split()#should split to number of characters
    
    #while predicate(generated):
    # generate the input tensor
    # from the last max_len characters generated so far
    while len(sentence) < max_length:
        x = np.zeros((1, max_length, len(words)))
        for t, word in enumerate(sentence):
            #print word
            #print sentence
            x[0, t, word_labels[word]] = 1.
    
        # this produces a probability distribution over characters
        probs = model.predict(x, verbose=0)[0]
    
        # sample the character to use based on the predicted probabilities
        next_idx = sample(probs, temperature)
        next_word = labels_words[next_idx]
    
        
        sentence.append(next_word)
        len(sentence)
    return sentence

def sample(probs, temperature):
    """samples an index from a vector of probabilities
    (this is not the most efficient way but is more robust)"""
    a = np.log(probs)/temperature
    dist = np.exp(a)/np.sum(np.exp(a))
    choices = range(len(probs))
    return np.random.choice(choices, p=dist)
    
for i in range(epochs):
    print('epoch', i)
    #Here we must cut the array in to parts, if we dont cut it we will run in to
    #some memory error.
    #Depending on the amount of memory you have you may have to cut it more than
    #two times.
    for cuts in range(0,data_cut):#number of cuts
        cutLocation = len(inputs)/data_cut*cuts#location of cut
        start = (cutLocation)
        end = (len(inputs)/data_cut) * (cuts + 1)
        ins = (inputs[start:end])#first half
        del X; del y;
        X = np.zeros((len(inputs)/data_cut, max_len, len(words)), dtype=np.bool)
        y = np.zeros((len(inputs)/data_cut, len(words)), dtype=np.bool)
        for i, example in enumerate(ins):
            for t, word in enumerate(example):
                X[i, t, word_labels[word]] = 1
            y[i, word_labels[outputs[i]]] = 1
        # set nb_epoch to 1 since we're iterating manually
        model.fit(X, y, batch_size=2048, nb_epoch=1)
              #callbacks=[ModelCheckpoint("rapgodWeights.h5", monitor='val_loss'),
              #verbose=0, save_best_only=True, mode='auto')])
        model.save_weights("rapgodWeights.h5", overwrite=True)

    # preview
    for temp in [0.2, 0.5, 1., 1.2]:
        print('temperature:', temp)
        print(' '.join(generate(temperature=temp, seed="what is", max_length=20)))

#appending tests
genStr = "what is"
for i in range(0, 10):
    genStr += ' '.join(generate(temperature=0.2, seed=(' '.join(genStr.split()[-20:])), max_length=20))
print(genStr)

