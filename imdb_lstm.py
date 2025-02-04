'''
#Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

**Notes**

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.

'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

from keras.models import model_from_json
from utils import save_model, load_model, convert_to_toy, train_test_split
import h5py
import numpy as np

# UNCOMMENT FOR MIMIC
#from create_model_dataset import load_dataset


#[Erik] added for some weird bug on my computer. 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


max_features = 10000
# cut texts after this number of words (among top max_features most common words)
maxlen = 250
batch_size = 32
num_epochs = 10
toy = False

mimic = False
if mimic:
	print("Loading for mimic model")
	subset_size = 10 #REPLACE WITH WHAT IT ACTUALLY IS
	data, labels = None, None
        # UNCOMMENT FOR MIMIC
        #data, labels = load_dataset(subset_size)
	(x_train, y_train), (x_test, y_test) = train_test_split(data, labels, test_frac  = 0.2)

else:

	print('Loading data from imdb...')
	(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

	if toy:
		print("Using toy...")
		x_train, y_train = convert_to_toy(x_train, labels = y_train)
		x_test, y_test = convert_to_toy(x_test, labels = y_test)

	print(len(x_train), 'train sequences')
	print(len(x_test), 'test sequences')

	print('Pad sequences (samples x time)')
	x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
	x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')
	print('x_train shape:', x_train.shape)
	print('x_test shape:', x_test.shape)

#IF should used saved data. In this case, we're building the mimic model

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

print("Saving model...")
save_model(model, filename = 'trained_model')
#loaded_model = load_model()

# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#score = loaded_model.evaluate(x_test, y_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
