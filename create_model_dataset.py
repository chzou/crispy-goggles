from utils import *
from keras.datasets import imdb
from keras.preprocessing import sequence

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def construct_dataset(model, full_data, subset_size = 1000, threshold = 0.5, dataset = 'imdb'):
	total_entries = full_data.shape[0]
	frac = subset_size/total_entries
	data_subset = convert_to_toy(full_data, frac = frac)
	#80 hard coded...
	data_subset = sequence.pad_sequences(data_subset, maxlen=80)
	predictions = model.predict(data_subset, batch_size = 1)
	bin_predictions = (predictions > threshold)

	data_filename = '{}_{}_data.pkl'.format(subset_size, dataset)
	label_filename = '{}_{}_labels.pkl'.format(subset_size, dataset)
	pickle_save(data_subset, data_filename)
	pickle_save(bin_predictions, label_filename)

def construct_imdb_dataset(model = None, subset_size = 1000, threshold = 0.5, max_features = 20000):
	(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
	if model is None:
		model = load_model()
	construct_dataset(model, x_train, subset_size = subset_size)

def load_dataset(subset_size, dataset = 'imdb'):
	data_filename = '{}_{}_data.pkl'.format(subset_size, dataset)
	label_filename = '{}_{}_labels.pkl'.format(subset_size, dataset)
	data = pickle_load(data_filename)
	labels = pickle_load(label_filename)
	return data, labels

construct_imdb_dataset(subset_size = 10)
load_dataset(10)