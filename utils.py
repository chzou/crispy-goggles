from keras.models import model_from_json
import h5py
import numpy as np
import pickle

def save_model(model, filename = 'model'):
	model_json = model.to_json()
	with open("{}.json".format(filename), "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("{}.h5".format(filename))
	print("Saved model to disk")

def pickle_save(obj, filename):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f)

def pickle_load(filename):
	with open(filename, 'rb') as f:
		obj = pickle.load(f)
		return obj

def load_model(filename = 'model'):
	# load json and create model
	json_file = open('{}.json'.format(filename), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("{}.h5".format(filename))
	print("Loaded model from disk")
	return loaded_model

def convert_to_toy(data, labels = None, frac = 0.01):
	current_size = data.shape[0]
	indices = np.random.choice(current_size, size = int(current_size * frac))
	toy_data = data[indices]
	if labels is not None:
		toy_labels = labels[indices]
		return toy_data, toy_labels
	return toy_data