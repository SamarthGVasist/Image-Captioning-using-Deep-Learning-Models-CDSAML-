from keras import backend as K
K.clear_session()
#predict
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from pickle import load
from numpy import argmax
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.models import load_model
from timeit import default_timer as timer
from Attention import Attention
 
# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = InceptionV3()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(299, 299))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature
 
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
#         print(yhat)
#         print("Before: ", np.shape(yhat))
#         yhat = beam_search_decoder(yhat, 3)
#         print(np.shape(yhat))
#         print(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text
 
# load the tokenizer
start = timer()
tokenizer = load(open('tokenizerStyleNew15000_3000.pkl', 'rb'))
print("Time to load tokenizer: ", timer()-start)
# pre-define the max sequence length (from training)
max_length = 32
inp = "../Flicker8k_Dataset/152029243_b3582c36fa.jpg"
start = timer()
photo = extract_features(inp)
print("Time to extract features: ", timer()-start)
# load the model
# start = timer()
# filename = "model-StyleNew3600-Batch128-Thresh10-Attention-0.5Dropout-1BiLSTM-End-relu-Adam-ep020-loss2.260-val_loss3.917.h5"
# model = load_model('/gdrive/My Drive/Datasets/models/' + filename , custom_objects={'Attention': Attention})
# print("Time to load model: ", timer()-start)
# load and prepare the photograph
# generate description
import os
epochs = []
models = "model-StyleNew15000_3000-Batch128-Thresh10-Attention-0.5Dropout-1BiLSTM-End-relu-Adam-ep0"
filenames = ["models/"+filename for filename in os.listdir('models') if filename.startswith(models)]
# x = [i+1 for i in range(0,25)]
print(filenames)
for filename in filenames:
#   print(filename)
  epoch = int((filename.split("-ep0"))[1].split("-")[0])
  epochs.append(epoch)
  model = load_model(filename, custom_objects={'Attention': Attention}, compile=False)
#   start = timer()
  description = generate_desc(model, tokenizer, photo, max_length)
#   print("Time to generate description: ", timer()-start)
  print("Epoch %d:"%epoch, description)
  # evaluate model
#   evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
  K.clear_session()
# import matplotlib.py
# start = timer()
# description = generate_desc(model, tokenizer, photo, max_length)
# print("Time to generate description: ", timer()-start)
# print(description)
