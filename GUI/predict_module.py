from keras import backend as K
# K.clear_session()
#predict
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.models import load_model
from timeit import default_timer as timer
from Attention import Attention
 
model = InceptionV3()
# re-structure the model
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
model._make_predict_function()
graph = tf.get_default_graph()
# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	# load the photo
	image = load_img(filename, target_size=(299, 299))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	with graph.as_default():
		feature = model.predict(image, verbose=0)
	return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length, graph):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	with graph.as_default():
		for i in range(max_length):
			# integer encode input sequence
			sequence = tokenizer.texts_to_sequences([in_text])[0]
			# pad input
			sequence = pad_sequences([sequence], maxlen=max_length)
			# predict next word
			yhat = model.predict([photo,sequence], verbose=0)
			# convert probability to integer
			yhat = argmax(yhat)
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
# start = timer()
tokenizer = load(open('../tokenizerStyleNew15000_5000.pkl', 'rb'))
# print("Time to load tokenizer: ", timer()-start)
# pre-define the max sequence length (from training)
max_length = 32
# load the model
# start = timer()
# filename = "model-StyleNew15000_3000-Batch128-Thresh10-Attention-0.5Dropout-1BiLSTM-End-relu-Adam-ep015-loss2.531-val_loss3.394.h5"
# model = load_model('models/' + filename , custom_objects={'Attention': Attention})
# print("Time to load model: ", timer()-start)
# load and prepare the photograph
# inp = "../Flicker8k_Dataset/2039457436_fc30f5e1ce.jpg"
# start = timer()
# photo = extract_features(inp)
# print("Time to extract features: ", timer()-start)
# generate description
# start = timer()
# description = generate_desc(model, tokenizer, photo, max_length)
# print("Time to generate description: ", timer()-start)
# print(description)
