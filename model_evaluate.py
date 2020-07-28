from keras import backend as K
K.clear_session()
#predict
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from pickle import dump
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from timeit import default_timer as timer
from rouge import rouge_n_summary_level
from rouge import rouge_w_summary_level
from rouge import rouge_l_summary_level
import nltk.translate.meteor_score
from math import log
from SPICE import Spice, PTBTokenizer
from CIDEr import Cider
from Attention import Attention
# from ptbtokenizer import PTBTokenizer

start = timer()
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)
 
# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions
 
# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features
 
# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
 
# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)
 
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
import sys
def update_progress(progress):
    barLength = 50 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
# for i in range(100):
#     time.sleep(0.1)
#     update_progress(i/100.0)

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
#         yhat = beam_search_decoder(yhat, 3)
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
 
# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    gts = {}
    res = {}
    count = 0
    length_desc = len(descriptions)
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)

        if key not in res:
            res[key] = []
        res[key].append({"caption": yhat})
        
        if key not in gts:
          gts[key] = []
        for desc in desc_list:
          gts[key].append({"caption": desc})

        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
        count+=1
        update_progress(count/length_desc)
    # gts = descriptions
    print("Ground Truths: ", gts)
    print("Predicted: ", res)

    print('tokenization...')
    tokenizer = PTBTokenizer()
    gts  = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print("SPICE: %.4f" % score)
    
    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)
    print("CIDEr: %.4f" % score)
    
#     print(actual)
#     print(predicted)
    total=[]
    for i in range(len(actual)):
        actual1=[' '.join(actual[i][0]),' '.join(actual[i][1]),' '.join(actual[i][2]),' '.join(actual[i][3]),' '.join(actual[i][4])]
        predicted1=' '.join(predicted[i])
#         print(actual1,predicted1)
        total.append(round(nltk.translate.meteor_score.meteor_score(actual1,predicted1),4))
# 	print(total)
    avg=sum(total)/len(total)
    print("METEOR: ",round(avg,4))
    
#     calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

    actual2 = [i[0] for i in actual]
    # print("ACTUAL: ", actual)
    # print("PREDICTED: ", predicted)
    _, _, rouge_1 = rouge_n_summary_level(predicted, actual2, 1)
    _, _, rouge_2 = rouge_n_summary_level(predicted, actual2, 2)
    print("ROUGE-1: %f" % rouge_1)
    print("ROUGE-2: %f" % rouge_2)
    _, _, rouge_l = rouge_l_summary_level(predicted, actual2)
    print('ROUGE-L: %f' % rouge_l)
    _, _, rouge_w = rouge_w_summary_level(predicted, actual2)
    print('ROUGE-W: %f' % rouge_w)

 
# prepare tokenizer on train set
 
# load training dataset (6K)
filename = 'TrainingImagesStyle5K.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptionsforstyle5K.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
# save the tokenizer
dump(tokenizer, open('tokenizerStyleNew15000_3000.pkl', 'wb'))
# obtain vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
 
# prepare test set

# load test set
filename = 'TestingImagesStyle5K.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptionsforstyle5K.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('featuresFlickr8k.pkl', test)
print('Photos: test=%d' % len(test_features))
 
# load the model
filename = 'models/weights/model-StyleNew15000_3000-Batch128-Thresh10-Attention-0.5Dropout-1BiLSTM-End-relu-Adam-ep020-loss2.555-val_loss3.695.h5'
model = load_model(filename, custom_objects={'Attention': Attention}, compile=False)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

print(timer() - start)