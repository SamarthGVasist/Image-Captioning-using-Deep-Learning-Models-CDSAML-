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
from ptbtokenizer import PTBTokenizer
from Attention import Attention
from SPICE import Spice
from CIDEr import Cider

import sys
def update_progress(progress):
    barLength = 50 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if progress >= 1:
        progress = 1
        status = "Done.\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

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
#         yhat = beam_search_decoder(yhat, 3)
#         map integer to word
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

bleu_1_scores = []
bleu_2_scores = []
bleu_3_scores = []
bleu_4_scores = []
rouge_1_scores = []
rouge_2_scores = []
rouge_l_scores = []
rouge_w_scores = []
meteor_scores = []
spice_scores = []
cider_scores = []
epochs = []
tokenizerptb = PTBTokenizer()
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
	
#     gts = descriptions
#     print("Ground Truths: ", gts)
#     print("Predicted: ", res)
#     print("Empty")
#     for key,value in res.items():
#         if not value[0]["caption"].endswith("endseq"):
#             print(key, value[0]["caption"])

#     print('tokenization...')
    gts = tokenizerptb.tokenize(gts)
    res = tokenizerptb.tokenize(res)
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
#     print("spiceeeeeee", score)
    spice = score
    spice_scores.append(spice)
#     print("SPICE: %.4f" % score)
    
    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)
    cider = score
    cider_scores.append(cider)
    #     print("CIDEr: %.4f" % score)
    
    
    total=[]
    for i in range(len(actual)):
        actual1=[' '.join(actual[i][0]),' '.join(actual[i][1]),' '.join(actual[i][2]),' '.join(actual[i][3]),' '.join(actual[i][4])]
        predicted1=' '.join(predicted[i])
#         print(actual1,predicted1)
        total.append(round(nltk.translate.meteor_score.meteor_score(actual1,predicted1),4))
# 	print(total)
#     avg=sum(total)/len(total)
    meteor=sum(total)/len(total)
    meteor_scores.append(meteor)
#     print("METEOR: ",round(avg,4))
    bleu_1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    bleu_4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    bleu_1_scores.append(bleu_1)
    bleu_2_scores.append(bleu_2)
    bleu_3_scores.append(bleu_3)
    bleu_4_scores.append(bleu_4)
    # calculate BLEU score
    print("Epoch %d: "% epoch, end="")
    print('BLEU-1: %f,' % (bleu_1), end="")
    print(' BLEU-2: %f,' % (bleu_2), end="")
    print(' BLEU-3: %f,' % (bleu_3), end="")
    print(' BLEU-4: %f,' % (bleu_4), end="")

#     print(photos)
    test = generate_desc(model, tokenizer, all_features["2882056260_4399dd4d7c"], max_length)

#     print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
#     print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
#     print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

    actual2 = [i[0] for i in actual]
    _, _, rouge_1 = rouge_n_summary_level(predicted, actual2, 1)
    _, _, rouge_2 = rouge_n_summary_level(predicted, actual2, 2)
    _, _, rouge_l = rouge_l_summary_level(predicted, actual2)
    _, _, rouge_w = rouge_w_summary_level(predicted, actual2)
    print(" ROUGE-1: %f," % rouge_1, end="")
    print(" ROUGE-2: %f," % rouge_2, end="")
    print(' ROUGE-L: %f,' % rouge_l, end="")
    print(' ROUGE-W: %f,' % rouge_w, end="")
    print(" METEOR: %f," % meteor, end="")
    print(" SPICE: %f," % spice, end="")
    print(" CIDEr: %f" % cider)
    rouge_1_scores.append(rouge_1)
    rouge_2_scores.append(rouge_2)
    rouge_l_scores.append(rouge_l)
    rouge_w_scores.append(rouge_w)
    out.write("Epoch %d: , BLEU-1: %f, BLEU-2: %f, BLEU-3: %f, BLEU-4: %f, ROUGE-1: %f, ROUGE-2: %f, ROUGE-L: %f, ROUGE-W: %f, METEOR: %f, SPICE: %f, CIDEr: %f\n"% (epoch, bleu_1, bleu_2, bleu_3, bleu_4, rouge_1, rouge_2, rouge_l, rouge_w, meteor, spice, cider))
    print(test)
    out.write(test + "\n")
#     _, _, rouge_l = rouge_l_summary_level(predicted, actual)
#     print('ROUGE-L: %f' % rouge_l)
#     _, _, rouge_w = rouge_w_summary_level(predicted, actual)
#     print('ROUGE-W: %f' % rouge_w)

 
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
dump(tokenizer, open('tokenizerStyleNew15000_5000.pkl', 'wb'))
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
# out = open('output.txt', 'w')
# load the model
# filenames = ['models/model-StyleNew3600-Batch128-Thresh10-Attention-1BiLSTM-End-ep001-loss4.915-val_loss4.549.h5',
#              "models/model-StyleNew3600-Batch128-Thresh10-Attention-1BiLSTM-End-ep001-loss4.915-val_loss4.549.h5",
#             "models/",
#             "models/",
#             "models/",
#             "models/"]
all_features = load(open('featuresFlickr8k.pkl', 'rb'))
import os
# models = "model-StyleNew15000_5000-4CuDNNLSTM-ResNetV2-Try2-ep0"
models = "model"
model_name = "4 LSTM nogpu"
# folder = "models/"
folder = "models/" + model_name + "/"
model_name = model_name.replace("/", " ")
out = open('output_{}.txt'.format(models+" "+model_name), 'w')
filenames = [folder+filename for filename in os.listdir(folder) if filename.startswith(models)]
filenames.sort(key = lambda x: int((x.split("-ep0"))[1].split("-")[0]))
# filenames = filenames[10:]
# x = [i+1 for i in range(0,25)]
print("Testing %d models" % len(filenames))
out.write("Testing %d models\n" % len(filenames))
for filename in filenames:
  print(filename)
  out.write(filename+"\n")
  epoch = int((filename.split("-ep0"))[1].split("-")[0])
  epochs.append(epoch)
  model = load_model(filename, custom_objects={'Attention': Attention}, compile=False)
  # evaluate model
  evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
  K.clear_session()
import matplotlib.pyplot as plt
# plt.plot(epochs, bleu_1_scores)
# plt.plot(epochs, bleu_2_scores)
# plt.plot(epochs, bleu_3_scores)
# plt.plot(epochs, bleu_4_scores)
# plt.savefig('scores_{}.png'.format(models+"4LSTM"))
# plt.show()
b1 = sorted(bleu_1_scores, key=lambda x: epochs[bleu_1_scores.index(x)])
b2 = sorted(bleu_2_scores, key=lambda x: epochs[bleu_2_scores.index(x)])
b3 = sorted(bleu_3_scores, key=lambda x: epochs[bleu_3_scores.index(x)])
b4 = sorted(bleu_4_scores, key=lambda x: epochs[bleu_4_scores.index(x)])
r1 = sorted(rouge_1_scores, key=lambda x: epochs[rouge_1_scores.index(x)])
r2 = sorted(rouge_2_scores, key=lambda x: epochs[rouge_2_scores.index(x)])
rl = sorted(rouge_l_scores, key=lambda x: epochs[rouge_l_scores.index(x)])
rw = sorted(rouge_w_scores, key=lambda x: epochs[rouge_w_scores.index(x)])
s = sorted(spice_scores, key=lambda x: epochs[spice_scores.index(x)])
c = sorted(cider_scores, key=lambda x: epochs[cider_scores.index(x)])
m = sorted(meteor_scores, key=lambda x: epochs[meteor_scores.index(x)])
e = sorted(epochs)

plt.plot(e, b1, label="BLEU-1")
plt.plot(e, b2, label="BLEU-2")
plt.plot(e, b3, label="BLEU-3")
plt.plot(e, b4, label="BLEU-4")
plt.legend(loc='best')
plt.savefig('bleu_scores_{}.png'.format(models+" "+model_name))
plt.show()
plt.close()

plt.plot(e, r1, label="ROUGE-1")
plt.plot(e, r2, label="ROUGE-2")
plt.plot(e, rl, label="ROUGE-L")
plt.plot(e, rw, label="ROUGE-W")
plt.legend(loc='best')
plt.savefig('rouge_scores_{}.png'.format(models+" "+model_name))
plt.show()
plt.close()

plt.plot(e, s, label="SPICE")
plt.plot(e, c, label="CIDEr")
plt.plot(e, m, label="METEOR")
plt.legend(loc='best')
plt.savefig('spice_cider_scores_{}.png'.format(models+" "+model_name))
plt.show()
plt.close()
print(timer() - start)
out.close()