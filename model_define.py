from keras import backend as K
K.clear_session()
#predict
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import GRU, LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from timeit import default_timer as timer
from Attention import Attention

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
 
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos):
    X1, X2, y = list(), list(), list()
    # walk through each image identifier
    for key, desc_list in descriptions.items():
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)

# # define the captioning model
# def define_model(vocab_size, max_length):
#     # feature extractor model
#     inputs1 = Input(shape=(2048,))
#     fe1 = Dropout(0.5)(inputs1)
# #     fe1 = inputs1
#     fe2 = Dense(256, activation='relu')(fe1)
#     fe3 = RepeatVector(max_length)(fe2)
#     # sequence model
#     inputs2 = Input(shape=(max_length,))
#     se1 = Embedding(vocab_size, 256)(inputs2)
#     # se2 = Dropout(0.5)(se1)
#     # se3 = GRU(256,return_sequences=True)(se2)
#     # se4 = LSTM(256,return_sequences=True)(se3)
#     # se5 = GRU(256,return_sequences=True)(se4)
#     # se6 = LSTM(256)(se5)
#     # decoder model
#     # decoder1 = add([fe2, se6])
#     # decoder2 = Dense(256, activation='relu')(decoder1)
#     # outputs = Dense(vocab_size, activation='softmax')(decoder2)
#     merged = concatenate([fe3, se1])
# #     lm1 = Conv1D(256, kernel_size=2)(merged)
# #     lm2 = GRU(256, return_sequences=True)(lm1)
# #     x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25,
# #                            recurrent_dropout=0.25))(merged)
# # #     x = AttentionDecoder(256, vocab_size)(x)
# #     x = attention_3d_block(x, max_length)
# #     x = Flatten()(x)
#     x = Bidirectional(LSTM(500, return_sequences=True, dropout=0.5,
#                            recurrent_dropout=0.5))(merged)
#     x = Attention(max_length)(x)
# #     x = Dense(256, activation="relu")(x)
# #     x = Dropout(0.25)(x)
# #     x = Dense(256, activation="sigmoid")(x)
#     x = Dense(256, activation='relu')(x)
#     x = Dense(vocab_size, activation="softmax")(x)
# #     x = Dense(256, activation="relu")(x)
# #     x = Dropout(0.25)(x)
# #     x = Dense(256, activation="sigmoid")(x)
# #     lm3 = LSTM(256)(lm2)
#     #lm3 = Dense(500, activation='relu')(lm2)
# #     outputs = Dense(vocab_size, activation='softmax')(x)
#     # tie it together [image, seq] [word]
#     model = Model(inputs=[inputs1, inputs2], outputs=x)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
#     # summarize model
#     print(model.summary())
#     plot_model(model, to_file='model.png', show_shapes=True)
#     input()
#     return model
def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256,return_sequences=True)(se2)
    se4 = LSTM(256,return_sequences=True)(se3)
    se5 = LSTM(256,return_sequences=True)(se4)
    se6 = LSTM(256,return_sequences=True)(se5)
#     se6 = CuDNNLSTM(256,return_sequences=True)(se5)
#     se7 = CuDNNLSTM(256,return_sequences=True)(se6)
    se7 = Attention(max_length)(se6)
    # decoder model
    decoder1 = add([fe2, se7])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model
 
# train dataset
 
# load training dataset (6K)
filename = 'TrainingImagesStyle5K.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptionsforstyle5K.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('featuresFlickr8k.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)
 
# dev dataset
 
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
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)
 
# fit model
 
# define the model
model = define_model(vocab_size, max_length)
# define checkpoint callback
filepath = 'models/model-Style-4LSTM-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
# fit model
model.fit([X1train, X2train], ytrain, epochs=20, verbose=1, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest), batch_size=128)

print(timer() - start)