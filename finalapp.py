import os

from uuid import uuid4

import jinja2
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
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.models import load_model
from timeit import default_timer as timer
from flask import render_template
from predict_module import extract_features,word_for_id,generate_desc
from Attention import Attention

from flask import Flask, request, render_template, send_from_directory

# __author__ = 'ibininja'

app = Flask(__name__)

# app = Flask(__name__, static_folder="images")


# app.jinja_loader=jinja2.FileSystemLoader(r'C:\Users\SAMARTH G VASIST\flask projects\templates')

APP_ROOT = os.path.dirname(os.path.abspath("__file__"))
all_features = load(open('featuresFlickr8k.pkl', 'rb'))


@app.route("/")
def index():

    return render_template("default.html")



@app.route("/upload", methods=["POST"])
def upload():

    target = os.path.join(APP_ROOT, 'images/')

    # target = os.path.join(APP_ROOT, 'static/')

    print(target)

    if not os.path.isdir(target):
            os.mkdir(target)

    else:
        print("Couldn't create upload directory: {}".format(target))

    print(request.files.getlist("file"))

    for upload in request.files.getlist("file"):

        print(upload)

        print("{} is the file name".format(upload.filename))

        filename = upload.filename

        destination = "/".join([target, filename])

        print ("Accept incoming file:", filename)

        print ("Save it to:", destination)

        upload.save(destination)

        print(filename)
        start = timer()
        tokenizer = load(open('tokenizerStyleNew15000_5000.pkl', 'rb'))
    #print("Time to load tokenizer: ", timer()-start)
# pre-define the max sequence length (from training)
        max_length = 32
# load the model
        start = timer()
        K.clear_session()
        model = load_model("models/4LSTM/model-StyleNew15000_3000-Batch128-Thresh10-Attention-0.5Dropout-1BiLSTM-End-relu-Adam-ep001-loss4.955-val_loss4.465.h5", custom_objects={'Attention': Attention})
        print("Time to load model: ", timer()-start)
# load and prepare the photograph
        inp = "../Flicker8k_Dataset/"
        inp=inp+filename
        # photo = extract_features(inp)
        start = timer()
        try:
            photo = all_features[filename.replace(".jpg", "")]
        except KeyError:
            photo = extract_features(destination)
        print("Time to extract features: ", timer()-start)
# generate description
        start = timer()
        description = generate_desc(model, tokenizer, photo, max_length)
        description = description.replace("startseq ", "").replace(" endseq", "")
        print("Time to generate description: ", timer()-start)
        #return description
        K.clear_session()

    # return send_from_directory("images", filename, as_attachment=True)

        return render_template("complete.html", image_name=filename,caption=description)

@app.route('/upload/<filename>')
def send_image(filename):
    #print(send_from_directory("images", filename))
    return send_from_directory("images", filename)

if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0')