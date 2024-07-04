from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.models import load_model

app = Flask(__name__)

model = load_model('handwriting_model_final/handwriting_model_final.h5', compile=False)

def preprocess(img):
    (h, w) = img.shape
    final_img = np.ones([64, 256]) * 255
    if w > 256:
        img = img[:, :256]
    if h > 64:
        img = img[:64, :]
    final_img[:h, :w] = img
    final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
    return final_img

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1: 
            break
        else:
            ret += alphabets[ch]
    return ret


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    prediction = ""
    file_path = ""
    file = ""
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = image / 255.
            pred = model.predict(image.reshape(1, 256, 64, 1))
            decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])
            prediction = num_to_label(decoded[0])
            file_path = file_path.replace("\\", "/")
        
    return render_template('upload.html', predict=prediction, file=file_path)

if __name__ == "__main__":
    app.run(debug=True)