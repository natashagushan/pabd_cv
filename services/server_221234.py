""" Module docstring """
from flask import Flask, request
import json
import tensorflow as tf
import numpy as np


app = Flask('Image classifier')
resnet = tf.keras.applications.ResNet101()
with open('../data/imgnet_cats_en.txt') as f:
    cats = f.readlines()

cats_en = [s.rstrip() for s in cats]


@app.route('/')
def home():
    """ Function docstring """
    return 'Home page'


@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_data()
    img = tf.io.decode_jpeg(data)
    img_t = tf.convert_to_tensor(
        img, dtype=None, dtype_hint=None, name=None
    )
    img_t = tf.expand_dims(img_t, axis=0)
    img_t = tf.image.resize(img_t, (224, 224))
    out = resnet(img_t)
    idxs = tf.argsort(out)[0][:3].numpy()
    out = ''.join([cats_en[int(i)] for i in idxs])
    return out


if __name__ == '__main__':
    app.run(port=1234)        #номер зачетки
    input()
