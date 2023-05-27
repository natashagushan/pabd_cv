import io
import unittest

import PIL.Image
import requests
from flack import Flask, request
import tensorflow as tf

class MyTestCase(unittest.TestCase):
    def setUp(self):
        app.run()

    def test_home(self):
        response = requests.request('GET', 'http://localhost:1774/')
        sample = response.content.decode()
        self.assertEqual(sample, 'Home page')

app = Flask('Image classifier')
resnet = tf.keras.applications.ResNet101()
with open('data/imgnet_cats_ru.txt', encoding="utf-8") as f:
    cats = f.readlines()

categories_ru = [s.rstrip() for s in cats]

model = tf.keras.models.load_model('models/dummy_model')

@app.route('/')
def home():
    return 'Home page'


@app.route('/classify/imgnet', methods = ['POST', 'GET'])
def classify():
    data = request.data
    img = tf.io.decode_jpeg(data)
    img_t = tf.expand_dims(img, axis=0)
    img_t = tf.image.resize(img_t, (224, 224))
    out = resnet(img_t)
    idxs = tf.argsort(out, direction='DESCENDING')[0][:3].numpy()
    out = ', '.join([categories_ru[int(i)] for i in idxs])
    return out

#img = request_to_img(request)
#

def predict_imagenet(img):
    out = resnet(img)
    idxs =  tf.argsort(out, direction='DESCENDING')[0][:3].numpy()
    return '. '.join([categories_ru[int(i)] for i in idxs])


@app.route('/classify/binary', methods = ['POST'])
def classify_binary():
    data = request.data
    img = tf.io.decode_jpeg(data)
    img_t = tf.expand_dims(img, axis=0)
    img_t = tf.image.resize(img_t, (180, 180))
    predictions = model.predict(img_t)
    dog_probability = float(predictions[0])
    print(dog_probability)
    idx = dog_probability > 0.5
    return ('Cat', 'Dog')[idx]

if __name__ == '__main__':
    app.run(port=1774)      # номер зачетки
    input()
