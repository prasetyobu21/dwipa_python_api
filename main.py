import base64

from flask import *
import json, time
from SPARQLWrapper import SPARQLWrapper, JSON
import ssl
from flask import Flask, jsonify, request
import numpy as np

from keras.models import load_model
import tensorflow as tf
from tensorflow import keras

import PIL
from PIL import Image

app = Flask(__name__)

model = load_model('model/dwipa_model3.h5')
class_name = ['borobudur', 'prambanan', 'stupa_borobudur']


@app.route('/ontology', methods=['GET'])
def ontology():
    ssl._create_default_https_context = ssl._create_unverified_context
    sparql = SPARQLWrapper("http://localhost:8890/sparql")
    sparql.setQuery("""
        select * 
        from <http://localhost:8890/dwipa>
        where {?s ?p ?o.} limit 100
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    data = {
        'result': results["results"]
    }
    json_dump = jsonify(data)

    return json_dump


@app.route('/predict', methods=['POST'])
def predict():
    # image = 'D:\Serius\Coding\DWIPA\py_model_test\model\img-testing.jpg'
    # image = 'D:\Serius\Coding\DWIPA\py_model_test\model\prambanan-testing.jpg'
    # img = keras.preprocessing.image.load_img(image, target_size=(244, 244))
    # img_array = keras.preprocessing.image.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0)

    image = request.files['file']
    image = Image.open(image)
    image = np.asarray(image.resize((244, 244)))
    image = image.reshape(1, 244, 244, 3)

    pred = model.predict(image)

    score = tf.nn.softmax(pred[0])
    result = class_name[np.argmax(score)]
    result2 = 100 * np.max(score)
    prediction = {'Class': result,
                  'Score': result2}

    query_result = ontology(result)
    return query_result

@app.route('/predictOnly', methods=['POST'])
def predictO():
    # image = 'D:\Serius\Coding\DWIPA\py_model_test\model\img-testing.jpg'
    # image = 'D:\Serius\Coding\DWIPA\py_model_test\model\prambanan-testing.jpg'
    image = request.files['file']
    image = Image.open(image)
    image = np.asarray(image.resize((244,244)))
    image = image.reshape(1,244,244,3)
    # im_b64 = request.json['file']
    # img_bytes = base64.b64decode(im_b64.encode('utf-8'))
    # image = Image.open

    # img = keras.preprocessing.image.load_img(image, target_size=(244, 244))
    # img_array = keras.preprocessing.image.img_to_array(image)
    # img_array = tf.expand_dims(img_array, 0)

    pred = model.predict(image)

    score = tf.nn.softmax(pred[0])
    result = class_name[np.argmax(score)]
    result2 = 100 * np.max(score)
    print(score)
    print(result)
    print(result2)
    prediction = {'Class': result,
                  'Score': result2}

    return prediction


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7777, debug=True)
