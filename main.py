from flask import *
import json, time
from SPARQLWrapper import SPARQLWrapper, JSON
import ssl
from flask import Flask, jsonify, request
import numpy as np

from keras.models import load_model
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

model = load_model('model/dwipa_model.h5')
class_name = ['borobudur', 'prambanan', 'stupa_borobudur']


@app.route('/ontology', methods=['GET'])
def ontology2():
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


@app.route('/predict', methods=['GET'])
def predict():
    image = 'D:\Serius\Coding\DWIPA\python_api\model\img-testing.jpg'
    img = keras.preprocessing.image.load_img(image, target_size=(244, 244))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    pred = model.predict(img_array)

    score = tf.nn.softmax(pred[0])
    result = class_name[np.argmax(score)]
    result2 = 100 * np.max(score)
    prediction = {'Class': result, 'Score': result2}
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(port=7777, debug=True)
    # predict()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
