import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictscore', methods=['POST'])
def predict1():

    input_features = [int(x) for x in request.form.values()]
    in_fea = [np.array(input_features)]
    pred = model.predict(in_fea)

    return render_template('after.html', data=pred[0])


if __name__ == '__main__':
    app.run(debug=True)
