from flask import Flask, render_template, jsonify
import os
from flask import request

import pickle


project_dir = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__)


filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))
featurizer = pickle.load(open("featurizer.pickle", 'rb'))




@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    input_mail = []
    mail = request.form["mail"]

    if len(mail) == 0:
        response = {"status": 200, "status_msg": "Mail is not entered"}
        return jsonify(response)

    input_mail += [mail]

    input_data_features = featurizer.transform(input_mail)


    prediction = model.predict(input_data_features)


    message = "Mail is Spam"
    
    if prediction[0] == 1:
        message = "Mail is Ham"

    response = {"status": 200, "status_msg": message}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)