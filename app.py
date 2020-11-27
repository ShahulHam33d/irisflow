import numpy as np
from flask import Flask,request,jsonify,render_template


app = Flask(__name__)
model = open("model.pkl","rb")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/presdict',methods = ["post"])
def predict():
    int_features =[int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    conv = {1 : "iris setosa",2 : 'iris versicolor', 3 : "iris verginica"}
    pred = conv[prediction]
    output = round(prediction[0],2)

    return render_template("index.html",prediction_text="the species is {}".format(output))

if __name__ == "__main__" :
    app.run(debug=True)