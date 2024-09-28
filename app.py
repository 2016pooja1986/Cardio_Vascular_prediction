from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('xgb_model.pkl','rb'))

@app.route("/",methods=['GET'])
def Home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():

    # gender = request.form.get("gender")
    # age = request.form.get("age")

    float_features = [float(x) for x in request.form.values()]
    float_features[0] = float_features[0]*365
        
    features = [np.array(float_features)]
    
    prediction = model.predict(features)

    if prediction[0] == 1:
        disease = 'CardioVascular Disease Present. Consult a Cardiologist immediately..'
    else:
        disease = 'CardioVascular Disease Absent'

    return  render_template("pred.html",prediction_text = f"{disease}")








if __name__ == "__main__":
    app.run(debug=True)