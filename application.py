from flask import Flask, request,jsonify, render_template
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
ridge_model=pickle.load(open("models/ridge.pkl","rb"))
standard_scaler=pickle.load(open("models/scaler.pkl","rb"))



application=Flask(__name__)
app=application
@app.route("/")
def welcome():
    return "Welcome to ML Practice on AWS!"
@app.route("/predict",methods=["GET","POST"])
def predict_new_data():
    if request.method=="POST":
        Temperature=request.form.get("Temperature")
        RH=request.form.get("RH")
        Ws=request.form.get("Ws")
        Rain=request.form.get("Rain")
        FFMC=request.form.get("FFMC")
        DMC=request.form.get("DMC")
        ISI=request.form.get("ISI")
        Classes=request.form.get("Classes")
        Region=request.form.get("Region")
        sclaled_data=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(sclaled_data)
        return render_template("predict.html",results=result[0])
    else:   
        return render_template("predict.html")
if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)