from flask import Flask,render_template,request
import pickle
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method =='POST':
        
        age = request.form['age']
        fever = request.form['fever']
        breath = request.form['breath']
        cold = request.form['cold']
        body = request.form['body']
       

        #print('#-------------------------------data is here-------------------------------------#')
        #print(age,body,fever,cold,breath)
        
        clf = pickle.load(open("corona.pkl", "rb"))
        #columns  -- Age,	Fever,	BodyPains,	RunnyNose,	Difficulty_in_Breath
        data = [[int(age),int(fever),int(body),int(cold),int(breath)]]
        predict = clf.predict(data)[0]
        #proba_score = clf.predict_proba([[60,100,0,1,0]])[0][0]
        #proba_score = clf.predict_proba([[age,fever,breath,cold,body]])[0][0]

        if predict==1:
            prediction='Positive'
            proba_score = clf.predict_proba([[age,fever,breath,cold,body]])[0][1]
        else:
            prediction = 'Negative'
            proba_score = clf.predict_proba([[age,fever,breath,cold,body]])[0][0]

        
        return render_template('index.html',prediction=prediction,proba_score=round(proba_score*100,2))
    else:
        
        return render_template('index.html',message='Something missed, Please follow the instructions..!')
              

if __name__ == '__main__':
    app.run(debug=True)
