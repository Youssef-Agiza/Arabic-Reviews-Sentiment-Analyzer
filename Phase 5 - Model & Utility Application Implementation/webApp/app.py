from flask import Flask, redirect, render_template, request, url_for,session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine

# import sys 
# sys.path.append(".")
from Model import NN, Layer
from Optimizer import AdamOptimizer
from Preprocessor import Preprocessor


import pandas as pd
import numpy as np

from  sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///data.db'
app.secret_key = "27eduCBA09"


vectorizer_file_path="tfidfVectorizerDump.joblib"
model_dump_file_path="modelDump.joblib"


db=SQLAlchemy(app)
class Sample(db.Model):
        id=db.Column(db.Integer,primary_key=True)
        label=db.Column(db.Integer,nullable=False)
        text=db.Column(db.String,nullable=False)
        def __repr__(self):
                return "<Sample %r>" %self.id



nn=NN()
nn.load_model(model_dump_file_path)
PP=Preprocessor(vectorizer_path=vectorizer_file_path)


def retrain(model,pp,df):
        print("Model will be retrained on: ",len(df))
        df["text"]=df['text'].values.astype('U')
        df_shuffled=shuffle(df,random_state=0)
        


        #Splitting data
        x=df_shuffled['text']
        y=np.expand_dims(df_shuffled['label'],axis=1)
        X_train,X_test,Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size=0.25,random_state=1)

        vectorizer= TfidfVectorizer()
        tf_x_train = vectorizer.fit_transform(X_train)
        tf_x_test = vectorizer.transform(X_test)
        tf_x_val=vectorizer.transform(X_val)


        
        adam= AdamOptimizer(beta1 = 0.9,beta2 = 0.99,alpha=0.1,eps=0.001)
        nn_temp = NN(optimizer=adam)

        nn_temp.add_layer(tf_x_train.shape[1],64,activation="relu",name="l1")
        nn_temp.add_layer(64,32,activation = "relu",name="l2")
        nn_temp.add_layer(32,8,activation = "relu",name="l4")
        nn_temp.add_layer(8,1,activation = "sigmoid",name="l5")


        # model.reset_layers()
        # print("optimizer: ",model.optimizer,model.optimizer.alpha)
        nn_temp.fit(tf_x_train,Y_train,validation_data=[tf_x_val,Y_val],batch_size=32,epochs=20)
        y_pred=nn_temp.predict(tf_x_test)
        print(np.unique(y_pred,return_counts=True))
        nn_temp.save_model(model_dump_file_path)
        with open(vectorizer_file_path,"wb") as f:
                pickle.dump(list, f)

        #update global objects
        nn.load_model(model_dump_file_path)
        pp.vectorizer=vectorizer



@app.route('/', methods=['POST','GET'])
def index():
        if request.method=="POST":
                try:
                    sentence=request.form["text"]
                    if not PP.detect_lang(sentence):
                           error="Language is not arabic"
                           print("\n\n\n\n",error,"\n\n\n")
                           return render_template("error.html",error=error)

                           
                    sentence=PP.preprocess(sentence)
                    tfidf=PP.convert_to_tfidf(sentence)

                    prediction=nn.predict(tfidf)
                    sentiment= "Positive" if prediction[0][0]==1 else "Negative"

                    session["sentence"]=sentence
                    session["sentiment"]=sentiment
                    return redirect("/result")
                except:
                        print("\n\nINSIDE ERRORRR\n\n\n")
                        return "There was an error predicting your sentence"
        else: 
                return render_template("index.html")

@app.route('/result',methods=['POST','GET'])
def result():
        if request.method=="POST": 
                feedback=request.form["feedback"]
                label= 1 if feedback=="yes" else 0
                sentence=session["sentence"]

                new_sample=Sample(label=label,text=sentence)
                try:
                        db.session.add(new_sample)
                        db.session.commit()
                        return redirect('/')
                except :
                        error="There was an error recording your feedback!"
                        return redirect("error.html",error=error)

        else:
         sentiment=session["sentiment"]
         print(sentiment)
         return render_template("result.html",sentiment=sentiment)

@app.route('/active-learn',methods=["POST","GET"])
def activeLearn():
        if request.method=="POST":
                cnx = create_engine('sqlite:///data.db').connect()
                df=pd.read_sql_table("Sample",cnx)
                print(df.head())
                retrain(nn,PP,df)
                return render_template('active-learn.html',finished_training=True,samples_count=len(df))

                # print(df.head())
        else: return render_template("active-learn.html")

if __name__ == "__main__":
    app.run(debug=True)