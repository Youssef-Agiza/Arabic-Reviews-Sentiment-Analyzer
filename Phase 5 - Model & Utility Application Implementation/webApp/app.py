from flask import Flask, redirect, render_template, request, url_for,session
from flask_sqlalchemy import SQLAlchemy

# import sys 
# sys.path.append(".")
from Model import NN, Layer
from Optimizer import AdamOptimizer
from Preprocessor import Preprocessor


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///data.db'
app.secret_key = "27eduCBA09"



db=SQLAlchemy(app)
class Sample(db.Model):
        id=db.Column(db.Integer,primary_key=True)
        label=db.Column(db.Integer,nullable=False)
        text=db.Column(db.String,nullable=False)
        def __repr__(self):
                return "<Sample %r>" %self.id



nn=NN()
nn.load_model("modelDump.joblib")
PP=Preprocessor(vectorizer_path="tfidfVectorizerDump.joblib")

    

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


        

if __name__ == "__main__":
    app.run(debug=True)