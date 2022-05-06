from flask import Flask, redirect, render_template, request
from flask_sqlalchemy import SQLAlchemy

# import sys 
# sys.path.append(".")
from Model import NN, Layer
from Optimizer import AdamOptimizer
from Preprocessor import Preprocessor


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///data.db'



db=SQLAlchemy(app)
class Sample(db.Model):
        id=db.Column(db.Integer,primary_key=True)
        label=db.Column(db.Integer,nullable=False)
        text=db.Column(db.String,nullable=False)
        def __repr__(self):
                return "<Sample %r>" %self.id



# nn=NN()
# nn.load_model("modelDump.joblib")
# PP=Preprocessor(vectorizer_path="tfidfVectorizerDump.joblib")

# class Todo(db.Model):
#     id=db.Column(db.Integer,primary_key=True)
#     content=db.Column(db.String(200),nullable=False)
#     date_created=db.Column(db.DateTime,default=datetime.utcnow)

#     def __repr__(self):
#         return "<Task %r>" %self.id
    

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
                    prediction=nn.predict(sentence)
                    sentiment= "Positive" if prediction[0][0]==1 else "Negative"
                    return render_template("result.html", sentiment=sentiment)
                except:
                        print("\n\nINSIDE ERRORRR\n\n\n")
                        return "There was an error predicting your sentence"
        else: 
                return render_template("index.html")
  


if __name__ == "__main__":
    app.run(debug=True)