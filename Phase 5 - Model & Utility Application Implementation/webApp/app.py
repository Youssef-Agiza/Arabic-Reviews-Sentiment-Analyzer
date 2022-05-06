from flask import Flask, redirect, render_template, request
from flask_sqlalchemy import SQLAlchemy

# import sys 
# sys.path.append(".")
from Model import NN, Layer
from Optimizer import AdamOptimizer
from Preprocessor import Preprocessor


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///test.db'



nn=NN()
nn.load_model("modelDump.joblib")
PP=Preprocessor(vectorizer_path="tfidfVectorizerDump.joblib")


db=SQLAlchemy(app)
# class Todo(db.Model):
#     id=db.Column(db.Integer,primary_key=True)
#     content=db.Column(db.String(200),nullable=False)
#     date_created=db.Column(db.DateTime,default=datetime.utcnow)

#     def __repr__(self):
#         return "<Task %r>" %self.id
    

@app.route('/', methods=['POST','GET'])
def index():
        if request.method=="POST":
                return "Predicting naaaaaw..."
        else: 
                return render_template("index.html")
  


if __name__ == "__main__":
    app.run(debug=True)