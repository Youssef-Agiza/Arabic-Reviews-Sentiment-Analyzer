from numpy import genfromtxt
from time import time
from datetime import datetime
from sqlalchemy import Column, Integer,  String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from flask import Flask, redirect, render_template, request
from flask_sqlalchemy import SQLAlchemy

def Load_Data(file_name):
    data = genfromtxt(file_name, delimiter=',', skip_header=1, converters={0: lambda s: str(s)})
    print(data)
    return data.tolist()


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///data.db'



db=SQLAlchemy(app)

class Sample(db.Model):
        id=Column(Integer,primary_key=True)
        label=Column(Integer,nullable=False)
        text=Column(String,nullable=False)
        def __repr__(self):
                return "<Sample %r>" %self.id

# import csv, sqlite3

# con = sqlite3.connect(":memory:") # change to 'sqlite:///your_filename.db'
# cur = con.cursor()
# cur.execute("CREATE TABLE t (col1, col2);") # use your column names here


# cur.executemany("INSERT INTO t (col1, col2) VALUES (?, ?);", to_db)
# con.commit()
# con.close()


if __name__ == "__main__":
    # samples= Sample.query.all()
    # print(type(samples[0].label))

    db.session.get
    t = time()

    #Create the database
    engine = create_engine('sqlite:///data.db')
    db.metadata.create_all(engine)

    #Create the session
    session = sessionmaker()
    session.configure(bind=engine)
    s = session()

    try:
        file_name = "cleanedText.csv" #sample CSV file used:  http://www.google.com/finance/historical?q=NYSE%3AT&ei=W4ikVam8LYWjmAGjhoHACw&output=csv
        # data = Load_Data(file_name) 
        with open('cleanedText.csv','r') as fin: # `with` statement available in 2.5+
            # csv.DictReader uses first line in file for column headings by default
            dr = csv.DictReader(fin) # comma is default delimiter
            to_db = [(i['label'], i['text']) for i in dr]

        for i in to_db:
            # print(data)
            # break
            record = Sample(**{
                'label':i[0],
                'text':i[1]
            })
            # print(record, i)
            # break
            s.add(record) #Add all the records

        s.commit() #Attempt to commit all the records
    except:
        s.rollback() #Rollback the changes on error
    finally:
        s.close() #Close the connection
    print ("Time elapsed: " + str(time() - t) + " s.") #0.091s