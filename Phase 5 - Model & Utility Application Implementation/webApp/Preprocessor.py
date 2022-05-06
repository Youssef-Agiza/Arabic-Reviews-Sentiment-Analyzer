import numpy as np
from langdetect import detect
from nltk.stem.isri import ISRIStemmer
from nltk import word_tokenize
import pyarabic.araby as araby
# from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class Preprocessor:
    def __init__(self,vectorizer_path=None):
        self.vectorizer=None 
        if vectorizer_path:
            self.load_tfidf_vectorizer(vectorizer_path)
    
    def load_tfidf_vectorizer(self,path):
        file=open(path,'rb')
        self.vectorizer=pickle.load(file)
        file.close()
    
    def save_tfidf_vectorizer(self,path):
        if not self.vectorizer:
            return None
        file=open(path,"wb")
        pickle.dump(file,self.vectorizer)
        file.close()
    
    def preprocess(self,sentence,vectorizer_path=""):
        if not self.vectorizer and vectorizer_path=="":
            return None

        # if not self.detect_lang(sentence):
        #     print("sentence Not arabic!")
        #     return None
        
        result=self.stem(sentence)
        result=self.normalize(result)
        result=self.remove_redundant_words(result)



        return result


    def detect_lang(self,sentence):
        try:
            print("\n\nhere\n\n")
            language = detect(sentence)
            print("\n\nhere2\n\n")

            if language != 'ar':
                return False
        except:
                return False
        print("returning true")
        return True

    def stem(self,sentence):
        st = ISRIStemmer()
        stemmed_sentence=""
        for a in word_tokenize(sentence):
            stemmed=st.stem(a)
            stemmed_sentence+=(stemmed+" ")
        return stemmed_sentence
    
    def normalize(self,sentence):
        normalized=sentence
        normalized=araby.strip_tashkeel(normalized)
        normalized= araby.strip_tatweel(normalized)
        normalized=araby.normalize_hamza(normalized)
        return normalized
    
    

    def remove_redundant_words(self,sentence):
        stop_words=['من','على','عن','ب','ك','ل','فى','و','ان','هذا','او','كتب','...','.','','الى','فيه','انه','قبل','//','..','،',':',"؟",'/']
        words=sentence.split()
        resultWords= [word for word in words if word not in stop_words]
        return ' '.join(resultWords)
    
    def convert_to_tfidf(self,sentence):
        if not self.vectorizer:
            return None
        return self.vectorizer.transform([sentence])

