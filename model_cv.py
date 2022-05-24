import os
from os import error
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from data import get_answer
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class LogisticRegression_Model(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("vect", TfidfVectorizer(analyzer='word', ngram_range=(1,2))),
            ("clf", LogisticRegression(C=212.10, max_iter=10000, solver='lbfgs', multi_class='auto'))
        ])
        return pipe_line

class SVM_Model(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("vect", TfidfVectorizer(analyzer='word', ngram_range=(1,2))),
            ("clf", SVC(kernel='sigmoid', C=500, gamma='scale', probability=True, class_weight='balanced'))
        ])
        return pipe_line

def train_model_conversation():
    try:
        df_train = pd.DataFrame(get_answer())
        logistic_model = LogisticRegression_Model()
        logistic_clf = logistic_model.clf.fit(df_train["Response"], df_train['Intent'])
        pickle.dump(logistic_clf, open("logistic_answer_model.pkl", "wb"))
        svm_model = SVM_Model()
        svm_clf = svm_model.clf.fit(df_train["Response"], df_train.Intent)
        pickle.dump(svm_clf, open("svm_answer_model.pkl", "wb"))
        # os.system("pkill gunicorn")
        # os.system("cd ~/chatbotapp")
        # os.system("source chatbotenv/bin/activate")
        # os.system("gunicorn --bind 0.0.0.0:8080 wsgi:app --daemon")
        # os.system("Train model ok")
        print('Success')
        return {"mess": "Train model thành công", "success":"true"}
    except error:
        print('Fail')
        return {"mess": "Lỗi khi train model", "success":"false"}

if __name__ == "__main__":
    train_model_conversation()
