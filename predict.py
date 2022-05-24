import threading
import numpy as np
import pandas as pd
import random
import math
import pickle
from text_preprocess import text_preprocess
from db_connect import get_collection


logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
answer = pickle.load(open('answer.pkl', 'rb'))
    
def get_answer(question):
    data_answer = pd.DataFrame(answer)
    df_question = pd.DataFrame([{"Question": (text_preprocess(question))}])
    logistic_predict = logistic_model.predict(df_question["Question"])
    svm_predict = svm_model.predict(df_question["Question"])
    maxLogisticPredictProb = (np.ndarray.max(logistic_model.predict_proba(df_question["Question"])))
    confused_answer = data_answer.loc[data_answer["tag"] == "boi_roi", 'response']
    logistic_predict_str = logistic_predict.tolist()[0]
    print(logistic_predict)
    print(svm_predict)
    print(maxLogisticPredictProb)
    try:
        if(logistic_predict[0] == "vo_nghia"):
            return {"mess": confused_answer.iat[0][math.trunc(random.random()*len(confused_answer.iat[0]))]}
    #     elif(maxLogisticPredictProb > 0.8 or (maxLogisticPredictProb > 0.7 and logistic_predict[0] == svm_predict[0])):
    #         s = data_answer.loc[data_answer['tag'] == " ".join(logistic_predict), 'response']
    #         if(isinstance(s.iat[0], list)):
    #             return {"mess": s.iat[0][math.trunc(random.random()*len(s.iat[0]))], "tag": logistic_predict_str}
    #         else:
    #             return {"mess": s.iat[0], "tag": logistic_predict_str}
    #     elif(maxLogisticPredictProb > 0.1):
    #             if(logistic_predict == svm_predict):
    #                 threading.Thread(target=insert_lowProb_question, args=[question,maxLogisticPredictProb, logistic_predict_str]).start()        
    #             else:
    #                 threading.Thread(target=insert_lowProb_question, args=[question,maxLogisticPredictProb, ""]).start()
    #     return {"mess": confused_answer.iat[0][math.trunc(random.random()*len(confused_answer.iat[0]))]}
    # except ValueError:
        elif(maxLogisticPredictProb > 0.7 and logistic_predict[0] == svm_predict[0]):
            s = data_answer.loc[data_answer['tag'] == " ".join(logistic_predict), 'response']
            if(isinstance(s.iat[0], list)):
                return {"mess": s.iat[0][math.trunc(random.random()*len(s.iat[0]))], "tag": logistic_predict_str}
            else:
                return {"mess": s.iat[0], "tag": logistic_predict_str}
        elif(maxLogisticPredictProb > 0.1):
            if(logistic_predict == svm_predict):
                threading.Thread(target=insert_lowProb_question, args=[question,maxLogisticPredictProb, logistic_predict_str]).start()        
            else:
                threading.Thread(target=insert_lowProb_question, args=[question,maxLogisticPredictProb, ""]).start()
        return {"mess": confused_answer.iat[0][math.trunc(random.random()*len(confused_answer.iat[0]))]}
    except ValueError:
        print(ValueError)
    
def insert_lowProb_question(question, maxLogisticPredictProb, tag_predict):
    question_collection = get_collection('questions')
    existed_question = question_collection.find_one({"question": question})
    if(existed_question):
        return
    else:
        question_collection.insert_one({"tag": tag_predict, "question": question, "prob": maxLogisticPredictProb})
