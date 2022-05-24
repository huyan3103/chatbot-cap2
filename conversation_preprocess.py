import numpy as np
import pandas as pd
import pickle
from text_preprocess import text_preprocess
from db_connect import get_collection

logistic_answer_model = pickle.load(open('logistic_answer_model.pkl', 'rb'))
    
def predict_question(question):
    df_question = pd.DataFrame([{"Question": (text_preprocess(question))}])
    logistic_predict = logistic_answer_model.predict(df_question["Question"])
    maxLogisticPredictProb = (np.ndarray.max(logistic_answer_model.predict_proba(df_question["Question"])))
    logistic_predict_str = logistic_predict.tolist()[0]
    try:
        return logistic_predict_str, maxLogisticPredictProb
    except ValueError:
        print(ValueError)
    
def insert_lowProb_question(question, maxLogisticPredictProb, tag_predict):
    question_collection = get_collection('questions')
    existed_question = question_collection.find_one({"question": question})
    if(existed_question):
        return
    else:
        question_collection.insert_one({"tag": tag_predict, "question": question, "prob": maxLogisticPredictProb})

def remove_unmeaning(sentence):
    df_sentence = pd.DataFrame([{"Sentence": text_preprocess(sentence)}])
    predict_prob = (np.ndarray.max(logistic_answer_model.predict_proba(df_sentence["Sentence"])))
    return predict_prob

def preprocess_sentence (user, expert):
  sentence_preprocessed = []
  for sentence in expert:
    sentenceTemp = text_preprocess(sentence)
    prob = remove_unmeaning(sentenceTemp)
    if(prob > 0.1):
        sentence_preprocessed.append(sentenceTemp)
  sentence_predict = ', '.join(sentence_preprocessed)
  sententce_final = ', '.join(user)
  finall_result = predict_question(sentence_predict)
  print(predict_question(sentence_predict))
  insert_lowProb_question(sententce_final, finall_result[1], finall_result[0])

def conversration_predict(conversations):
    try:
        for data in conversations:
            preprocess_sentence(data['user'], data['expert'])
        return {"mess": "Done"}
    except Exception as e:
        return {"mess": e}
# fake_data = [{"user": ['quên mật khẩu, mất mật khẩu', 'không nhớ mật khẩu'], 'expert': ['lên phòng cse tầng 7 Quang Trung', 'cầm theo thẻ sinh viên', 'nhập mail cá nhân ở myDtu']}, {"expert": ["mình không hiểu bạn nói gì", "ádasdasd"], "user": ["asdasdasdasdasd", "ádasdasdasd"]}]
# conversration_predict(fake_data)
