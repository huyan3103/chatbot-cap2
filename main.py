from predict import get_answer
from conversation_preprocess import conversration_predict
from model import train
from model_cv import train_model_conversation

from flask import Flask, jsonify, request
from flask_cors import CORS
# Init app
app = Flask(__name__)
CORS(app)

@app.route("/", methods=['POST'])
async def receiveAnswer():
    try:
        json_data = request.json
        question = json_data['question']
        answer = get_answer(question)
        return answer
    except Exception as e:
        return jsonify(e)

@app.route("/predict-conversation", methods=['POST'])
async def predictConversation():
    try:
        json_data = request.json
        conversations = json_data['conversations']
        return conversration_predict(conversations)
    except Exception as e:
        return jsonify(e)

@app.route("/", methods=['GET'])
async def home():
    return 'Đây là home'

@app.route("/train", methods=['GET'])
async def trainModel():
    return train()

@app.route("/train-conversation", methods=['GET'])
async def trainModelConversation():
    return train_model_conversation()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

