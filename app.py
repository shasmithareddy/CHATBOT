''' import os
from flask import Flask, render_template, request, jsonify
from chatbot import get_response  # Assuming you have this method defined in chatbot.py

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    user_text = request.get_json().get("message")
    response = get_response(user_text)
    return jsonify({"response": response})

if __name__ == "__main__":
    # Dynamically get the port from environment variable, defaulting to 5000 if not set
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
'''
from flask import Flask, render_template, request, jsonify
from chatbot import get_response
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message")
    if not message:
        return jsonify({"reply": "Please enter a message."}), 400
    response = get_response(message)
    return jsonify({"reply": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
