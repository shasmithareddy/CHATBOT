import os
from flask import Flask, render_template, request, jsonify
from chatbot import get_response  # assuming chatbot.py has get_response()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
   data = request.get_json()
    user_message = data.get('message')
    reply = get_response(user_message)
    return jsonify({'reply': reply})

if __name__ == "__main__":
    # Dynamically get the port from the environment variable, defaulting to 5000 if not set
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
