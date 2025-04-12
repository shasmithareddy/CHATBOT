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
import os
from flask import Flask, render_template, request, jsonify
from chatbot import get_response  # assuming chatbot.py has get_response()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # Get the JSON data from the POST request
    data = request.get_json()
    user_message = data.get("message")

    # If no message is provided, return a response
    if not user_message:
        return jsonify({"reply": "Please enter a message."}), 400

    # Get the chatbot's reply
    reply = get_response(user_message)

    # Return the reply as a JSON response
    return jsonify({"reply": reply})

if __name__ == "__main__":
    # Dynamically get the port from the environment variable, defaulting to 5000 if not set
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
