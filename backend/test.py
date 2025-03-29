from flask import Flask, jsonify, request, make_response

from spell_corrector import get_close_matches
from flask_cors import CORS 
import json
from trie import Trie
from backbone import get_predicted_class

app = Flask(__name__)



CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})


trie = Trie()

@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    

    respnse_for_tag = get_predicted_class(user_input) 
    classification_result = "spam" if int(respnse_for_tag) == 1 else "ham"
    
    print(respnse_for_tag)
    return jsonify({"response": classification_result})


@app.route('/suggest', methods=['GET'])
def suggest():
    prefix = request.args.get('prefix', '')
    suggestions = trie.search_prefix(prefix)
    response = make_response(jsonify(suggestions))
    response.headers.add('Access-Control-Allow-Origin', '*')  
    return response

@app.route('/spellcorrect', methods=['GET'])
def spell_correct():
    word = request.args.get('word', "").strip().lower()
    if not word:
        return jsonify([])

    corrections = get_close_matches(word,  n=3, cutoff=0.6) 
    response = make_response(jsonify(corrections))
    response.headers.add('Access-Control-Allow-Origin', '*') 
    return response

if __name__ == "__main__":
    app.run(debug=True)
