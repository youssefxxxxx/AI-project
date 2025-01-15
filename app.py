# app.py
import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from marshmallow import Schema, fields, ValidationError
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS

from search_exercises import search_exercises
from generate_response import generate_response

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# We only validate "query" now; no "top_k" field
class SearchSchema(Schema):
    query = fields.Str(required=True, validate=lambda s: len(s.strip()) > 0)

search_schema = SearchSchema()

@app.route('/search', methods=['POST'])
@limiter.limit("10 per minute")
def search_endpoint():
    try:
        data = search_schema.load(request.get_json())
    except ValidationError as err:
        logger.warning(f"Validation Error: {err.messages}")
        return jsonify(err.messages), 400

    user_query = data["query"].strip()

    logger.info(f"Received query: '{user_query}'")

    # Always get 4 relevant exercises
    try:
        exercises, exercise_embeddings, query_embedding = search_exercises(user_query)
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error searching exercises: {e}")
        return jsonify({"error": "Internal error searching exercises."}), 500

    if not exercises:
        logger.info("No relevant exercises found.")
        return jsonify({"query": user_query, "response": "No relevant exercises found."}), 200

    try:
        response_text = generate_response(user_query, exercises, exercise_embeddings, query_embedding)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        response_text = "Error generating response."

    return jsonify({
        "query": user_query,
        "response": response_text,
        "exercises": exercises
    }), 200

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad Request"}), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
