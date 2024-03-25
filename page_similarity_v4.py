#!/usr/bin/python3

# Import libraries
import re
import json
import faiss
import openai
import logging
import psycopg2
import datetime
import numpy as np

from sqlalchemy import text
from functools import wraps
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from database_manager import DatabaseManager
from sentence_transformers import SentenceTransformer
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from flask_jwt_extended import jwt_required, verify_jwt_in_request, get_jwt_identity, JWTManager
from sqlalchemy import create_engine, MetaData, inspect, Table, Column, Integer, String, Boolean, ARRAY, Float, Sequence, JSON

# JWT Exception Function
def jwt_required_except_options(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if request.method != 'OPTIONS':
            try:
                verify_jwt_in_request()
            except Exception as e:
                logging.error(f"Error when verifying JWT token: {e}")
                return jsonify({"message": "Error when verifying JWT token"}), 401  # Unauthorized
        return fn(*args, **kwargs)
    return wrapper

# APP Config
app = Flask(__name__)
jwt = JWTManager(app)
app.config['JWT_SECRET_KEY'] = '1234' # todo: real value
app.config['JWT_HEADER_NAME'] = 'Authorization'
app.config['JWT_HEADER_TYPE'] = 'Bearer'
app.config['JWT_TOKEN_LOCATION'] = ['headers']
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
logging.basicConfig(filename='proccessed', encoding='utf-8', level=logging.DEBUG)

# PostgreSQL DB Manager
db_manager = DatabaseManager(
    username="postgres", 
    password='Kwantx11!!', 
    host='internaldb.c0rz9kyyn4jp.us-east-2.rds.amazonaws.com', 
    db_name="user_db"
)
user_query = Table('user_query', db_manager.meta, autoload_with=db_manager.engine)
parsed_pages = Table('parsed_pages_2', db_manager.meta, autoload_with=db_manager.engine) # debug

encoder = SentenceTransformer("distilbert-base-nli-mean-tokens")
index = faiss.IndexFlatL2(768)

# OpenAI Config
openai.api_key = "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"
model_engine = "gpt-4"

def chat_with_ai(user_input):
    print("OpenAI API Request:", user_input)

    conversation_history = [
        {"role": "system", "content": "You are an AI assistant that provides helpful answers."},
        {"role": "user", "content": user_input},
    ]

    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=conversation_history,
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.7,
    )

    ai_response = response.choices[0].message['content']
    print("OpenAI API Response:", response)
    # app.logger.info("OpenAI API Response: %s", response)

    return ai_response

@app.route('/selected_papers', methods=['POST'])
@cross_origin()
# debug @jwt_required_except_options

def handle_selected_papers():
    app.logger.info('Processing /selected_papers request...') 
    start_time = datetime.datetime.now()
    # debug token = request.headers.get('Authorization')
    # debug print('Token in server: ', token)
    # debug logging.info('Token in server: %s', token)

    # debug if request.method != 'OPTIONS':
    # debug     current_user = get_jwt_identity()
    # debug     if current_user is None:
    # debug         return jsonify({"message": "Error: You are not authorized."}), 401  # Unauthorized
    # debug else:
    current_user = None # debug

    data = request.get_json()
    print("Inside /selected_papers route...")
    logging.info("Inside /selected_papers route...")
    print(f"Data received: {data}")
    logging.info("Data received: %s", data)

    user_id = data.get('user_id')
    query = data.get('query')
    selected_paper_ids = data.get('selected_paper_ids')

    if not user_id or not query or not selected_paper_ids:
        return jsonify({"message": "Error: Missing required parameters."}), 400  # Bad Request

    print(f"User ID: {user_id}, Query: {query}, Selected Paper IDs: {selected_paper_ids}")
    logging.info("User ID: %s, Query: %s, Selected Paper IDs: %s", user_id, query, selected_paper_ids)

    k = 10  # number of closest pages to return

    print("Fetching user query embeddings...")
    logging.info("Fetching user query embeddings...")
    query_embeddings_ok = False
    with db_manager.conn.begin():
        result = db_manager.conn.execute(text(f"SELECT query_embeddings FROM user_query WHERE user_id = '{user_id}' AND query = '{query}'"))
        for row in result:
            query_embeddings = row[0]
            query_embeddings_ok = True
    if query_embeddings_ok == False:
        logging.error("Error: Your query is not exists.")
        return jsonify({"message": "Error: Your query is not exists."}), 402
    _vector = np.array(query_embeddings, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(_vector)
    print("User query embeddings: ", _vector.size)

    print("Fetching page embeddings for selected papers and building the index...")
    logging.info("Fetching page embeddings for selected papers and building the index...")
    parsed_pages = Table('parsed_pages_2', db_manager.meta, autoload_with=db_manager.engine) # debug
    page_id_to_idx = {}
    idx_to_page_id = {}
    idx = 0
    with db_manager.conn.begin():
        for paper_id in selected_paper_ids:
            result = db_manager.conn.execute(text(f"SELECT paper_id, page_embeddings FROM parsed_pages_2 WHERE paper_id = {paper_id}")) # debug ???_page_number
            for row in result:
                id, embeddings_list = row
                embeddings = np.array(embeddings_list).reshape(1, -1)
                index.add(embeddings)
                page_id_to_idx[id] = idx
                idx_to_page_id[idx] = id
                idx += 1

    if not idx_to_page_id:
        logging.error("Error: No matched page embedding exists.")
        return jsonify({"message": "Error: No matched page embedding exists."}), 403

    print("Searching the index...")
    logging.info("Searching the index...")
    D, I = index.search(_vector, k)
    # closest_page_ids = [idx_to_page_id[i] for i in I[0]]
    closest_page_ids = []
    for i in I[0]:
        if i > -1 and i < len(idx_to_page_id):
            closest_page_ids.append(idx_to_page_id[i])

    print(f"Closest page IDs: {closest_page_ids}")
    logging.info("Closest page IDs: %s", closest_page_ids)

    print("Closest page IDs have been found. Proceeding with feature extraction...")
    logging.info("Closest page IDs have been found. Proceeding with feature extraction...")


    # Get the contents of these pages and extract the features
    features = []
    page_features = Table('page_features', db_manager.meta, autoload_with=db_manager.engine)

    with db_manager.conn.begin():
        for page_id in closest_page_ids:
            result = db_manager.conn.execute(text(f"SELECT content FROM parsed_pages_2 WHERE paper_id = '{page_id}'")) # _page_number??? #debug
            for row in result:
                content_text = row[0][:1000]  # limit to 1000 characters

                prompt = f"""
                As an AI, I'm instructed to extract any features referenced on the specific page of a research paper. The features must be generated from stock market price and/or volume data such as (open, high, low, close, volume, trades). Extract every feature from a page and each of the four items, if the page doesn't specific each of these items then use your best knowledge to provide them given your knowledge and the paper context:

                1A) Feature Name: Name of the stock indicator feature
                1B) Feature Formula: The mathematical formula
                1C) Feature Definition: A brief definition
                1D) Feature Parameters: Parameters or variables used in the feature formula

                Use the text on this page to extract the information:

                {content_text}

                Provide the response for each feature on the page. For 1A surround the response with '~~' such as '~~text~~'. For 1B use '@@', for 1C use '$$', for 1D use '%%'. Only provide each of the features information as a response, with no other explanatory text.
                """

                ai_chat = chat_with_ai(prompt)

                # Split the ai_chat into individual feature information
                features_info = ai_chat.split('\n\n')
                for f in features_info:
                    if f.count("~~") != 2 or f.count("@@") != 2 or f.count("$$") != 2 or f.count("%%") != 2:
                        continue
                        x = txt.split('~~')[1]
                    feature_name = f.split('~~')[1]
                    feature_formula = f.split('@@')[1]
                    feature_definition = f.split('$$')[1]
                    feature_parameters = f.split('%%')[1].replace('\"', '')

                    if not feature_name or not feature_formula or not feature_definition or not feature_parameters:
                        continue

                    # Prepare a feature dictionary
                    feature = {
                        "user_id": user_id,
                        "paper_id": page_id, # _page_number???
                        "feature_name": feature_name,
                        "feature_definition": feature_definition,
                        "feature_formula": feature_formula,
                        "feature_parameters": feature_parameters# json.loads(feature_parameters)  # Assuming feature_parameters is a valid JSON string
                    }
                    features.append(feature)

                    # Inserting into 'page_features' table
                    ins = page_features.insert().values(feature)
                    db_manager.conn.execute(ins)

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for /selected_papers route: {elapsed_time}")
    logging.info("Elapsed time for /selected_papers route: %s", elapsed_time)

    return jsonify(features), 200  # OK


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5010)