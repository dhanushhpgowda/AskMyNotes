import os
from flask import Flask, request, session, jsonify, render_template
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from pymilvus import Collection, utility
import processor  # Importing your updated processor.py

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()
CORS(app)

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Establish Milvus connection (Port 19530 from your Docker)
processor.connect_to_milvus()

@app.route('/')
def index():
    return render_template('index.html')

# --- NEW: LIST ALL PREVIOUS NOTE COLLECTIONS ---
@app.route('/sessions', methods=['GET'])
def list_sessions():
    """Returns a list of all existing note collections in Milvus."""
    try:
        all_collections = utility.list_collections()
        # Filter to only show collections created by this app
        note_sessions = [c for c in all_collections if c.startswith("notes_")]
        return jsonify({"sessions": note_sessions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_notes():
    if 'files' not in request.files:
        return jsonify({"error": "No files"}), 400
    
    files = request.files.getlist('files')
    
    # 2. UNIQUE COLLECTIONS: Create a unique ID for this specific upload
    note_id = os.urandom(4).hex()
    collection_name = processor.create_session_collection(note_id)
    
    for file in files:
        temp_path = f"./temp_{file.filename}"
        file.save(temp_path)
        processor.process_and_store(temp_path, collection_name)
        os.remove(temp_path)
    
    # Set this as the active collection in the current session
    session['current_collection'] = collection_name
    
    return jsonify({
        "status": "Success", 
        "collection": collection_name,
        "note_id": note_id
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    
    # 3. HISTORY: Use selected collection from frontend, else use latest session
    collection_name = data.get('collection_name') or session.get('current_collection')
    
    if not collection_name:
        return jsonify({"error": "Please select a note session first"}), 400

    try:
        # Get query vector using HF model defined in .env
        query_vector = processor.client.feature_extraction(
            question, 
            model=os.getenv("EMBEDDING_MODEL_ID")
        )
        
        if hasattr(query_vector, 'tolist'):
            query_vector = query_vector.tolist()
        if isinstance(query_vector[0], float):
            query_vector = [query_vector]

        # Search the specific collection
        col = Collection(collection_name)
        col.load()
        
        search_res = col.search(
            data=query_vector, 
            anns_field="vector", 
            param={"metric_type": "L2", "params": {"nprobe": 10}}, 
            limit=3, 
            output_fields=["text"]
        )
        
        context = "\n---\n".join([hit.entity.get('text') for hit in search_res[0]])
        
        # Use Groq for intelligence
        completion = groq_client.chat.completions.create(
            model=os.getenv("LLM_MODEL_ID"), # llama-3.3-70b
            messages=[
                {"role": "system", "content": "You are a helpful college assistant. Use the provided context to answer questions precisely."},
                {"role": "user", "content": f"Context from notes:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.4
        )
        
        return jsonify({
            "answer": completion.choices[0].message.content,
            "session": collection_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup():
    # Only drop the collection specified in the request
    data = request.json
    collection_name = data.get('collection_name')
    if collection_name and utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    return jsonify({"status": "Collection Deleted"}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)