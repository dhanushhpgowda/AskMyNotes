import os
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from huggingface_hub import login, InferenceClient

load_dotenv()

# Setup Models
HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_ID") # sentence-transformers/all-mpnet-base-v2

login(token=HF_TOKEN)
client = InferenceClient(api_key=HF_TOKEN)

def connect_to_milvus():
    connections.connect(
        "default", 
        host=os.getenv("MILVUS_HOST"), 
        port=os.getenv("MILVUS_PORT")
    )

def create_session_collection(note_id):
    collection_name = f"notes_{note_id.replace('-', '_')}"
    
    # We don't drop the collection anymore so we can keep HISTORY
    if utility.has_collection(collection_name):
        return collection_name
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # all-mpnet-base-v2 requires dimension 768
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, "Stored student notes history")
    collection = Collection(collection_name, schema)
    
    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    collection.create_index(field_name="vector", index_params=index_params)
    return collection_name

def process_and_store(file_path, collection_name):
    # 1. Get Text
    text = ""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = "".join([p.extract_text() for p in PdfReader(file_path).pages])
    elif ext == ".docx":
        text = "\n".join([para.text for para in Document(file_path).paragraphs])
    else:
        with open(file_path, 'r') as f: text = f.read()

    # 2. Chunking
    chunks = [text[i:i+500] for i in range(0, len(text), 300)]
    
    # 3. Embedding (Using the HF model from .env)
    embeddings = client.feature_extraction(chunks, model=EMBEDDING_MODEL)
    
    if hasattr(embeddings, 'tolist'):
        embeddings = embeddings.tolist()
    
    # 4. Insert into Milvus
    collection = Collection(collection_name)
    collection.insert([embeddings, chunks])
    collection.flush()
    collection.load() 
    return True