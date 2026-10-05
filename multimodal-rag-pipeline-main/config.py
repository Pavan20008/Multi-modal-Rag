import os

# ---- Base directories ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# ---- Models (all free / open-source, run locally) ----
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"          # sentence-transformers
CLIP_MODEL_NAME = "ViT-B-32"                    # open_clip architecture
CLIP_PRETRAINED = "laion2b_s34b_b79k"           # open_clip pretrained weights
VLM_MODEL = "qwen2.5vl:7b"                       # served locally via Ollama

# ---- Chunking ----
CHUNK_SIZE = 800        # characters per text chunk
CHUNK_OVERLAP = 100

# ---- Retrieval ----
TOP_K_TEXT = 4
TOP_K_IMAGES = 2

# ---- Chroma collection names ----
TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "image_chunks"

# ---- BM25 (keyword) retrieval ----
BM25_TOP_K = 8