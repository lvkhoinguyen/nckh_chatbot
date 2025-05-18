from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

# Cache directory setup
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def save_cache(file_name, data):
    """Save data to cache"""
    with open(os.path.join(CACHE_DIR, file_name), "wb") as f:
        pickle.dump(data, f)

def load_cache(file_name):
    """Load data from cache"""
    file_path = os.path.join(CACHE_DIR, file_name)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None

def create_embeddings_with_cache(chunks, model_name='all-MiniLM-L6-v2', cache_file=None):
    """Create embeddings for text chunks with caching"""
    if cache_file is None:
        cache_file = f"embeddings_{model_name}.pkl"
    cached_data = load_cache(cache_file)
    if cached_data:
        return cached_data
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    save_cache(cache_file, (embeddings, model))
    return embeddings, model

def find_relevant_chunks_with_cache(question, chunks, embeddings, model, top_n=3):
    """Find relevant chunks for a question using cached embeddings"""
    # Create embedding for the question
    question_embedding = model.encode([question])

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Search for top N relevant chunks
    _, top_indices = index.search(question_embedding, top_n)
    relevant_chunks = [chunks[i] for i in top_indices[0]]
    return relevant_chunks

def re_rank_chunks(question, chunks, model_name='all-MiniLM-L6-v2'):
    """Re-rank chunks based on semantic similarity to question"""
    from sentence_transformers import util

    model = SentenceTransformer(model_name)
    question_embedding = model.encode(question, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    # Compute cosine similarity scores
    scores = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
    scored_chunks = list(zip(chunks, scores.tolist()))
    
    # Sort by score in descending order
    return sorted(scored_chunks, key=lambda x: x[1], reverse=True)
