import os
import sqlite3
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = "jinaai/jina-embeddings-v5-text-nano-retrieval"
LANG = "en"
DB_PATH = f"local_embeddings_{MODEL_NAME.replace("/", "-")}.db"
OVERLAP_PERCENTAGE = 0.15 
BATCH_SIZE = 64  # Adjust based on your GPU's VRAM (64-256 is usually good)

# ==========================================
# Native Database Setup (Two Tables)
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Table 1: Tweets
    c.execute('''
        CREATE TABLE IF NOT EXISTS tweets (
            id TEXT PRIMARY KEY,
            content TEXT,
            embedding BLOB
        )
    ''')
    
    # Table 2: Papers (Titles and Chunks)
    c.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            pubkey TEXT,
            sub_type TEXT,       -- 'title' or 'abstract_chunk'
            content TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    return conn

def get_existing_ids(conn, table_name):
    """Fetches all existing IDs from a specific table so we can skip them instantly."""
    c = conn.cursor()
    c.execute(f'SELECT id FROM {table_name}')
    # Return as a set for O(1) lightning-fast lookups
    return {row[0] for row in c.fetchall()}

# ==========================================
# Text Processing Functions
# ==========================================
def get_average_length(texts):
    if not texts: return 200
    return int(sum(len(t) for t in texts) / len(texts))

def chunk_text(text, target_char_length, overlap_ratio):
    """Chunks text by word boundaries to prevent cutting words in half."""
    if not text: 
        return []
    
    words = text.split()
    
    # Estimate how many words fit into the target character length
    # (Using 5 characters per word + 1 space as a rough average)
    words_per_chunk = max(1, target_char_length // 6)
    overlap_words = int(words_per_chunk * overlap_ratio)
    step = max(1, words_per_chunk - overlap_words)
    
    chunks = []
    for i in range(0, len(words), step):
        # Join the words back together into a single string
        chunk_str = " ".join(words[i:i + words_per_chunk])
        chunks.append(chunk_str)
        
    return chunks

# ==========================================
# Main Execution
# ==========================================
def run_pipeline():
    print("1. Initializing Database...")
    conn = init_db()
    
    existing_tweet_ids = get_existing_ids(conn, "tweets")
    existing_paper_ids = get_existing_ids(conn, "papers")

    print("2. Loading Datasets...")
    collection_data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "collection")["collection"]
    tweets_data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", LANG)["train"]

    print(f"3. Loading Model: {MODEL_NAME}")
    model = SentenceTransformer(
        MODEL_NAME, 
        model_kwargs={"dtype": torch.bfloat16},
        trust_remote_code=True
    )

    # --- Step 4: Process Tweets (Batched) ---
    print("\n4. Processing Tweets...")
    tweet_texts = tweets_data["text"]
    avg_tweet_len = get_average_length(tweet_texts)
    print(f"   -> Calculated average tweet length: {avg_tweet_len} characters.")

    pending_tweets = []
    for i, text in enumerate(tweet_texts):
        tweet_id = f"tweet_{i}"
        if tweet_id not in existing_tweet_ids:
            pending_tweets.append((tweet_id, text))

    if pending_tweets:
        print(f"   -> Embedding {len(pending_tweets)} new tweets...")
        # Extract just the texts for the model
        texts_to_embed = [t[1] for t in pending_tweets]
        embeddings = model.encode(texts_to_embed, batch_size=BATCH_SIZE, prompt_name="query", show_progress_bar=True)
        
        # Prepare data for bulk insert
        db_records = [
            (t[0], t[1], emb.astype(np.float32).tobytes()) 
            for t, emb in zip(pending_tweets, embeddings)
        ]
        
        c = conn.cursor()
        c.executemany('INSERT INTO tweets (id, content, embedding) VALUES (?, ?, ?)', db_records)
        conn.commit()
        print(f"   -> Saved {len(db_records)} tweets!")
    else:
        print("   -> All tweets already embedded. Skipping.")


    # --- Step 5: Process Papers (Batched) ---
    print("\n5. Processing Research Papers...")
    total_papers = len(collection_data["pubkey"])
    
    pending_papers = []

    # Helper function to embed and save a batch of papers so we don't run out of RAM
    def flush_paper_batch(paper_batch):
        if not paper_batch: return
        texts = [p['content'] for p in paper_batch]
        embs = model.encode(texts, batch_size=BATCH_SIZE, prompt_name="document")
        
        records = [
            (p['id'], p['pubkey'], p['sub_type'], p['content'], emb.astype(np.float32).tobytes())
            for p, emb in zip(paper_batch, embs)
        ]
        
        c = conn.cursor()
        c.executemany('INSERT INTO papers (id, pubkey, sub_type, content, embedding) VALUES (?, ?, ?, ?, ?)', records)
        conn.commit()

    for i in range(total_papers):
        pubkey = collection_data["pubkey"][i]
        title = collection_data["title"][i]
        abstract = collection_data["abstract"][i]
        
        # 5a. Queue Title
        title_id = f"{pubkey}_title"
        if title and title_id not in existing_paper_ids:
            pending_papers.append({
                'id': title_id, 'pubkey': pubkey, 'sub_type': 'title', 'content': title
            })
            
        # 5b. Queue Abstract Chunks
        if abstract:
            chunks = chunk_text(abstract, target_char_length=avg_tweet_len, overlap_ratio=OVERLAP_PERCENTAGE)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{pubkey}_chunk_{chunk_idx}"
                if chunk_id not in existing_paper_ids:
                    pending_papers.append({
                        'id': chunk_id, 'pubkey': pubkey, 'sub_type': 'abstract_chunk', 'content': chunk
                    })
        
        # Flush to database whenever our queue hits a multiple of BATCH_SIZE
        if len(pending_papers) >= BATCH_SIZE * 10:  # Queue up 10 batches at a time
            flush_paper_batch(pending_papers)
            pending_papers.clear()
            
        if (i + 1) % 500 == 0:
            print(f"   Scanned {i+1}/{total_papers} papers...")

    # Flush any remaining items in the queue
    if pending_papers:
        flush_paper_batch(pending_papers)

    print("\nAll data successfully embedded and saved locally!")
    conn.close()

if __name__ == "__main__":
    run_pipeline()