import os
import sqlite3
import numpy as np
#import torch
from datasets import load_dataset
#from sentence_transformers import SentenceTransformer

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = "jinaai-jina-embeddings-v5-text-nano-retrieval"
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
    tweet_ids = tweets_data["index"]
    avg_tweet_len = get_average_length(tweet_texts)
    print(f"   -> Calculated average tweet length: {avg_tweet_len} characters.")

    pending_tweets = []
    for tweet_id, text in zip(tweet_ids,tweet_texts):
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
    

def fix_tweet_ids():
    """
    Creates a new table, populates it with the correct IDs and existing embeddings,
    then replaces the old table.
    """
    import sqlite3
    from datasets import load_dataset
    
    print("Starting ID fix via new table...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. Load dataset
    tweets_data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", LANG)["train"]
    
    c.execute("SELECT content, embedding FROM tweets order by rowid")  # Fetch embeddings in the same order as the dataset
    old_embeddings = c.fetchall()
    
    # 2. Build the mapping, using the original index directly
    # content_to_id = {}
    # for text, orig_id, embeddings in zip(tweets_data["text"], tweets_data["index"], old_embeddings):
    #     new_id = str(orig_id)  # Convert to string if not already
    #     content_to_id[text] = new_id
        
    new_records = []
    for text, orig_id in zip(tweets_data["text"], tweets_data["index"]):
        # Add LIMIT 1 just to be safe in case of duplicate texts
        c.execute("SELECT embedding FROM tweets WHERE content = ? LIMIT 1", (text,))
        
        # fetchone() returns a tuple: (blob,)
        row = c.fetchone()
        
        # Extract the actual raw bytes from the tuple
        embedding_blob = row[0] if row else None 
        
        new_id = str(orig_id)
        new_records.append((new_id, text, embedding_blob))
    
    # assert the embeddings and text are in the same order as in the original dataset
    for i, (text, orig_id) in enumerate(zip(tweets_data["text"], tweets_data["index"])):
        expected_id = str(orig_id)
        actual_id = new_records[i][0]
        assert expected_id == actual_id, f"ID mismatch at index {i}: expected {expected_id}, got {actual_id}"
        
        # check the text also matches to ensure we didn't accidentally shuffle the order
        expected_text = text
        actual_text = new_records[i][1]
        assert expected_text == actual_text, f"Text mismatch at index {i}: expected '{expected_text}', got '{actual_text}'"
    
    # 3. Create the new temporary table
    c.execute('''
        CREATE TABLE IF NOT EXISTS tweets_fixed (
            id TEXT PRIMARY KEY,
            content TEXT,
            embedding BLOB
        )
    ''')
    # Clear it just in case it already exists from a previous failed run
    c.execute('DELETE FROM tweets_fixed') 
    
    
    # 6. Insert into the new table
    try:
        print(f"Inserting {len(new_records)} records into new table...")
        c.executemany('INSERT INTO tweets_fixed (id, content, embedding) VALUES (?, ?, ?)', new_records)
        
        # 7. Swap the tables (Commented out for safe testing)
        print("Swapping tables (currently skipped for testing)...")
        c.execute('ALTER TABLE tweets RENAME TO tweets_broken')
        c.execute('ALTER TABLE tweets_fixed RENAME TO tweets')
        
        conn.commit()
        print(f"ID Fix Completed: Successfully rebuilt table with {len(new_records)} records.")
        
    except sqlite3.IntegrityError as e:
        print(f"Database Error (Likely duplicate IDs): {e}")
        conn.rollback()
        
    finally:
        conn.close()
    
    

if __name__ == "__main__":
    #run_pipeline()
    fix_tweet_ids()
    
    