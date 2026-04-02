import numpy as np
import polars as pl
import json
from datasets import load_dataset
from sklearn.neighbors import NearestNeighbors
def main():
    # system = rankingSystem(
    #     topN=10,
    #     databaseURL = "local_embeddings_jinaai-jina-embeddings-v5-text-nano-retrieval.db", 
    #     title_model_name = 'llama3.2',
    #     embedding_model_name = "jinaai/jina-embeddings-v5-text-nano-retrieval"
    # )
    

    # # load papers, tweets, testing tweets etc.
    # system.load_data()
    
    # # train the classifier on training set
    # system.train_system()
    
    # system.run_system()
    
    # system.evaluate_system()
    
    db_path = "C:\Pluggmapp\check-that-clef-2026\local_embeddings_jinaai-jina-embeddings-v5-text-nano-retrieval.db"

    print(query_db(db_path, "SELECT * FROM papers"))

    # step 1. load embeddings for paper chunks
    paper_chunks = load_paper_chunks(db_path)
    print("Paper chunks with embeddings:")
    print(paper_chunks.head())
    print("-" * 50)
    
    # step 2. load testing tweets, and get their embeddings
    testing_tweets = load_tweets(db_path)
    print("Testing tweets with embeddings:")
    print(testing_tweets.head())
    print("-" * 50)
    
    
    
    
    # step 3. for each testing tweet,
    # calculate cosine similarity with all paper chunks,
    # and get top N most similar paper chunks
    
    NN = NearestNeighbors(n_neighbors=10, metric='cosine')
    NN.fit(paper_chunks["embedding"].to_list())
    print("Nearest neighbors model fitted on paper chunk embeddings.")
    
    # step 4. for each testing tweet,
    # calculate the relevance score based on the top N most similar paper chunks
    # and get the final ranking of papers for each testing tweet.
    print("Calculating nearest neighbors for testing tweets...")
    subset_testing_tweets = testing_tweets.head(5)  # For demonstration, we will only use the first 5 testing tweets
    distances, indices = NN.kneighbors(subset_testing_tweets["embedding"].to_list())
    
    print("Loading ground truth")

    # evaluate result using ground truth data
    languages = ["en"]
    tweet_to_pubkey = load_tweet_pubkeys(languages=languages)
    print(tweet_to_pubkey["en"])
    for i, tweet_id in enumerate(subset_testing_tweets["id"]):
        print(f"Testing tweet ID: {tweet_id}")
        clean_id = str(tweet_id).replace("tweet_", "")
        print(f"True pubkey: {tweet_to_pubkey['en'][clean_id]}")
        print("Top 10 most similar paper chunks:")
        for idx in indices[i]:
            print(paper_chunks["pubkey"][idx])
        print("-" * 50)
        
def parse_embeddings(dataframe: pl.DataFrame) -> pl.DataFrame:
    return dataframe.with_columns(
        pl.col("embedding").map_elements(
            # Assuming your embeddings were saved as 32-bit floats. 
            # If they look garbled or are the wrong length, change to np.float64
            lambda x: np.frombuffer(x, dtype=np.float32).tolist(), 
            return_dtype=pl.List(pl.Float32)
        )
    )

def query_db(chunk_path: str, query: str) -> pl.DataFrame:
    uri = "sqlite:///" + chunk_path
    return pl.read_database_uri(query=query, uri=uri)

def load_paper_chunks(chunk_path: str) -> pl.DataFrame:
    """
    Load paper chunks from the given path and return a DataFrame.
    
    Returns:
        A DataFrame containing the paper chunks with their embeddings.
        Each row corresponds to a paper chunk and includes the following columns:
        - pubkey: The unique identifier for the paper.
        - embedding: The embedding vector for the chunk, parsed from a JSON string to a list
    """
    query = "SELECT pubkey, embedding FROM papers WHERE sub_type = \"abstract_chunk\";"
    df = query_db(chunk_path, query)
    df_abstracts = parse_embeddings(df)
    return df_abstracts

def load_tweets(chunk_path: str) -> pl.DataFrame:
    """
    Load tweets from the given path and return a DataFrame.
    Returns:
        A DataFrame containing the tweets with their embeddings.
        Each row corresponds to a tweet and includes the following columns:
        - pubkey: The unique identifier for the tweet.
        - subtype: The type of the chunk (e.g., "tweet").
        - embedding: The embedding vector for the tweet, parsed from a JSON string to a list
    """
    query = "SELECT id, embedding FROM tweets;"
    df = query_db(chunk_path, query)
    df_tweets = parse_embeddings(df)
    return df_tweets

def load_tweet_pubkeys(languages:list = ["en", "de", "fr"]) -> dict[str, dict[str, str]]:
    """
    Load the ground truth data for the given languages and return a dictionary mapping tweet IDs to paper pubkeys.
    """
    tweet_to_pubkey = {}
    for lang in languages:
        collection_data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", lang).get("train")
        tweet_to_pubkey[lang] = {}
        
        # the collection data contains the following columns: "text", "pubkey", "index"
        # pubkey points to the paper
        for pubkey, id in zip(collection_data["pubkey"], collection_data["index"]):
            tweet_to_pubkey[lang][str(id)] = pubkey
    
    return tweet_to_pubkey
        

if __name__ == "__main__":
    main()