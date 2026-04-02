from sklearn.neighbors import NearestNeighbors
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
import sqlite3
import polars as pl
import numpy as np
import json
import ollama
from sentence_transformers import SentenceTransformer
import torch


class rankingSystem():
    
    def __init__(self, 
                 topN = 10, 
                 databaseURL = "local_embeddings_jinaai-jina-embeddings-v5-text-nano-retrieval.db", 
                 title_model_name = 'llama3.2',
                 embedding_model_name = "jinaai/jina-embeddings-v5-text-nano-retrieval"):
        self.abstractChunksNN = NearestNeighbors(n_neighbors = topN, metric = "cosine")
        self.titleNN = NearestNeighbors(n_neighbors = topN, metric = "cosine")
        self.classifier = RandomForestClassifier()
        self.databaseURL = databaseURL

        try:
            ollama.show(title_model_name)
            print("✅ Model is available locally.")
        except Exception:
            print(f"⏳ Model not found. Pulling '{title_model_name}'...")
            ollama.pull(title_model_name)
            print("✅ Model pulled successfully!")
        self.title_model_name = title_model_name

        self.embeddingModel = SentenceTransformer(
            embedding_model_name, 
            model_kwargs={"dtype": torch.bfloat16},
            trust_remote_code=True
        )
    
    def trainSystem(self):
        


    def runSystem(self):


    def getLLMExtraction(self, tweet_text):
        json_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
            },
            "required": ["title"]
        }
        prompt = (
            "You are an expert researcher."
            "Read the following tweet and extract the referenced research paper. "
            "Return the only the title, if you dont know which paper, make an educated guess. "
            "\n\n"
            f"Tweet: {tweet_text}"
        )
        
        try:
            response = ollama.chat(
                model=self.title_model_name,
                messages=[{'role': 'user', 'content': prompt}],
                format=json_schema, 
                options={"temperature": 0.0} 
            )
            result = json.loads(response['message']['content'])
            return result.get('title', '')
        except Exception as e:
            print(f"Extraction Error: {e}")
            return "", "", []
        
    def loadData(self, languages = ["en", "de", "fr"]):
        self.languages = languages
        self.loadPaperData()
        self.loadTweets()
        self.loadTrainingTweets()
        self.loadAndTrainNN()
        
    def loadPaperData(self):
        collection_data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "collection")["collection"]
        self.title_to_pubkey = {}
        self.pubkey_to_title = {}
        self.pubkey_to_abstract = {}
        self.pubkey_to_authors = {}

        for title, pubkey, abstract, authors in zip(collection_data["title"], collection_data["pubkey"], collection_data["abstract"], collection_data["authors"]):
            if title:
                self.title_to_pubkey[title.lower()] = pubkey
                self.pubkey_to_title[pubkey] = title
                self.pubkey_to_abstract[pubkey] = abstract if abstract else ""
                self.pubkey_to_authors[pubkey] = authors
    
    def loadTweets(self):
        self.tweet_to_text = {}
        self.tweet_to_pubkey = {}
        for lang in self.languages:
            collection_data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", lang)["train"]
            self.tweet_to_text[lang] = {}
            self.tweet_to_pubkey[lang] = {}
            for text, pubkey, id in zip(collection_data["text"], collection_data["pubkey"], collection_data["index"]):
                self.tweet_to_text[lang][id] = text
                self.tweet_to_pubkey[lang][id] = pubkey
        
    def loadTrainingTweets(self):
        self.test_tweet_to_text = {}
        
        for lang in self.languages:
            collection_data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", lang)["dev"]
            self.test_tweet_to_text[lang] = {}
            for id, text in zip(collection_data["index"], collection_data["text"]):
                self.test_tweet_to_text[lang][id] = text

    def loadAndTrainNN(self):
        uri = "sqlite:///" + self.databaseURL
        query = "SELECT pubkey, subtype, embedding FROM papers;"
        df = pl.read_database(query=query, connection=uri)
        df_titles = df.filter(pl.col("subtype") == "title")
        df_abstracts = df.filter(pl.col("subtype") == "abstract_chunk")

        def parse_embeddings(dataframe):
            return dataframe.with_columns(
                pl.col("embedding").map_elements(
                    lambda x: json.loads(x), 
                    return_dtype=pl.List(pl.Float64)
                )
            )

        df_titles = parse_embeddings(df_titles)
        df_abstracts = parse_embeddings(df_abstracts)

        self.NNTitlesPubkeys = df_titles["pubkeys"].to_list()
        self.NNabstractPubkeys = df_abstracts["pubkeys"].to_list()

        X_titles = np.vstack(df_titles["embedding"].to_list())
        X_abstracts = np.vstack(df_abstracts["embedding"].to_list())

        self.titleNN.fit(X_titles)
        self.abstractChunksNN(X_abstracts)
    
    def getPubkeyNNTitle(self, index):
        return self.NNTitlesPubkeys[index]
    
    def getPubkeyNNAbstract(self, index):
        return self.NNabstractsPubkeys[index]

        
        
    
    
    
    
