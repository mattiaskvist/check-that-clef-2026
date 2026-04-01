from sklearn.neighbors import NearestNeighbors
from datasets import load_dataset

# title likeness -> author likeness -> embedding likeness -> combined methods

EMBEDDING_WEIGHT = 0.5
KEYWORD_WEIGHT = 0.01
TITLE_WEIGHT = 0.1

EMBEDDING_TYPE = "embedding"
KEYWORD_TYPE = "keyword"
TITLE_TYPE = "title"

weightMap = {
    EMBEDDING_TYPE : EMBEDDING_WEIGHT,
    KEYWORD_TYPE : KEYWORD_WEIGHT,
    TITLE_TYPE : TITLE_WEIGHT
}

class scoreType():
    def _init_(self, scoreType, score):
        self.scoreType = scoreType
        self.score = score
        
    
class multiScore():
    
    def _init_(self):
        self.scores = {}
        self.totalScore = 0
        
    def addScore(self, score: scoreType):
        self.scores[score.scoreType] = score.score
        self.calculateScore()
        
    def calculateScore(self):
        self.totalScore = 0
        for score in self.scores.keys():
            self.totalScore += weightMap[score.scoreType]*self.scores[score]
            
        
            
            
class scorer():
    
    def __init__(self):
        self.NNtrainer = NearestNeighbors(n_neighbors = 10)
    
    def runSystem(self):
        
    def loadData(self, languages = ["en", "de", "fr"]):
        self.loadPaperData()
        self.loadTweets(languages)
        self.loadTrainingTweets(languages)
        
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
    
    def loadTweets(self, languages):
        self.languages = languages
        self.tweet_to_text = {}
        self.tweet_to_pubkey = {}
        for lang in self.languages:
            collection_data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "collection")[lang]
            self.tweet_to_text[lang] = {}
            self.tweet_to_pubkey[lang] = {}
            for text, pubkey, id in zip(collection_data["text"], collection_data["pubkey"], collection_data["index"]):
                self.tweet_to_text[lang][id] = text
                self.tweet_to_pubkey[lang][id] = pubkey
        self.
        
    def loadTrainingTweets(self, languages):
        
    
class rankingSystem():
    
    def _init_(self):
        pass
        
        
    
    
    
    
