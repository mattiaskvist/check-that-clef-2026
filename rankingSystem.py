from sklearn.neighbors import NearestNeighbors

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
    
class rankingSystem():
    
    def _init_(self):
        self.NNtrainer = NearestNeighbors(n_neighbors = 10)
        
        
    def loadData(self):
        
        
    
    
    
    
