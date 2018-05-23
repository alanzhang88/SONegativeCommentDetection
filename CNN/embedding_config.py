EMBEDDING_MODELS = ["word2vec","glove_twitter"]
class Config:
    def __init__(self, embedding_file):
        self.embedding_file = embedding_file

class Embedding_Config:
    def __init__(self, embedding_models):
        for model in embedding_models:
            if model == EMBEDDING_MODELS[0]:
                ind = Config('./embeddings/word2vec_vec')
            elif model == EMBEDDING_MODELS[1]:
                ind = Config('./embeddings/glove.twitter.27B/glove.twitter.27B.100d.txt')
            self.embedding_configs.append(ind)
    
    embedding_configs = []
    embedding_file = ""


config = Embedding_Config(EMBEDDING_MODELS[0:1])