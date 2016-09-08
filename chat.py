from utilities.cornell_movie_dialogs import load_data
from utilities.model_eval import converse

__author__ = 'uyaseen'

if __name__ == '__main__':
    voc, _ = load_data(path='data/cornell movie-dialogs corpus/dataset.pkl')
    emb = load_data(path='data/cornell movie-dialogs corpus/w2vec.pkl')
    converse(vocabulary=voc, embeddings=emb, m_path='data/cornell movie-dialogs corpus/models/tr-best_model.pkl',
             enc='gru', dec='gru', hidden_dim=1024)
