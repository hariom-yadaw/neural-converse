from utilities.loaddata import load_pickled_data
from utilities.model_eval import converse

__author__ = 'uyaseen'

if __name__ == '__main__':
    voc, data = load_pickled_data(path='data/cornell movie-dialogs corpus/dataset.pkl')
    emb = load_pickled_data(path='data/cornell movie-dialogs corpus/w2vec.pkl')
    max_len = len(data[0][0][0])
    converse(vocabulary=voc, embeddings=emb, m_path='data/cornell movie-dialogs corpus/models/tr-best_model.pkl',
             enc='gru', dec='gru', max_len=max_len, hidden_dim=1024)
