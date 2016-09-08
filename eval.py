from utilities.cornell_movie_dialogs import load_data
from utilities.model_eval import sanity_check

__author__ = 'uyaseen'


if __name__ == '__main__':
    voc, data = load_data(path='data/cornell movie-dialogs corpus/dataset.pkl')
    emb = load_data(path='data/cornell movie-dialogs corpus/w2vec.pkl')
    sanity_check(data, voc, emb, m_path='data/cornell movie-dialogs corpus/models/tr-best_model.pkl',
                 enc='gru', dec='gru', hidden_dim=1024, sample_count=10)
    print('... done')
