import os
import shutil
import urllib
import zipfile
from utilities.cornell_movie_dialogs import pickle_cornell
from utilities.embeddings import pickle_w2vec

__author__ = 'uyaseen'


def fetch_and_extract(b_path, dataset):
    dataset_url = 'http://www.mpi-sws.org/~cristian/data/' + dataset
    print('downloading data from %s' % dataset_url)
    if not os.path.exists(b_path):
        os.makedirs(b_path)
    urllib.urlretrieve(dataset_url, b_path + dataset)
    with zipfile.ZipFile(b_path + dataset, 'r') as z:
        z.extractall(b_path)
    # delete extra directory
    shutil.rmtree(b_path + '__MACOSX')


if __name__ == '__main__':
    _dir = 'data/'
    _data_dir = 'cornell movie-dialogs corpus/'
    _file = 'cornell_movie_dialogs_corpus.zip'
    _w2vec = 'GoogleNews-vectors-negative300.bin.gz'
    if not os.path.isfile(_dir + _file):
        fetch_and_extract(b_path=_dir, dataset=_file)
    pickle_cornell(b_path=_dir + _data_dir, mv_lines='movie_lines.txt',
                   mv_converse='movie_conversations.txt', dataset_size=50000,
                   va_split=0.1, te_split=0.1)
    assert os.path.exists(_dir + _w2vec), True
    pickle_w2vec(w2vec=_dir + _w2vec, dataset=_dir + _data_dir + 'dataset.pkl',
                 emb_path=_dir + _data_dir + 'w2vec.pkl')
    print('... done')
