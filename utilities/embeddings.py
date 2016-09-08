import gensim
import numpy as np
import cPickle as pkl

from utilities.cornell_movie_dialogs import load_data
from utilities.initializations import get

__author__ = 'uyaseen'


# w2vec pre-trained model contains embeddings for 3 million words, to save memory we will extract
# embedding vector only for 'our' vocabulary
def pickle_w2vec(w2vec, dataset, emb_path, emb_dim=300):
    model = gensim.models.Word2Vec.load_word2vec_format(w2vec, binary=True)
    voc, _ = load_data(path=dataset)
    vocab, words_to_ix, _ = voc
    emb = [0] * len(vocab)
    vocab.remove('EOS')
    vocab.remove('UNKNOWN_TOKEN')
    # initialize randomly for 'EOS' & 'UNKNOWN_TOKEN'
    emb[words_to_ix['EOS']] = get(identifier='emb', shape=(emb_dim,), scale=np.sqrt(3))
    emb[words_to_ix['UNKNOWN_TOKEN']] = get(identifier='emb', shape=(emb_dim,), scale=np.sqrt(3))
    unk_count = 2
    for word in vocab:
        if word in model.vocab:
            emb[words_to_ix[word]] = model[word]
        else:
            unk_count += 1
            emb[words_to_ix[word]] = get(identifier='emb', shape=(emb_dim,), scale=np.sqrt(3))
    print('... embeddings initialized randomly for %d words' % unk_count)
    # pickle our mini embeddings
    with open(emb_path, 'wb') as f:
        pkl.dump(emb, f, pkl.HIGHEST_PROTOCOL)
    print('... %s created' % emb_path)
