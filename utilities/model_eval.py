import cPickle as pkl
import os.path
from random import randint

import theano
import theano.tensor as T

from model.encoder.rnn import RnnEnc, BiRnnEnc
from model.encoder.gru import GruEnc, BiGruEnc
from model.encoder.lstm import LstmEnc, BiLstmEnc
from model.decoder.rnn import RnnDec
from model.decoder.gru import GruDec
from model.decoder.lstm import LstmDec

from utils import tokenize, remove_symbols

__author__ = 'uyaseen'


# sample responses from the model, for random queries from test set.
def sanity_check(dataset, vocabulary, embeddings, m_path, enc, dec,
                 emb_dim=300, hidden_dim=1024, max_response=50,
                 sample_count=50):
    print('sanity_check(..)')
    assert os.path.isfile(m_path), True
    vocab, words_to_ix, ix_to_words = vocabulary
    eos_token = words_to_ix['EOS']
    test_set_x, test_set_y = dataset[2]
    n_test_examples = len(test_set_x)
    x = T.ivector('x')
    seq_len = T.iscalar('seq_len')  # not required at test time, but still need to be provided to decoder :/
    vocab_size = len(vocab)
    with open(m_path, 'rb') as f:
                enc_params, dec_params = pkl.load(f)

    if enc == 'rnn':
        encoder = RnnEnc(input=x, emb_mat=embeddings, emb_dim=emb_dim,
                         hidden_dim=hidden_dim, params=enc_params)
    elif enc == 'bi-rnn':
        encoder = BiRnnEnc(input=x, emb_mat=embeddings, emb_dim=emb_dim,
                           hidden_dim=hidden_dim, params=enc_params)
    elif enc == 'gru':
        encoder = GruEnc(input=x, emb_mat=embeddings, emb_dim=emb_dim,
                         hidden_dim=hidden_dim, params=enc_params)
    elif enc == 'bi-gru':
        encoder = BiGruEnc(input=x, emb_mat=embeddings, emb_dim=emb_dim,
                           hidden_dim=hidden_dim, params=enc_params)
    elif enc == 'lstm':
        encoder = LstmEnc(input=x, emb_mat=embeddings, emb_dim=emb_dim,
                          hidden_dim=hidden_dim, params=enc_params)
    elif enc == 'bi-lstm':
        encoder = BiLstmEnc(input=x, emb_mat=embeddings, emb_dim=emb_dim,
                            hidden_dim=hidden_dim, params=enc_params)
    else:
        print('Only supported encoders are:\n'
              'rnn, bi-rnn, gru, bi-gru, lstm, bi-lstm')
        raise TypeError
    if dec == 'rnn':
        decoder = RnnDec(enc_h=encoder.h, seq_len=seq_len, emb_mat=embeddings,
                         vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim,
                         eos_token=eos_token, max_response=max_response, params=dec_params)
    elif dec == 'gru':
        decoder = GruDec(enc_h=encoder.h, seq_len=seq_len, emb_mat=embeddings,
                         vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim,
                         eos_token=eos_token, max_response=max_response, params=dec_params)
    elif dec == 'lstm':
        decoder = LstmDec(enc_h=encoder.h, seq_len=seq_len, emb_mat=embeddings,
                          vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim,
                          eos_token=eos_token, max_response=max_response, params=dec_params)
    else:
        print('Only supported decoders are:\n'
              'rnn, gru, lstm')
        raise TypeError

    get_pred = theano.function(
        inputs=[x],
        outputs=decoder.sample()
    )
    for i in xrange(sample_count):
        print('%i.' % (i + 1))
        seed = randint(0, n_test_examples - 1)
        seedling = test_set_x[seed][0:-1]
        if len(seedling) > 1:
            query = ' '.join(ix_to_words[ix] for ix in seedling)
            print('query    :: %s' % query)
            fruit = get_pred(seedling)[0:-1]
            response = ' '.join(ix_to_words[ix] for ix in fruit)
            print('response :: %s' % response)


# converse with the user :-)
def converse(vocabulary, embeddings, m_path, enc, dec,
             emb_dim=300, hidden_dim=1024, max_response=50):
    assert os.path.isfile(m_path), True
    vocab, words_to_ix, ix_to_words = vocabulary
    eos_token = words_to_ix['EOS']
    x = T.ivector('x')
    seq_len = T.iscalar('seq_len')  # not required at test time, but still need to be provided to decoder :/
    vocab_size = len(vocab)
    with open(m_path, 'rb') as f:
        enc_params, dec_params = pkl.load(f)

    if enc == 'rnn':
        encoder = RnnEnc(input=x, emb_mat=embeddings, emb_dim=emb_dim,
                         hidden_dim=hidden_dim, params=enc_params)
    elif enc == 'bi-rnn':
        encoder = BiRnnEnc(input=x, emb_mat=embeddings, emb_dim=emb_dim,
                           hidden_dim=hidden_dim, params=enc_params)
    elif enc == 'gru':
        encoder = GruEnc(input=x, emb_mat=embeddings, emb_dim=emb_dim,
                         hidden_dim=hidden_dim, params=enc_params)
    elif enc == 'bi-gru':
        encoder = BiGruEnc(input=x, emb_mat=embeddings, emb_dim=emb_dim,
                           hidden_dim=hidden_dim, params=enc_params)
    elif enc == 'lstm':
        encoder = LstmEnc(input=x, emb_mat=embeddings, emb_dim=emb_dim,
                          hidden_dim=hidden_dim, params=enc_params)
    elif enc == 'bi-lstm':
        encoder = BiLstmEnc(input=x, emb_mat=embeddings, emb_dim=emb_dim,
                            hidden_dim=hidden_dim, params=enc_params)
    else:
        print('Only supported encoders are:\n'
              'rnn, bi-rnn, gru, bi-gru, lstm, bi-lstm')
        raise TypeError
    if dec == 'rnn':
        decoder = RnnDec(enc_h=encoder.h, seq_len=seq_len, emb_mat=embeddings,
                         vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim,
                         eos_token=eos_token, max_response=max_response, params=dec_params)
    elif dec == 'gru':
        decoder = GruDec(enc_h=encoder.h, seq_len=seq_len, emb_mat=embeddings,
                         vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim,
                         eos_token=eos_token, max_response=max_response, params=dec_params)
    elif dec == 'lstm':
        decoder = LstmDec(enc_h=encoder.h, seq_len=seq_len, emb_mat=embeddings,
                          vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim,
                          eos_token=eos_token, max_response=max_response, params=dec_params)
    else:
        print('Only supported decoders are:\n'
              'rnn, gru, lstm')
        raise TypeError

    get_pred = theano.function(
        inputs=[x],
        outputs=decoder.sample()
    )
    unk = 'UNKNOWN_TOKEN'
    while True:
        query = raw_input('Human >> ')
        query = [words_to_ix[wd] if wd in vocab else words_to_ix[unk]
                 for wd in tokenize(remove_symbols(query.lower()))]
        if len(query) > 0:
            fruit = get_pred(query)[0:-1]
            response = ' '.join(ix_to_words[ix] for ix in fruit)
            print('Machine >> %s' % response)
        else:
            response = 'at-least say a word :/ ...'
            print('Machine >> %s' % response)
