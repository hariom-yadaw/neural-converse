import cPickle as pkl
import os.path
import numpy as np
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
from loaddata import shared_data

__author__ = 'uyaseen'


# sample responses from the model, for random queries from test set.
def sanity_check(dataset, vocabulary, embeddings, m_path, enc, dec,
                 emb_dim=300, hidden_dim=1024, max_response=50,
                 sample_count=50, batch_size=50):
    print('sanity_check(..)')
    assert os.path.isfile(m_path), True
    vocab, words_to_ix, ix_to_words = vocabulary
    eos_token = words_to_ix['EOS']
    pad_token = words_to_ix['PADDING']
    test_set_x, test_set_y, test_set_y_mask = shared_data(dataset[2])
    n_test_examples = test_set_x.get_value(borrow=True).shape[0]
    x = T.imatrix('x')
    mask = T.imatrix('mask')  # only needed for decoder
    max_len = len(test_set_x.get_value(borrow=True)[0])
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
        decoder = RnnDec(enc_h=encoder.h, mask=mask, emb_mat=embeddings,
                         vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim,
                         eos_token=eos_token, batch_size=batch_size, max_len=max_len,
                         params=dec_params, max_response=max_response)
    elif dec == 'gru':
        decoder = GruDec(enc_h=encoder.h, mask=mask, emb_mat=embeddings,
                         vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim,
                         eos_token=eos_token, batch_size=batch_size, max_len=max_len,
                         params=dec_params, max_response=max_response)
    elif dec == 'lstm':
        decoder = LstmDec(enc_h=encoder.h, mask=mask, emb_mat=embeddings,
                          vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim,
                          eos_token=eos_token, batch_size=batch_size, max_len=max_len,
                          params=dec_params, max_response=max_response)
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
        seedling_t = test_set_x.get_value(borrow=True)[seed]
        query = ' '.join(ix_to_words[ix] for ix in seedling_t if ix != pad_token)
        print('query    :: %s' % query)
        seedling = np.empty((1, max_len), dtype='int32')
        seedling[0] = seedling_t
        fruit = get_pred(seedling)
        response = ' '.join(ix_to_words[ix] for ix in fruit)
        print('response :: %s' % response)


# converse with the user :-)
def converse(vocabulary, embeddings, m_path, enc, dec, max_len,
             emb_dim=300, hidden_dim=1024, max_response=50, batch_size=50):
    assert os.path.isfile(m_path), True
    vocab, words_to_ix, ix_to_words = vocabulary
    eos_token = words_to_ix['EOS']
    pad_token = words_to_ix['PADDING']
    x = T.imatrix('x')
    mask = T.imatrix('mask')  # only needed for decoder
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
        decoder = RnnDec(enc_h=encoder.h, mask=mask, emb_mat=embeddings,
                         vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim,
                         eos_token=eos_token, batch_size=batch_size, max_len=max_len,
                         params=dec_params, max_response=max_response)
    elif dec == 'gru':
        decoder = GruDec(enc_h=encoder.h, mask=mask, emb_mat=embeddings,
                         vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim,
                         eos_token=eos_token, batch_size=batch_size, max_len=max_len,
                         params=dec_params, max_response=max_response)
    elif dec == 'lstm':
        decoder = LstmDec(enc_h=encoder.h, mask=mask, emb_mat=embeddings,
                          vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim,
                          eos_token=eos_token, batch_size=batch_size, max_len=max_len,
                          params=dec_params, max_response=max_response)
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
        query_t = raw_input('Human >> ')
        query_t = [words_to_ix[wd] if wd in vocab else words_to_ix[unk]
                   for wd in tokenize(remove_symbols(query_t.lower()))]
        if len(query_t) > 0:
            query = np.empty((1, max_len), dtype='int32')
            query[0] = [pad_token] * (max_len - len(query_t)) + query_t
            fruit = get_pred(query)
            response = ' '.join(ix_to_words[ix] for ix in fruit)
            print('Machine >> %s' % response)
        else:
            response = 'at-least say a word :/ ...'
            print('Machine >> %s' % response)
