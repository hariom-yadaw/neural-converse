import cPickle as pkl
import timeit
import os

import theano
import theano.tensor as T

from random import randint
import numpy as np

from model.encoder.rnn import RnnEnc, BiRnnEnc
from model.encoder.gru import GruEnc, BiGruEnc
from model.encoder.lstm import LstmEnc, BiLstmEnc
from model.decoder.rnn import RnnDec
from model.decoder.gru import GruDec
from model.decoder.lstm import LstmDec

from utilities.optimizers import get_optimizer
from utilities.utils import shuffle_pair
from utilities.cornell_movie_dialogs import load_data

__author__ = 'uyaseen'


def seq2seq(dataset, vocabulary, embeddings, b_path, enc, dec,
            use_existing_model=True, optimizer='rmsprop', emb_dim=300,
            hidden_dim=1024, n_epochs=100, max_response=50):
    print('seq2seq(..)')
    vocab, words_to_ix, ix_to_words = vocabulary
    eos_token = words_to_ix['EOS']
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]

    print('... building the model')
    # allocate symbolic variables for the input & output sequences
    x = T.ivector('x')
    y = T.ivector('y')
    seq_len = T.iscalar('seq_len')  # need to sample only till target sequence's length while 'training'
    vocab_size = len(vocab)
    m_path = b_path + 'models/va-best_model.pkl'

    enc_params = None
    dec_params = None
    if not os.path.exists(b_path + 'models/'):
        os.makedirs(b_path + 'models/')
    if use_existing_model:
        if os.path.isfile(m_path):
            with open(m_path, 'rb') as f:
                enc_params, dec_params = pkl.load(f)
        else:
            print('Unable to load existing model %s, initializing model with random weights' % m_path)

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

    all_params = encoder.params + decoder.params
    cost = decoder.negative_log_likelihood(y)
    updates = get_optimizer(identifier=optimizer, cost=cost, params=all_params)
    train_model = theano.function(
        inputs=[x, seq_len, y],
        outputs=cost,
        updates=updates
    )
    get_cost = theano.function(
        inputs=[x, seq_len, y],
        outputs=cost
    )
    get_pred = theano.function(
        inputs=[x],
        outputs=decoder.sample()
    )
    validation_freq = 1  # check the 'error/cost' on validation set after going through these many epochs
    shuffle_freq = 1
    sampling_freq = 1
    best_train_error = np.inf
    best_valid_error = np.inf
    n_train_examples = len(train_set_x)
    n_valid_examples = len(valid_set_x)
    n_test_examples = len(test_set_x)
    print('Vocabulary: %i' % vocab_size)
    print('Size of training set: %i' % n_train_examples)
    print('Size of validation set: %i' % n_valid_examples)
    print('Size of test set: %i' % n_test_examples)
    print('encoder -- %s' % enc)
    print('decoder -- %s' % dec)
    print('... training')
    epoch = 0
    start_time = timeit.default_timer()
    while epoch < n_epochs:
        epoch += 1
        ep_start_time = timeit.default_timer()
        if epoch % shuffle_freq == 0:
            train_set_x, train_set_y = shuffle_pair(train_set_x, train_set_y)
        train_cost = 0.
        for i in xrange(n_train_examples):
            if len(train_set_x[i][0:-1]) > 1 and len(train_set_y[i]) > 1:
                # encoder should get the whole input except for 'eos' token
                train_cost += train_model(train_set_x[i][0:-1], len(train_set_y[i]), train_set_y[i])

        print('epoch: %i/%i, cost: %0.8f, /epoch: %.4fm' %
              (epoch, n_epochs, train_cost/n_train_examples,
               (timeit.default_timer() - ep_start_time) / 60.))

        # save the best 'train' model (helpful if there is not enough validation data)
        if train_cost/n_train_examples < best_train_error:
            best_train_error = train_cost/n_train_examples
            with open(b_path + 'models/tr-best_model.pkl', 'wb') as f:
                    dump_params = encoder.params, decoder.params
                    pkl.dump(dump_params, f, pkl.HIGHEST_PROTOCOL)
        # sample responses from the model now and then
        if epoch % sampling_freq == 0:
            # pick a random example from the training set
            seed = randint(0, n_train_examples - 1)
            seedling = train_set_x[seed][0:-1]
            fruit = get_pred(seedling)[0:-1]
            query = ' '.join(ix_to_words[ix] for ix in seedling)
            response = ' '.join(ix_to_words[ix] for ix in fruit)
            print('<sample>')
            print('query    :: %s' % query)
            print('response :: %s' % response)
            print('</sample>')

        if epoch % validation_freq == 0:
            valid_cost = 0.
            for i in xrange(n_valid_examples):
                if len(valid_set_x[i][0:-1]) > 1 and len(valid_set_y[i]) > 1:
                    valid_cost += get_cost(valid_set_x[i][0:-1], len(valid_set_y[i]), valid_set_y[i])
            # save the current best 'valid' model
            if valid_cost/n_valid_examples < best_valid_error:
                best_valid_error = valid_cost/n_valid_examples
                with open(m_path, 'wb') as f:
                    dump_params = encoder.params, decoder.params
                    pkl.dump(dump_params, f, pkl.HIGHEST_PROTOCOL)
    print('The code ran for %.2fm' % ((timeit.default_timer() - start_time) / 60.))
    return 0

if __name__ == '__main__':
    voc, data = load_data(path='data/cornell movie-dialogs corpus/dataset.pkl')
    emb = load_data(path='data/cornell movie-dialogs corpus/w2vec.pkl')
    seq2seq(data, voc, emb, b_path='data/cornell movie-dialogs corpus/',
            enc='gru', dec='gru', use_existing_model=True,
            optimizer='rmsprop', emb_dim=300, hidden_dim=1024,
            n_epochs=20, max_response=100)
    print('... done')
