import cPickle as pkl
import timeit
import os.path

import theano
import theano.tensor as T

import numpy as np
from random import randint

from model.encoder.rnn import RnnEnc, BiRnnEnc
from model.encoder.gru import GruEnc, BiGruEnc
from model.encoder.lstm import LstmEnc, BiLstmEnc
from model.decoder.rnn import RnnDec
from model.decoder.gru import GruDec
from model.decoder.lstm import LstmDec

from utilities.optimizers import get_optimizer
from utilities.loaddata import load_pickled_data, load_data

__author__ = 'uyaseen'


def seq2seq(dataset, vocabulary, embeddings, b_path, enc, dec,
            use_existing_model=True, optimizer='rmsprop', emb_dim=300,
            hidden_dim=1024, n_epochs=100, max_response=50, batch_size=50):
    print('seq2seq(..)')
    vocab, words_to_ix, ix_to_words = vocabulary
    eos_token = words_to_ix['EOS']
    pad_token = words_to_ix['PADDING']
    train, valid, test = load_data(dataset)
    train_set_x, train_set_y, train_set_y_mask = train
    valid_set_x, valid_set_y, valid_set_y_mask = valid
    test_set_x, test_set_y, test_set_y_mask = test
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    max_len = len(train_set_x.get_value(borrow=True)[0])

    print('... building the model')
    # allocate symbolic variables for the input & output sequences
    x = T.imatrix('x')
    y = T.imatrix('y')
    mask = T.imatrix('mask')  # only needed for decoder
    index = T.lscalar('index')
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

    all_params = encoder.params + decoder.params
    cost = decoder.negative_log_likelihood(y)
    updates = get_optimizer(identifier=optimizer, cost=cost, params=all_params)
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            mask: train_set_y_mask[index * batch_size: (index + 1) * batch_size]
        },
        updates=updates
    )
    get_cost = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            mask: valid_set_y_mask[index * batch_size: (index + 1) * batch_size]
        },
    )
    get_pred = theano.function(
        inputs=[x],
        outputs=decoder.sample()
    )
    validation_freq = 1  # check the 'error/cost' on validation set after going through these many epochs
    sampling_freq = 1
    best_train_error = np.inf
    best_valid_error = np.inf
    n_train_examples = train_set_x.get_value(borrow=True).shape[0]
    n_valid_examples = valid_set_x.get_value(borrow=True).shape[0]
    n_test_examples = test_set_x.get_value(borrow=True).shape[0]
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
        train_cost = 0.
        for i in xrange(n_train_batches):
            train_cost += train_model(i)

            print('epoch: %i/%i, minibatch: %i/%i, cost: %0.8f, /sample: %.4fm' %
                  (epoch, n_epochs, i, n_train_batches, train_cost/(i + 1),
                   (timeit.default_timer() - ep_start_time) / 60.))

        # save the best 'train' model (helpful if there is not enough validation data)
        if train_cost/n_train_batches < best_train_error:
            best_train_error = train_cost/n_train_batches
            with open(b_path + 'models/tr-best_model.pkl', 'wb') as f:
                dump_params = encoder.params, decoder.params
                pkl.dump(dump_params, f, pkl.HIGHEST_PROTOCOL)
        # sample responses from the model now and then
        if epoch % sampling_freq == 0:
            # pick a random example from the training set
            seed = randint(0, n_train_examples - 1)
            seedling = np.empty((1, max_len), dtype='int32')
            seedling[0] = train_set_x.get_value(borrow=True)[seed]
            fruit = get_pred(seedling)
            query = ' '.join(ix_to_words[ix] for ix in seedling[0]
                             if ix != pad_token)
            response = ' '.join(ix_to_words[ix] for ix in fruit)
            # remove EOS & PADDING TOKEN
            print('<sample>')
            print('query    :: %s' % query)
            print('response :: %s' % response)
            print('</sample>')

        if epoch % validation_freq == 0:
            valid_cost = 0.
            for i in xrange(n_valid_batches):
                valid_cost += get_cost(i)
            # save the current best 'validation' model
            if valid_cost/n_valid_examples < best_valid_error:
                best_valid_error = valid_cost/n_valid_examples
                with open(m_path, 'wb') as f:
                    dump_params = encoder.params, decoder.params
                    pkl.dump(dump_params, f, pkl.HIGHEST_PROTOCOL)
    print('The code ran for %.2fm' % ((timeit.default_timer() - start_time) / 60.))
    return 0

if __name__ == '__main__':
    voc, data = load_pickled_data(path='data/cornell movie-dialogs corpus/dataset.pkl')
    emb = load_pickled_data(path='data/cornell movie-dialogs corpus/w2vec.pkl')
    seq2seq(data, voc, emb, b_path='data/cornell movie-dialogs corpus/',
            enc='gru', dec='gru', use_existing_model=True,
            optimizer='rmsprop', emb_dim=300, hidden_dim=1024,
            n_epochs=30, max_response=100, batch_size=50)
    print('... done')
