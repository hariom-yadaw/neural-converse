import theano
import theano.tensor as T
import theano.scan_module

import numpy as np

from utilities.initializations import get


__author__ = 'uyaseen'


class RnnDec(object):
    def __init__(self, enc_h, seq_len, emb_mat, vocab_size, emb_dim, hidden_dim, eos_token,
                 init='uniform', inner_init='orthonormal', activation=T.tanh, params=None,
                 max_response=100):
        self.enc_h = enc_h
        self.eos_token = eos_token
        self.activation = activation
        self.max_response = max_response
        if params is None:
            self.emb = theano.shared(value=np.asarray(emb_mat, dtype=theano.config.floatX),
                                     name='emb', borrow=True)
            self.W = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                   name='W', borrow=True)
            self.U = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                   name='U', borrow=True)
            self.V = theano.shared(value=get(identifier=init, shape=(hidden_dim, vocab_size)),
                                   name='V', borrow=True)
            self.bh = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                    name='bh', borrow=True)
            self.by = theano.shared(value=get(identifier='zero', shape=(vocab_size,)),
                                    name='by', borrow=True)
            # to weight 'context' from encoder
            self.c_h = theano.shared(value=get(identifier=init, shape=(hidden_dim, hidden_dim)),
                                     name='c_h', borrow=True)
            self.c_y = theano.shared(value=get(identifier=init, shape=(hidden_dim, vocab_size)),
                                     name='c_y', borrow=True)
            # to weight 'y_t-1' for decoder's 'y'
            self.y_t1 = theano.shared(value=get(identifier=init, shape=(emb_dim, vocab_size)),
                                      name='y_t1', borrow=True)
        else:
            self.emb, self.W, self.U, self.V, self.bh, self.by, self.c_h, self.c_y, self.y_t1 = params

        self.params = [self.emb,
                       self.W, self.U, self.V,
                       self.bh, self.by,
                       self.c_h, self.c_y,
                       self.y_t1]

        self.h0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='h0', borrow=True)
        # y(t-1) from encoder will always be 'eos' token
        self.y0 = theano.shared(value=self.eos_token, name='y0', borrow=True)

        # remember for decoder both h_t and y_t are conditioned on 'enc_h' & 'y_t-1'.
        def recurrence(h_tm_prev, y_tm_prev):
            h_t = self.activation(T.dot(self.emb[y_tm_prev], self.W) +
                                  T.dot(h_tm_prev, self.U) +
                                  T.dot(self.enc_h, self.c_h) + self.bh)
            # needed to back-propagate errors
            y_d = T.nnet.softmax(T.dot(h_t, self.V) +
                                 T.dot(self.enc_h, self.c_y) +
                                 T.dot(self.emb[y_tm_prev], self.y_t1) +
                                 self.by)[0]
            y_t = T.argmax(y_d)
            return h_t, y_d, y_t

        [_, y_dist, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None, self.y0],
            n_steps=seq_len
        )

        self.y = y
        self.y_dist = y_dist

    def negative_log_likelihood(self, y):
        return T.sum(T.nnet.categorical_crossentropy(self.y_dist, y))

    '''
    Since for generating responses from the decoder at 'test time', we do not know the sequence length;
    We will sample till 'eos' token or max_responses, and therefore, need a different scan loop.
    '''
    def sample(self):

        def step(h_tm_prev, y_tm_prev):
            h_t = self.activation(T.dot(self.emb[y_tm_prev], self.W) +
                                  T.dot(h_tm_prev, self.U) +
                                  T.dot(self.enc_h, self.c_h) + self.bh)
            # needed to back-propagate errors
            y_dist = T.nnet.softmax(T.dot(h_t, self.V) +
                                    T.dot(self.enc_h, self.c_y) +
                                    T.dot(self.emb[y_tm_prev], self.y_t1) +
                                    self.by)[0]
            y_t = T.argmax(y_dist)
            return (h_t, y_t), theano.scan_module.until(T.eq(y_t, self.eos_token))

        [_, y], _ = theano.scan(
            fn=step,
            outputs_info=[self.h0, self.y0],
            n_steps=self.max_response
        )

        return y
