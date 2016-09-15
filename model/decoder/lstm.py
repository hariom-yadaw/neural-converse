import theano
import theano.tensor as T
import theano.scan_module

import numpy as np

from utilities.initializations import get


__author__ = 'uyaseen'


class LstmDec(object):
    def __init__(self, enc_h, mask, emb_mat, vocab_size, emb_dim, hidden_dim, eos_token,
                 batch_size, max_len, init='uniform', inner_init='orthonormal',
                 inner_activation=T.nnet.hard_sigmoid, activation=T.tanh,
                 params=None, max_response=100):
        self.enc_h = enc_h
        self.mask = mask
        self.eos_token = eos_token
        self.batch_size = batch_size
        self.inner_activation = inner_activation
        self.activation = activation
        self.max_response = max_response
        if params is None:
            self.emb = theano.shared(value=np.asarray(emb_mat, dtype=theano.config.floatX),
                                     name='emb', borrow=True)
            # input gate
            self.W_i = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_i', borrow=True)
            self.U_i = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_i', borrow=True)
            self.b_i = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                     name='b_i', borrow=True)
            # forget gate
            self.W_f = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_f', borrow=True)
            self.U_f = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_f', borrow=True)
            self.b_f = theano.shared(value=get(identifier='one', shape=(hidden_dim, )),
                                     name='b_f', borrow=True)
            # memory
            self.W_c = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_c', borrow=True)
            self.U_c = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_c', borrow=True)
            self.b_c = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                     name='b_c', borrow=True)
            # output gate
            self.W_o = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_o', borrow=True)
            self.U_o = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_o', borrow=True)
            self.b_o = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                     name='b_o', borrow=True)
            # weights pertaining to output neuron
            self.V = theano.shared(value=get(identifier=init, shape=(hidden_dim, vocab_size)),
                                   name='V', borrow=True)
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
            self.emb, self.W_i, self.U_i, self.b_i, self.W_f, self.U_f, self.b_f, \
                self.W_c, self.U_c, self.b_c, self.W_o, self.U_o, self.b_o, \
                self.V, self.by, self.c_h, self.c_y, self.y_t1 = params

        self.params = [self.emb,
                       self.W_i, self.U_i, self.b_i,
                       self.W_f, self.U_f, self.b_f,
                       self.W_c, self.U_c, self.b_c,
                       self.W_o, self.U_o, self.b_o,
                       self.V, self.by,
                       self.c_h, self.c_y,
                       self.y_t1]

        self.c0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='c0', borrow=True)
        self.h0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='h0', borrow=True)
        # y(t-1) from encoder will always be 'eos' token
        self.y0 = theano.shared(value=np.asarray(np.full((batch_size, ),
                                                         self.eos_token), dtype='int32'), name='y0', borrow=True)

        # remember for decoder both h_t and y_t are conditioned on 'enc_h' & 'y_t-1'.
        def recurrence(msk, c_tm_prev, h_tm_prev, y_tm_prev):
            x_i = T.dot(self.emb[y_tm_prev], self.W_i) + self.b_i
            x_f = T.dot(self.emb[y_tm_prev], self.W_f) + self.b_f
            x_c = T.dot(self.emb[y_tm_prev], self.W_c) + self.b_c
            x_o = T.dot(self.emb[y_tm_prev], self.W_o) + T.dot(self.enc_h, self.c_h) + self.b_o

            i_t = self.inner_activation(x_i + T.dot(h_tm_prev, self.U_i))
            f_t = self.inner_activation(x_f + T.dot(h_tm_prev, self.U_f))
            c_t = f_t * c_tm_prev + i_t * self.activation(x_c + T.dot(h_tm_prev, self.U_c))  # internal memory
            o_t = self.inner_activation(x_o + T.dot(h_tm_prev, self.U_o))
            h_t = o_t * self.activation(c_t)  # actual hidden state

            # needed to back-propagate errors
            y_d_t = T.dot(h_t, self.V) + T.dot(self.enc_h, self.c_y) + T.dot(self.emb[y_tm_prev], self.y_t1) + self.by
            # ignore padded tokens
            y_d_t = T.batched_dot(y_d_t, msk)
            y_d = T.clip(T.nnet.softmax(y_d_t),
                         0.0001, 0.9999)
            y_t = T.argmax(y_d, axis=1)
            return c_t, h_t, y_d, T.cast(y_t.flatten(), 'int32')

        [_, _, y_dist, y], _ = theano.scan(
            fn=recurrence,
            sequences=mask.dimshuffle(1, 0),  # ugly, but we have to go till the end
            outputs_info=[T.alloc(self.c0, self.enc_h.shape[0], hidden_dim),
                          T.alloc(self.h0, self.enc_h.shape[0], hidden_dim),
                          None,
                          T.alloc(self.y0, self.enc_h.shape[0])],
            n_steps=max_len
        )

        self.y = y.dimshuffle(1, 0)
        self.y_dist = y_dist.dimshuffle(1, 0, 2)

    def negative_log_likelihood(self, y):

        def compute_cost(y_dist, target):
            return T.nnet.categorical_crossentropy(y_dist, target)

        batched_cost, _ = theano.scan(
            fn=compute_cost,
            sequences=[self.y_dist, y],
            outputs_info=None
        )

        return T.mean(T.sum(batched_cost, axis=1))  # naive normalization

    '''
    Since for generating responses from the decoder at 'test time', we do not know the sequence length;
    We will sample till 'eos' token or max_responses, and therefore, need a different scan loop.
    '''
    def sample(self):

        y0 = theano.shared(value=self.eos_token)

        def step(c_tm_prev, h_tm_prev, y_tm_prev):
            x_i = T.dot(self.emb[y_tm_prev], self.W_i) + self.b_i
            x_f = T.dot(self.emb[y_tm_prev], self.W_f) + self.b_f
            x_c = T.dot(self.emb[y_tm_prev], self.W_c) + self.b_c
            x_o = T.dot(self.emb[y_tm_prev], self.W_o) + T.dot(self.enc_h[-1], self.c_h) + self.b_o

            i_t = self.inner_activation(x_i + T.dot(h_tm_prev, self.U_i))
            f_t = self.inner_activation(x_f + T.dot(h_tm_prev, self.U_f))
            c_t = f_t * c_tm_prev + i_t * self.activation(x_c + T.dot(h_tm_prev, self.U_c))  # internal memory
            o_t = self.inner_activation(x_o + T.dot(h_tm_prev, self.U_o))
            h_t = o_t * self.activation(c_t)  # actual hidden state

            # needed to back-propagate errors
            y_d = T.nnet.softmax(T.dot(h_t, self.V) +
                                 T.dot(self.enc_h[-1], self.c_y) +
                                 T.dot(self.emb[y_tm_prev], self.y_t1) +
                                 self.by)[0]
            y_t = T.argmax(y_d)
            return (c_t, h_t, y_t), theano.scan_module.until(T.eq(y_t, self.eos_token))

        [_, _, y], _ = theano.scan(
            fn=step,
            outputs_info=[self.c0, self.h0, y0],
            n_steps=self.max_response
        )

        return y
