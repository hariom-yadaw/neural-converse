import theano
import theano.tensor as T

import numpy as np

from utilities.initializations import get


__author__ = 'uyaseen'


class RnnEnc(object):
    def __init__(self, input, emb_mat, emb_dim, hidden_dim, init='uniform', inner_init='orthonormal',
                 activation=T.tanh, params=None):
        input = input.dimshuffle(1, 0)
        if params is None:
            self.emb = theano.shared(value=np.asarray(emb_mat, dtype=theano.config.floatX),
                                     name='emb', borrow=True)
            self.W = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                   name='W', borrow=True)
            self.U = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                   name='U', borrow=True)
            self.bh = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                    name='bh', borrow=True)
        else:
            self.emb, self.W, self.U, self.bh = params

        self.h0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='h0', borrow=True)
        self.params = [self.emb,
                       self.W, self.U,
                       self.bh]

        def recurrence(x_t, h_tm_prev):
            h_t = activation(T.dot(self.emb[x_t], self.W) +
                             T.dot(h_tm_prev, self.U) + self.bh)
            return h_t

        h, _ = theano.scan(
            fn=recurrence,
            sequences=input,
            outputs_info=T.alloc(self.h0, input.shape[1], hidden_dim)
        )

        # 'hidden state + prediction' at last time-step need to be passed to the decoder;
        # prediction at last-time step will always be 'eos' therefore, ignored
        self.h = h[-1]


class BiRnnEnc(object):
    def __init__(self, input, emb_mat, emb_dim, hidden_dim, init='uniform', inner_init='orthonormal',
                 activation=T.tanh, params=None, merge_mode='sum'):
        if params is None:
            self.emb = theano.shared(value=np.asarray(emb_mat, dtype=theano.config.floatX),
                                     name='emb', borrow=True)
            # weights for forward rnn
            self.W_f = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_f', borrow=True)
            self.U_f = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_f', borrow=True)
            self.b_f = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                     name='b_f', borrow=True)
            # weights for backward rnn
            self.W_b = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_b', borrow=True)
            self.U_b = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_b', borrow=True)
            self.b_b = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                     name='b_b', borrow=True)

        else:
            self.emb, self.W_f, self.U_f, self.b_f, self.W_b, self.U_b, self.b_b = params

        self.hf = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='hf', borrow=True)
        self.hb = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='hb', borrow=True)
        self.params = [self.emb,
                       self.W_f, self.U_f, self.b_f,
                       self.W_b, self.U_b, self.b_b]

        input_f = input.dimshuffle(1, 0)
        input_b = input[::-1].dimshuffle(1, 0)

        # forward rnn
        def recurrence_f(xf_t, hf_tm):
            hf_t = activation(T.dot(self.emb[xf_t], self.W_f) +
                              T.dot(hf_tm, self.U_f) + self.b_f)
            return hf_t

        h_f, _ = theano.scan(
            fn=recurrence_f,
            sequences=input_f,
            outputs_info=T.alloc(self.hf, input_f.shape[1], hidden_dim)
        )

        # backward rnn
        def recurrence_b(xb_t, hb_tm):
            hf_b = activation(T.dot(self.emb[xb_t], self.W_b) +
                              T.dot(hb_tm, self.U_b) + self.b_b)
            return hf_b

        h_b, _ = theano.scan(
            fn=recurrence_b,
            sequences=input_b,
            outputs_info=T.alloc(self.hb, input_b.shape[1], hidden_dim)
        )

        if merge_mode == 'sum':
            h = h_f[-1] + h_b[-1]
        elif merge_mode == 'multiply':
            h = h_f[-1] * h_b[-1]
        elif merge_mode == 'average':
            h = (h_f[-1] + h_b[-1]) / 2
        elif merge_mode == 'concat':
            h = T.concatenate([h_f, h_b])
        else:
            print('Supported "merge_mode" for forward + backward rnn are: "sum", "multiply", "average" & "concat".')
            raise NotImplementedError

        # 'hidden state + prediction' at last time-step need to be passed to the decoder;
        # prediction at last-time step will always be 'eos' therefore, ignored
        self.h = h
