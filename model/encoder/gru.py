import theano
import theano.tensor as T

import numpy as np

from utilities.initializations import get


__author__ = 'uyaseen'


class GruEnc(object):
    def __init__(self, input, emb_mat, emb_dim, hidden_dim, init='uniform', inner_init='orthonormal',
                 inner_activation=T.nnet.hard_sigmoid, activation=T.tanh,
                 params=None):
        if params is None:
            self.emb = theano.shared(value=np.asarray(emb_mat, dtype=theano.config.floatX),
                                     name='emb', borrow=True)
            # update gate
            self.W_z = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_z', borrow=True)
            self.U_z = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_z', borrow=True)
            self.b_z = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                     name='b_z', borrow=True)
            # reset gate
            self.W_r = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_r', borrow=True)
            self.U_r = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_r', borrow=True)
            self.b_r = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                     name='b_r', borrow=True)
            # hidden state
            self.W_h = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_h', borrow=True)
            self.U_h = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_h', borrow=True)
            self.b_h = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                     name='b_h', borrow=True)
        else:
            self.emb, self.W_z, self.U_z, self.b_z, self.W_r, self.U_r, self.b_r, \
                self.W_h, self.U_h, self.b_h = params

        self.h0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='h0', borrow=True)
        self.params = [self.emb,
                       self.W_z, self.U_z, self.b_z,
                       self.W_r, self.U_r, self.b_r,
                       self.W_h, self.U_h, self.b_h]

        def recurrence(x_t, h_tm_prev):
            x_z = T.dot(self.emb[x_t], self.W_z) + self.b_z
            x_r = T.dot(self.emb[x_t], self.W_r) + self.b_r
            x_h = T.dot(self.emb[x_t], self.W_h) + self.b_h

            z_t = inner_activation(x_z + T.dot(h_tm_prev, self.U_z))
            r_t = inner_activation(x_r + T.dot(h_tm_prev, self.U_r))
            hh_t = activation(x_h + T.dot(r_t * h_tm_prev, self.U_h))
            h_t = (T.ones_like(z_t) - z_t) * hh_t + z_t * h_tm_prev

            return h_t

        h, _ = theano.scan(
            fn=recurrence,
            sequences=input,
            outputs_info=self.h0
        )

        # 'hidden state + prediction' at last time-step need to be passed to the decoder;
        # prediction at last-time step will always be 'eos' therefore, ignored
        self.h = h[-1]


class BiGruEnc(object):
    def __init__(self, input, emb_mat, emb_dim, hidden_dim, init='uniform', inner_init='orthonormal',
                 inner_activation=T.nnet.hard_sigmoid, activation=T.tanh,
                 params=None, merge_mode='sum'):
        if params is None:
            self.emb = theano.shared(value=np.asarray(emb_mat, dtype=theano.config.floatX),
                                     name='emb', borrow=True)
            # Forward GRU
            # update gate
            self.Wf_z = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wf_z', borrow=True)
            self.Uf_z = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Uf_z', borrow=True)
            self.bf_z = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                      name='bf_z', borrow=True)
            # reset gate
            self.Wf_r = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wf_r', borrow=True)
            self.Uf_r = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Uf_r', borrow=True)
            self.bf_r = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                      name='bf_r', borrow=True)
            # hidden state
            self.Wf_h = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wf_h', borrow=True)
            self.Uf_h = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Uf_h', borrow=True)
            self.bf_h = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                      name='bf_h', borrow=True)

            # Backward GRU
            # update gate
            self.Wb_z = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wb_z', borrow=True)
            self.Ub_z = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Ub_z', borrow=True)
            self.bb_z = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                      name='bb_z', borrow=True)
            # reset gate
            self.Wb_r = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wb_r', borrow=True)
            self.Ub_r = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Ub_r', borrow=True)
            self.bb_r = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                      name='bb_r', borrow=True)
            # hidden state
            self.Wb_h = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wb_h', borrow=True)
            self.Ub_h = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Ub_h', borrow=True)
            self.bb_h = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                      name='bb_h', borrow=True)

        else:
            self.emb, self.Wf_z, self.Uf_z, self.bf_z, self.Wf_r, self.Uf_r, self.bf_r, \
                self.Wf_h, self.Uf_h, self.bf_h, self.Wb_z, self.Ub_z, self.bb_z, self.Wb_r, \
                self.Ub_r, self.bb_r, self.Wb_h, self.Ub_h, self.bb_h = params

        self.hf = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='hf', borrow=True)
        self.hb = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='hb', borrow=True)
        self.params = [self.emb,
                       self.Wf_z, self.Uf_z, self.bf_z,
                       self.Wf_r, self.Uf_r, self.bf_r,
                       self.Wf_h, self.Uf_h, self.bf_h,

                       self.Wb_z, self.Ub_z, self.bb_z,
                       self.Wb_r, self.Ub_r, self.bb_r,
                       self.Wb_h, self.Ub_h, self.bb_h]

        input_f = input
        input_b = input[::-1]

        # forward gru
        def recurrence_f(xf_t, hf_tm):
            xf_z = T.dot(self.emb[xf_t], self.Wf_z) + self.bf_z
            xf_r = T.dot(self.emb[xf_t], self.Wf_r) + self.bf_r
            xf_h = T.dot(self.emb[xf_t], self.Wf_h) + self.bf_h

            zf_t = inner_activation(xf_z + T.dot(hf_tm, self.Uf_z))
            rf_t = inner_activation(xf_r + T.dot(hf_tm, self.Uf_r))
            hhf_t = activation(xf_h + T.dot(rf_t * hf_tm, self.Uf_h))
            hf_t = (T.ones_like(zf_t) - zf_t) * hhf_t + zf_t * hf_tm

            return hf_t

        h_f, _ = theano.scan(
            fn=recurrence_f,
            sequences=input_f,
            outputs_info=self.hf
        )

        # backward gru
        def recurrence_b(xb_t, hb_tm):
            xb_z = T.dot(self.emb[xb_t], self.Wb_z) + self.bb_z
            xb_r = T.dot(self.emb[xb_t], self.Wb_r) + self.bb_r
            xb_h = T.dot(self.emb[xb_t], self.Wb_h) + self.bb_h

            zb_t = inner_activation(xb_z + T.dot(hb_tm, self.Ub_z))
            rb_t = inner_activation(xb_r + T.dot(hb_tm, self.Ub_r))
            hhb_t = activation(xb_h + T.dot(rb_t * hb_tm, self.Ub_h))
            hb_t = (T.ones_like(zb_t) - zb_t) * hhb_t + zb_t * hb_tm

            return hb_t

        h_b, _ = theano.scan(
            fn=recurrence_b,
            sequences=input_b,
            outputs_info=self.hb
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
            print('Supported "merge_mode" for forward + backward gru are: "sum", "multiply", "average" & "concat".')
            raise NotImplementedError

        # 'hidden state + prediction' at last time-step need to be passed to the decoder;
        # prediction at last-time step will always be 'eos' therefore, ignored
        self.h = h
