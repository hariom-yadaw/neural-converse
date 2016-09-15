import os.path
import theano
import numpy as np
import cPickle as pkl


__author__ = 'uyaseen'


def shared_data(data_xy, borrow=True):
    data_x, _, data_y, data_y_mask = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype='int32'),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype='int32'),
                             borrow=borrow)
    shared_y_mask = theano.shared(np.asarray(data_y_mask,
                                             dtype='int32'))

    return shared_x, shared_y, shared_y_mask


def load_data(dataset):
    train_set, valid_set, test_set = dataset

    print('... transferring data to the %s' % theano.config.device)
    train_set_x, train_set_y, train_set_y_mask = shared_data(train_set)
    valid_set_x, valid_set_y, valid_set_y_mask = shared_data(valid_set)
    test_set_x, test_set_y, test_set_y_mask = shared_data(test_set)

    return [[train_set_x, train_set_y, train_set_y_mask],
            [valid_set_x, valid_set_y, valid_set_y_mask],
            [test_set_x, test_set_y, test_set_y_mask]]


def load_pickled_data(path):
    assert os.path.isfile(path), True
    curr_dir = os.getcwd()
    with open(path, 'rb') as f:
        dump = pkl.load(f)
    os.chdir(curr_dir)
    return dump
