import os.path
import cPickle as pkl

import codecs
import ast

from utils import shuffle_pair, shuffle_list, tokenize, remove_symbols

__author__ = 'uyaseen'


def load_data(path):
    assert os.path.isfile(path), True
    curr_dir = os.getcwd()
    with open(path, 'rb') as f:
        dump = pkl.load(f)
    os.chdir(curr_dir)
    return dump


# 'dataset_size' can be used to adjust the '#' of conversations; (helpful for training large datasets)
def pickle_cornell(b_path='data/cornell movie-dialogs corpus/', mv_lines='movie_lines.txt',
                   mv_converse='movie_conversations.txt', dataset_size=10000,
                   va_split=0.1, te_split=0.5):
    assert os.path.isfile(b_path + mv_lines), True
    assert os.path.isfile(b_path + mv_converse), True
    print('... parsing "Cornell Movie--Dialogs Corpus" ...')
    pattern = ' +++$+++ '
    vocab = set()
    movie_lines = {}
    movie_conversations = []  # contain all the conversations
    conversation_mapping = []  # only two conversations (this is 'x', 'y' pair)
    dialogue_x = []
    dialogue_y = []
    eos_token = 'EOS'
    unknown_token = 'UNKNOWN_TOKEN'
    curr_dir = os.getcwd()
    print('... -> %s' % mv_lines)
    with codecs.open(b_path + mv_lines, 'r', encoding='us-ascii', errors='ignore') as f:
        lines = [line.strip('\n') for line in f.readlines()]
    for line in lines:
        l = line.split(pattern)
        movie_lines[l[0]] = l[-1]

    print('... -> %s' % mv_converse)
    with codecs.open(b_path + mv_converse, 'r', encoding='us-ascii', errors='ignore') as f:
        lines = [line.strip('\n') for line in f.readlines()]
    for line in lines:
        l = line.split(pattern)
        movie_conversations.append(l[-1])
    print('... mapping conversations')
    for converse in movie_conversations:
        converse = ast.literal_eval(converse)
        for c in xrange(0, len(converse)-1):
            conversation_mapping.append((converse[c], converse[c+1]))
    # build vocabulary from a subset of conversation_mapping
    shuffle_list(conversation_mapping)
    conversation_mapping = conversation_mapping[0: dataset_size]
    print('... building vocabulary')
    for converse in conversation_mapping:
        for query in tokenize(remove_symbols(movie_lines[converse[0]])):
            vocab.add(query)
        for response in tokenize(remove_symbols(movie_lines[converse[1]])):
            vocab.add(response)
    vocab.add(unknown_token)
    vocab.add(eos_token)
    words_to_ix = {wd: i for i, wd in enumerate(vocab)}
    ix_to_words = {i: wd for i, wd in enumerate(vocab)}
    print('... vocabulary size: %i' % len(vocab))

    for converse in conversation_mapping:
        x = tokenize(remove_symbols(movie_lines[converse[0]]) + ' ' + eos_token)
        y = tokenize(remove_symbols(movie_lines[converse[1]]) + ' ' + eos_token)
        x = [words_to_ix[wd] if wd in vocab else words_to_ix[unknown_token]
             for wd in x]
        y = [words_to_ix[wd] if wd in vocab else words_to_ix[unknown_token]
             for wd in y]
        dialogue_x.append(x)
        dialogue_y.append(y)

    # split data into train, validation & test set
    dialogue_x, dialogue_y = shuffle_pair(dialogue_x, dialogue_y)
    # test/train split
    te_idx = int(len(dialogue_x) * te_split)
    test_x = dialogue_x[0: te_idx]
    test_y = dialogue_y[0: te_idx]
    train_x = dialogue_x[te_idx:]
    train_y = dialogue_y[te_idx:]
    # validation split
    va_idx = int(len(train_x) * va_split)
    train_x, train_y = shuffle_pair(train_x, train_y)
    valid_x = train_x[0: va_idx]
    valid_y = train_y[0: va_idx]
    del train_x[0: va_idx]
    del train_y[0: va_idx]

    print('... creating persistence storage')
    vocabulary = [vocab, words_to_ix, ix_to_words]
    data = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    dump = [vocabulary, data]
    w_path = b_path + 'dataset.pkl'
    with open(w_path, 'wb') as f:
        pkl.dump(dump, f, pkl.HIGHEST_PROTOCOL)
    print('... %s created' % w_path)
    os.chdir(curr_dir)
