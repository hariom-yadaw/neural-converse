import os.path
import cPickle as pkl

import codecs
import ast

from utils import shuffle_four, shuffle_list, tokenize, remove_symbols

__author__ = 'uyaseen'


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
    mask_x = []
    mask_y = []
    eos_token = 'EOS'
    unknown_token = 'UNKNOWN_TOKEN'
    pad_token = 'PADDING'
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
    words_to_ix = {wd: i+1 for i, wd in enumerate(vocab)}
    ix_to_words = {i+1: wd for i, wd in enumerate(vocab)}
    # enforce 'PADDING' index to be 0
    words_to_ix.update({pad_token: 0})
    ix_to_words.update({0: pad_token})
    vocab.add(pad_token)
    print('... vocabulary size: %i' % len(vocab))

    max_len = -1
    for converse in conversation_mapping:
        x = tokenize(remove_symbols(movie_lines[converse[0]]) + ' ' + eos_token)
        y = tokenize(remove_symbols(movie_lines[converse[1]]) + ' ' + eos_token)
        x = [words_to_ix[wd] if wd in vocab else words_to_ix[unknown_token]
             for wd in x]
        y = [words_to_ix[wd] if wd in vocab else words_to_ix[unknown_token]
             for wd in y]
        dialogue_x.append(x)
        dialogue_y.append(y)
        max_len = len(x) if len(x) > max_len else max_len
        max_len = len(y) if len(y) > max_len else max_len

    # (left) pad the sequences
    pad_idx = words_to_ix[pad_token]
    assert pad_idx == 0
    N = len(dialogue_x)
    for idx in xrange(N):
        mask_x.append([0] * (max_len - len(dialogue_x[idx])) + [1] * len(dialogue_x[idx]))
        dialogue_x[idx] = [pad_idx] * (max_len - len(dialogue_x[idx])) + dialogue_x[idx]
        mask_y.append([0] * (max_len - len(dialogue_y[idx])) + [1] * len(dialogue_y[idx]))
        dialogue_y[idx] = [pad_idx] * (max_len - len(dialogue_y[idx])) + dialogue_y[idx]

    # split data into train, validation & test set
    dialogue_x, mask_x, dialogue_y, mask_y = shuffle_four(dialogue_x, mask_x, dialogue_y, mask_y)
    # test/train split
    te_idx = int(len(dialogue_x) * te_split)
    test_x = dialogue_x[0: te_idx]
    test_x_mask = mask_x[0: te_idx]
    test_y = dialogue_y[0: te_idx]
    test_y_mask = mask_y[0: te_idx]
    train_x = dialogue_x[te_idx:]
    train_x_mask = mask_x[te_idx:]
    train_y = dialogue_y[te_idx:]
    train_y_mask = mask_y[te_idx:]
    # validation split
    va_idx = int(len(train_x) * va_split)
    train_x, train_x_mask, train_y, train_y_mask = shuffle_four(train_x, train_x_mask, train_y, train_y_mask)
    valid_x = train_x[0: va_idx]
    valid_x_mask = train_x_mask[0: va_idx]
    valid_y = train_y[0: va_idx]
    valid_y_mask = train_y_mask[0: va_idx]
    del train_x[0: va_idx]
    del train_x_mask[0: va_idx]
    del train_y[0: va_idx]
    del train_y_mask[0: va_idx]
    print('... max_len: %d' % max_len)
    print('... creating persistence storage')
    vocabulary = [vocab, words_to_ix, ix_to_words]
    data = [(train_x, train_x_mask, train_y, train_y_mask),
            (valid_x, valid_x_mask, valid_y, valid_y_mask),
            (test_x, test_x_mask, test_y, test_y_mask)]
    dump = [vocabulary, data]
    w_path = b_path + 'dataset.pkl'
    with open(w_path, 'wb') as f:
        pkl.dump(dump, f, pkl.HIGHEST_PROTOCOL)
    print('... %s created' % w_path)
    os.chdir(curr_dir)
