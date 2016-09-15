# neural-converse
Theano implementation of the paper [A Neural Conversational Model](http://arxiv.org/abs/1506.05869) by Oriol Vinyals and Quoc V. Le, which is based on [Sequence to Sequence Learning with Neural Networks](http://arxiv.org/abs/1409.3215) by Sutskever et al. (2014). Encoder can be RNN, GRU, LSTM, BiRNN, BiGRU and BiLSTM. Decoder can be RNN, GRU and LSTM.
![seq2seq](https://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s1600/2TFstaticgraphic_alt-01.png)  
Source: http://googleresearch.blogspot.ca/2015/11/computer-respond-to-this-email.htm

## Dependencies
* Python 2.7
* [Theano](http://deeplearning.net/software/theano/)
* [Numpy](http://www.numpy.org/)
* [Gensim](https://radimrehurek.com/gensim/index.html)

## Dataset
* [Cornell Movie Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

## Usage

### Generating Data
Download [word2vec](https://code.google.com/archive/p/word2vec/) word embeddings from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) and copy the file `GoogleNews-vectors-negative300.bin.gz` in `data` directory i.e `neural-converse/data/GoogleNews-vectors-negative300.bin.gz`.

```bash
python data.py
```
Note that `python data.py` will randomly sample fixed number of conversations (specified by `dataset_size` parameter of `pickle_cornell(..)` in `utilities/cornell_move_dialogs.py`), therefore, each run might change vocabulary size, and your old trained model might not work. So along with trained model `data/cornell movie-dialogs_corpus/models/*-best_model.pkl`, also do take back-up of `data/cornell movie-dialogs_corpus/dataset.pkl` and `data/cornell movie-dialogs_corpus/w2vec.pkl`.

### Training
```bash
python seq2seq.py
```

#### Sanity Check
How appropriate are responses on the test data ?
```bash
python eval.py
```

### Chat
```bash
python chat.py
```

### Note
If you are interested in minimal code, then browse to an [older version](https://github.com/uyaseen/neural-converse/tree/9e6d8dd81d8e16315df5903b86759b77bf3df169) of this repository.
