"""This module implements word vectors and their similarity look-ups.

Since trained word vectors are independent from the way they were trained (:class:`~gensim.models.word2vec.Word2Vec`,
:class:`~gensim.models.fasttext.FastText`, :class:`~gensim.models.wrappers.wordrank.WordRank`,
:class:`~gensim.models.wrappers.varembed.VarEmbed` etc), they can be represented by a standalone structure,
as implemented in this module.

The structure is called "KeyedVectors" and is essentially a mapping between *entities*
and *vectors*. Each entity is identified by its string id, so this is a mapping between {str => 1D numpy array}.

The entity typically corresponds to a word (so the mapping maps words to 1D vectors),
but for some models, they key can also correspond to a document, a graph node etc. To generalize
over different use-cases, this module calls the keys **entities**. Each entity is
always represented by its string id, no matter whether the entity is a word, a document or a graph node.

Why use KeyedVectors instead of a full model?
=============================================

+---------------------------+--------------+------------+-------------------------------------------------------------+
|        capability         | KeyedVectors | full model |                               note                          |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| continue training vectors | ❌           | ✅         | You need the full model to train or update vectors.         |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| smaller objects           | ✅           | ❌         | KeyedVectors are smaller and need less RAM, because they    |
|                           |              |            | don't need to store the model state that enables training.  |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| save/load from native     |              |            | Vectors exported by the Facebook and Google tools           |
| fasttext/word2vec format  | ✅           | ❌         | do not support further training, but you can still load     |
|                           |              |            | them into KeyedVectors.                                     |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| append new vectors        | ✅           | ✅         | Add new entity-vector entries to the mapping dynamically.   |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| concurrency               | ✅           | ✅         | Thread-safe, allows concurrent vector queries.              |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| shared RAM                | ✅           | ✅         | Multiple processes can re-use the same data, keeping only   |
|                           |              |            | a single copy in RAM using                                  |
|                           |              |            | `mmap <https://en.wikipedia.org/wiki/Mmap>`_.               |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| fast load                 | ✅           | ✅         | Supports `mmap <https://en.wikipedia.org/wiki/Mmap>`_       |
|                           |              |            | to load data from disk instantaneously.                     |
+---------------------------+--------------+------------+-------------------------------------------------------------+

TL;DR: the main difference is that KeyedVectors do not support further training.
On the other hand, by shedding the internal data structures necessary for training, KeyedVectors offer a smaller RAM
footprint and a simpler interface.

How to obtain word vectors?
===========================

Train a full model, then access its `model.wv` property, which holds the standalone keyed vectors.
For example, using the Word2Vec algorithm to train the vectors

>>> from gensim.test.utils import common_texts
>>> from gensim.models import Word2Vec
>>>
>>> model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
>>> word_vectors = model.wv

Persist the word vectors to disk with

>>> from gensim.test.utils import get_tmpfile
>>> from gensim.models import KeyedVectors
>>>
>>> fname = get_tmpfile("vectors.kv")
>>> word_vectors.save(fname)
>>> word_vectors = KeyedVectors.load(fname, mmap='r')

The vectors can also be instantiated from an existing file on disk
in the original Google's word2vec C format as a KeyedVectors instance

>>> from gensim.test.utils import datapath
>>>
>>> wv_from_text = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)  # C text format
>>> wv_from_bin = KeyedVectors.load_word2vec_format(datapath("euclidean_vectors.bin"), binary=True)  # C binary format

What can I do with word vectors?
================================

You can perform various syntactic/semantic NLP word tasks with the trained vectors.
Some of them are already built-in

>>> import gensim.downloader as api
>>>
>>> word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
>>>
>>> result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
>>> print("{}: {:.4f}".format(*result[0]))
queen: 0.7699
>>>
>>> result = word_vectors.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
>>> print("{}: {:.4f}".format(*result[0]))
queen: 0.8965
>>>
>>> print(word_vectors.doesnt_match("breakfast cereal dinner lunch".split()))
cereal
>>>
>>> similarity = word_vectors.similarity('woman', 'man')
>>> similarity > 0.8
True
>>>
>>> result = word_vectors.similar_by_word("cat")
>>> print("{}: {:.4f}".format(*result[0]))
dog: 0.8798
>>>
>>> sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
>>> sentence_president = 'The president greets the press in Chicago'.lower().split()
>>>
>>> similarity = word_vectors.wmdistance(sentence_obama, sentence_president)
>>> print("{:.4f}".format(similarity))
3.4893
>>>
>>> distance = word_vectors.distance("media", "media")
>>> print("{:.1f}".format(distance))
0.0
>>>
>>> sim = word_vectors.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
>>> print("{:.4f}".format(sim))
0.7067
>>>
>>> vector = word_vectors['computer']  # numpy vector of a word
>>> vector.shape
(100,)
>>>
>>> vector = word_vectors.wv.word_vec('office', use_norm=True)
>>> vector.shape
(100,)

Correlation with human opinion on word similarity

>>> from gensim.test.utils import datapath
>>>
>>> similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))

And on word analogies

>>> analogy_scores = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))

and so on.
"""
