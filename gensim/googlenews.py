from gensim.models.keyedvectors import KeyedVectors
word_vectors=KeyedVectors.load_word2vec_format('/home/freeman/Downloads/GoogleNews-vectors-negative300.bin',binary=True)
word_vectors['human']
print(word_vectors.similarity('man', 'women'))
