
import gensim.downloader as api

word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data

result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))
#queen: 0.7699

result = word_vectors.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))
#queen: 0.8965

print(word_vectors.doesnt_match("breakfast cereal dinner lunch".split()))
#cereal

similarity = word_vectors.similarity('woman', 'man')
similarity > 0.8
#True

result = word_vectors.similar_by_word("cat")
print("{}: {:.4f}".format(*result[0]))
#dog: 0.8798

sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
sentence_president = 'The president greets the press in Chicago'.lower().split()

similarity = word_vectors.wmdistance(sentence_obama, sentence_president)
print("{:.4f}".format(similarity))
#3.4893

distance = word_vectors.distance("media", "media")
print("{:.1f}".format(distance))
#0.0

sim = word_vectors.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
print("{:.4f}".format(sim))
#0.7067

vector = word_vectors['computer']  # numpy vector of a word
vector.shape
#(100,)

vector = word_vectors.wv.word_vec('office', use_norm=True)
vector.shape
#(100,)

#Correlation with human opinion on word similarity--------------------

from gensim.test.utils import datapath

similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))

#And on word analogies

analogy_scores = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
