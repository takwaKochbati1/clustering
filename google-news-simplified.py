import gensim.models as gm

'''
======================= Read me =============================
 This code snippet is used to simplify the word2vec model in
 order to speed up Gensim word2vec model load time.
 
 1. 1) Download pre-trained word2vec model,
    GoogleNews-vectors-negative300.bin.gz,
    from https://code.google.com/archive/p/word2vec/
    2) Or use a word2vec model trained with other corpus by yourself.
 2. Use this code snippet to simplify it.
=============================================================
'''

#model = gm.KeyedVectors.load_word2vec_format('C:/Users/TK257812/.spyder-py3/GoogleNews-vectors-negative300.bin', binary=True)
#model.init_sims(replace=True)  # remove syn1, replace syn0
#model.save('C:/Users/TK257812/.spyder-py3/GoogleNews-simplified')  # stores only syn0, not syn0norm
model = gm.KeyedVectors.load('C:/Users/TK257812/.spyder-py3/GoogleNews-simplified',mmap='r')  # mmap the large matrix as read-only
model.syn0norm = model.syn0  # no need to call init_sims

print('Check whether the simplified model works or not:')
print(model.similarity("woman", "man"))
print(model.similarity("homme", "man"))
print(model.similarity("configure", "configuration"))
print(model.most_similar('appointment'))
#print(model.similarity("word", "world"))
#print(model.similarity("malware", "antimalware"))
#print(model.similarity("open", "close"))
