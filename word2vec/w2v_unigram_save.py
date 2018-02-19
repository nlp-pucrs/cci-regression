import pickle
import sys

chunks = int(sys.argv[1])
size = int(sys.argv[2])
#max_vocab = int(sys.argv[3])

sentences_load = []
cont = 0
with open("sentences.txt", "rb") as rfp:   # Unpickling
    while 1:
        try:
            sentences_load.extend(pickle.load(rfp))
            print(cont,',')
            cont += 1
            if cont >= chunks:
                break
        except EOFError:
            rfp.close()
            break
        
print(len(sentences_load))

filename = 'health_w2v_unigram_' + str(size) 
import gensim
model = gensim.models.Word2Vec(sentences_load, size=size, workers=32, iter=10, negative=5, min_count=10)
model.init_sims(replace=True)
model.save(filename + '.mdl')
model.wv.save_word2vec_format(filename + '.bin', binary=True)

w2v = dict(zip(model.wv.index2word, model.wv.syn0))
print ("Number of tokens in Word2Vec:", len(w2v.keys()))
print(model.most_similar_cosmul('dpoc',topn=10))
