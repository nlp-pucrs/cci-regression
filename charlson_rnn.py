
# coding: utf-8

# ### Basic Imports

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import time
import sys

# ### Load Data

# In[2]:


import pandas as pd

patients = pd.read_csv('internacoes_charlson_zero.csv.gz', compression='gzip', nrows=None)
target = patients['target'].values
print(patients.shape)
sys.stdout.flush()


# ### Split a Smaller Set

# In[3]:


from sklearn.model_selection import StratifiedKFold

split_kfold = StratifiedKFold(n_splits=2, shuffle=True)
for trash, used in split_kfold.split(patients.index.values, target):
    break
    
target_set = np.asarray(patients.iloc[used]['target'].values)
text_set = patients.iloc[used]['text'].values

print('Data Size:', len(used))
print('Mean Tokens:', np.mean(patients.iloc[used]['wc'].values))
sys.stdout.flush()

# ### Load Word2Vec Model

# In[4]:


from gensim.models.word2vec import KeyedVectors
w2v_model = KeyedVectors.load_word2vec_format('health_w2v_unigram_150.bin', binary=True)
print(len(w2v_model.vocab))
sys.stdout.flush()


# ### Tokenize Clinical Notes
# Remove accents and stopwords, It take a while...

# In[5]:


import unicodedata
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def tokenizer(text):
    return_text = []
    sw_port = stopwords.words("portuguese")
    for sentence in text:
        reg_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        tokens = reg_tokenizer.tokenize(sentence)
        return_text.append(' '.join([remove_accents(w.lower()) for w in tokens if w not in sw_port]))
        
    return return_text


# In[6]:


start = time.time()
tokens_set = tokenizer(text_set)
print('Takes ', round(time.time() - start), ' s for', len(used), ' instances')
sys.stdout.flush()

# ### Text Vector Representation

# In[7]:


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

start = time.time()

max_words = len(w2v_model.vocab)
tokenize = Tokenizer(num_words=max_words)
tokenize.fit_on_texts(tokens_set)

max_length = 5000
sequences = tokenize.texts_to_sequences(tokens_set)
data_matrix = pad_sequences(sequences, maxlen=max_length)

print('Takes ', round(time.time() - start), ' s for', len(used), ' instances')
sys.stdout.flush()

# In[8]:


len(data_matrix[0][data_matrix[0]!=0]), len(tokens_set[0])


# ### Setup Words Weights for Embedding Layer

# In[9]:


vocab_dim = len(w2v_model.word_vec('0'))
word_index = tokenize.word_index
n_symbols = min(max_words, len(word_index))+1

embedding_weights = np.zeros((n_symbols, vocab_dim))
for word, i in word_index.items():
    if i >= n_symbols: break
    if word in w2v_model.vocab:
        embedding_weights[i] = w2v_model.word_vec(word)

print('Symbols', n_symbols)
print('Weights', embedding_weights.shape)
sys.stdout.flush()

# ### Setup RNN Layers

# In[10]:


from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv1D, Bidirectional, Flatten, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential

conv_0 = Conv1D(50, 3, activation='relu', name='conv0')

lstm_0 = LSTM(units=50, recurrent_activation="hard_sigmoid", activation="sigmoid", name='lstm0')

bi_lstm_0 = Bidirectional(lstm_0, name='bilstm0')

embedding_layer = Embedding(embedding_weights.shape[0],
                            embedding_weights.shape[1],
                            weights=[embedding_weights],
                            input_length=max_length)


# ### Create Model, Train and Evaluate

# In[ ]:


print ('Defining a RNN Model...')
   
kfold = StratifiedKFold(n_splits=6)
cvscores = []
times = []

print('Data Shape', data_matrix.shape)
print('Weights', embedding_weights.shape)
sys.stdout.flush()
for i, (train, test) in enumerate(kfold.split(data_matrix, target_set)):
    
    if i > 2: break
    
    start = time.time()
    print('Creating model...')
    sys.stdout.flush()
    # create model
    model = Sequential()
    model.add(embedding_layer)
    
    #model.add(Conv1D(128, 5, activation='relu'))
    #model.add(MaxPooling1D(2))
    #model.add(Conv1D(128, 5, activation='relu'))
    #model.add(MaxPooling1D(5))
    
    #model.add(conv_0)
    model.add(LSTM(128, recurrent_activation="hard_sigmoid", activation="sigmoid"))
    #model.add(bi_lstm_0)
    #model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))    
    model.add(Dense(1))
    model.add(Activation('relu'))
    
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='sgd',metrics=['mae'])
    
    # Fit the model
    model.fit(data_matrix[train], target_set[train], epochs=25, batch_size=100, verbose=2)

    # evaluate the model
    scores = model.evaluate(data_matrix[test], target_set[test], verbose=0)
    print(model.metrics_names[1], scores[1])
    cvscores.append(scores[1])
    times.append(time.time() - start)
    sys.stdout.flush()

# Time / Epoch  |  (2251, 1000)
# - Flat = 2s
# - LSTM = 120s
# - BiLSTM = 120s
# - CONV = 5s

# Flatten, Data 2K = 8s
# Flatten, Data 5K = 12s

# Conv1D, Data  2K = 15s
# Conv1D, Data  5K = 26s

# RandomForestRegressor (2452, 2000) = 1.6523, 1.9503
# RandomForestRegressor (2452, 5000) = 1.5573, 4.6613

print('Mean: ', np.mean(cvscores), 'Std: ', np.std(cvscores))
print('Time: ', np.mean(times))
sys.stdout.flush()

