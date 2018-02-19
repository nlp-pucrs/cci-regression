
# coding: utf-8

# In[8]:


import numpy as np
import gzip, pickle
import pandas as pd
import sys

patients = pd.read_csv('internacoes_charlson_zero.csv.gz', compression='gzip', nrows=None, usecols=['target'])
target = np.asarray(patients['target'].values)
print(patients.shape)
sys.stdout.flush()

# In[9]:


#with gzip.open("data_100_lda.npy.gz", "rb") as wfp:   #Pickling
#    lda = pickle.load(wfp)
#    wfp.close()
    
with gzip.open("data_10k_multigram.npy.gz", "rb") as wfp:   #Pickling
    grams = pickle.load(wfp)
    wfp.close()
    
data = grams[:,:5000].todense()
print(data.shape, target.shape)
sys.stdout.flush()

# In[10]:


import time
import warnings
warnings.filterwarnings('ignore')


# In[11]:


from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv1D, Bidirectional, Flatten, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential

conv_0 = Conv1D(50, 3, activation='relu', name='conv0')

lstm_0 = LSTM(units=50, recurrent_activation="hard_sigmoid", activation="relu", name='lstm0')

bi_lstm_0 = Bidirectional(lstm_0, name='bilstm0')

embedding_layer = Embedding(output_dim=128,
                    input_dim=5000,
                    input_length=5000)


# In[13]:


from sklearn.model_selection import StratifiedKFold

split_kfold = StratifiedKFold(n_splits=2)
for trash, used in split_kfold.split(data, target):
    break

print ('Defining a RNN Model...')
sys.stdout.flush()

kfold = StratifiedKFold(n_splits=6)
cvscores = []
times = []

target1 = target[used]
data1 = data[used]

print('Data Shape', data1.shape)
sys.stdout.flush()

for i, (train, test) in enumerate(kfold.split(data1, target1)):
    
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
    
    #model.add(Conv1D(50, 3, activation='relu'))
    model.add(LSTM(128, recurrent_activation="hard_sigmoid", activation="relu"))
    #model.add(bi_lstm_0)
    #model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))    
    model.add(Dense(1))
    model.add(Activation('relu'))
    
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='sgd',metrics=['mae'])
    
    # Fit the model
    model.fit(data1[train], target1[train], epochs=20, batch_size=100, verbose=2)

    # evaluate the model
    scores = model.evaluate(data1[test], target1[test], verbose=0)
    print(model.metrics_names[1], scores[1])
    cvscores.append(scores[1])
    times.append(time.time() - start)
    sys.stdout.flush()

# Data Shape (4898, 5000)
# batch 100 , epochs 10 , Flatten = 1.63
# batch 100 , epochs 10 , Conv1D = 1.90
# batch 100 , epochs 10 , LSTM = 
    
# Shape: (2454, 2000)
# 'KNeighborsRegressor': '2.2277, 2.2231', 
# 'GradientBoostingRegressor': '1.6322, 37.7982'
# 'RandomForestRegressor': '1.6893, 3.9377'
    
print('Mean: ', np.mean(cvscores), 'Std: ', np.std(cvscores))
print('Time: ', np.mean(times))

