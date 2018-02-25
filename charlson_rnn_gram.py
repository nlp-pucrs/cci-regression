# ### Load Pickles

# In[1]:


from sklearn.model_selection import StratifiedKFold
import time, sys, numpy as np
import warnings
warnings.filterwarnings('ignore')
import gzip, pickle

with gzip.open("data_5k_unigram.npy.gz", "rb") as d_wfp:   #Pickling
    grams = pickle.load(d_wfp)
    d_wfp.close()
data_matrix = grams.todense()
    
with gzip.open("target_set.pkl.gz", "rb") as t_wfp:   #Pickling
    target_set = pickle.load(t_wfp)
    t_wfp.close()
    
print(data_matrix.shape, target_set.shape)
sys.stdout.flush()

# ### Setup RNN Layers

# In[3]:


from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv1D, Bidirectional, Flatten, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential

conv_0 = Conv1D(50, 3, activation='relu', name='conv0')

lstm_0 = LSTM(units=50, recurrent_activation="hard_sigmoid", activation="sigmoid", name='lstm0')

bi_lstm_0 = Bidirectional(lstm_0, name='bilstm0')

max_length = data_matrix.shape[1]
embedding_layer = Embedding(output_dim=128,
                    input_dim=max_length,
                    input_length=max_length)


# ### Create Model, Train and Evaluate

# In[5]:


print ('Defining a RNN Model...')
sys.stdout.flush()

kfold = StratifiedKFold(n_splits=10)
cvscores = []
times = []

print('Data Shape', data_matrix.shape)
sys.stdout.flush()

for i, (train, test) in enumerate(kfold.split(data_matrix, target_set)):
    
    #if i > 0: break
    
    start = time.time()
    print('Creating model... #',i)
    sys.stdout.flush()
    # create model
    model = Sequential()
    model.add(embedding_layer)
    
    #model.add(Conv1D(128, 5, activation='relu'))
    #model.add(MaxPooling1D(2))
    #model.add(Conv1D(128, 5, activation='relu'))
    #model.add(MaxPooling1D(5))
    
    #model.add(SimpleRNN(100))
    #model.add(Conv1D(50, 3, activation='relu'))
    #model.add(lstm_0)
    #model.add(bi_lstm_0)
    model.add(Flatten())
    
    model.add(Dense(128, activation='linear'))
    model.add(Dropout(0.5))    
    model.add(Dense(1))
    model.add(Activation('linear'))
    # softmax, elu , selu, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, linear
    # 2.67   , 2.40, 2.36, 2.33    , 2.78    , 2.79, 2.68, 2.74   , 2.73        , 2.16
    
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='sgd',metrics=['mae'])
    
    # Fit the model
    model.fit(data_matrix[train], target_set[train], epochs=20, batch_size=100, verbose=2)

    # evaluate the model
    scores = model.evaluate(data_matrix[test], target_set[test], verbose=0)
    print(model.metrics_names[1], scores[1])
    sys.stdout.flush()
    cvscores.append(scores[1])
    times.append(time.time() - start)

#Shape: (4871, 5000)
# batch 10  , epochs 10 , Dense(100) = 2.26
# batch 100 , epochs 10 , Dense(100) = 1.57
# batch 100 , epochs 20 , Dense(100) = 1.59
# batch 100 , epochs 10 , Conv1+Conv1+Dense = 1.44
# batch 100 , epochs 10 , SimpleRNN+Dense = ?

#Shape: (3271, 2000)
#{'RandomForestRegressor': '1.5543, 2.3715'}
    
print('Mean: ', np.mean(cvscores), 'Std: ', np.std(cvscores))
print('Time: ', np.mean(times))
sys.stdout.flush()

