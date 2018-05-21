
# coding: utf-8

# In[2]:


import gensim


# In[3]:


import pandas as pd
import numpy as np


# In[24]:


def word2vec_gen(fileName):
    data = pd.read_csv(fileName,names=['Comment','Label'])
    arr = data['Comment'].values
    tokenarr = [l.split() for l in arr]
    model = gensim.models.word2vec.Word2Vec(tokenarr,min_count=2)
    model.save(fname_or_handle='./word2vec_model')
    model.wv.save_word2vec_format('./word2vec_vec')
    print('Finished model generation')


# In[25]:


def get_embedding(fileName='./word2vec_vec'):
    embed_dict = dict()
    with open(fileName) as f:
        f.readline()
        for l in f:
            l = l.split()
            key = l[0]
            vec = np.array(l[1:],dtype='float32')
            embed_dict[key] = vec
    return embed_dict


# In[27]:


if __name__ == '__main__':
    word2vec_gen('./processed.csv')

