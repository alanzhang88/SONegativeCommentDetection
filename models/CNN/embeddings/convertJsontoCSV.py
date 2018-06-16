
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


import pandas as pd


# In[4]:


import json
import sys


# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-fn", dest="filePath", type=str, required=True)
# args = parser.parse_args()

# In[5]:


# filePath = '../../labeled_document2.json'


# In[6]:

with open(sys.argv[1]) as f:
    data = json.load(f)

with open(sys.argv[2]) as f:
    data_iter2 = json.load(f)


# In[10]:


negativeComment = None
positiveComment = None
for i in range(len(data['Comment'])):
    if data['CommentLabel'][i] == 0:
        negativeComment = np.array([[data['Comment'][i],0]]) if negativeComment is None else np.append(negativeComment,[[data['Comment'][i],0]],0)
    else:
        positiveComment = np.array([[data['Comment'][i],1]]) if positiveComment is None else np.append(positiveComment,[[data['Comment'][i],1]],0)


for i in range(len(data_iter2['Comment'])):
    if data_iter2['CommentLabel'][i] == 0:
        negativeComment = np.append(negativeComment,[[data_iter2['Comment'][i],0]],0)
    else:
        positiveComment = np.append(positiveComment,[[data_iter2['Comment'][i],1]],0)


# In[11]:
indexes = np.random.choice(positiveComment.shape[0],negativeComment.shape[0],replace=False)

concatComment = np.concatenate((positiveComment[indexes],negativeComment),axis=0)
# concatComment = np.concatenate((positiveComment,negativeComment),axis=0)

# print (len(positiveComment) + len(negativeComment))
dataframe = pd.DataFrame(concatComment)
dataframe.to_csv('./labeled_comments.csv',header=False,index=False)
