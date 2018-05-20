
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import json


# In[28]:


filePath = '../../labeled_document2.json'


# In[29]:


with open(filePath) as f:
    data = json.load(f)


# In[38]:


negativeComment = None
positiveComment = None
for i in range(len(data['Comment'])):
    if data['CommentLabel'][i] == 0:
        negativeComment = np.array([[data['Comment'][i],0]]) if negativeComment is None else np.insert(negativeComment,negativeComment.shape[0],[data['Comment'][i],0],0)
    else:
        positiveComment = np.array([[data['Comment'][i],1]]) if positiveComment is None else np.insert(positiveComment,positiveComment.shape[0],[data['Comment'][i],1],0)


# In[45]:


indexes = np.random.choice(positiveComment.shape[0],negativeComment.shape[0],replace=False)
concatComment = np.concatenate((positiveComment[indexes],negativeComment),axis=0)
dataframe = pd.DataFrame(concatComment)
dataframe.to_csv('./labeled_comments.csv',header=False,index=False)

