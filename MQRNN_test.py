#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/tianchen101/MQRNN/blob/master/MQRNN_test.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[15]:


#get_ipython().system('rm -rf MQRNN/')
#get_ipython().system('git clone  https://github.com/tianchen101/MQRNN.git')


# In[16]:


import Encoder
import Decoder
from MQRNN import MQRNN 
from data import MQRNN_dataset,read_df


# In[17]:


config = {
    'horizon_size':40,
    'hidden_size':50,
    'quantiles': [0.8], 
    'columns': [1],
    'dropout': 0.3,
    'layer_size':2,
    'by_direction':False,
    'lr': 1e-3,
    'batch_size': 1,
    'num_epochs':500,
    'context_size': 10,
}


# In[18]:


import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[19]:


train_target_df, test_target_df, train_covariate_df, test_covariate_df = read_df(config)
train_target_df = train_target_df.iloc[-500:,:]
train_covariate_df = train_covariate_df.iloc[-500:,:]


# In[20]:


train_covariate_df


# In[21]:


from data import MQRNN_dataset
dset = MQRNN_dataset(train_target_df,train_covariate_df,40,1 )


# In[22]:


config['covariate_size'] = train_covariate_df.shape[1]
config['device'] = device
config 


# In[23]:


import torch 
horizon_size = config['horizon_size']
hidden_size = config['hidden_size']
quantiles = config['quantiles']
quantile_size = len(quantiles)
columns = config['columns']
dropout = config['dropout']
layer_size = config['layer_size']
by_direction = config['by_direction']
lr = config['lr']
batch_size= config['batch_size']
num_epochs = config['num_epochs']
context_size = config['context_size']
covariate_size = config['covariate_size']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[24]:


net = MQRNN(horizon_size,hidden_size,quantiles,columns,dropout,layer_size,by_direction,lr,batch_size,num_epochs,context_size,covariate_size,device)


# In[25]:


train_dataset = MQRNN_dataset(train_target_df,train_covariate_df,horizon_size,quantile_size)


# In[26]:


net.train(train_dataset)


# In[28]:


predict_result = net.predict(train_target_df,train_covariate_df,test_covariate_df,1 )
predict_result 


# In[ ]:


test_target_df


# In[31]:


import matplotlib.pyplot as plt
plt.plot(predict_result[0.8], color = 'r', label='prediction')
plt.plot(test_target_df[1].to_list(), color= 'b', label='real')
plt.legend()


# In[ ]:




