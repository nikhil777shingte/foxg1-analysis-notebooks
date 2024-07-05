#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import loadmat

mat = loadmat('/Users/nikhil/Documents/eeg-mne-analysis/foxg1-eeg/eeg-24h-2-data.mat', struct_as_record=True)


# In[2]:


mat


# In[3]:
    
print(mat.keys())

# In[5]:


print(type(mat['d']))
print(mat['d'].shape)