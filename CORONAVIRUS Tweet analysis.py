#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re 
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[8]:


import os
os.chdir('C:\\Users\\win10')


# In[9]:


d=pd.read_csv('Corona_NLP_test.csv')


# # Reading the Data

# In[10]:


d.head()


# In[11]:


d.shape


# # Checking for missing Values

# In[12]:


d.isnull().sum()


# In[13]:


d.head()


# In[14]:





# In[15]:


from sklearn.model_selection import train_test_split


# # Data Cleaning

# In[19]:


def Tokenizer(string):
    words=nltk.word_tokenize(string)
    return ' '.join(words)

def Removestopwords(string):
    for i in punctuation:
        string=string.replace(i,'')
    words=nltk.word_tokenize(string)
    eng_stop=stopwords.words('english')
    k=[]
    for each in words:
        if each not in eng_stop:
            k.append(each.lower())
    return ' '.join(k)

def Lammetization(string):
    words=nltk.word_tokenize(string)
    ws=WordNetLemmatizer()
    l=[]
    for each in words:
        l.append(ws.lemmatize(each))
    return ' '.join(l)


# In[20]:


def Refine(string):
    return Lammetization(Removestopwords(Tokenizer(string)))


# In[21]:


d.head()


# In[22]:


d['Processed']=d['OriginalTweet'].apply(lambda x: Refine(x))


# In[23]:


d.head()


# In[59]:


x=d['Processed']
y=d['Sentiment']


# # Spliting the data

# In[25]:


from sklearn.preprocessing import LabelEncoder


# In[26]:


enc=LabelEncoder()


# In[60]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)


# In[61]:


y_train=enc.fit_transform(y_train)


# In[62]:


y_test=enc.transform(y_test)


# In[63]:


y_train


# In[33]:


from keras.models import Sequential
from keras.layers import Embedding,Dense,LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[64]:


tokens=Tokenizer(num_words=50000)


# In[65]:


tokens.fit_on_texts(x_train)


# In[66]:


x_train=tokens.texts_to_sequences(x_train)


# In[67]:


x_train=pad_sequences(x_train,maxlen=25)


# In[68]:


x_train


# In[69]:


x_test=tokens.texts_to_sequences(x_test)
x_test=pad_sequences(x_test,maxlen=25)


# In[70]:


x_test


# In[71]:


from keras.utils import to_categorical


# In[72]:


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


# In[73]:


y_train


# # Using glove word2vect

# In[78]:


con=open('glove.6B.50d.txt',encoding='utf-8')
import numpy as np
matrix=np.zeros((50000,50))


# In[79]:


index={}
for line in con:
    values=line.split()
    word=values[0]
    vect=np.asarray(values[1:],dtype='float32')
    index[word]=vect


# In[80]:


for word,i in tokens.word_index.items():
    if i <50000:
        vect=index.get(word)
        if vect is not None:
            matrix[i]=vect


# In[74]:


from keras.layers import Flatten


# In[85]:


from keras.layers import Dropout


# In[86]:


from keras.layers import SimpleRNN


# # Building SimpleRnn Model

# In[89]:


model2=Sequential()
model2.add(Embedding(input_dim=50000,output_dim=50,input_length=25,weights=[matrix]))
model2.add(SimpleRNN(150,return_sequences=True))
model2.add(SimpleRNN(150))
model2.add(Dense(5,activation='softmax'))
model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[90]:


model2.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test),batch_size=32)


# # Final Results

# In[91]:


results=pd.DataFrame(model2.history.history)


# In[93]:


results.plot()


# In[ ]:




