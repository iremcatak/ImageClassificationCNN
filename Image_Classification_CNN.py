#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
X_train.shape


# In[4]:


X_test.shape


# In[39]:


y_train.shape


# In[7]:


y_train[:10]


# In[8]:


y_train = y_train.reshape(-1,)
y_train[:10]


# In[9]:


y_test = y_test.reshape(-1,)


# In[12]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[13]:


def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[42]:


classes[8]


# In[15]:


plot_sample(X_train, y_train, 8)


# In[16]:


plot_sample(X_train, y_train, 4)


# In[17]:


plot_sample(X_train, y_train, 6)


# In[20]:


plot_sample(X_train, y_train, 5)


# In[43]:


X_train[0]


# In[21]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[22]:


ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=10)


# In[23]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[24]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[25]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[26]:


cnn.fit(X_train, y_train, epochs=10)


# In[27]:


cnn.evaluate(X_test,y_test)


# In[28]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[31]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:10]


# In[32]:


y_test[:10]


# In[34]:


plot_sample(X_test, y_test,21)


# In[36]:


plot_sample(X_test, y_test,3156)


# In[37]:


classes[y_classes[21]]


# In[38]:


classes[y_classes[3156]]


# In[ ]:




