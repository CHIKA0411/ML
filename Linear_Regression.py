#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv("C:\\Users\\Asus\\OneDrive\\Desktop\\COURSE\\machine_learning\\Housing.csv")
dataset.head()


# In[3]:


dataset.info()


# In[4]:


dataset['bedrooms'].value_counts()


# In[5]:


x_train=dataset['area']
y_train=dataset['price']
print(f"Area ={x_train}")
print(f"price ={y_train}")


# In[7]:


print("x_train shape:",x_train.shape)
m=x_train.shape[0]
print("no.of training example ",m)


# In[8]:


m=len(x_train)
print("no.of training example ",m)


# In[23]:


from sklearn.preprocessing import LabelEncoder


# In[24]:


i=0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")


# In[27]:


# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()


# In[34]:


w = 1000
b = 1000
print(f"w: {w}")
print(f"b: {b}")


# In[35]:


def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


# In[36]:


tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()


# In[ ]:




