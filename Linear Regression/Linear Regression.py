#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import math
import os
os.getcwd()


# In[4]:


#LMS
def LMS(data, w, b):
    #print("y:", y)
    #print("x:", x)
    #data = pd.DataFrame(data)
    #print(data)
    x = data.loc[:,:"Fine Aggr"]

    y = data.loc[:,"output"]
    
    LMS = np.sum(0.5*(y - w.dot(x.T) - b)**2)
    return LMS



def grad_wrt_w(w, x, y, b):

    #gradient
    grad_j_wrt_grad_w = -(y - w.dot(x)-b)*x
    return grad_j_wrt_grad_w

def grad_wrt_b(w, x, y, b):

    #gradient
    grad_j_wrt_grad_b = -(y - w.dot(x)-b)
    return grad_j_wrt_grad_b


# In[5]:


#predict

from sklearn.metrics import mean_squared_error
  

  

def predict(test_data_m, w, b):
    prediction_list = []
    actual_list = []

    x = test_data_m.loc[:,:"Fine Aggr"]

    y = test_data_m.loc[:,"output"]

    for i in range(0, x.shape[0]):
            #calculate gradients
            #print(i)
            x_i = np.array(x.iloc[i])
            actual_list.append(y[i])

            prediction = w.dot(x_i)+b
            prediction_list.append(prediction)
            #print("prediction:", prediction)
            #print("actual:", y[i])


    #print("prediction list:", pd.Series(prediction_list))
    #print("actual values:", y)

    # Calculation of Mean Squared Error (MSE)
    MSE = mean_squared_error(y,prediction_list)
    print("MSE:", MSE)
    return MSE


# In[ ]:





# In[6]:


train_data_m = pd.read_csv('./concrete/train.csv', names = ["Cement","Slag","Fly ash","Water","SP","Coarse Aggr","Fine Aggr", 'output'], header = None)

test_data_m = pd.read_csv('./concrete/test.csv', names = ["Cement","Slag","Fly ash","Water","SP","Coarse Aggr","Fine Aggr", 'output'], header = None)


# In[7]:


train_data_m


# In[8]:


test_data_m


# In[ ]:





# In[9]:


import random

def stochastic_grad_dec_LMS(data, w, b, lr):
    x = data.loc[:,:"Fine Aggr"]

    y = data.loc[:,"output"]
    
    #loss_dict = {}
    #loss = 0

    for i in range(0, x.shape[0]):
        #calculate gradients
        #print(i)
        
        #select random datapoint for data
        rand_i = random.randint(0, x.shape[0]-1)
        #print("rand_i:", rand_i)
        
        x_i = np.array(x.iloc[rand_i])
        grad_wrt_w_i = grad_wrt_w(w, x_i, y[rand_i], b)
        #print("grad_wrt_w_i:", grad_wrt_w_i)

        grad_wrt_b_i = grad_wrt_b(w, x_i, y[rand_i], b)
        #print("grad_wrt_b_i:", grad_wrt_b_i)

        #calculate loss
        #loss += LMS(w, x_i, y[i], b)
        
        #store loss
        #loss_dict[i] = loss

        #update weights and bias
        w = w - lr*grad_wrt_w_i

        #print("w:",w)
        b = b - lr*grad_wrt_b_i
        #print("b:",b)
    return w, b



# In[10]:


import math
def batch_grad_dec_LMS(data, w, b, lr, batch_size = 1):
    
    #dist_w_dict = {}
    #loss_dict = {}
    #loss = 0
    
    x = data.loc[:,:"Fine Aggr"]

    y = data.loc[:,"output"]
    
    grad_wrt_w_sum = 0
    grad_wrt_b_sum = 0

    for i in range(0, x.shape[0]):
        #calculate gradients
        #print(i)
        x_i = np.array(x.iloc[i])
        grad_wrt_w_i = grad_wrt_w(w, x_i, y[i], b)
        #print("grad_wrt_w_i:", grad_wrt_w_i)

        grad_wrt_b_i = grad_wrt_b(w, x_i, y[i], b)
        #print("grad_wrt_b_i:", grad_wrt_b_i)

        #add gradients
        grad_wrt_w_sum = grad_wrt_w_sum + grad_wrt_w_i
        grad_wrt_b_sum = grad_wrt_b_sum + grad_wrt_b_i
        
        
        
        #calculate loss
        #loss += LMS(w, x_i, y[i], b)
        
        #store loss
        #loss_dict[i] = loss
        
        if i%batch_size == 0:
            #update weights and bias
            #w_prev = w
            #print('w_prev:', w_prev)
            w = w - lr*grad_wrt_w_sum

            #print("w:",w)
            #b_prev = b
            b = b - lr*grad_wrt_b_sum
            #print("b:",b)
            
            #dist_w_dict[i/batch_size] = np.sqrt(np.sum((w-w_prev)**2))
            #print("Distance between W{t} and w_{-1}:", np.sqrt(np.sum((w-w_prev)**2)))
            
            
            grad_wrt_w_sum = 0
            grad_wrt_b_sum = 0
            
            
    return w, b


# In[ ]:





# In[11]:


import matplotlib.pyplot as plt

def plot_dict(dist_w):
    names = list(dist_w.keys())
    values = list(dist_w.values())

    plt.plot(names, values)
    plt.show()


# In[12]:




def train_lin_reg(train_data_m, w, b, lr, max_epochs, method, batch_size = 1):

    dist_w_dict = {}
    cost_dict = {}
    for i in range(0, max_epochs):
        #print("Iteration", i, "of", max_epochs)
        w_prev = w

        if method == batch_grad_dec_LMS:
            w, b = method(train_data_m, w, b, lr, batch_size = 1)
        if method == stochastic_grad_dec_LMS:
            w, b = method(train_data_m, w, b, lr)
        
        w_dist = np.sqrt(np.sum((w-w_prev)**2))
        
        dist_w_dict[i] = w_dist
        
        cost = LMS(train_data_m, w, b)
        
        cost_dict[i] = cost
        
        if w_dist < 1e-6:
                plot_dict(dist_w_dict)
                plot_dict(cost_dict)
                #print("cost_dict:", cost_dict)
                return w, b, i
        elif i == max_epochs-1:
                plot_dict(dist_w_dict)
                plot_dict(cost_dict)
                #print("cost_dict:", cost_dict)
                print("Didn't Converge within the max number of epoches")
                return w, b, i
        #print("distance between w:",w_dist)
        #print("w:", w)
        #print("b:", b)








# In[13]:


lr_list = [.125, .0625, .03125, 0.015625, .0078125, .00390625]

for lr in lr_list:
    print('')
    print('lr:', lr)

    w = np.array([0,0,0,0,0,0,0])

    b = 0
    max_epochs = 10000
    method = batch_grad_dec_LMS
    batch_size = train_data_m.shape[0]

    w, b, epoch_number = train_lin_reg(train_data_m, w, b, lr, max_epochs, method, batch_size )
    print("w:", w)
    print("b:", b)
    print("epoch_number:", epoch_number)
    predict(test_data_m, w, b)


# In[14]:


train_data_m.shape[0]


# In[ ]:





# In[37]:





lr_list = [.00390625, .00195312, .00097656]

for lr in lr_list:
    print('')
    print('lr:', lr)

    w = np.array([0,0,0,0,0,0,0])

    b = 0
    max_epochs = 30000
    method = stochastic_grad_dec_LMS


    w, b, epoch_number = train_lin_reg(train_data_m, w, b, lr, max_epochs, method, batch_size )
    print("w:", w)
    print("b:", b)
    print("epoch_number:", epoch_number)
    predict(test_data_m, w, b)


# In[ ]:





# In[16]:


#Part 2 - 4 c
train_data_m


# In[30]:


x = train_data_m.loc[:,:"Fine Aggr"]

y = train_data_m.loc[:,"output"]


# In[31]:


#add column of ones to x to account for bias
x["bias"] = np.ones(x.shape[0])
x


# In[32]:


a = x.T.dot(x)
a


# In[33]:


b = np.linalg.inv(a)
b


# In[34]:


c = b.dot(x.T)


# In[35]:


c.dot(y)


# In[ ]:




