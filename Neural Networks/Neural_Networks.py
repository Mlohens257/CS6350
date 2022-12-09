#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import os
os.getcwd()


# In[2]:


#upload and process data


#Bank Data
train_data_m = pd.read_csv('./bank-note/train.csv', names = ["variance", "skewness", "curtosis", "entropy", "label"], header = None)

test_data_m = pd.read_csv('./bank-note/test.csv', names = ["variance", "skewness", "curtosis", "entropy", "label"], header = None)

#add label
label = "label"

#seperate train data from the label
X_train = train_data_m.iloc[:,:-1]

#Convert the label to -1 and 1
Y_train = train_data_m.iloc[:,-1:]
Y_train = Y_train + ((Y_train == 0)*-1) #convert y values to be -1 or 1 instead of 0 or 1


#seperate train data from the label
X_test = test_data_m.iloc[:,:-1]

#Convert the label to -1 and 1
Y_test = test_data_m.iloc[:,-1:]
Y_test = Y_test + ((Y_test == 0)*-1) #convert y values to be -1 or 1 instead of 0 or 1



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


def accuracy(y_i, y_true):
    return sum((np.sign(y_i) == y_true.reshape((y_i.shape[0])))*1)/y_i.shape[0]

#accuracy(y_i, y_true)


# In[11]:


def predict(x, w_1, b_1, x_0, w_2, b_2, z_01, w_3, b_3, z_02):
    z1 = np.matmul(x, w_1 + b_1*x_0)
    #print("z1:", z1)
    z1_out = sigmoid_activation(z1)
    z1_out
    #print("z1_out:", z1_out)
    z2 = np.matmul(z1_out, w_2) + b_2*z_01
    z2_out = sigmoid_activation(z2)
    z2_out
    #print("z2_out.shape:", z2_out.shape)
    #print("w_3.shape:", w_3.shape)
    #print("b_3.shape:", b_3.shape)
    #print("z_02.shape:", z_02.shape)
    #print("z2_out:", z2_out)
    #print("w_3:", w_3)
    #print("np.matmul(z2_out, w_3)",np.matmul(z2_out, w_3))
    #print("b_3*z_02",b_3*z_02)
    y_i = np.matmul(z2_out, w_3) + b_3*z_02
    return np.sign(y_i)

#y_i = predict(x, w_1, b_1, x_0, w_2, b_2, z_01, w_3, b_3, z_02)
#y_i.shape


# In[12]:


#function to shuffle data

from sklearn.utils import shuffle
import numpy as np

def Shuffle(X, y):

    X, y = shuffle(X, y)
    return X, y


# In[13]:



## sigmoid activation function using pytorch
def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))


# In[14]:


## function to calculate the derivative of activation
def sigmoid_delta(x):
  return x * (1 - x)


# # Part 2 
# ## Problem 2 (a)

# In[15]:


x = X_train
y = Y_train

lr = .1

## initialize tensor for inputs, and outputs 

x = np.array(x )


y_true = np.array(y)





x_0 = np.array(1. )
z_01 = np.array(1.)
z_02 = np.array(1.)


## initialize tensor variables for weights 
w_1 = [[0., 0, 0, 0],[0, 0, 0, 0], [0., 0, 0, 0],[0, 0, 0, 0]]
w_1 = np.array(w_1)
w_1



b_1 = np.array([0., 0, 0, 0])
b_1

w_2 = [[0., 0, 0, 0],[0, 0, 0, 0], [0., 0, 0, 0],[0, 0, 0, 0]]
w_2 = np.array(w_2)
w_2


b_2 = np.array([0., 0, 0, 0])
b_2


w_3 = [0., 0, 0, 0]
w_3 = np.array(w_3)
w_3


b_3 = np.array([0.])
b_3


i = 0
#print("x[i]:", x[i])

y_true_i = y_true[i]
#print("y_true_i:", y_true_i)

print("x:", x[i])

print("y:", y_true_i)

z1 = np.matmul(x[i], w_1) + b_1*x_0
z1_out = sigmoid_activation(z1)
z1_out

z2 = np.matmul(z1_out, w_2) + b_2*z_01
z2_out = sigmoid_activation(z2)
z2_out

y_i = np.matmul(z2_out, w_3) + b_3*z_02
y_i

loss = .5*(y_i - y_true[i] )**2
#print("loss:", loss)

dloss_wrt_y = (y_i - y_true_i )
dloss_wrt_y
#print("dloss_wrt_y:", dloss_wrt_y)


#Layer 2 back prop

dloss_wrt_b_3 = dloss_wrt_y*z_02

print("dL/db3:",dloss_wrt_b_3)



dloss_wrt_w_3 = dloss_wrt_y*z2_out

print("dL/dw3:",dloss_wrt_w_3)


dloss_wrt_z2 = dloss_wrt_y*w_3*sigmoid_delta(z2_out)


#Layer 1 back prop

dloss_wrt_b_2 = dloss_wrt_z2*z_01

print("dL/db2:",dloss_wrt_b_2)



dloss_wrt_w_2 = dloss_wrt_z2*z1_out.reshape(4,1)

print("dL/dw2:",dloss_wrt_w_2)

dloss_wrt_z1 = np.matmul(w_2, dloss_wrt_z2.T)*sigmoid_delta(z1_out)

#Layer 0 back prop

dloss_wrt_b_1 = dloss_wrt_z1*x_0

print("dL/db1:",dloss_wrt_b_1)



dloss_wrt_w_1 = dloss_wrt_z1*x[i].reshape(4,1)

print("dL/dw1:",dloss_wrt_w_1)







# # Part 2 
# ## Problem 2 (b)

# In[17]:


###Code


hidden_layer_width_list = [5, 10, 25, 50, 100]
lr_0_list = [.01, .05, .15, .25,.5]
d_list = [.1, .25,.5, 1,2,3,4,5]

#hidden_layer_width = 20
for hidden_layer_width in hidden_layer_width_list:
    print("Hidden Layer Width:", hidden_layer_width)
    
    input_width = X_train.shape[1]

    x = X_train
    y = Y_train
    
    best_acc = 0
    best_lr_0 = 0
    best_d = 0
    best_epoch = 0

    for lr_0 in lr_0_list:
        for d in d_list:
            #learning rate schedule constants

            #print("lr_0:", lr_0)
            #print("d:", d)



            ## initialize tensor for inputs, and outputs 

            x = np.array(x )


            y_true = np.array(y)

            x_0 = np.array(1. )
            z_01 = np.array(1.)
            z_02 = np.array(1.)


            ## initialize tensor variables for weights 
            w_1 = [[0., 0, 0, 0],[0, 0, 0, 0], [0., 0, 0, 0],[0, 0, 0, 0]]
            w_1 = np.zeros((input_width, hidden_layer_width))
            w_1 = np.random.normal(0, 1, size=(input_width, hidden_layer_width))
            w_1 = np.array(w_1)
            w_1



            b_1 = np.array([0., 0, 0, 0])
            b_1 = np.zeros((hidden_layer_width))
            b_1 = np.random.normal(0, 1, size=(hidden_layer_width))
            b_1

            w_2 = [[0., 0, 0, 0],[0, 0, 0, 0], [0., 0, 0, 0],[0, 0, 0, 0]]
            w_2 = np.zeros((hidden_layer_width, hidden_layer_width))
            w_2 = np.random.normal(0, 1, size=(hidden_layer_width, hidden_layer_width))
            w_2 = np.array(w_2)
            w_2


            b_2 = np.array([0., 0, 0, 0])
            b_2 = np.zeros((hidden_layer_width))
            b_2 = np.random.normal(0, 1, size=(hidden_layer_width))
            b_2


            w_3 = [0., 0, 0, 0]
            w_3 = np.zeros((hidden_layer_width))
            w_3 = np.random.normal(0, 1, size=(hidden_layer_width))
            w_3 = np.array(w_3)
            w_3


            b_3 = np.array([0.])
            b_3 = np.random.normal(0, 1, size=1)
            b_3
            



            for epoch in range(10):
              #print("epoch:", epoch)


              #shuffle the data
              x, y = Shuffle(x, y)

              #update lr
              lr = lr_0/(1 +(lr_0/d)*epoch)  


    

              for i in range(x.shape[0]):
                  #print("x[i]:", x[i])

                  y_true_i = y_true[i]
                  #print("y_true_i:", y_true_i)

                  z1 = np.matmul(x[i], w_1) + b_1*x_0
                  z1_out = sigmoid_activation(z1)
                  z1_out

                  z2 = np.matmul(z1_out, w_2) + b_2*z_01
                  z2_out = sigmoid_activation(z2)
                  z2_out

                  y_i = np.matmul(z2_out, w_3) + b_3*z_02
                  y_i

                  loss = .5*(y_i - y_true[i] )**2
                  #print("loss:", loss)

                  dloss_wrt_y = (y_i - y_true_i )
                  dloss_wrt_y
                  #print("dloss_wrt_y:", dloss_wrt_y)
                  #print("dloss_wrt_y:",torch.autograd.grad(loss, y_i, retain_graph=True))

                  #Layer 2 back prop

                  dloss_wrt_b_3 = dloss_wrt_y*z_02
                  #print("dL/db3:",torch.autograd.grad(loss, b_3, retain_graph=True))
                  #print("dL/db3:",dloss_wrt_b_3)

                  dloss_wrt_z_02 = dloss_wrt_y*b_3
                  #print("dL/dz_02:",torch.autograd.grad(loss, z_02, retain_graph=True))
                  #print("dL/dz_02:",dloss_wrt_z_02)

                  dloss_wrt_w_3 = dloss_wrt_y*z2_out
                  #print("dL/dw3:",torch.autograd.grad(loss, w_3, retain_graph=True))
                  #print("dL/dw3:",dloss_wrt_w_3)

                  #print("z2_out:", z2_out )

                  dloss_wrt_z2 = dloss_wrt_y*w_3*sigmoid_delta(z2_out)
                  #dloss_wrt_z2 = dloss_wrt_y*w_3.reshape(4)*(sigmoid_delta(z2_out))
                  #print("dL/dz2:",torch.autograd.grad(loss, z2, retain_graph=True))
                  #print("dL/dz2:",dloss_wrt_z2)

                  #Layer 1 back prop

                  dloss_wrt_b_2 = dloss_wrt_z2*z_01
                  #print("dL/db2:",torch.autograd.grad(loss, b_2, retain_graph=True))
                  #print("dL/db2:",dloss_wrt_b_2)


                  dloss_wrt_z_01 = np.matmul(dloss_wrt_z2,b_2)
                  #print("dL/dz_01:",torch.autograd.grad(loss, z_01, retain_graph=True))
                  #print("dL/dz_01:",dloss_wrt_z_01)

                  dloss_wrt_w_2 = dloss_wrt_z2*z1_out.reshape(hidden_layer_width,1)
                  #print("dL/dw2:",torch.autograd.grad(loss, w_2, retain_graph=True))
                  #print("dL/dw2:",dloss_wrt_w_2)

                  dloss_wrt_z1 = np.matmul(w_2, dloss_wrt_z2.T)*sigmoid_delta(z1_out)
                  #dloss_wrt_z1 = dloss_wrt_z2*w_2*(sigmoid_delta(z1_out))
                  #print("dL/dz1:",torch.autograd.grad(loss, z1, retain_graph=True))
                  #print("dL/dz1:",dloss_wrt_z1)

                  #Layer 0 back prop

                  dloss_wrt_b_1 = dloss_wrt_z1*x_0
                  #print("dL/db1:",torch.autograd.grad(loss, b_1, retain_graph=True))
                  #print("dL/db1:",dloss_wrt_b_1)


                  dloss_wrt_x_0 = np.matmul(dloss_wrt_z1,b_1)
                  #print("dL/dx_0:",torch.autograd.grad(loss, x_0 , retain_graph=True))
                  #print("dL/dx_0:",dloss_wrt_x_0 )

                  dloss_wrt_w_1 = dloss_wrt_z1*x[i].reshape(input_width,1)
                  #print("dL/dw1:",torch.autograd.grad(loss, w_1, retain_graph=True))
                  #print("dL/dw1:",dloss_wrt_w_1)


                  dloss_wrt_x = np.sum(dloss_wrt_z1*w_1, axis = 1)
                  #print("dL/dx:",torch.autograd.grad(loss, x, retain_graph=True))
                  #print("dL/dx:",dloss_wrt_x)



                  ## update weights 

                  w_1 = w_1 - lr*dloss_wrt_w_1
                  #print("w_1:", w_1)

                  b_1 = b_1 - lr*dloss_wrt_b_1
                  #print("b_1:", b_1)

                  w_2 = w_2 - lr*dloss_wrt_w_2
                 #print("w_2:", w_2)

                  b_2 = b_2 - lr*dloss_wrt_b_2
                  #print("b_2:", b_2)

                  w_3 = w_3 - lr*dloss_wrt_w_3
                  #print("w_3:", w_3)

                  b_3 = b_3 - lr*dloss_wrt_b_3
                  #print("b_3:", b_3)    
              x = X_train
              y_true = Y_train
              #print("X_test.shape:", x.shape)
              #print("y_true.shape:", y_true.shape)


              x = np.array(x)
              y_true = np.array(y_true)
              y_i = predict(x, w_1, b_1, x_0, w_2, b_2, z_01, w_3, b_3, z_02)
              y_i 

              #print("training accuracy:",accuracy(y_i, y_true))
              
              if best_acc < accuracy(y_i, y_true):

                  best_acc = accuracy(y_i, y_true)
                  best_epoch = epoch
                  best_lr_0 = lr_0
                  best_d = d
                
                


                  y_i = predict(np.array(X_test), w_1, b_1, x_0, w_2, b_2, z_01, w_3, b_3, z_02)
                  y_i 
                  best_test_acc = accuracy(y_i, np.array(Y_test))


                    
    print("best training accuracy is:", best_acc )
    print("best testing accuracy  is:", best_test_acc )
    print("best epoch is:", best_epoch )
    print("best lr_0 is:", best_lr_0 )
    print("best d is:", best_d )
    print(" " )


# # Part 2 
# ## Problem 2 (c)

# In[18]:


###Code


hidden_layer_width_list = [5, 10, 25, 50, 100]
lr_0_list = [.01, .05, .15, .25,.5]
d_list = [.1, .25,.5, 1,2,3,4,5]

#hidden_layer_width = 20
for hidden_layer_width in hidden_layer_width_list:
    print("Hidden Layer Width:", hidden_layer_width)
    
    input_width = X_train.shape[1]

    x = X_train
    y = Y_train
    
    best_acc = 0
    best_lr_0 = 0
    best_d = 0
    best_epoch = 0

    for lr_0 in lr_0_list:
        for d in d_list:
            #learning rate schedule constants

            #print("lr_0:", lr_0)
            #print("d:", d)



            ## initialize tensor for inputs, and outputs 

            x = np.array(x )


            y_true = np.array(y)

            x_0 = np.array(1. )
            z_01 = np.array(1.)
            z_02 = np.array(1.)


            ## initialize tensor variables for weights 
            w_1 = [[0., 0, 0, 0],[0, 0, 0, 0], [0., 0, 0, 0],[0, 0, 0, 0]]
            w_1 = np.zeros((input_width, hidden_layer_width))
            w_1 = np.array(w_1)
            w_1



            b_1 = np.array([0., 0, 0, 0])
            b_1 = np.zeros((hidden_layer_width))
            b_1

            w_2 = [[0., 0, 0, 0],[0, 0, 0, 0], [0., 0, 0, 0],[0, 0, 0, 0]]
            w_2 = np.zeros((hidden_layer_width, hidden_layer_width))
            w_2 = np.array(w_2)
            w_2


            b_2 = np.array([0., 0, 0, 0])
            b_2 = np.zeros((hidden_layer_width))
            b_2


            w_3 = [0., 0, 0, 0]
            w_3 = np.zeros((hidden_layer_width))
            w_3 = np.array(w_3)
            w_3


            b_3 = np.array([0.])
            b_3
            



            for epoch in range(10):
              #print("epoch:", epoch)


              #shuffle the data
              x, y = Shuffle(x, y)

              #update lr
              lr = lr_0/(1 +(lr_0/d)*epoch)  


    

              for i in range(x.shape[0]):
                  #print("x[i]:", x[i])

                  y_true_i = y_true[i]
                  #print("y_true_i:", y_true_i)

                  z1 = np.matmul(x[i], w_1) + b_1*x_0
                  z1_out = sigmoid_activation(z1)
                  z1_out

                  z2 = np.matmul(z1_out, w_2) + b_2*z_01
                  z2_out = sigmoid_activation(z2)
                  z2_out

                  y_i = np.matmul(z2_out, w_3) + b_3*z_02
                  y_i

                  loss = .5*(y_i - y_true[i] )**2
                  #print("loss:", loss)

                  dloss_wrt_y = (y_i - y_true_i )
                  dloss_wrt_y
                  #print("dloss_wrt_y:", dloss_wrt_y)
                  #print("dloss_wrt_y:",torch.autograd.grad(loss, y_i, retain_graph=True))

                  #Layer 2 back prop

                  dloss_wrt_b_3 = dloss_wrt_y*z_02
                  #print("dL/db3:",torch.autograd.grad(loss, b_3, retain_graph=True))
                  #print("dL/db3:",dloss_wrt_b_3)

                  dloss_wrt_z_02 = dloss_wrt_y*b_3
                  #print("dL/dz_02:",torch.autograd.grad(loss, z_02, retain_graph=True))
                  #print("dL/dz_02:",dloss_wrt_z_02)

                  dloss_wrt_w_3 = dloss_wrt_y*z2_out
                  #print("dL/dw3:",torch.autograd.grad(loss, w_3, retain_graph=True))
                  #print("dL/dw3:",dloss_wrt_w_3)

                  #print("z2_out:", z2_out )

                  dloss_wrt_z2 = dloss_wrt_y*w_3*sigmoid_delta(z2_out)
                  #dloss_wrt_z2 = dloss_wrt_y*w_3.reshape(4)*(sigmoid_delta(z2_out))
                  #print("dL/dz2:",torch.autograd.grad(loss, z2, retain_graph=True))
                  #print("dL/dz2:",dloss_wrt_z2)

                  #Layer 1 back prop

                  dloss_wrt_b_2 = dloss_wrt_z2*z_01
                  #print("dL/db2:",torch.autograd.grad(loss, b_2, retain_graph=True))
                  #print("dL/db2:",dloss_wrt_b_2)


                  dloss_wrt_z_01 = np.matmul(dloss_wrt_z2,b_2)
                  #print("dL/dz_01:",torch.autograd.grad(loss, z_01, retain_graph=True))
                  #print("dL/dz_01:",dloss_wrt_z_01)

                  dloss_wrt_w_2 = dloss_wrt_z2*z1_out.reshape(hidden_layer_width,1)
                  #print("dL/dw2:",torch.autograd.grad(loss, w_2, retain_graph=True))
                  #print("dL/dw2:",dloss_wrt_w_2)

                  dloss_wrt_z1 = np.matmul(w_2, dloss_wrt_z2.T)*sigmoid_delta(z1_out)
                  #dloss_wrt_z1 = dloss_wrt_z2*w_2*(sigmoid_delta(z1_out))
                  #print("dL/dz1:",torch.autograd.grad(loss, z1, retain_graph=True))
                  #print("dL/dz1:",dloss_wrt_z1)

                  #Layer 0 back prop

                  dloss_wrt_b_1 = dloss_wrt_z1*x_0
                  #print("dL/db1:",torch.autograd.grad(loss, b_1, retain_graph=True))
                  #print("dL/db1:",dloss_wrt_b_1)


                  dloss_wrt_x_0 = np.matmul(dloss_wrt_z1,b_1)
                  #print("dL/dx_0:",torch.autograd.grad(loss, x_0 , retain_graph=True))
                  #print("dL/dx_0:",dloss_wrt_x_0 )

                  dloss_wrt_w_1 = dloss_wrt_z1*x[i].reshape(input_width,1)
                  #print("dL/dw1:",torch.autograd.grad(loss, w_1, retain_graph=True))
                  #print("dL/dw1:",dloss_wrt_w_1)


                  dloss_wrt_x = np.sum(dloss_wrt_z1*w_1, axis = 1)
                  #print("dL/dx:",torch.autograd.grad(loss, x, retain_graph=True))
                  #print("dL/dx:",dloss_wrt_x)



                  ## update weights 

                  w_1 = w_1 - lr*dloss_wrt_w_1
                  #print("w_1:", w_1)

                  b_1 = b_1 - lr*dloss_wrt_b_1
                  #print("b_1:", b_1)

                  w_2 = w_2 - lr*dloss_wrt_w_2
                 #print("w_2:", w_2)

                  b_2 = b_2 - lr*dloss_wrt_b_2
                  #print("b_2:", b_2)

                  w_3 = w_3 - lr*dloss_wrt_w_3
                  #print("w_3:", w_3)

                  b_3 = b_3 - lr*dloss_wrt_b_3
                  #print("b_3:", b_3)    
              x = X_train
              y_true = Y_train
              #print("X_test.shape:", x.shape)
              #print("y_true.shape:", y_true.shape)


              x = np.array(x)
              y_true = np.array(y_true)
              y_i = predict(x, w_1, b_1, x_0, w_2, b_2, z_01, w_3, b_3, z_02)
              y_i 

              #print("training accuracy:",accuracy(y_i, y_true))
              
              if best_acc < accuracy(y_i, y_true):

                  best_acc = accuracy(y_i, y_true)
                  best_epoch = epoch
                  best_lr_0 = lr_0
                  best_d = d
                
                


                  y_i = predict(np.array(X_test), w_1, b_1, x_0, w_2, b_2, z_01, w_3, b_3, z_02)
                  y_i 
                  best_test_acc = accuracy(y_i, np.array(Y_test))


                    
    print("best training accuracy is:", best_acc )
    print("best testing accuracy  is:", best_test_acc )
    print("best epoch is:", best_epoch )
    print("best lr_0 is:", best_lr_0 )
    print("best d is:", best_d )
    print(" " )


# In[ ]:





# In[ ]:





# In[ ]:




