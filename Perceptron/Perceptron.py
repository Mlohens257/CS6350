#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import math
import os
os.getcwd()


# ### Homework 3

#  

# In[3]:


#preprocess data so that its binarized
def numer_to_binary(train_data, label):

    attribute_list = list(train_data.columns) #list of attributes
    attribute_list.remove(label) #remove the label column since we don't want to alter that




    for attribute in attribute_list: #iterate through the list
        #print(attribute)
        #print(type(train_data[attribute][0]))
        if type(train_data[attribute][0]) != type(''): #if values for an attribute is not a string type
            #print(np.median(train_data[attribute]))
            train_data[attribute] = (train_data[attribute] >= np.median(train_data[attribute]))*1 #binarize based of the median value
            #print((train_data[attribute] >= np.median(train_data[attribute]))*1)
        
    return train_data


# In[4]:


#process data to replace unknowns with the most comon label for a particular attribute
        
import statistics
from statistics import mode

def replace_unknowns(train_data, label, unknown_label = "unknown"):
    print(label)
    attribute_list = list(train_data.columns) #list of attributes
    attribute_list.remove(label) #remove the label column since we don't want to alter that




    for attribute in attribute_list: #iterate through the list
        bool_array =(train_data[attribute] == unknown_label)
        bool_index = bool_array[bool_array].index.values
        if bool_index != []:
            if unknown_label == mode(train_data[attribute]):
                bool_array_2 = (train_data_m[attribute] != unknown_label)
                bool_index_2 = bool_array_2[bool_array_2].index.values #gets index for values that are not unknown


                train_data[attribute][bool_index] = mode(train_data_m[attribute][bool_index_2])
            train_data[attribute][bool_index] = mode(train_data[attribute])
            print(attribute)
        #print((train_data[attribute] == "unknown"))
        #print(bool_index[bool_index].index.values)
        #if type(train_data[attribute][0]) != type(''): #if values for an attribute is not a string type
            #print(np.median(train_data[attribute]))
            #train_data[attribute] = (train_data[attribute] >= np.median(train_data[attribute]))*1 #binarize based of the median value
            #print((train_data[attribute] >= np.median(train_data[attribute]))*1)
        
    return train_data


# In[ ]:





# In[5]:


#function to shuffle data

from sklearn.utils import shuffle
import numpy as np

def Shuffle(X, y):

    X, y = shuffle(X, y)
    return X, y


# In[6]:


#Initialize weights - weights size should be the size of the 
#number of features plus 1 for the bias

def init_weights(train_data_m):
    num_features = train_data_m.shape[1] + 1 
    weights = np.zeros(num_features)
    return weights


# In[ ]:





# In[7]:


#function to predict outputs based on weights and input data
#outputs a series object of 1s and -1s (1 is yes, -1 is no)
def predict(weights, x):
    return np.sign(x.dot(weights[1:]) + weights[0])


# In[8]:



#function that counts the correct predictions. This is used in 
# the voted perceptron
def count(x_test, y_test, weights):
    count_correct = np.sum((y_test.values == predict(weights, x_test).values.reshape(-1,1))*1)
    return count_correct


# In[9]:


import copy



#train function 
#inputs training data, epochs, learning rate (lr), and method
#outputs updated weights

def train(x_train, y_train, epochs, lr = .01, method = "standard"):
    
    if method == "standard":
        #Intialize weights
        weights = init_weights(x_train)
        print(weights)

        for epoch in range(1, epochs+1):
            #print(epoch)

            #shuffle data
            x_train, y_train = Shuffle(x_train, y_train)  

            #get predictions for current epoch
            prediction = predict(weights, x_train)


            #only update incorrect predictions
            weight_update_mask = (y_train.iloc[:, 0]*prediction <= 0)*1
            weight_update_mask = weight_update_mask.values.reshape((-1, 1)) #This aids in later multiplication

            update_variable = y_train.values*x_train.values*weight_update_mask

            weights[1:] = weights[1:] + lr*np.sum(update_variable, axis = 0)
            
            weights[0] = weights[0] + lr*(np.sum(y_train.values*weight_update_mask, axis = 0))
            
            print(weights)
            
            print(evaluate(x_train, y_train, weights))
        return weights
    
    if method == "voted":
        voted_dict = {}
        #Intialize weights
        weights = init_weights(x_train)

        for epoch in range(1, epochs+1):
            #print(epoch)

            #shuffle data
            #x_train, y_train = Shuffle(x_train, y_train)  

            #get predictions for current epoch
            prediction = predict(weights, x_train)


            #only update incorrect predictions
            weight_update_mask = (y_train.iloc[:, 0]*prediction <= 0)*1
            weight_update_mask = weight_update_mask.values.reshape((-1, 1)) #This aids in later multiplication

            update_variable = y_train.values*x_train.values*weight_update_mask

            weights[1:] = weights[1:] + lr*np.sum(update_variable, axis = 0)
            weights[0] = weights[0] + lr*(np.sum(y_train.values*weight_update_mask, axis = 0))
            #print(weights)
            weights_copy = weights.view()
            voted_dict[str(epoch)] = (np.array(weights_copy) , count(x_train, y_train, weights))
        
        return voted_dict

    if method == "averaged":
        #Intialize weights
        
        weights = init_weights(x_train)
        averaged_weights = np.array(weights)
        for epoch in range(1, epochs+1):
            #print(epoch)

            #shuffle data
            #x_train, y_train = Shuffle(x_train, y_train)  

            #get predictions for current epoch
            prediction = predict(weights, x_train)


            #only update incorrect predictions
            weight_update_mask = (y_train.iloc[:, 0]*prediction <= 0)*1
            weight_update_mask = weight_update_mask.values.reshape((-1, 1)) #This aids in later multiplication

            update_variable = y_train.values*x_train.values*weight_update_mask

            weights[1:] = weights[1:] + lr*np.sum(update_variable, axis = 0)
            weights[0] = weights[0] + lr*(np.sum(y_train.values*weight_update_mask, axis = 0))
            
            #print(weights)
            averaged_weights += np.array(weights)
        
        return averaged_weights


# In[ ]:





# In[10]:


#function to create accuracy of certain weights
def evaluate(x_test, y_test, weights):
    accuracy = np.sum((y_test.values == predict(weights, x_test).values.reshape(-1,1))*1)/y_test.shape[0]
    return accuracy


# In[11]:


#function to create accuracy of certain weights for the voted perceptron
def evaluate_voted(x_test, y_test, voted_dict):
    #print(voted_dict['1'][0].shape[0])
    voted_prediction = predict(voted_dict["1"][0],x_test)*0 
    for i in voted_dict:
        
        #print(i)
        #print(voted_dict[i][0], voted_dict[i][1])
        weights = voted_dict[i][0]
        votes  = voted_dict[i][1]
        #print("prediction:",predict(weights,x_test))
        prediction = votes* predict(weights,x_test)
        #print("prediction*votes:", prediction)
        voted_prediction += prediction
        #print("sum(voted_prediction*votes):", voted_prediction)
    #accuracy = np.sum((y_test.values == predict(weights, x_test).values.reshape(-1,1))*1)/y_test.shape[0]
    #return accuracy
    #print("np.sign(voted_prediction)",np.sign(voted_prediction))
    accuracy = np.sum((y_test.values == np.sign(voted_prediction).values.reshape(-1,1))*1)/y_test.shape[0]
    return accuracy


# In[12]:


lr = 1
epochs = 10


#Bank Data
train_data_m = pd.read_csv('./bank-note/train.csv', names = ["variance", "skewness", "curtosis", "entropy", "label"], header = None)

test_data_m = pd.read_csv('./bank-note/test.csv', names = ["variance", "skewness", "curtosis", "entropy", "label"], header = None)

#add label
label = "label"

x_train = train_data_m.iloc[:,:-1]


y_train = train_data_m.iloc[:,-1:]
y_train = y_train + ((y_train == 0)*-1) #convert y values to be -1 or 1 instead of 0 or 1



x_test = test_data_m.iloc[:,:-1]


y_test = test_data_m.iloc[:,-1:]
y_test = y_test + ((y_test == 0)*-1) #convert y values to be -1 or 1 instead of 0 or 1
y_test


#############
weights = train(x_train, y_train, epochs, lr, method = "standard")
evaluate(x_test, y_test, weights)

print("The standard perceptron error is", 1-evaluate(x_test, y_test, weights))
print("The standard perceptron weights is", weights)


# In[ ]:





# In[ ]:





# In[13]:


voted_dict = train(x_train, y_train, epochs, lr, method = "voted")
voted_dict
    
print("The voted perceptron error is", 1- evaluate_voted(x_test, y_test, voted_dict))

print("The voted perceptron weight and count dictionary is")#, voted_dict)
voted_dict


# In[14]:


weights = train(x_train, y_train, epochs, lr, method = "averaged")
weights
evaluate(x_test, y_test, weights)
weights

print("The averaged perceptron error is", 1-evaluate(x_test, y_test, weights))

print("The averaged perceptron weight and count dictionary is", weights)


# In[ ]:





# In[ ]:





# In[ ]:




