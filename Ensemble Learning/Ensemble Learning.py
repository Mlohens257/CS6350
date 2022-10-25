#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import os
os.getcwd()


# In[2]:


### From Homework 1


# In[3]:


#### Below functions were created per the direction of the homework 1 instructions.


# In[20]:


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


# In[46]:


#process data to replace unknowns with the most comon label for a particular attribute

import statistics
from statistics import mode

def replace_unknowns(train_data, label, unknown_label = "unknown"):
    #print(label)
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
        #print((train_data[attribute] == \"unknown\"))
        #print(bool_index[bool_index].index.values)
        #if type(train_data[attribute][0]) != type(''): #if values for an attribute is not a string type
            #print(np.median(train_data[attribute]))
            #train_data[attribute] = (train_data[attribute] >= np.median(train_data[attribute]))*1 #binarize based of the median value
            #print((train_data[attribute] >= np.median(train_data[attribute]))*1)
        
    return train_data


# In[22]:


#helper function to get dictionary depth
def dict_depth(dic, level = 1):
     
    if not isinstance(dic, dict) or not dic:
        return level
    return max(dict_depth(dic[key], level + 1)
                               for key in dic)


# In[23]:


import matplotlib.pyplot as plt

def plot_dict(input_dict1, input_dict2):
    names = list(input_dict1.keys())
    values1 =  np.array(list(input_dict1.values()))
    values2 =  np.array(list(input_dict2.values()))
    #print("names:", names)
    
    #print("values1:", values1)
    #print("values2:", values2)
    
    plt.plot(names, values1, names, values2)
    #plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()


# In[24]:


import matplotlib.pyplot as plt

def plot_dict1(input_dict1, input_dict2):
    names = list(input_dict1.keys())
    values1 = 1 - np.array(list(input_dict1.values()))
    values2 = 1 - np.array(list(input_dict2.values()))
    print("names:", names)
    
    print("values1:", values1)
    print("values2:", values2)
    
    plt.plot(names, values1, names, values2)
    #plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()


# In[25]:


def tot_entropy(train_data, label, label_list):
    tot_row = len(train_data.index) #the tot size of the dataset
    tot_entr = 0
    #print('train_data', train_data)
    #print(\"tot_row:\",tot_row,\"(This should not be 0!)\")
    
    for l in label_list: #for each label in the label
        #print(\"label:\", l)
        tot_label_count = len(train_data[train_data[label] == l].index) #number of the label
        #print(\"tot_entropy - index\", train_data[train_data[label] == l].index)
        #print(\"tot_label_count\",tot_label_count)
        tot_label_entr = - (tot_label_count/tot_row)*np.log2((tot_label_count+1e-7)/tot_row) #entropy of the label
        
        #print(\"tot_label_entr\",tot_label_entr)
        tot_entr += tot_label_entr #adding the label entropy to the tot entropy of the dataset
    #print(\"tot_entr\",tot_entr)
    return tot_entr


# In[26]:


def tot_entropy_weights(train_data, label, label_list, weights):
    tot_row = len(train_data.index) #the tot size of the dataset
    tot_entr = 0
    #print('train_data', train_data)
    #print(\"tot_row:\",tot_row,\"(This should not be 0!)\")
    
    for l in label_list: #for each label in the label
        #print(\"label:\", l)
        tot_label_count = len(train_data[train_data[label] == l].index) #number of the label
        #print(\"tot_entropy - index\", train_data[train_data[label] == l].index)
        #print(\"tot_label_count\",tot_label_count)
        #p_l = (tot_label_count+1e-7)/tot_row
        p_l = float(np.sum(weights.loc[train_data[train_data[label] == l].index])) + 1e-7
        tot_label_entr = - p_l*np.log2(p_l) #entropy of the label
        
        #print(\"tot_label_entr\",tot_label_entr)
        tot_entr += tot_label_entr #adding the label entropy to the tot entropy of the dataset
    #print(\"tot_entr\",tot_entr)
    return tot_entr


# In[27]:


def tot_GI(train_data, label, label_list):
    tot_row = len(train_data.index) #the tot size of the dataset
    tot_gi = 0
    #print('train_data', train_data)
    #print(\"tot_row:\",tot_row,\"(This should not be 0!)\")
    
    for l in label_list: #for each label in the label
        #print(\"label:\", l)
        tot_label_count = len(train_data[train_data[label] == l].index) #number of the label
        #print(\"tot_label_count\",tot_label_count)
        tot_label_gi =  np.square((tot_label_count)/tot_row) #entropy of the label
        

        tot_gi += tot_label_gi #adding the label entropy to the tot entropy of the dataset

    GI_tot = 1- tot_gi
    return GI_tot


# In[28]:


def tot_ME(train_data, label, label_list):
    tot_row = len(train_data.index) #the tot size of the dataset
    tot_ME = 0

    max_count = -1
    for l in label_list: #for each label in the label

        
        tot_label_count = len(train_data[train_data[label] == l].index) #number of the label
        #print(\"tot_label_count\",tot_label_count)
        if max_count <= tot_label_count:
            max_count = tot_label_count

    tot_ME = (tot_row - max_count)/tot_row
    return tot_ME


# In[29]:


def entropy(attribute_value_data, label, label_list):
    label_count = len(attribute_value_data.index)
    #print('label_count', label_count, len(attribute_value_data.index))\n",
    ent = 0
    #print(\"label_count\",label_count)
    
    for l in label_list:
        label_label_count = len(attribute_value_data[attribute_value_data[label] == l].index) #row count of label c
        
        #print(\"entropy - index\", attribute_value_data[attribute_value_data[label] == l].index)
        ent_l = 0
        if label_label_count != 0:
            #print(\"label_label_count != 0\")
            prob_l = label_label_count/label_count #probability
            #print(\"prob_l\", prob_l)
            ent_l = - prob_l * np.log2(prob_l)  #entropy
            #print(\"ent_l\", ent_l)
        ent += ent_l
        #print(\"ent_l\", ent_l)
    return ent


# In[30]:


def entropy_weights(attribute_value_data, label, label_list, weights):
    label_count = len(attribute_value_data.index)
    #print('label_count', label_count, len(attribute_value_data.index))
    ent = 0
    #print(\"label_count\",label_count)
    
    for l in label_list:
        label_label_count = len(attribute_value_data[attribute_value_data[label] == l].index) #row count of label c 
        #print("attribute_value_data[label] == l", attribute_value_data[label] == l)
        #print("attribute_value_data", attribute_value_data)
        #print("weights:", weights)
        #print(\"entropy - index\", attribute_value_data[attribute_value_data[label] == l].index)
        ent_l = 0
        if label_label_count != 0:
            #print(\"l\", l)
            #print(\"label_label_count != 0\")
            #print(\"weights for entropy label:\", float(np.sum(weights.loc[attribute_value_data[attribute_value_data[label] == l].index])))
            #prob_l = label_label_count/label_count #probability
            #print(\"attribute_value_data[label] == l:\", attribute_value_data[label] == l)
            #print(\"weights[attribute_value_data[label] == l]:\", weights[attribute_value_data[label] == l])
            #print(\"weights:\", weights)
            #print(\"attribute_value_data[attribute_value_data[label] == l].index:\", attribute_value_data[attribute_value_data[label] == l].index)
            prob_l = float(np.sum(weights.loc[attribute_value_data[attribute_value_data[label] == l].index])) + 1e-7
            #print(\"prob_l\", prob_l)
            ent_l = - prob_l * np.log2(prob_l)  #entropy
            #print(\"ent_l\", ent_l)
        ent += ent_l
        #print(\"ent_l\", ent_l)
    return ent


# In[31]:


def GI(attribute_value_data, label, label_list):
    label_count = len(attribute_value_data.index)

    gi = 0

    for l in label_list:
        label_label_count = len(attribute_value_data[attribute_value_data[label] == l].index) #row count of label c 
        gi_l = 0
        if label_label_count != 0:
            #print(\"label_label_count != 0\")
            prob_l = label_label_count/label_count #probability
            #print(\"prob_l\", prob_l)
            gi_l =   np.square(prob_l)  #entropy
         
        gi += gi_l
       
    GI = 1-gi
    return GI


# In[32]:


def ME(attribute_value_data, label, label_list):
    label_count = len(attribute_value_data.index)
    #print('label_count', label_count, len(attribute_value_data.index))
    ent = 0
    #print(\"label_count\",label_count)
    max_count = -1
    for l in label_list:
        label_label_count = len(attribute_value_data[attribute_value_data[label] == l].index) #row count of label c 
        
        if max_count <= label_label_count:
            max_count = label_label_count
        
    ME = (label_count - max_count)/label_count
    return ME


# In[33]:


def information_gain_weights(attribute_name, train_data, label, label_list, weights, gain_method, tot_gain_method ):
    attribute_value_list = train_data[attribute_name].unique() #unique values of the attribute
    tot_row = len(train_data.index)
    attribute_info = 0.0
    #print(\"attribute_name\", attribute_name
    for attribute_value in attribute_value_list:
        #print(\"attribute_value\",attribute_value)
        #print(\"train_data\", train_data)

        
        attribute_value_data = train_data[train_data[attribute_name] == attribute_value] #update data to only include data that has the designated attribute and value
        #print(\"attribute_value_data\", attribute_value_data)
        attribute_value_count = len(attribute_value_data.index)
        #print(\"attribute_value_count\", attribute_value_count)
        
        attribute_value_gain = gain_method(attribute_value_data, label, label_list, weights) #gain for the attribute value
        #print(\"attribute_value_entropy\", attribute_value_entropy)
                    
        p_attribute = float(np.sum(weights.loc[attribute_value_data.index]))+1e-7            
        #print(\"p_attribute:\", float(p_attribute))
        #attribute_info += (attribute_value_count/tot_row) * attribute_value_gain #information of the attribute value
        attribute_info += p_attribute * attribute_value_gain #information of the attribute value
        #print(\"attribute_info\", attribute_info)
        info_gain = tot_gain_method(train_data, label, label_list, weights) - attribute_info #information gain
        #if info_gain < 0:
            #print(\"info_gain is less than zero. SOMETHING IS WRONG. Information Gain:\", info_gain)
    return info_gain


# In[34]:


def information_gain(attribute_name, train_data, label, label_list, gain_method, tot_gain_method ):
    attribute_value_list = train_data[attribute_name].unique() #unique values of the attribute
    tot_row = len(train_data.index)
    attribute_info = 0.0
    #print(\"attribute_name\", attribute_name)
    for attribute_value in attribute_value_list:
        #print(\"attribute_value\",attribute_value)
        #print(\"train_data\", train_data)

        
        attribute_value_data = train_data[train_data[attribute_name] == attribute_value] #update data to only include data that has the designated attribute and value
        #print(\"attribute_value_data\", attribute_value_data)
        attribute_value_count = len(attribute_value_data.index)
        #print(\"attribute_value_count\", attribute_value_count)
        
        attribute_value_gain = gain_method(attribute_value_data, label, label_list) #gain for the attribute value
        #print(\"attribute_value_entropy\", attribute_value_entropy)
                    
                    

        attribute_info += (attribute_value_count/tot_row) * attribute_value_gain #information of the attribute value
        #print(\"attribute_info\", attribute_info)
        info_gain = tot_gain_method(train_data, label, label_list) - attribute_info #information gain
        #if info_gain < 0:
            #print("info_gain is less than zero. SOMETHING IS WRONG. Information Gain:", info_gain)
    return info_gain


# In[37]:



def get_best_attribute(train_data, label, method, label_list, weights):
    #print("train_data.columns", train_data.columns)
    attribute_list = train_data.columns.drop(label) #list of attributes
    #print(\"method\", method)                                    
    max_info_gain = -1
    max_info_attribute = None
    
    for attribute in attribute_list:  #for each attribute in the dataset
    
        #select gain calculation method - 'entropy', 'ME', or \"ME\"
        if method == "entropy" and weights is not None:
            #print(\"Entered1\")\n",
            attribute_info_gain = information_gain_weights(attribute, train_data, label, label_list, weights, entropy_weights, tot_entropy_weights)
        elif method == "entropy" and weights == None:
            #print(\"Entered2\")
            attribute_info_gain = information_gain(attribute, train_data, label, label_list, entropy, tot_entropy)
        
        elif method == "ME":
            attribute_info_gain = information_gain(attribute, train_data, label, label_list, ME, tot_ME)
        elif method == "GI":
            attribute_info_gain = information_gain(attribute, train_data, label, label_list, GI, tot_GI)
        else:
            print("Gain information calculation method hasn't been specified!")
        
        #print('attribute:', attribute)
        #print(\"max_info_gain\",max_info_gain)
        #print(\"attribute_info_gain:\", attribute_info_gain)

        if max_info_gain < attribute_info_gain: #store \"best\" gain
            max_info_gain = attribute_info_gain
            max_info_attribute = attribute
    #print(\"max_info_attribute\",max_info_attribute)        
    return max_info_attribute


# In[39]:



def sub_tree(attribute_name, train_data, label, label_list, weights, depth, depth_limit):
    attribute_value_count_dict = train_data[attribute_name].value_counts(sort=False) #dictionary of the count of unqiue attribute value
    #print(\"attribute_value_count_dict\",attribute_value_count_dict)
    tree = {} #sub tree or node
    #print(\"DEPTH =\", depth)
    #print(\"DEPTH LIMIT =\", depth_limit)
    #print(\"attribute_name\", attribute_name)
    #print(\"depth_limit:\",depth_limit)
    #print(\"attribute_name\", attribute_name)
    for attribute_value, count in attribute_value_count_dict.iteritems():
        attribute_value_data = train_data[train_data[attribute_name] == attribute_value] #dataset with only attribute_name = attribute_value
        
        assigned_to_node = False #flag for tracking attribute_value is pure label or not
        for l in label_list: #for each label
            label_count = len(attribute_value_data[attribute_value_data[label] == l].index) #count of label c

            if label_count == count: #count of attribute_value = count of label (pure label)
                tree[attribute_value] = l #adding node to the tree
                train_data = train_data[train_data[attribute_name] != attribute_value] #removing rows with attribute_value
                assigned_to_node = True
            
        if (not assigned_to_node) and (depth >= (depth_limit - 2)):
            tree[attribute_value] = train_data[label].mode()[0] #add most common value to the tree for the specified attibute
            train_data = train_data[train_data[attribute_name] != attribute_value] #removing rows with attribute_value
            assigned_to_node = True
            
        if not assigned_to_node: #not leaf node\n",
            tree[attribute_value] = "Place_Holder" #mark branch with "Place_Holder" for future expansion. Not a leaf node.
    
        

    return tree, train_data


# In[43]:



def create_tree(root,  prev_attribute_value, train_data, label, method, label_list, weights, random_forest_num, depth = 0, depth_limit = 6):
    #print(\"base root:\", root)
    #print(\"base root depth:\",dict_depth(root) )
    #print(\"len(train_data.index):\", len(train_data.index))

    #print(\"depth_limit:\",depth_limit)
    
    #print(\"depth\", depth)
    #print(\"create_tree data index:\", train_data.index)   
    #print(\"create_tree weights index:\", weights.index)     
    #print(\"prev_attribute_value\", prev_attribute_value)

    if len(train_data.index) != 0: #enter if dataset



        #print(\"prev_attribute_value\",prev_attribute_value)
        if random_forest_num == -1:
            max_info_attribute = get_best_attribute(train_data, label, method, label_list, weights) #most informative attribute
        else:
            rf_column_list = random.sample(list(train_data.columns[:-1]), k=random_forest_num)
            rf_column_list.append(label)
            #rf_column_list
            rf_data = train_data[rf_column_list]
            max_info_attribute = get_best_attribute(rf_data, label, method, label_list, weights) #most informative attribute
        #print(\"max_info_attribute\",max_info_attribute)
        #print(\"train_data\",train_data)
        #print(\"label\", label)
        #print(\"method\", method)
        #print(\"label_list\", label_list)
        #print(\"depth\" ,depth)
        #print(\"depth_limit\",depth_limit)
        
        tree, train_data = sub_tree(max_info_attribute, train_data, label, label_list, weights, depth , depth_limit) #getting tree node and updated dataset
        next_root = None
        #print(\"max_info_attribute\",max_info_attribute)
        #print('tree depth', dict_depth(tree))

        
        if prev_attribute_value != None: #add to intermediate node of the tree
            root[prev_attribute_value] = dict()
            root[prev_attribute_value][max_info_attribute] = tree
            next_root = root[prev_attribute_value][max_info_attribute]
            #print(\"tree\",tree)
        else: #add to root of the tree


            
            root[max_info_attribute] = tree
            next_root = root[max_info_attribute]
            #print(\"tree\",tree)
    

        
        place_holder_count = 0

        #print(\"Iterate Tree Node\")
        #print(\"len(list(next_root.items())):\",len(list(next_root.items())))
        for node, branch in list(next_root.items()): #iterating the tree node
            #print(\"node:\", node)
            #print(\"branch:\", branch)
            

            if branch == "Place_Holder": #if it is expandable
                
                
                #root_depth_tracker = root_depth_tracker + dict_depth(root)
                #print(\"root_depth_tracker:\",root_depth_tracker )
                
                
                
                #place_holder_count +=1
                #print(\"place_holder_count\", place_holder_count)
                
                
                attribute_value_data = train_data[train_data[max_info_attribute] == node] #using the updated dataset
                #print(\"attribute_value_data.index\", attribute_value_data.index)
                #print(\"weights.index\", weights.index)
                if  weights is not None:
                    attribute_value_updated_weights = weights.loc[attribute_value_data.index] #using only the weights of the updated dataset
                else:
                    attribute_value_updated_weights = None
                #print(\"attribute_value_updated_weights\", attribute_value_updated_weights)
                #print(\"attribute_value_updated_weights\", attribute_value_updated_weights)
                create_tree(next_root, node, attribute_value_data, label, method, label_list,attribute_value_updated_weights,depth +2, depth_limit) #recursive call with updated dataset
            #else:
                #print(\"branch doesn't equal placeholder\")
              


# In[44]:



def id3(train_data_m, label, method , depth_limit, weights = None, random_forest_num = -1):
    train_data = train_data_m.copy() #getting a copy of the dataset
    tree = {} #tree which will be updated
    #print(\"weights.shape:\", weights.shape, train_data_m.shape)
    
    #check if weights are the correct size. If not return an error.
    if (weights is not None) and (weights.shape[0] !=train_data_m.shape[0]):
        print("ERROR: The weights shape doesn't match the input data shape.")
        return
    
    label_list = train_data[label].unique() #getting unqiue labels
    print("Start Recursion")
    create_tree(tree, None, train_data_m, label, method, label_list, weights, random_forest_num, depth_limit = depth_limit) #start calling recursion
    print("End Recursion")
    return tree


# In[45]:


def predict(tree, instance):
    if not isinstance(tree, dict): #if it is leaf node
        return tree #return the value
    else:
        root_node = next(iter(tree)) #getting first key/feature name of the dictionary
        feature_value = instance[root_node] #value of the feature
        #print(root_node)
        #print(tree)
        if feature_value in tree[root_node]: #checking the feature value in current tree node
            return predict(tree[root_node][feature_value], instance) #goto next feature
        else:
            return None


# In[47]:


def evaluate(tree, test_data_m, label):
    correct_predict = 0
    wrong_predict = 0
    #print(test_data_m)
    for index, row in test_data_m.iterrows(): #for each row in the dataset\n",
        #print(index)
        #print(test_data_m.loc[index])
        result = predict(tree, test_data_m.loc[index]) #predict the row\n",
        if result == test_data_m[label].loc[index]: #predicted value and expected value is same or not\n",
            correct_predict += 1 #increase correct count
        else:
            wrong_predict += 1 #increase incorrect count
    accuracy = correct_predict / (correct_predict + wrong_predict) #calculating accuracy\n",
    #print(\"Accuracy is_a\", accuracy)
    return accuracy


# In[48]:


#we need to get the results so that we can update the weights of the Adaboost algorithm. Therefore, 
#the evaluate function is being copied and altered so that the results are returned. The new function 
#\"evaluate_and_return_results\" is created for use in the Adaboost algorithm.

#This function returns 1's for correct predictions and -1's for incorrect prediction. Only for Binary Labels!

def evaluate_and_return_results_adaboost(tree, test_data_m, label):
    #correct_predict = 0
    #wrong_predict = 0
    result_list = []
    for index, row in test_data_m.iterrows(): #for each row in the dataset
        result = predict(tree, test_data_m.loc[index]) #predict the row
        if result == test_data_m[label].loc[index]: #predicted value and expected value is same or not
            #correct_predict += 1 #increase correct count
            result_list.append(-1.)
        else:
            #wrong_predict += 1 #increase incorrect count
            result_list.append(1.)
    #accuracy = correct_predict / (correct_predict + wrong_predict) #calculating accuracy
    #print(\"Accuracy is_a\", accuracy)
    return result_list


# In[49]:


#we need to get the results so that we can implement the bagging algorithm. Therefore, 
#the evaluate function is being copied and altered so that the results are returned. The new function 
#\"evaluate_and_return_results\" is created for use in the Adaboost algorithm.

#This function the results of the tree

def evaluate_and_return_results_bagging(tree, test_data_m, label):
    #correct_predict = 0
    #wrong_predict = 0
    result_list = []
    for index, row in test_data_m.iterrows(): #for each row in the dataset
        result = predict(tree, test_data_m.loc[index]) #predict the row
        #if result == test_data_m[label].loc[index]: #predicted value and expected value is same or not
            #correct_predict += 1 #increase correct count
            #result_list.append(1)
        #else:
            #wrong_predict += 1 #increase incorrect count
        result_list.append(result)
    #accuracy = correct_predict / (correct_predict + wrong_predict) #calculating accuracy
    #print(\"Accuracy is_a\", accuracy)
    return result_list


# In[50]:


def evaluate_bagging(bagging_list, test_data_m, label):
    correct_predict = 0
    wrong_predict = 0

    #print(test_data_m)
    for index, row in test_data_m.iterrows(): #for each row in the dataset
        #print(index)
        #print(test_data_m.loc[index])
        #result = predict(tree, test_data_m.loc[index]) #predict the row
        #print(\"bagging_list[index]\",bagging_list[index])
        #print(\"test_data_m[label].loc[index]\",test_data_m[label].loc[index])
        if bagging_list[index] == test_data_m[label].loc[index]: #predicted value and expected value is same or not
            correct_predict += 1 #increase correct count
        else:
            wrong_predict += 1 #increase incorrect count
    accuracy = correct_predict / (correct_predict + wrong_predict) #calculating accuracy
    #print(\"Accuracy is_a\", accuracy)
    return accuracy


# In[ ]:


### Homework 2 Implement the boosting and bagging algorithms using decision stumps

