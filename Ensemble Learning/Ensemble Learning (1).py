#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import math
import os
os.getcwd()


# In[6]:


### From Homework 1


# In[7]:


#### Below functions were created per the direction of the homework 1 instructions.


# In[8]:


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


# In[9]:


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
            #print(attribute)
        #print((train_data[attribute] == \"unknown\"))
        #print(bool_index[bool_index].index.values)
        #if type(train_data[attribute][0]) != type(''): #if values for an attribute is not a string type
            #print(np.median(train_data[attribute]))
            #train_data[attribute] = (train_data[attribute] >= np.median(train_data[attribute]))*1 #binarize based of the median value
            #print((train_data[attribute] >= np.median(train_data[attribute]))*1)
        
    return train_data


# In[10]:


#helper function to get dictionary depth
def dict_depth(dic, level = 1):
     
    if not isinstance(dic, dict) or not dic:
        return level
    return max(dict_depth(dic[key], level + 1)
                               for key in dic)


# In[11]:


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


# In[38]:


import matplotlib.pyplot as plt

def plot_dict1(input_dict1, input_dict2):
    names = list(input_dict1.keys())
    values1 = 1 - np.array(list(input_dict1.values()))
    values2 = 1 - np.array(list(input_dict2.values()))
    #print("names:", names)
    
    #print("values1:", values1)
    #print("values2:", values2)
    
    plt.plot(names, values1, names, values2)
    #plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()


# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[18]:


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


# In[19]:


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


# In[20]:


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


# In[21]:


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


# In[22]:


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


# In[23]:



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


# In[24]:



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


# In[25]:



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
              


# In[39]:



def id3(train_data_m, label, method , depth_limit, weights = None, random_forest_num = -1):
    train_data = train_data_m.copy() #getting a copy of the dataset
    tree = {} #tree which will be updated
    #print(\"weights.shape:\", weights.shape, train_data_m.shape)
    
    #check if weights are the correct size. If not return an error.
    if (weights is not None) and (weights.shape[0] !=train_data_m.shape[0]):
        #print("ERROR: The weights shape doesn't match the input data shape.")
        return
    
    label_list = train_data[label].unique() #getting unqiue labels
    print("Start Recursion")
    create_tree(tree, None, train_data_m, label, method, label_list, weights, random_forest_num, depth_limit = depth_limit) #start calling recursion
    print("End Recursion")
    return tree


# In[27]:


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


# In[28]:


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


# In[29]:


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


# In[30]:


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


# In[31]:


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


# In[32]:


### Homework 2 Implement the boosting and bagging algorithms using decision stumps


# In[42]:


###Part 2 - Problem 2a


def adaboost_predict(data, classifier_dict, label, num_of_classifiers):
    #create a result dataframe, which will be used later to find the bagging algorithm result.
    bagging_result_df = pd.DataFrame()
    for i in range(0, num_of_classifiers+1):
        #print(i)
        #print(classifier_dict[i])
        tree = classifier_dict[i]
        bagging_result_df[i] = evaluate_and_return_results_bagging(tree, data, label)
    
  
    
    #get the median answers for all test points and then convert to yes or no answers. Store into a list.
    unique_values_list = set(bagging_result_df.values.flatten().tolist())
    most_common_list = []
    for index in bagging_result_df.index:
        #print("index", index)
        max_occur = 0
        #print("bagging_result_df.loc[index,:].unique()", bagging_result_df.loc[index,:].unique())
        for pat in bagging_result_df.loc[index,:].unique().astype(str):
            #print("pat", pat)
            #bagging_result_df.iloc[0,:].str.count()
            #print(index)
           #print(pat)
            #print(bagging_result_df.loc[index,:])
            #print(sum(bagging_result_df.loc[index,:].str.count(str(pat))))
            count =sum(bagging_result_df.loc[index,:].astype(str).str.count(str(pat)))
            #print(count)
            if max_occur < count:
                max_occur_val = pat
                #print("max_occur_val", max_occur_val)

                max_occur = count
                #print("max_occur", max_occur)
        #evaluate_bagging(bagging_result_list, test_data_m, label)
        most_common_list.append(max_occur_val)
        max_occur_val
    return most_common_list

#Bank Data
train_data_m = pd.read_csv('./bank/train.csv', names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "Output variable"], header = None)

test_data_m = pd.read_csv('./bank/test.csv', names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "Output variable"], header = None)

#add label
label = "Output variable"

#binarize data
train_data_m = numer_to_binary(train_data_m, label)
test_data_m =numer_to_binary(test_data_m, label)

T = 20

batch_size = 25

train_data_sample = train_data_m.iloc[:batch_size]

#initialize weights as a single column dataframe
weights = pd.DataFrame()
weights[0] = np.ones(train_data_sample.shape[0])/train_data_sample.shape[0]
print(train_data_sample.shape[0])
print(weights.shape)

alpha_stored ={}
classifier_dict = {}
test_accuracy_dict = {}
train_accuracy_dict = {}

test_accuracy_dict_combined = {}
train_accuracy_dict_combined = {}
depth_limit = 1
method = "entropy"
#for depth in range(2, 2*(max_depth+1), 2):
for t in range(0, T):
    
    #print("weights(in for loop):", weights[t])
    ###Run Algorithm
    tree = id3(train_data_sample, label, method, depth_limit, weights[t])
    classifier_dict[t] = tree
    results = evaluate_and_return_results_adaboost(tree, train_data_sample, label)
    #print("results:", results)
    #print(" ")
    error_t = 0.5 - 0.5*(np.sum(weights[t]*results))
    #print("error_t",error_t)
    #print(" ")
    #print("weights*results", weights[t]*results)
    #print(" ")
    alpha = 0.5*np.log((1-error_t)/error_t)
    alpha_stored[t] = alpha
    #print("alpha:", alpha)
    #print(" ")
    results = pd.DataFrame(results)
    alpha1 = np.exp(-1.*alpha*results)

    #weights = 
    weights_unnormalized = pd.DataFrame(weights[t])*alpha1
    weights_unnormalized = weights[t]*alpha1[0]
    
    #print("alpha1[0]:", alpha1[0])
    #print("weights_unnormalized:",weights_unnormalized)
    t_plus_1 = int(t+1)
    weights[t_plus_1] = weights_unnormalized/np.sum(weights_unnormalized)
    #print("weights:", weights[t_plus_1])
    #print(" ")
    acc_test = evaluate(tree, test_data_m, label)
    acc_train = evaluate(tree, train_data_sample, label)
    test_accuracy_dict[t] = (1 - acc_test)
    train_accuracy_dict[t] = (1 - acc_train)

    #print(results)
    #print(" ")
    #print("Bank Data - accuracy_dict for entropy with unknowns:", accuracy_dict) 
    #print(" ")
    #print("Tree:", tree)
    #print("classifier_dict:", classifier_dict)
    #print("label:", label)
    #print("t:", t)
    most_common_list_train = adaboost_predict(train_data_m, classifier_dict, label, t)
    most_common_list_test = adaboost_predict(test_data_m, classifier_dict, label, t)
    #print("most_common_list_train:",most_common_list_train)
    #print("most_common_list_test:",most_common_list_test)
    test_acc_combined =  evaluate_bagging(most_common_list_test, test_data_m, label)
    train_acc_combined = evaluate_bagging(most_common_list_train, train_data_m, label)
    #print("test_acc_combined:", test_acc_combined, (1-test_acc_combined ))
    #print("train_acc_combined:", train_acc_combined, (1-train_acc_combined))

    test_accuracy_dict_combined[t] = (1 - test_acc_combined)
    train_accuracy_dict_combined[t] = (1 - train_acc_combined)


# In[43]:


###Part 2 - Problem 2a


def adaboost_predict(data, classifier_dict, label, num_of_classifiers):
    #create a result dataframe, which will be used later to find the bagging algorithm result.
    bagging_result_df = pd.DataFrame()
    for i in range(0, num_of_classifiers+1):
        #print(i)
        #print(classifier_dict[i])
        tree = classifier_dict[i]
        bagging_result_df[i] = evaluate_and_return_results_bagging(tree, data, label)
    
  
    
    #get the median answers for all test points and then convert to yes or no answers. Store into a list.
    unique_values_list = set(bagging_result_df.values.flatten().tolist())
    most_common_list = []
    for index in bagging_result_df.index:
        #print("index", index)
        max_occur = 0
        #print("bagging_result_df.loc[index,:].unique()", bagging_result_df.loc[index,:].unique())
        for pat in bagging_result_df.loc[index,:].unique().astype(str):
            #print("pat", pat)
            #bagging_result_df.iloc[0,:].str.count()
            #print(index)
           #print(pat)
            #print(bagging_result_df.loc[index,:])
            #print(sum(bagging_result_df.loc[index,:].str.count(str(pat))))
            count =sum(bagging_result_df.loc[index,:].astype(str).str.count(str(pat)))
            #print(count)
            if max_occur < count:
                max_occur_val = pat
                #print("max_occur_val", max_occur_val)

                max_occur = count
                #print("max_occur", max_occur)
        #evaluate_bagging(bagging_result_list, test_data_m, label)
        most_common_list.append(max_occur_val)
        max_occur_val
    return most_common_list

#Bank Data
train_data_m = pd.read_csv('./bank/train.csv', names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "Output variable"], header = None)

test_data_m = pd.read_csv('./bank/test.csv', names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "Output variable"], header = None)

#add label
label = "Output variable"

#binarize data
train_data_m = numer_to_binary(train_data_m, label)
test_data_m =numer_to_binary(test_data_m, label)

T = 20

batch_size = 25

train_data_sample = train_data_m.iloc[:batch_size]

#initialize weights as a single column dataframe
weights = pd.DataFrame()
weights[0] = np.ones(train_data_sample.shape[0])/train_data_sample.shape[0]
print(train_data_sample.shape[0])
print(weights.shape)

alpha_stored ={}
classifier_dict = {}
test_accuracy_dict = {}
train_accuracy_dict = {}

test_accuracy_dict_combined = {}
train_accuracy_dict_combined = {}
depth_limit = 1
method = "entropy"
#for depth in range(2, 2*(max_depth+1), 2):
for t in range(0, T):
    
    print("weights(in for loop):", weights[t])
    ###Run Algorithm
    tree = id3(train_data_sample, label, method, depth_limit, weights[t])
    classifier_dict[t] = tree
    results = evaluate_and_return_results_adaboost(tree, train_data_sample, label)
    print("results:", results)
    print(" ")
    error_t = 0.5 - 0.5*(np.sum(weights[t]*results))
    print("error_t",error_t)
    #print(" ")
    #print("weights*results", weights[t]*results)
    #print(" ")
    alpha = 0.5*np.log((1-error_t)/error_t)
    alpha_stored[t] = alpha
    #print("alpha:", alpha)
    #print(" ")
    results = pd.DataFrame(results)
    alpha1 = np.exp(-1.*alpha*results)

    #weights = 
    weights_unnormalized = pd.DataFrame(weights[t])*alpha1
    weights_unnormalized = weights[t]*alpha1[0]
    
    print("alpha1[0]:", alpha1[0])
    #print("weights_unnormalized:",weights_unnormalized)
    t_plus_1 = int(t+1)
    weights[t_plus_1] = weights_unnormalized/np.sum(weights_unnormalized)
    #print("weights:", weights[t_plus_1])
    #print(" ")
    acc_test = evaluate(tree, test_data_m, label)
    acc_train = evaluate(tree, train_data_sample, label)
    test_accuracy_dict[t] = (1 - acc_test)
    train_accuracy_dict[t] = (1 - acc_train)

    #print(results)
    #print(" ")
    print("Bank Data - accuracy_dict for entropy with unknowns:", accuracy_dict) 
    #print(" ")
    print("Tree:", tree)
    print("classifier_dict:", classifier_dict)
    print("label:", label)
    print("t:", t)
    most_common_list_train = adaboost_predict(train_data_m, classifier_dict, label, t)
    most_common_list_test = adaboost_predict(test_data_m, classifier_dict, label, t)
    print("most_common_list_train:",most_common_list_train)
    print("most_common_list_test:",most_common_list_test)
    test_acc_combined =  evaluate_bagging(most_common_list_test, test_data_m, label)
    train_acc_combined = evaluate_bagging(most_common_list_train, train_data_m, label)
    print("test_acc_combined:", test_acc_combined, (1-test_acc_combined ))
    print("train_acc_combined:", train_acc_combined, (1-train_acc_combined))

    test_accuracy_dict_combined[t] = (1 - test_acc_combined)
    train_accuracy_dict_combined[t] = (1 - train_acc_combined)


# In[ ]:


#by just guessing \"no\" for everything the classifier can get an accuracy of 88%
np.sum((train_data_m[label] == \"no\")*1)/train_data_m.shape[0]


# In[ ]:


plot_dict(test_accuracy_dict_combined,train_accuracy_dict_combined)


# In[ ]:


plot_dict(test_accuracy_dict,train_accuracy_dict)


# In[ ]:


test_accuracy_dict_combined


# In[ ]:


train_accuracy_dict_combined


# In[ ]:


test_accuracy_dict


# In[ ]:


train_accuracy_dict


# In[ ]:


"###Part 2 - Problem 2b\n",
"\n",
"import random\n",
"\n",
"\n",
"\n",
"def create_classifier_dict(train_data_m, batch_size, T, depth_limit = 16, random_forest_num = -1):\n",
"\n",
"    classifier_dict = {}\n",
"    print(\"depth_limit:\",depth_limit)\n",
"\n",
"    train_acc_dict = {}\n",
"    test_acc_dict = {}\n",
"\n",
"    accuracy_dict = {}\n",
"\n",
"    for t in range(1, T):\n",
"\n",
"        #sample\n",
"\n",
"        rand_i = random.randint(0, train_data_m.shape[0]-batch_size-1)\n",
"        train_data_sample  = train_data_m.iloc[rand_i:rand_i+batch_size]\n",
"\n",
"        #train and store the classifier into a list\n",
"\n",
"\n",
"\n",
"\n",
"        method = \"entropy\"\n",
"\n",
"        ###Run Algorithm\n",
"        tree = id3(train_data_sample, label, method, depth_limit = depth_limit, weights = None, random_forest_num = random_forest_num)\n",
"        #print(tree)\n",
"        acc_test = evaluate(tree, test_data_m, label)\n",
"        #results = evaluate_and_return_results(tree, test_data_m, label)\n",
"        #acc_train = evaluate(tree, train_data_sample, label)\n",
"        #accuracy_dict[t] = (1 - acc_test), (1 - acc_train)\n",
"        #print(results)\n",
"        #print(\"Bank Data - accuracy_dict for entropy with unknowns:\", accuracy_dict) \n",
"        #train_data_m\n",
"        classifier_dict[t] = tree\n",
"        #print(tree)\n",
"        \n",
"        \n",
"\n",
"\n",
"\n",
"        num_of_classifiers = t\n",
"\n",
"\n",
"        most_common_list_train = bagging_predict(train_data_m, classifier_dict, label, num_of_classifiers)\n",
"        #print(\"len(most_common_list_train):\", len(most_common_list_train))\n",
"        most_common_list_test = bagging_predict(test_data_m, classifier_dict, label, num_of_classifiers)   \n",
"\n",
"        train_acc =  evaluate_bagging(most_common_list_train, train_data_m, label)   \n",
"        test_acc =  evaluate_bagging(most_common_list_test, test_data_m, label)\n",
"\n",
"        train_acc_dict[t] = 1-train_acc\n",
"        test_acc_dict[t] = 1-test_acc\n",
"        print(\"train acc:\", train_acc)    \n",
"        print(\"test acc:\", test_acc)\n",
"        print(\"train err:\", 1-train_acc)    \n",
"        print(\"test err:\", 1-test_acc)        \n",
"        \n",
"        \n",
"    return classifier_dict, train_acc_dict, test_acc_dict\n",
"#predict test data for each classifier and take the average\n",
"\n",
"def bagging_predict(data, classifier_dict, label, num_of_classifiers):\n",
"    #create a result dataframe, which will be used later to find the bagging algorithm result.\n",
"    bagging_result_df = pd.DataFrame()\n",
"    for i in range(1, num_of_classifiers+1):\n",
"        print(i)\n",
"        print(classifier_dict[i])\n",
"        tree = classifier_dict[i]\n",
"        bagging_result_df[i] = evaluate_and_return_results_bagging(tree, data, label)\n",
"    \n",
"  \n",
"    \n",
"    #get the median answers for all test points and then convert to yes or no answers. Store into a list.\n",
"    unique_values_list = set(bagging_result_df.values.flatten().tolist())\n",
"    most_common_list = []\n",
"    for index in bagging_result_df.index:\n",
"        print(\"index\", index)\n",
"        max_occur = 0\n",
"        print(\"bagging_result_df.loc[index,:].unique()\", bagging_result_df.loc[index,:].unique())\n",
"        for pat in bagging_result_df.loc[index,:].unique().astype(str):\n",
"            #print(\"pat\", pat)\n",
"            #bagging_result_df.iloc[0,:].str.count()\n",
"            #print(index)\n",
"            #print(pat)\n",
"            #print(bagging_result_df.loc[index,:])\n",
"            #print(sum(bagging_result_df.loc[index,:].str.count(str(pat))))\n",
"            count =sum(bagging_result_df.loc[index,:].astype(str).str.count(str(pat)))\n",
"            #print(count)\n",
"            if max_occur < count:\n",
"                max_occur_val = pat\n",
"                #print(\"max_occur_val\", max_occur_val)\n",
"\n",
"                max_occur = count\n",
"                #print(\"max_occur\", max_occur)\n",
"        #evaluate_bagging(bagging_result_list, test_data_m, label)\n",
"        most_common_list.append(max_occur_val)\n",
"        max_occur_val\n",
"    return most_common_list\n",
"    \n"


# In[ ]:


###Part 2 - Problem 2b

import random



def create_classifier_dict(train_data_m, batch_size, T, depth_limit = 16, random_forest_num = -1):

    classifier_dict = {}
    print("depth_limit:",depth_limit)

    train_acc_dict = {}
    test_acc_dict = {}

    accuracy_dict = {}

    for t in range(1, T):

        #sample

        rand_i = random.randint(0, train_data_m.shape[0]-batch_size-1)
        train_data_sample  = train_data_m.iloc[rand_i:rand_i+batch_size]

        #train and store the classifier into a list




        method = "entropy"

        ###Run Algorithm
        tree = id3(train_data_sample, label, method, depth_limit = depth_limit, weights = None, random_forest_num = random_forest_num)
        #print(tree)
        acc_test = evaluate(tree, test_data_m, label)
        #results = evaluate_and_return_results(tree, test_data_m, label)
        #acc_train = evaluate(tree, train_data_sample, label)
        #accuracy_dict[t] = (1 - acc_test), (1 - acc_train)
        #print(results)
        #print("Bank Data - accuracy_dict for entropy with unknowns:", accuracy_dict) 
        #train_data_m
        classifier_dict[t] = tree
        #print(tree)
        
        



        num_of_classifiers = t


        most_common_list_train = bagging_predict(train_data_m, classifier_dict, label, num_of_classifiers)
        #print("len(most_common_list_train):", len(most_common_list_train))
        most_common_list_test = bagging_predict(test_data_m, classifier_dict, label, num_of_classifiers)   

        train_acc =  evaluate_bagging(most_common_list_train, train_data_m, label)   
        test_acc =  evaluate_bagging(most_common_list_test, test_data_m, label)

        train_acc_dict[t] = 1-train_acc
        test_acc_dict[t] = 1-test_acc
        print("train acc:", train_acc)    
        print("test acc:", test_acc)
        print("train err:", 1-train_acc)    
        print("test err:", 1-test_acc)        
        
        
    return classifier_dict, train_acc_dict, test_acc_dict
#predict test data for each classifier and take the average

def bagging_predict(data, classifier_dict, label, num_of_classifiers):
    #create a result dataframe, which will be used later to find the bagging algorithm result.
    bagging_result_df = pd.DataFrame()
    for i in range(1, num_of_classifiers+1):
        print(i)
        print(classifier_dict[i])
        tree = classifier_dict[i]
        bagging_result_df[i] = evaluate_and_return_results_bagging(tree, data, label)
    
  
    
    #get the median answers for all test points and then convert to yes or no answers. Store into a list.
    unique_values_list = set(bagging_result_df.values.flatten().tolist())
    most_common_list = []
    for index in bagging_result_df.index:
        print("index", index)
        max_occur = 0
        print("bagging_result_df.loc[index,:].unique()", bagging_result_df.loc[index,:].unique())
        for pat in bagging_result_df.loc[index,:].unique().astype(str):
            #print("pat", pat)
            #bagging_result_df.iloc[0,:].str.count()
            #print(index)
            #print(pat)
            #print(bagging_result_df.loc[index,:])
            #print(sum(bagging_result_df.loc[index,:].str.count(str(pat))))
            count =sum(bagging_result_df.loc[index,:].astype(str).str.count(str(pat)))
            #print(count)
            if max_occur < count:
                max_occur_val = pat
                #print("max_occur_val", max_occur_val)

                max_occur = count
                #print("max_occur", max_occur)
        #evaluate_bagging(bagging_result_list, test_data_m, label)
        most_common_list.append(max_occur_val)
        max_occur_val
    return most_common_list
 


# In[ ]:


"#Bank Data\n",
"train_data_m = pd.read_csv('./bank/train.csv', names = [\"age\", \"job\", \"marital\", \"education\", \"default\", \"balance\", \"housing\", \"loan\", \"contact\", \"day\", \"month\", \"duration\", \"campaign\", \"pdays\", \"previous\", \"poutcome\", \"Output variable\"], header = None)\n",
"\n",
"test_data_m = pd.read_csv('./bank/test.csv', names = [\"age\", \"job\", \"marital\", \"education\", \"default\", \"balance\", \"housing\", \"loan\", \"contact\", \"day\", \"month\", \"duration\", \"campaign\", \"pdays\", \"previous\", \"poutcome\", \"Output variable\"], header = None)\n",
"\n",
"#add label\n",
"label = \"Output variable\"\n",
"\n",
"#binarize data\n",
"train_data_m = numer_to_binary(train_data_m, label)\n",
"test_data_m =numer_to_binary(test_data_m, label)\n",
"\n",
"T = 20 #number of classifiers\n",
"\n",
"batch_size = 10\n",
"\n",
"classifier_dict, train_acc_dict, test_acc_dict = create_classifier_dict(train_data_m, batch_size, T, depth_limit = 15)\n",
"\n",
"plot_dict(train_acc_dict, test_acc_dict)    "


# In[ ]:


#Bank Data
train_data_m = pd.read_csv('./bank/train.csv', names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "Output variable"], header = None)

test_data_m = pd.read_csv('./bank/test.csv', names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "Output variable"], header = None)

#add label
label = "Output variable"

#binarize data
train_data_m = numer_to_binary(train_data_m, label)
test_data_m =numer_to_binary(test_data_m, label)

T = 20 #number of classifiers

batch_size = 10

classifier_dict, train_acc_dict, test_acc_dict = create_classifier_dict(train_data_m, batch_size, T, depth_limit = 15)

plot_dict(train_acc_dict, test_acc_dict)


# In[ ]:


plot_dict1(train_acc_dict, test_acc_dict)


# In[ ]:


train_acc_dict


# In[ ]:





# In[ ]:





# In[ ]:


"# Part 2 - 2d random forest\n",
"\n",
"#Bank Data\n",
"train_data_m = pd.read_csv('./bank/train.csv', names = [\"age\", \"job\", \"marital\", \"education\", \"default\", \"balance\", \"housing\", \"loan\", \"contact\", \"day\", \"month\", \"duration\", \"campaign\", \"pdays\", \"previous\", \"poutcome\", \"Output variable\"], header = None)\n",
"\n",
"test_data_m = pd.read_csv('./bank/test.csv', names = [\"age\", \"job\", \"marital\", \"education\", \"default\", \"balance\", \"housing\", \"loan\", \"contact\", \"day\", \"month\", \"duration\", \"campaign\", \"pdays\", \"previous\", \"poutcome\", \"Output variable\"], header = None)\n",
"\n",
"#add label\n",
"label = \"Output variable\"\n",
"\n",
"#binarize data\n",
"train_data_m = numer_to_binary(train_data_m, label)\n",
"test_data_m =numer_to_binary(test_data_m, label)\n",
"\n",
"T = 20 #number of classifiers\n",
"\n",
"batch_size = 6#len(train_data_m.index)-1\n",
"\n",
"classifier_dict, train_acc_dict, test_acc_dict = create_classifier_dict(train_data_m, batch_size, T, depth_limit = 1, random_forest_num = 4)\n",
"\n",
"plot_dict1(train_acc_dict, test_acc_dict)"


# In[ ]:


# Part 2 - 2d random forest

#Bank Data
train_data_m = pd.read_csv('./bank/train.csv', names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "Output variable"], header = None)

test_data_m = pd.read_csv('./bank/test.csv', names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "Output variable"], header = None)

#add label
label = "Output variable"

#binarize data
train_data_m = numer_to_binary(train_data_m, label)
test_data_m =numer_to_binary(test_data_m, label)

T = 20 #number of classifiers

batch_size = 6#len(train_data_m.index)-1

classifier_dict, train_acc_dict, test_acc_dict = create_classifier_dict(train_data_m, batch_size, T, depth_limit = 1, random_forest_num = 4)

plot_dict1(train_acc_dict, test_acc_dict)


# In[ ]:


classifier_dict, train_acc_dict, test_acc_dict = create_classifier_dict(train_data_m, batch_size, T, depth_limit = 1, random_forest_num = 2)

plot_dict1(train_acc_dict, test_acc_dict)


# In[ ]:




