#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import os
os.getcwd()


# ### Homework 1

# #### Below functions were created per the direction of the homework instructions.

# In[426]:


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


# In[448]:


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


# In[428]:


#helper function to get dictionary depth
def dict_depth(dic, level = 1):
     
    if not isinstance(dic, dict) or not dic:
        return level
    return max(dict_depth(dic[key], level + 1)
                               for key in dic)


# In[429]:


def tot_entropy(train_data, label, label_list):
    tot_row = len(train_data.index) #the tot size of the dataset
    tot_entr = 0
    #print('train_data', train_data)
    #print("tot_row:",tot_row,"(This should not be 0!)")
    
    for l in label_list: #for each label in the label
        #print("label:", l)
        tot_label_count = len(train_data[train_data[label] == l].index) #number of the label
        #print("tot_label_count",tot_label_count)
        tot_label_entr = - (tot_label_count/tot_row)*np.log2((tot_label_count+1e-7)/tot_row) #entropy of the label
        
        #print("tot_label_entr",tot_label_entr)
        tot_entr += tot_label_entr #adding the label entropy to the tot entropy of the dataset
    #print("tot_entr",tot_entr)
    return tot_entr


# In[430]:


def tot_GI(train_data, label, label_list):
    tot_row = len(train_data.index) #the tot size of the dataset
    tot_gi = 0
    #print('train_data', train_data)
    #print("tot_row:",tot_row,"(This should not be 0!)")
    
    for l in label_list: #for each label in the label
        #print("label:", l)
        tot_label_count = len(train_data[train_data[label] == l].index) #number of the label
        #print("tot_label_count",tot_label_count)
        tot_label_gi =  np.square((tot_label_count)/tot_row) #entropy of the label
        

        tot_gi += tot_label_gi #adding the label entropy to the tot entropy of the dataset

    GI_tot = 1- tot_gi
    return GI_tot


# In[431]:


def tot_ME(train_data, label, label_list):
    tot_row = len(train_data.index) #the tot size of the dataset
    tot_ME = 0

    max_count = -1
    for l in label_list: #for each label in the label

        
        tot_label_count = len(train_data[train_data[label] == l].index) #number of the label
        #print("tot_label_count",tot_label_count)
        if max_count <= tot_label_count:
            max_count = tot_label_count

    tot_ME = (tot_row - max_count)/tot_row
    return tot_ME


# In[432]:


def entropy(attribute_value_data, label, label_list):
    label_count = len(attribute_value_data.index)
    #print('label_count', label_count, len(attribute_value_data.index))
    ent = 0
    #print("label_count",label_count)
    
    for l in label_list:
        label_label_count = len(attribute_value_data[attribute_value_data[label] == l].index) #row count of label c 
        ent_l = 0
        if label_label_count != 0:
            #print("label_label_count != 0")
            prob_l = label_label_count/label_count #probability
            #print("prob_l", prob_l)
            ent_l = - prob_l * np.log2(prob_l)  #entropy
            #print("ent_l", ent_l)
        ent += ent_l
        #print("ent_l", ent_l)
    return ent


# In[433]:


def GI(attribute_value_data, label, label_list):
    label_count = len(attribute_value_data.index)

    gi = 0

    
    for l in label_list:
        label_label_count = len(attribute_value_data[attribute_value_data[label] == l].index) #row count of label c 
        gi_l = 0
        if label_label_count != 0:
            #print("label_label_count != 0")
            prob_l = label_label_count/label_count #probability
            #print("prob_l", prob_l)
            gi_l =   np.square(prob_l)  #entropy
         
        gi += gi_l
       
    GI = 1-gi
    return GI


# In[434]:


def ME(attribute_value_data, label, label_list):
    label_count = len(attribute_value_data.index)
    #print('label_count', label_count, len(attribute_value_data.index))
    ent = 0
    #print("label_count",label_count)
    max_count = -1
    for l in label_list:
        label_label_count = len(attribute_value_data[attribute_value_data[label] == l].index) #row count of label c 
        
        if max_count <= label_label_count:
            max_count = label_label_count
        
    ME = (label_count - max_count)/label_count
    return ME


# In[444]:


def information_gain(attribute_name, train_data, label, label_list, gain_method, tot_gain_method ):
    attribute_value_list = train_data[attribute_name].unique() #unique values of the attribute
    tot_row = len(train_data.index)
    attribute_info = 0.0
    #print("attribute_name", attribute_name)
    for attribute_value in attribute_value_list:
        #print("attribute_value",attribute_value)
        #print("train_data", train_data)

        
        attribute_value_data = train_data[train_data[attribute_name] == attribute_value] #update data to only include data that has the designated attribute and value
        #print("attribute_value_data", attribute_value_data)
        attribute_value_count = len(attribute_value_data.index)
        #print("attribute_value_count", attribute_value_count)
        
        attribute_value_gain = gain_method(attribute_value_data, label, label_list) #gain for the attribute value
        #print("attribute_value_entropy", attribute_value_entropy)
                    
                    

        attribute_info += (attribute_value_count/tot_row) * attribute_value_gain #information of the attribute value
        #print("attribute_info", attribute_info)
        info_gain = tot_gain_method(train_data, label, label_list) - attribute_info #information gain
        #if info_gain < 0:
            #print("info_gain is less than zero. SOMETHING IS WRONG. Information Gain:", info_gain)
    return info_gain


# In[436]:



def get_best_attribute(train_data, label, method, label_list):
    attribute_list = train_data.columns.drop(label) #list of attributes
    #print("method", method)                                    
    max_info_gain = -1
    max_info_attribute = None
    
    for attribute in attribute_list:  #for each attribute in the dataset
        
        #select gain calculation method - 'entropy', 'ME', or "ME"
        if method == "entropy":
            #print("Entered")
            attribute_info_gain = information_gain(attribute, train_data, label, label_list, entropy, tot_entropy)
        elif method == "ME":
            attribute_info_gain = information_gain(attribute, train_data, label, label_list, ME, tot_ME)
        elif method == "GI":
            attribute_info_gain = information_gain(attribute, train_data, label, label_list, GI, tot_GI)
        else:
            print("Gain information calculation method hasn't been specified!")
        
        #print('attribute:', attribute)
        #print("max_info_gain",max_info_gain)
        #print("attribute_info_gain", attribute_info_gain)

        if max_info_gain < attribute_info_gain: #store "best" gain
            max_info_gain = attribute_info_gain
            max_info_attribute = attribute
    #print("max_info_attribute",max_info_attribute)        
    return max_info_attribute


# In[437]:



def sub_tree(attribute_name, train_data, label, label_list, depth, depth_limit):
    attribute_value_count_dict = train_data[attribute_name].value_counts(sort=False) #dictionary of the count of unqiue attribute value
    #print("attribute_value_count_dict",attribute_value_count_dict)
    tree = {} #sub tree or node
    #print("DEPTH =", depth)
    #print("DEPTH LIMIT =", depth_limit)
    #print("attribute_name", attribute_name)
    
    #print("attribute_name", attribute_name)
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
            
        if not assigned_to_node: #not leaf node
            tree[attribute_value] = "Place_Holder" #mark branch with "Place_Holder" for future expansion. Not a leaf node.
        
        
   
    return tree, train_data


# In[ ]:





# In[445]:


def create_tree(root,  prev_attribute_value, train_data, label, method, label_list, depth = 0, depth_limit = 6):
    #print("base root:", root)
    #print("base root depth:",dict_depth(root) )
    #print("len(train_data.index):", len(train_data.index))

    #print("depth", depth)
        
        
    #print("prev_attribute_value", prev_attribute_value)

    if len(train_data.index) != 0: #enter if dataset


       
        #print("prev_attribute_value",prev_attribute_value)
        
        max_info_attribute = get_best_attribute(train_data, label, method, label_list) #most informative attribute
        
        #print("max_info_attribute",max_info_attribute)
        #print("train_data",train_data)
        #print("label", label)
        #print("method", method)
        #print("label_list", label_list)
        #print("depth" ,depth)
        #print("depth_limit",depth_limit)
        
        tree, train_data = sub_tree(max_info_attribute, train_data, label, label_list, depth , depth_limit) #getting tree node and updated dataset
        next_root = None
        #print("max_info_attribute",max_info_attribute)
        #print('tree depth', dict_depth(tree))

        
        if prev_attribute_value != None: #add to intermediate node of the tree
            root[prev_attribute_value] = dict()
            root[prev_attribute_value][max_info_attribute] = tree
            next_root = root[prev_attribute_value][max_info_attribute]
            #print("tree",tree)
        else: #add to root of the tree


            
            root[max_info_attribute] = tree
            next_root = root[max_info_attribute]
            #print("tree",tree)
    

        
        place_holder_count = 0

        #print("Iterate Tree Node")
        #print("len(list(next_root.items())):",len(list(next_root.items())))
        for node, branch in list(next_root.items()): #iterating the tree node
            #print("node:", node)
            #print("branch:", branch)
            

            if branch == "Place_Holder": #if it is expandable
                
                
                #root_depth_tracker = root_depth_tracker + dict_depth(root)
                #print("root_depth_tracker:",root_depth_tracker )
                
                
                
                #place_holder_count +=1
                #print("place_holder_count", place_holder_count)
                
                
                attribute_value_data = train_data[train_data[max_info_attribute] == node] #using the updated dataset
                create_tree(next_root, node, attribute_value_data, label, method, label_list,depth +2, depth_limit) #recursive call with updated dataset
            #else:
                #print("branch doesn't equal placeholder")
              


# In[439]:



def id3(train_data_m, label, method , depth):
    train_data = train_data_m.copy() #getting a copy of the dataset
    tree = {} #tree which will be updated
    
    label_list = train_data[label].unique() #getting unqiue labels
    print("Start Recursion")
    create_tree(tree, None, train_data_m, label, method, label_list, depth_limit = depth) #start calling recursion
    print("End Recursion")
    return tree


# In[ ]:





# In[ ]:





# In[ ]:





# In[440]:


def predict(tree, instance):
    if not isinstance(tree, dict): #if it is leaf node
        return tree #return the value
    else:
        root_node = next(iter(tree)) #getting first key/feature name of the dictionary
        feature_value = instance[root_node] #value of the feature
        if feature_value in tree[root_node]: #checking the feature value in current tree node
            return predict(tree[root_node][feature_value], instance) #goto next feature
        else:
            return None


# In[441]:


def evaluate(tree, test_data_m, label):
    correct_predict = 0
    wrong_predict = 0
    for index, row in test_data_m.iterrows(): #for each row in the dataset
        result = predict(tree, test_data_m.iloc[index]) #predict the row
        if result == test_data_m[label].iloc[index]: #predicted value and expected value is same or not
            correct_predict += 1 #increase correct count
        else:
            wrong_predict += 1 #increase incorrect count
    accuracy = correct_predict / (correct_predict + wrong_predict) #calculating accuracy
    print("Accuracy is", accuracy)
    return accuracy


# In[ ]:





# In[ ]:





# In[ ]:





# In[442]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[453]:


###Part 2 - Problem 2B


train_data_m = pd.read_csv('./car/train.csv', names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"], header = None)
test_data_m = pd.read_csv('./car/test.csv', names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"], header = None)

label = 'label'


# In[456]:



accuracy_dict = {}
max_depth = 6
method = "entropy"
for depth in range(2, 2*(max_depth+1), 2):
    print(depth/2)
    ###Run Algorithm
    tree = id3(train_data_m, label, method, depth)
    acc_test = evaluate(tree, test_data_m, label)
    acc_train = evaluate(tree, train_data_m, label)
    accuracy_dict[depth/2] = (1 - acc_test), (1 - acc_train)
    
print("Car Data - accuraccy_dict for entropy:", accuracy_dict)   


# In[457]:


accuracy_dict = {}
max_depth = 6
method = "ME"
for depth in range(2, 2*(max_depth+1), 2):
    print(depth/2)
    ###Run Algorithm
    tree = id3(train_data_m, label, method, depth)
    acc_test = evaluate(tree, test_data_m, label)
    acc_train = evaluate(tree, train_data_m, label)
    accuracy_dict[depth/2] = (1 - acc_test), (1 - acc_train)
    
print("Car Data - accuraccy_dict for Majority Error:", accuracy_dict)  


# In[458]:


accuracy_dict = {}
max_depth = 6
method = "GI"
for depth in range(2, 2*(max_depth+1), 2):
    print(depth/2)
    ###Run Algorithm
    tree = id3(train_data_m, label, method, depth)
    acc_test = evaluate(tree, test_data_m, label)
    acc_train = evaluate(tree, train_data_m, label)
    accuracy_dict[depth/2] = (1 - acc_test), (1 - acc_train)
    
print("Car Data - accuraccy_dict for Gini Index:", accuracy_dict)  


# In[459]:


###Part 2 - Problem 3a

#Bank Data
train_data_m = pd.read_csv('./bank/train.csv', names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "Output variable"], header = None)

test_data_m = pd.read_csv('./bank/test.csv', names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "Output variable"], header = None)

#add label
label = "Output variable"

#binarize data
train_data_m = numer_to_binary(train_data_m, label)
test_data_m =numer_to_binary(test_data_m, label)

accuracy_dict = {}
max_depth = 16
method = "entropy"
for depth in range(2, 2*(max_depth+1), 2):
    print(depth/2)
    ###Run Algorithm
    tree = id3(train_data_m, label, method, depth)
    acc_test = evaluate(tree, test_data_m, label)
    acc_train = evaluate(tree, train_data_m, label)
    accuracy_dict[depth/2] = (1 - acc_test), (1 - acc_train)
    
print("Bank Data - accuraccy_dict for entropy with unknowns:", accuracy_dict) 


# In[460]:


accuracy_dict = {}
max_depth = 16
method = "ME"
for depth in range(2, 2*(max_depth+1), 2):
    print(depth/2)
    ###Run Algorithm
    tree = id3(train_data_m, label, method, depth)
    acc_test = evaluate(tree, test_data_m, label)
    acc_train = evaluate(tree, train_data_m, label)
    accuracy_dict[depth/2] = (1 - acc_test), (1 - acc_train)
    
print("Bank Data - accuraccy_dict for Majority Error with unknowns:", accuracy_dict) 


# In[461]:


accuracy_dict = {}
max_depth = 16
method = "GI"
for depth in range(2, 2*(max_depth+1), 2):
    print(depth/2)
    ###Run Algorithm
    tree = id3(train_data_m, label, method, depth)
    acc_test = evaluate(tree, test_data_m, label)
    acc_train = evaluate(tree, train_data_m, label)
    accuracy_dict[depth/2] = (1 - acc_test), (1 - acc_train)
    
print("Bank Data - accuraccy_dict for Gini Index with unknowns:", accuracy_dict) 


# In[462]:


###Part 2 - Problem 3b



train_data_m = pd.read_csv('./bank/train.csv', names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "Output variable"], header = None)

test_data_m = pd.read_csv('./bank/test.csv', names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "Output variable"], header = None)

#add label
label = "Output variable"

#replace unknowns
train_data_m = replace_unknowns(train_data_m, label)

#binarize data
train_data_m = numer_to_binary(train_data_m, label)
test_data_m =numer_to_binary(test_data_m, label)

accuracy_dict = {}
max_depth = 16
method = "entropy"
for depth in range(2, 2*(max_depth+1), 2):
    print(depth/2)
    ###Run Algorithm
    tree = id3(train_data_m, label, method, depth)
    acc_test = evaluate(tree, test_data_m, label)
    acc_train = evaluate(tree, train_data_m, label)
    accuracy_dict[depth/2] = (1 - acc_test), (1 - acc_train)
    
print("Bank Data - accuraccy_dict for entropy without unknowns:", accuracy_dict) 


# In[463]:


accuracy_dict = {}
max_depth = 16
method = "ME"
for depth in range(2, 2*(max_depth+1), 2):
    print(depth/2)
    ###Run Algorithm
    tree = id3(train_data_m, label, method, depth)
    acc_test = evaluate(tree, test_data_m, label)
    acc_train = evaluate(tree, train_data_m, label)
    accuracy_dict[depth/2] = (1 - acc_test), (1 - acc_train)
    
print("Bank Data - accuraccy_dict for ME without unknowns:", accuracy_dict) 


# In[464]:


accuracy_dict = {}
max_depth = 16
method = "GI"
for depth in range(2, 2*(max_depth+1), 2):
    print(depth/2)
    ###Run Algorithm
    tree = id3(train_data_m, label, method, depth)
    acc_test = evaluate(tree, test_data_m, label)
    acc_train = evaluate(tree, train_data_m, label)
    accuracy_dict[depth/2] = (1 - acc_test), (1 - acc_train)
    
print("Bank Data - accuraccy_dict for GI without unknowns:", accuracy_dict) 


# In[ ]:





# In[ ]:




