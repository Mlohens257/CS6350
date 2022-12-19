#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.support.ui import Select
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.keys import Keys


# In[2]:


import time
import pandas as pd
import numpy as np
from random import randint


# In[3]:


#DATA CREATION

#RETRIEVE DATA - ONLY NEED TO RUN THIS ONCE - IF YOU HAVE THE DATASTORED ON YOUR LOCAL MACHINE 
#YOU DON'T NEED TO RUN AGAIN.

#This is commented out to make it easier for the report graders.


# driver = webdriver.Chrome(ChromeDriverManager().install())

# #Go to the SEC website.
# driver.get("https://www.sec.gov/dera/data/form-345")



# #Use BeautifulSoup to get information from the page
# content = driver.page_source
# soup = BeautifulSoup(content)


# href_dict = {}
# index = 0
# #Store data
# prev_href = ''

# #create list of links to data
# for a in soup.find_all('a', href=True):

#     if a['href'][:58] == "/files/structureddata/data/insider-transactions-data-sets/":
#         if prev_href != a['href']: #prevent duplicates links
#             prev_href = a['href']
#             href_dict[index] = a['href']
#             index += 1
            
#             print(a['href'][-18:-4])

            
# #Download information from each link and unzip to a unique folder on local machine            
# import requests, zipfile, io

# folder_list = {}
# for i in href_dict:


#     address = 'https://www.sec.gov' + href_dict[i]
    
#     folder = href_dict[i][-18:-4]
#     folder_list[i] = folder 
#     r = requests.get(address)
#     z = zipfile.ZipFile(io.BytesIO(r.content))
#     z.extractall("./data/" + folder)
#     print("folder:", folder)
    


# In[4]:


#DATA CREATION

#Identify files   
import glob 
import os

#Identify files
list_of_files = glob.glob('./data/*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
#print(latest_file)

#get the list of folders. Folders contain data for each quarter dating back to 2006.

for i, j in enumerate(os.walk('./data/')):
    #print("i:", i)
    #print(" ")
    if i == 0:
        folder_list = j[1]
        
    #print("i[1]:", i[1])

    #print("j:", j)
    #print(" ")
folder_list.sort()


#Breaks the list into three parts because the program was having trouble storying all the information into one big
#dataframe

folder_list1 = folder_list[:15] #break list into parts because the follow code kept crashing when trying to do it all at once.
folder_list2 = folder_list[15:30]
folder_list3 = folder_list[30:45]
folder_list4 = folder_list[45:60]
folder_list5 = folder_list[60:]
#len(folder_list1 + folder_list2 + folder_list3)


# In[ ]:





# In[5]:


#DATA CREATION

#create dataframes of the SEC data. Each .tsv file recieved from SEC is uploaded from the local computer
#and stored into its own dataframe.
DERIV_TRANS_DF1 = pd.DataFrame()
SUBMISSION_DF1 = pd.DataFrame()
NONDERIV_TRANS_DF1 = pd.DataFrame()
DERIV_HOLDING_DF1 = pd.DataFrame()
OWNER_SIGNATURE_DF1 = pd.DataFrame()
NONDERIV_HOLDING_DF1 = pd.DataFrame()
REPORTINGOWNER_DF1 = pd.DataFrame()
FOOTNOTES_DF1 = pd.DataFrame()

for subfolder in folder_list5:
    

    print(subfolder)
    list_of_files = glob.glob('./data/'+subfolder+'/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    #print(list_of_files)


    for i in list_of_files:
        file_path = i
        file_name = i.split('/')[3].split('.')[0]
        #print(i.split('/')[3].split('.')[0])
        if file_name == 'DERIV_TRANS':
            DERIV_TRANS_df = pd.read_csv(i,sep='\t')
            DERIV_TRANS_DF1 = pd.concat([DERIV_TRANS_DF1, DERIV_TRANS_df], axis=0)
        if file_name == 'SUBMISSION':
            SUBMISSION_df = pd.read_csv(i,sep='\t')
            SUBMISSION_DF1 = pd.concat([SUBMISSION_DF1, SUBMISSION_df], axis=0)
        if file_name == 'NONDERIV_TRANS':
            NONDERIV_TRANS_df = pd.read_csv(i,sep='\t')
            NONDERIV_TRANS_DF1 = pd.concat([NONDERIV_TRANS_DF1, NONDERIV_TRANS_df], axis=0)
        if file_name == 'DERIV_HOLDING':
            DERIV_HOLDING_df = pd.read_csv(i,sep='\t')
            DERIV_HOLDING_DF1 = pd.concat([DERIV_HOLDING_DF1, DERIV_HOLDING_df], axis=0)
        if file_name == 'OWNER_SIGNATURE':
            OWNER_SIGNATURE_df = pd.read_csv(i,sep='\t')
            OWNER_SIGNATURE_DF1 = pd.concat([OWNER_SIGNATURE_DF1, OWNER_SIGNATURE_df], axis=0)
        if file_name == 'NONDERIV_HOLDING':
            NONDERIV_HOLDING_df = pd.read_csv(i,sep='\t')
            NONDERIV_HOLDING_DF1 = pd.concat([NONDERIV_HOLDING_DF1, NONDERIV_HOLDING_df], axis=0)
        if file_name == 'REPORTINGOWNER':
            REPORTINGOWNER_df = pd.read_csv(i,sep='\t')
            REPORTINGOWNER_DF1 = pd.concat([REPORTINGOWNER_DF1, REPORTINGOWNER_df], axis=0)
        if file_name == 'FOOTNOTES':
            FOOTNOTES_df = pd.read_csv(i,sep='\t')
            FOOTNOTES_DF1 = pd.concat([FOOTNOTES_DF1, FOOTNOTES_df], axis=0)


# In[6]:


DERIV_TRANS_df


# In[ ]:





# In[7]:


#DATA CREATION

#Get Stock Symbol to be combine with Non Derivative data later
df = SUBMISSION_DF1[["ACCESSION_NUMBER", "ISSUERTRADINGSYMBOL"]].set_index('ACCESSION_NUMBER')
df


#Add Trading Symbol to the nonderiv_trans_df
df1_nonderiv = NONDERIV_TRANS_DF1.set_index('ACCESSION_NUMBER').merge(df, left_index=True, right_index=True).reset_index()


# In[ ]:





# In[8]:


#Helper Function

from time import process_time
def timer(func):
    t1_start = time.time()
    func
    t1_stop = time.time()
    
    #print(func)
    print("The process took", t1_stop - t1_start, "seconds.")
    return func


# In[9]:


#DATA CREATION


import pandas_datareader as pdr
import datetime 


import datetime 
import numpy as np
import pandas as pd

#This function returns the 1 year percentage change 


def pct_chg_year(transaction):
    


    ###INPUTS

    month_dict = {"JAN": 1,"FEB": 2,"MAR": 3,"APR": 4,"MAY": 5,"JUN": 6,"JUL": 7,"AUG": 8,"SEP": 9,"OCT": 10,"NOV": 11,"DEC": 12}

    date = transaction["TRANS_DATE"]
    ticker = transaction["ISSUERTRADINGSYMBOL"]

    #print(date)
    
    
    
    Day = int(date.split("-")[0])
    Month = int(month_dict[date.split("-")[1]])
    Year = int(date.split("-")[2])

    #print("Day:", Day)
    #print("Month:", Month)
    #print("Year:", Year)
    #print(datetime.datetime(Year+1, Month, Day))

    try:  #This code is used to handle leap year errors. For example 29-FEB-2008 is a 
          #valid date, but 29-FEB-2009 is not.  
        testDate = datetime.datetime(Year+1,Month,Day)
        correctDate = True
        #print("try Entered")
    except ValueError:
        correctDate = False
        Day = Day - 1
        #print("except Entered: Data has been adjusted to account for leap year")

    stock = pdr.get_data_yahoo(ticker, 
                              start=datetime.datetime(Year, Month, Day), 
                              end=datetime.datetime(Year+1, Month, Day))   

    yr_pctchng = (float(stock.tail(1)["Adj Close"])/float(stock.head(1)["Adj Close"])) -1
    return yr_pctchng
#transaction


# In[ ]:





# In[10]:


#DATA CREATION

#save dataframe to csv

df1_nonderiv
filepath = './data/' + 'df1_nonderiv.csv'

timer(df1_nonderiv.to_csv(filepath))
df1_nonderiv


# In[11]:


#DATA CREATION

#copy df to another variable

df1_nonderiv_ = df1_nonderiv
df1_nonderiv


# In[12]:


#DATA CREATION


#This kernel creates the dataset by appending the new_column (One year percentage change). Adding this information 
#takes alot of time and the dataset is huge. Therefore it was useful to add code that was able to store and retreive
#the most uptodate dataset. The code works as follows
#-Check if the new_column has been added to the saved dataset(stored in filepath)
#-If it hasn't then start from scratch add the new column and start adding data to the column.
#-If it has then, then use the saved dataset.
#-Looking at the dataset in 10 point chucks, determine if the new_column has already been populated
#--If it has then skip to the next 10 points
#--If it has not then add the data.

new_column = "1YEARPCTCHNG"

desired_datapoints = 80000   #desired datapoints to be created.

#Storing the dataset with out the new column to be compared with the saved dataset later.
df1_nonderiv = NONDERIV_TRANS_DF1.set_index('ACCESSION_NUMBER').merge(df, left_index=True, right_index=True).reset_index()
df1_nonderiv_ = df1_nonderiv

#Specify we're the saved dataset will be stored to and retreived from.
filepath = './data/' + 'df1_nonderiv_.csv'
#print(df1_nonderiv_)

#create NaN array that can be added to the new column 
NaN_array = np.empty(df1_nonderiv_.shape[0])
NaN_array[:] = np.NaN
NaN_array 

#upload stored dataset. If the dataset hasn't been stored yet, then you'll need to comment this out initially.    
df1_nonderiv_1 = pd.read_csv(filepath,index_col=0)

#print(df1_nonderiv_1)
#print("df1_nonderiv_1.shape != df1_nonderiv_.shape:", df1_nonderiv_1.shape != df1_nonderiv_.shape)

#Check if the new_column has been added to the saved dataset(stored in filepath)
#if df1_nonderiv_1.shape != df1_nonderiv_.shape:  #If the datasets aren't the same shape it is assumed that
if new_column in df1_nonderiv_1.columns: #check if                                                  #

    print("Start with the stored dataset.")
    df1_nonderiv_ = df1_nonderiv_1
else:

    print("Build dataset from scratch.")


    df1_nonderiv_[new_column] = NaN_array



print(df1_nonderiv_)

for j in range(0,int(desired_datapoints/10)):

    t1_start = time.time()
    
    print("j=", j)

    
    if str(pd.DataFrame(NaN_array)[0].unique()[0]) == str(df1_nonderiv_.iloc[j*10:j*10 + 10][new_column].unique()[0]):
        for i in range(j*10, j*10 + 10):
            transaction = df1_nonderiv_.iloc[i]
            
            
            try:  #sometimes the yearly percentage change return an error. I assign a the average median in that case.
                yr_pctchange = pct_chg_year(transaction)

                transaction[new_column] = yr_pctchange
            except:
                transaction[new_column] = np.mean(df1_nonderiv_[new_column])
            #print(i)

            df1_nonderiv_.loc[i] = transaction[0:]
            #print("Building Dataset")

    
        #print(df1_nonderiv_)
    
        df1_nonderiv_.to_csv(filepath)
    t1_stop = time.time()
    
    #print("This took", t1_stop - t1_start,"seconds")
    
    #print("index=", i)


# In[13]:


df1_nonderiv_.shape


# In[14]:


#DATA CREATION

#create list of unique stocks
unique_stock_list = df1_nonderiv_.dropna(subset=['1YEARPCTCHNG']).dropna(subset=['ISSUERTRADINGSYMBOL'])["ISSUERTRADINGSYMBOL"].unique()

acceptable_stock_list = []

for stock in unique_stock_list:
    stock_count = df1_nonderiv_.dropna(subset=['1YEARPCTCHNG'])["ISSUERTRADINGSYMBOL"].str.count(stock).sum()
    
    if stock_count > 25:  #Only include stocks that have more than XX data points
        acceptable_stock_list.append(stock)  #stock all acceptable stocks
        print(stock, ":", stock_count)
acceptable_stock_list


# In[15]:


#DATA CREATION

#filter dataframe to only include acceptable stocks and drop NaN variables
df1_nonderiv_acceptable = df1_nonderiv_.dropna(subset=['1YEARPCTCHNG'])[df1_nonderiv_['ISSUERTRADINGSYMBOL'].isin(acceptable_stock_list)]


# In[16]:


print("Max data date: ",df1_nonderiv_acceptable["TRANS_DATE"].max())


# In[17]:


print("Min data date: ",df1_nonderiv_acceptable["TRANS_DATE"].min())


# In[18]:


#DATA CREATION

#shuffle data frame
df1_nonderiv_acceptable_sh = df1_nonderiv_acceptable.sample(frac=1)


# In[19]:


df1_nonderiv_acceptable_sh.shape


# In[20]:


#DATA CREATION

#preprocess data

from sklearn import preprocessing

from sklearn.preprocessing import OrdinalEncoder


def preprocess_SEC_insider_data(X):
    
    #fill NaN values with 'NaN'
    X = X.fillna('NaN')  
    
    
    for column_name in X:
        
        #print(column_name)
        #print(X[column_name].dtype)
        
        if column_name == "EQUITY_SWAP_INVOLVED":
            X[column_name] = X[column_name].astype('str')
        
        
        if X[column_name].dtype == "object": #if the column dtype is an object then encode it so that SKlearn can use it
                                             #SKlearn only allows numerical features.
            le = preprocessing.LabelEncoder()
#             enc = OrdinalEncoder()
            #print(X[column_name].unique())
            try:
                temp = le.fit_transform(X[column_name])
#                 temp = enc.fit_transform(X[column_name])
                
                X[column_name] = temp
                #print('try')
            except:
                #print("Exception was taken. NaN was replaced with the median data(Not including Nan).")
                #print(X[column_name])
                temp = le.fit_transform(X[column_name].replace('NaN', np.mean(X[column_name][X[column_name] != 'NaN'])))
                #temp = enc.fit_transform(X[column_name].replace('NaN', np.mean(X[column_name][X[column_name] != 'NaN'])))
                
                X[column_name] = temp
            
    
  
    return X
#le


# In[ ]:





# In[ ]:





# In[21]:


### REMOVE TRANS_DATE to as future predictions can't know how the current transaction date will affect the future.   
##Remove unnessary information specifically the ACCESSION_NUMBER and NONDERIV_TRANS_SK
#important features were discovered by a random forest tree to get the following features



data_columns =  ['SECURITY_TITLE', 'TRANS_CODE',
       'EQUITY_SWAP_TRANS_CD_FN', 'TRANS_SHARES', 'TRANS_SHARES_FN',
       'TRANS_PRICEPERSHARE', 'TRANS_ACQUIRED_DISP_CD',
       'SHRS_OWND_FOLWNG_TRANS', 'SHRS_OWND_FOLWNG_TRANS_FN',
       'NATURE_OF_OWNERSHIP', 'ISSUERTRADINGSYMBOL']

data_columns =  ['TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'SHRS_OWND_FOLWNG_TRANS','ISSUERTRADINGSYMBOL']

#allocate first 80% of the data to training

X = df1_nonderiv_acceptable_sh.iloc[:int(df1_nonderiv_acceptable_sh.shape[0]*.8), :].loc[:,data_columns]
X
y = df1_nonderiv_acceptable_sh.iloc[:int(df1_nonderiv_acceptable_sh.shape[0]*.8),:].loc[:,'1YEARPCTCHNG':]
#print(float(np.sum((y > 0)*1)/len(y))*100, "percent are positive returns after one year of the insiders purchase.")



X_train = preprocess_SEC_insider_data(X)

y['1YEARPCTCHNG'] = (y['1YEARPCTCHNG'] > 0)*1
y_train = y.to_numpy().reshape(len(y),)


#allocate last 20% of the data to testing
#process test data
X_test = df1_nonderiv_acceptable_sh.iloc[int(df1_nonderiv_acceptable_sh.shape[0]*.8):, :].loc[:,data_columns]
#X_test
y_test = df1_nonderiv_acceptable_sh.iloc[int(df1_nonderiv_acceptable_sh.shape[0]*.8):,:].loc[:,'1YEARPCTCHNG':]
#y_test

X_test = preprocess_SEC_insider_data(X_test)


# In[ ]:





# In[22]:


print("Training input data is size:", X_train.shape)
print("Testing input data is size:", X_test.shape)


# In[23]:


print("Total input data is size:", X_test.shape[0] +X_train.shape[0], ",", X_test.shape[1] )


# In[24]:


X_test.shape[0] +X_train.shape[0]


# In[25]:



print("The data's percentage of positive 1 year percentage change is", (np.sum(y_train) + np.sum(((y_test > 0)*1).reset_index()['1YEARPCTCHNG']))/(X_test.shape[0] +X_train.shape[0]))


# In[26]:


# %matplotlib inline
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np


# df = X_train.copy()
# df["label"] = y_train
# df

#Update df to remove outliers
#updated_index = df.index[(df.TRANS_PRICEPERSHARE *df.TRANS_SHARES < 100000000)].tolist()
#df = df.iloc[updated_index,:]

##Visualize features relationship to eachother

# for i in df.columns:
        
#     column_x = 'ISSUERTRADINGSYMBOL'
#     #for j in df.columns:
#     column_y = i

#     plt.figure(figsize=(12, 9))
#     plt.scatter(df[column_x], df[column_y], s=4, c=df.label)
#     #plt.legend()
#     plt.xlabel(column_x)
#     plt.ylabel(column_y)
#     plt.title('Comparison of {0} and {1}'.format(column_x, column_y))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#train SVM

from sklearn import svm

clf = svm.SVC()
clf.fit(X_train, y_train)



#predict with SVM


y_pred = clf.predict(X_train).copy()
y_actual = ((y_train > 0)*1).copy()
print("The accuracy of a SVM on the TRAIN data is", np.sum((y_pred == y_actual)*1)/len(y_pred))

y_pred = clf.predict(X_test).copy()
y_actual = ((y_test > 0)*1).reset_index()['1YEARPCTCHNG'].copy()
print("The accuracy of a SVM on the TEST data is", np.sum((y_pred == y_actual)*1)/len(y_pred))

print("Percentage of positive predictions:",sum(y_pred)/y_pred.shape[0])


#Result Note:
##Seems to get a train accuracy  of 55.7% and a test accuracy of 56.4%


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#Decision Tree
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=30)
clf = clf.fit(X_train, y_train)


#predict

y_pred_train = clf.predict(X_train).copy()
y_actual_train = ((y_train > 0)*1).copy()
print("The accuracy of a Decision Tree on the TRAIN data is", np.sum((y_pred_train == y_actual_train)*1)/len(y_pred_train))

y_pred = clf.predict(X_test).copy()
y_actual = ((y_test > 0)*1).reset_index()['1YEARPCTCHNG'].copy()
print("The accuracy of a Decision Tree on the TEST data is", np.sum((y_pred == y_actual)*1)/len(y_pred))

print("Percentage of positive predictions:",sum(y_pred)/y_pred.shape[0])


#Result Note:
##Seems to get a train accuracy  of 100% and a test accuracy of 57.2%


# In[ ]:





# In[ ]:



#Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)


#predict

y_pred_train = clf.predict(X_train).copy()
y_actual_train = ((y_train > 0)*1).copy()
print("The accuracy of a Random Forest on the TRAIN data is", np.sum((y_pred_train == y_actual_train)*1)/len(y_pred_train))

y_pred = clf.predict(X_test).copy()
y_actual = ((y_test > 0)*1).reset_index()['1YEARPCTCHNG'].copy()
print("The accuracy of a Random Forest on the TEST data is", np.sum((y_pred == y_actual)*1)/len(y_pred))

print("Percentage of positive predictions:",sum(y_pred)/y_pred.shape[0])


#Result Note:
##Seems to get a train accuracy  of 100% and a test accuracy of 71.5%


# In[ ]:


important_columns = X_train.columns[(clf.feature_importances_ > 1.0e-01)]
print("Important feature columns according to the RandomForestClassifier:",important_columns)


# In[ ]:


#clf.feature_importances_


# In[ ]:





# In[ ]:


#SGD Classifier

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Always scale the input. The most convenient way is to use a pipeline.
clf = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-3))
clf.fit(X_train, y_train)

#predict

y_pred = clf.predict(X_train).copy()
y_actual = ((y_train > 0)*1).copy()
print("The accuracy of a SGD Classifier on the TRAIN data is", np.sum((y_pred == y_actual)*1)/len(y_pred))

y_pred = clf.predict(X_test).copy()
y_actual = ((y_test > 0)*1).reset_index()['1YEARPCTCHNG'].copy()
print("The accuracy of a SGD Classifier on the TEST data is", np.sum((y_pred == y_actual)*1)/len(y_pred))

print("Percentage of positive predictions:",sum(y_pred)/y_pred.shape[0])


#Result Note:
##Seems to get a train accuracy  of 57.78% and a test accuracy of 58.06%


# In[ ]:





# In[ ]:


#Run a nueral network algorithm using data features that are greater than 0 in importance
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(30,), random_state=1, max_iter = 20000)

X_train_imp_feat = X_train[important_columns]
X_test_imp_feat = X_test[important_columns]

clf.fit(X_train_imp_feat, y_train)


y_pred = clf.predict(X_train_imp_feat).copy()
y_actual = ((y_train > 0)*1).copy()
print("The accuracy of a Nueral Net on the TRAIN data is", np.sum((y_pred == y_actual)*1)/len(y_pred))

y_pred = clf.predict(X_test_imp_feat).copy()
y_actual = ((y_test > 0)*1).reset_index()['1YEARPCTCHNG'].copy()
print("The accuracy of a Nueral Net on the TEST data is", np.sum((y_pred == y_actual)*1)/len(y_pred))

print("Percentage of positive predictions:",sum(y_pred)/y_pred.shape[0])


#Result Note:
##Seems to get a train accuracy  of 55.1% and a test accuracy of 54.7%


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




