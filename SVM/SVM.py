#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import os
os.getcwd()


# In[2]:


#function to shuffle data

from sklearn.utils import shuffle
import numpy as np

def Shuffle(X, y):

    X, y = shuffle(X, y)
    return X, y


# In[3]:


def accuracy(x, y, w):
    return sum((np.sign(x.dot(w)) == y.reshape(-1, )))/y.shape[0]


# In[4]:


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





# In[5]:


#SVM_t is an primal SVM function that performs weight updates for one epoch
#Input: Training data
#Output: Updated weights
def SVM_t(lr, x, w, y, N, C):
    for i in range(N):
      #print("lr[i]:", lr)
      #print("x[i]:", x[i])
      #print("w:", w) 
      #print("y[i]:", y[i])


      if max(0, 1-y[i]*w.dot(x[i])) == 0:
        dj_dw = np.zeros(w.shape)

        #print("max(0, 1-y[i]*w.dot(x[i])) == 0")
      else:
        dj_dw = w-C*N*y[i]*x[i]


        #print("max(0, 1-y[0]*w.dot(x[0])) != 0")
      #print("dj_dw:", dj_dw)
      w = w - lr*dj_dw
      #print("w:", w)  
      #print(" ")  
    #print("w:", w) 
    return w


# In[ ]:





# In[6]:



def SVM_a(x, w, y, N, C, T, lr_0, a):
    for epoch in range(T):
        #shuffle the data
        x, y = Shuffle(x, y)

        #update lr
        lr_t = lr_0/(1 +(lr_0/a)*epoch)

        w = SVM_t(lr_t, x, w, y, N, C)
    #print("Training accuracy is", accuracy(x, y, w))
    return accuracy(x, y, w), w


# In[7]:


#Part 2 Question 2 (a)
#convert data to an array from a dataframe
x_test = np.array(X_test)
x_test = np.hstack((x_test, np.ones([x_test.shape[0], 1])))
x_test.shape

#convert data to an array from a dataframe
y_test = np.array(Y_test)
y_test.shape

#convert data to an array from a dataframe
x_train = np.array(X_train)
x_train = np.hstack((x_train, np.ones([x_train.shape[0], 1])))
x_train.shape

#convert data to an array from a dataframe
y_train = np.array(Y_train)
y_train.shape

#extract shape information for later use
N, D = x_train.shape

#convert data to an array from a dataframe
w = np.array(np.zeros(D))
w.shape
w





T = 100

C_list = {100/873, 500/873, 700/873}
lr_0_list = [.25,.5, 1,2,3,4,5]
a_list = [.25,.5, 1,2,3,4,5]


###


for C in C_list:
    
    max_acc = 0
    for lr_0 in lr_0_list:
        for a in a_list:
            #print("lr_0", lr_0)
            #print("a", a)
            acc, w_epoch = SVM_a(x_train, w, y_train, N, C, T, lr_0, a)
            #print(acc)
            if acc > max_acc:
                max_acc = acc
                lr_0_max = lr_0
                a_max = a
                w_max = w_epoch

                
    print(" ")
    print("Maximum training error is", 100*(1-max_acc), "with the following hyper parameters:")
    print("C", C)
    print("lr_0", lr_0_max)
    print("a", a_max)
    print("w", w_max)
    print(" ")
    print("Testing error with the above parameters is", 100*(1 - accuracy(x_test, y_test, w_max)))


# In[ ]:





# In[ ]:





# In[8]:



def SVM_b(x, w, y, N, C, T, lr_0):
    for epoch in range(T):
        #shuffle the data
        x, y = Shuffle(x, y)

        #update lr
        lr_t = lr_0/(1 +epoch)

        w = SVM_t(lr_t, x, w, y, N, C)
    #print("Training accuracy is", accuracy(x, y, w))
    return accuracy(x, y, w), w


# In[9]:


#Part 2 Question 2 (b)
#convert data to an array from a dataframe
x_test = np.array(X_test)
x_test = np.hstack((x_test, np.ones([x_test.shape[0], 1])))
x_test.shape

#convert data to an array from a dataframe
y_test = np.array(Y_test)
y_test.shape

#convert data to an array from a dataframe
x_train = np.array(X_train)
x_train = np.hstack((x_train, np.ones([x_train.shape[0], 1])))
x_train.shape

#convert data to an array from a dataframe
y_train = np.array(Y_train)
y_train.shape

#extract shape information for later use
N, D = x_train.shape

#convert data to an array from a dataframe
w = np.array(np.zeros(D))
w.shape
w




T = 100

C_list = {100/873, 500/873, 700/873}
lr_0_list = [.25,.5, 1,2,3,4,5]
#a_list = [.25,.5, 1,2,3,4,5]


###


for C in C_list:
    
    max_acc = 0
    for lr_0 in lr_0_list:
        #for a in a_list:
            #print("lr_0", lr_0)
            #print("a", a)
            acc, w_epoch = SVM_b(x_train, w, y_train, N, C, T, lr_0)
            #print(acc)
            if acc > max_acc:
                max_acc = acc
                lr_0_max = lr_0
                #a_max = a
                w_max = w_epoch

                
    print(" ")
    print("Maximum training error is", 100*(1-max_acc), "with the following hyper parameters:")
    print("C", C)
    print("lr_0", lr_0_max)
    #print("a", a_max)
    print("w", w_max)
    print(" ")
    print("Testing error with the above parameters is", 100*(1-accuracy(x_test, y_test, w_max)))


# In[10]:


1-0.9873853211009175


# In[ ]:





# ### Part 2 - Question 2 (c)
# 
# NEED TO UPDATE:
# For the rate schedule from question 2 (a), The different values of C have the following affect:
# 
# lr_0: Tends to increase as C gets larger.
# 
# a: tends to fluctualte more randomly.
# 
# weights: Seem to get larger as C gets larger. This is likely because the minimization of the slack variables have a larger influences as C gets larger.
# 
# training error: Very little change - all about 4%
# 
# testing error: Very little change - all about 5%
# 
# For the rate schedule from question 2 (b), The different values of C have the following affect:
# 
# lr_0: No apparent trend
# 
# weights: Seem to get larger as C gets larger. This is likely because the minimization of the slack variables have a larger influences as C gets larger.
# 
# training error: Very little change - all about 5%
# 
# testing error: Very little change - all about 6%
# 

# In[11]:


#convert data to an array from a dataframe
x_test = np.array(X_test)
#x_test = np.hstack((x_test, np.ones([x_test.shape[0], 1])))
x_test.shape

#convert data to an array from a dataframe
y_test = np.array(Y_test)
y_test.shape

#convert data to an array from a dataframe
x_train = np.array(X_train)
#x_train = np.hstack((x_train, np.ones([x_train.shape[0], 1])))
x_train.shape

#convert data to an array from a dataframe
y_train = np.array(Y_train)
y_train.shape

#extract shape information for later use
N, D = x_train.shape

#convert data to an array from a dataframe
w = np.array(np.zeros(D))
w.shape
w


# In[12]:


#References:
#Mathieu Blondel's Code  http://www.mblondel.org/journal/2010/09/19/support-vector-machines-in-python/


import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
             
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, g=2):
    return np.exp(-linalg.norm(x-y)**2 / (g))

class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None, gamma = None):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        #print("self.a.shape:", self.a.T.shape)
        #print("self.sv.shape:", self.sv.T.shape)
        #print("self.sv_y.shape:", self.sv_y.shape)
        #print("weight:", np.sum(np.multiply(
        #    np.multiply(
        #        self.sv.T, 
        #        self.a),
        #        self.sv_y), axis = 1))
        
        print("C:", self.C)
        print("gamma:", self.gamma)
        
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        print("Support vectors:", np.where(sv)[0])
        

        
        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)
        

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
            #print("weight:", self.w)
        else:
            self.w = None
            #print("weight:", np.sum(np.multiply(
            #np.multiply(
            #    self.sv.T, 
            #    self.a),
            #    self.sv_y), axis = 1))
            
        #print("self.b:", self.b)  


        

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv, self.gamma)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))
    

if __name__ == "__main__":
    import pylab as pl

     
    def bank_note_data(x = x_train, y = y_train):
        
        index_neg = np.where(y < 0)[0]
        index_pos = np.where(y > 0)[0]
        x_train_neg = x[index_neg]
        x_train_pos = x[index_pos]
        X1 = x_train_neg
 
        y1 = -1*np.ones(x_train_neg.shape[0])
 
        X2 = x_train_pos

        y2 = np.ones(x_train_pos.shape[0])

        X_train = np.vstack((X1, X2))
        y_train = np.hstack((y1, y2))
        
        
        return X_train, y_train

 
        
    def test_non_linear(x_train_, y_train_, x_test_, y_test_):
        
        X_train, y_train  = bank_note_data(x = x_train_, y = y_train_)     
        X_test, y_test = bank_note_data(x = x_test_, y = y_test_)

        clf = SVM(C=1000.1)
        clf = SVM(gaussian_kernel)
        clf.fit(X_train, y_train)
        
        
        y_predict = clf.predict(X_train)
        correct = np.sum(y_predict == y_train)
        print("Training Accuracy: %d out of %d predictions correct" % (correct, len(y_predict)), "(",100*(correct/ len(y_predict)) ,"%)")


        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)), "(",100*(correct/ len(y_predict)) ,"%)")



    def test_soft(x_train_, y_train_, x_test_, y_test_):
        X_train, y_train  = bank_note_data(x = x_train_, y = y_train_)     
        X_test, y_test = bank_note_data(x = x_test_, y = y_test_)

        clf = SVM(C=1000.1)
        clf.fit(X_train, y_train)
        
        y_predict = clf.predict(X_train)
        correct = np.sum(y_predict == y_train)
        print("Training Accuracy: %d out of %d predictions correct" % (correct, len(y_predict)), "(",100*(correct/ len(y_predict)) ,"%)")


        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)), "(",100*(correct/ len(y_predict)) ,"%)")

    def test_soft_3a(x_train_, y_train_, x_test_, y_test_):
        
        X_train, y_train  = bank_note_data(x = x_train_, y = y_train_)     
        X_test, y_test = bank_note_data(x = x_test_, y = y_test_)
        for c in {100/873, 500/873, 700/873}:
            print("C:", c)
            clf = SVM(C=c)
            #clf = SVM(kernel = gaussian_kernel)
            clf.fit(X_train, y_train)
            
            
            y_predict = clf.predict(X_train)
            correct = np.sum(y_predict == y_train)
            print("Training Error: %d out of %d predictions correct" % (correct, len(y_predict)), "(",100 - 100*(correct/ len(y_predict)) ,"%)")


            y_predict = clf.predict(X_test)
            correct = np.sum(y_predict == y_test)
            print("Testing Error: %d out of %d predictions correct" % (correct, len(y_predict)), "(",100 -100*(correct/ len(y_predict)) ,"%)")

    def test_non_lin_3b(x_train_, y_train_, x_test_, y_test_):
        
        X_train, y_train  = bank_note_data(x = x_train_, y = y_train_)     
        X_test, y_test = bank_note_data(x = x_test_, y = y_test_)
        for c in {100/873, 500/873, 700/873}:
            for gamma in {0.1,0.5,1,5,100}:
                #print("C:", c)
                #print("gamma:", gamma)
                clf = SVM(C=c, kernel = gaussian_kernel, gamma = gamma)
                #clf = SVM(kernel = gaussian_kernel, gamma = gamma)

                #clf = SVM(gamma = gamma)
                clf.fit(X_train, y_train)


                y_predict = clf.predict(X_train)

                correct = np.sum(y_predict == y_train)
                print("Training Error: %d out of %d predictions correct" % (correct, len(y_predict)), "(",100-100*(correct/ len(y_predict)) ,"%)")


                y_predict = clf.predict(X_test)
                correct = np.sum(y_predict == y_test)
                print("Testing Error: %d out of %d predictions correct" % (correct, len(y_predict)), "(",100-100*(correct/ len(y_predict)) ,"%)")

                
    def test_non_lin_3c(x_train_, y_train_, x_test_, y_test_):
        
        X_train, y_train  = bank_note_data(x = x_train_, y = y_train_)     
        X_test, y_test = bank_note_data(x = x_test_, y = y_test_)
        for c in {100/873, 500/873, 700/873}:
            for gamma in {0.1,0.5,1,5,100}:
                #print("C:", c)
                #print("gamma:", gamma)
                clf = SVM(C=c, kernel = gaussian_kernel, gamma = gamma)
                #clf = SVM(kernel = gaussian_kernel, gamma = gamma)

                #clf = SVM(gamma = gamma)
                clf.fit(X_train, y_train)


                y_predict = clf.predict(X_train)

                correct = np.sum(y_predict == y_train)
                #print("Training Error: %d out of %d predictions correct" % (correct, len(y_predict)), "(",100-100*(correct/ len(y_predict)) ,"%)")


                y_predict = clf.predict(X_test)
                correct = np.sum(y_predict == y_test)
                #print("Testing Error: %d out of %d predictions correct" % (correct, len(y_predict)), "(",100-100*(correct/ len(y_predict)) ,"%)")



    #test_soft(x_train, y_train, x_test, y_test)


# In[ ]:





# In[13]:


test_soft_3a(x_train, y_train, x_test, y_test)


# In[14]:


test_non_lin_3b(x_train, y_train, x_test, y_test)


# In[15]:


test_non_lin_3c(x_train, y_train, x_test, y_test)


# In[ ]:




