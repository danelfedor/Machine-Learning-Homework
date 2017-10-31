import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import os
#***core function with sklearn-tree***#
def depth_test(train_X,train_Y,valid_X,valid_Y,max_depth):
    train_err=[]
    valid_err=[]
    for i in range(1,max_depth+1):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        clf = clf.fit(train_X, train_Y)
        pred_train=clf.predict(train_X)
        pred_valid=clf.predict(valid_X)
        train_err.append(sum(pred_train!=train_Y.iloc[:,0])/train_Y.shape[0])
        valid_err.append(sum(pred_valid!=valid_Y.iloc[:,0])/valid_Y.shape[0])
    return train_err,valid_err

#***Data1 Madelon***#
os.chdir(r'G:\Users\Hu Wenqi\Dropbox\Machine Learning Data\binary\MADELON')
train_X=pd.read_csv('madelon_train.data',sep=' ',header=None)
train_Y=pd.read_csv('madelon_train.labels',header=None)
valid_X=pd.read_csv('madelon_valid.data',sep=' ',header=None)
valid_Y=pd.read_csv('madelon_valid.labels',header=None)

train_X=train_X.iloc[:,:-1]
valid_X=valid_X.iloc[:,:-1]


train_err,valid_err=depth_test(train_X,train_Y,valid_X,valid_Y,12)

depth=range(1,13)
plt.figure(figsize=(16,9))  
plt.plot(depth,train_err)
plt.plot(depth,valid_err)

#***Data1 Gisette***#
os.chdir(r'G:\Users\Hu Wenqi\Dropbox\Machine Learning Data\binary\Gisette')
train_X=pd.read_csv('gisette_train.data',sep=' ',header=None)
train_Y=pd.read_csv('gisette_train.labels',header=None)
valid_X=pd.read_csv('gisette_valid.data',sep=' ',header=None)
valid_Y=pd.read_csv('gisette_valid.labels',header=None)
train_X=train_X.iloc[:,:-1]
valid_X=valid_X.iloc[:,:-1]

train_err,valid_err=depth_test(train_X,train_Y,valid_X,valid_Y,6)
depth=range(1,7)

plt.figure(figsize=(16,9))  
plt.plot(depth,train_err)
plt.plot(depth,valid_err)

#***Minibone***#
os.chdir(r'G:\Users\Hu Wenqi\Dropbox\Machine Learning Data\binary\Miniboone')
data_r=pd.read_csv(r'MiniBooNE_PID.txt',header=None)
num_Y=data_r.iloc[0,:][0].split(' ')[1:]
Y_1=np.zeros(int(num_Y[0]))+1
Y_0=np.zeros(int(num_Y[1]))
Y=Y_1.tolist()
Y.extend(Y_0.tolist())
X=[]
for i in range(1,data_r.shape[0]):
    a=data_r.iloc[i,0].split(' ')
    temp=[]     
    for j in a:
        if j !=' ' and j!='':
           temp.append(float(j)) 
    X.append(temp)

X=np.array(X)
Y=np.array(Y)

def cross_val(X,Y,max_depth):
    length=int(len(Y)/4)
    rand_ind=np.random.choice(range(len(Y)),size=len(Y), replace=False)
    for i in range(4):
        valid_X=pd.DataFrame(X[rand_ind[length*i:length*(i+1)]])
        valid_Y=pd.DataFrame(Y[rand_ind[length*i:length*(i+1)]])
        train_X=X[rand_ind[0:length*i],:].tolist()
        train_X.extend(X[rand_ind[length*(i+1):]].tolist())
        train_X=pd.DataFrame(train_X)
        train_Y=Y[rand_ind[0:length*i]].tolist()
        train_Y.extend(Y[rand_ind[length*(i+1):]].tolist())
        train_Y=pd.DataFrame(train_Y)
        train_err,valid_err=depth_test(train_X,train_Y,valid_X,valid_Y,max_depth)#this line change with request mathod
        depth=range(1,max_depth+1)
        plt.figure(figsize=(16,9))  
        plt.plot(depth,train_err)
        plt.plot(depth,valid_err)

cross_val(X,Y,6)    
    
    

