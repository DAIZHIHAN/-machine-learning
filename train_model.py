# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:08:17 2019

@author: Troy

"""
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score  
from sklearn.tree import DecisionTreeClassifier



df=pd.read_excel("D:/Users/Troy/Desktop/-machine-learning/data.xlsx",header=None)

y=df.iloc[1:,16]  

y=LabelEncoder().fit_transform(y)

x=df.iloc[1:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]] #這邊是條X-Y軸的特徵 此資料有16個維度

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)#切割80%,20%

def pca_Dimensionality_reduction(x_train,x_test):
    pca = PCA(n_components=6)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    return x_train,x_test

def lda_Dimensionality_reduction(x_train,x_test,y_train):
    lda = LinearDiscriminantAnalysis(n_components=6)
    lda.fit(x_train, y_train)
    x_train = lda.transform(x_train)
    x_test = lda.transform(x_test)
    return x_train,x_test,lda
    
def TSNE_Dimensionality_reduction(x_train,x_test):
    x_train = TSNE(n_components=2).fit_transform(x_train)
    x_test = TSNE(n_components=2).fit_transform(x_test)
    return x_train,x_test
def KNeighbors_fit(x_train, y_train):
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x_train, y_train)
    return neigh

def Decision_Trees(x_train, y_train):
    tree = DecisionTreeClassifier().fit(x_train, y_train)
    return tree

def svm_cross_validation(train_x, train_y):    
    from sklearn.model_selection import GridSearchCV 
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10,20,30, 100, 1000], 'gamma': [0.001, 0.0001,0.5,0.4,0.1]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(train_x, train_y)    
    return model

def svm_fit(x_train,y_train):
    svm=SVC(kernel='rbf',random_state=0,gamma=0.5,C=5) #這邊條參數
    svm.fit(x_train,y_train)
    return svm

def Visualization(x_in,y_in):
    ax = plt.axes()
    colors = ['red', 'blue', 'green','cyan','yellow','black','magenta']
    for i in range(len(colors)):
        x = x_in[:,0][y_in == i]   
        y = x_in[:,1][y_in == i] 
        ax.scatter(x, y,c=colors[i])
    

    
if __name__=="__main__":
    #訓練加測試
    x_train,x_test,lda=lda_Dimensionality_reduction(x_train,x_test,y_train)
    Visualization(x_train,y_train)
    model=svm_fit(x_train,y_train) 
    y_pred=model.predict(x_test)  
    kk=(y_test!=y_pred).sum()           
    print('Accuracy: {:f} %'.format((1-(kk/y_test.shape[0]))*100)) #判斷錯誤率   

    #驗證
    # df2=pd.read_excel('D:/Users/Troy/Desktop/-machine-learning/test.xlsx')
    # y_v=df2.iloc[1:,16]
    # y_v=LabelEncoder().fit_transform(y_v) 
    # reduced_data = lda.transform(df2.iloc[1:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
    # y_pred2=model.predict(reduced_data) 
    # kk=(y_v!=y_pred2).sum()           
    # print('Accuracy: {:f} %'.format((1-(kk/y_v.shape[0]))*100)) #判斷錯誤率      

    


