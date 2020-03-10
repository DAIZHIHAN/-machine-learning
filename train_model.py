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
        
       
df=pd.read_excel("D:/Users/Troy/Desktop/流量計分類(ML)/data_init.xlsx",header=None)


y=df.iloc[1:,16]  #抓取該網站前100筆資料蘭花的名稱
# y=np.array(y)
# y=np.where(y=='D',1,0) #為D類和其他類
y=LabelEncoder().fit_transform(y)

x=df.iloc[1:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]] #這邊是條X-Y軸的特徵 此資料有16個維度

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

svm=SVC(kernel='rbf',random_state=0,gamma=0.5,C=38) #這邊條參數

svm.fit(x_train,y_train)

y_pred=svm.predict(x_test)  

k=(y_test!=y_pred).sum()           
print('Accuracy: {:f} %'.format((1-(k/y_test.shape[0]))*100)) #判斷錯誤率       

df2=pd.read_excel('D:/Users/Troy/Desktop/流量計分類(ML)/test_init.xlsx')
y_v=df2.iloc[1:,16]
y_v=LabelEncoder().fit_transform(y_v)

y_pred2=svm.predict(df2.iloc[1:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]) 

k=(y_v!=y_pred2).sum()           
print('Accuracy: {:f} %'.format((1-(k/y_v.shape[0]))*100)) #判斷錯誤率    

# lda = LinearDiscriminantAnalysis(n_components=2)
# X_r2 = lda.fit(x_test, y_test).transform(x_test)
# print(X_r2.shape)
ax121 = plt.subplot(1,1,1)

# # X_embedded = TSNE(n_components=3).fit_transform(x_test)

pca = PCA(n_components=2)

# # # # # Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(x_test)

colors = ['red', 'blue', 'green','cyan','yellow','black','magenta']

for i in range(len(colors)):
    x = reduced_data_pca[:,0][y_test == i]
    
    y = reduced_data_pca[:,1][y_test == i]
    # z = reduced_data_pca[:,2][y_test == i]
    ax121.scatter(x, y,c=colors[i])
   
    
# # ax121.set_zlabel('Z-axis') #z軸名稱
    
# # ax121.view_init(10,150)

# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')

plt.title("PCA Scatter Plot")
plt.show()