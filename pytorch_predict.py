# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 14:36:30 2019

@author: Troy
"""
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch.utils.data as Data
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),           
            )
        self.hidden3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),            
            )
        self.hidden4 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),            
            )
        self.hidden5 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(), 
            
            )        
        self.hidden6 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(), 
            
            )   
        self.output = nn.Sequential(
            nn.Linear(512, 7),
            
            
            )

    def forward(self, x):
           
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.output(x)  
       
        return x
    
model = Model()
model.load_state_dict(torch.load("./model.pkl"))

df=pd.read_excel("D:/Users/Troy/Desktop/流量計分類(ML)/test_init.xlsx",header=None)

#標籤處理
y_v=df.iloc[1:,16]
y_v=LabelEncoder().fit_transform(y_v)

x=df.iloc[1:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
x=np.array(x,dtype=np.float32)

x=Variable(torch.from_numpy(x)) #要把他包成variable才能進行反向傳播

outputs = model(x.float())

_,preds_tensor=torch.max(outputs,1)
preds=np.squeeze(preds_tensor.numpy())

print('Predicted',preds[:])
k=(y_v!=preds).sum()           
print('Accuracy: {:f} %'.format((1-(k/y_v.shape[0]))*100)) #判斷錯誤率   


















