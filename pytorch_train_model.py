
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

train_number=10

df=pd.read_excel("D:/Users/Troy/Desktop/流量計分類(ML)/data_init.xlsx",header=None)

y=df.iloc[1:,16]  #抓取該網站前100筆資料蘭花的名稱

y=LabelEncoder().fit_transform(y)

x=df.iloc[1:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]] #這邊是條X-Y軸的特徵 此資料有16個維度

x=np.array(x,dtype=np.float32)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


x_train=Variable(torch.from_numpy(x_train)) 

y_train=torch.from_numpy(y_train)
y_train = torch.tensor(y_train, dtype=torch.long)
y_train=Variable(y_train)


x_test=Variable(torch.from_numpy(x_test))

y_test=torch.from_numpy(y_test)
y_test = torch.tensor(y_test, dtype=torch.long)
y_test=Variable(y_test)


torch_dataset_train = Data.TensorDataset(x_train, y_train)
torch_dataset_test = Data.TensorDataset(x_test, y_test)

trainloader = DataLoader(dataset=torch_dataset_train, batch_size=512,shuffle=True)
validloader = DataLoader(dataset=torch_dataset_test, batch_size=512,shuffle=True)

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

learning_rate=0.001
model = Model()
model.train()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

losses,train_accuracy,test_accuracy=[],[],[]

train_loss = []

for epoch in range(1, train_number): 
    correct1=0
    total1 = 0
    for data, target in trainloader:
        
        optimizer.zero_grad()
        
        output = model(data.float())
        
        _, predicted = torch.max(output.data, 1) 
        
        total1 += target.size(0)
        
        correct1 += (predicted == target).sum()
        
        loss = loss_function(output, target)
        
        loss.backward()
            
        optimizer.step()
        
    train_accuracy.append(100 * correct1 / total1)
    
    losses.append(loss.item())
    
    print('Epoch:', epoch, 'Training Loss: ', loss.item())
       
    with torch.no_grad():
        correct = 0
        total = 0
        
        for images, labels in validloader:           
 
            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            

            correct += (predicted == labels).sum()
        
        test_accuracy.append(100 * correct / total)
        
        print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


torch.save(model.state_dict(), "./model.pkl")   


fig = plt.figure()

plt.subplot(2,1,1)
x1 = range(0,train_number-1)
x2 = range(0,train_number-1)

line1, = plt.plot(x1, train_accuracy, color = 'red')

line2, = plt.plot(x2, test_accuracy, color = 'blue')

plt.title('train and test accuracy and loss')
plt.xlabel('epoch')
plt.ylabel('Accuracy (%)')

plt.subplot(2,1,2)
x3 = range(0,train_number-1)
line1, = plt.plot(x3, losses, color = 'blue')

plt.xlabel('epoch')
plt.ylabel('loss')

plt.show()




