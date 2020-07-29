from residual_3dcnn import *
import numpy as np
from torch.utils.data import DataLoader
from torch.utils import data
import os
import numpy as np
import datetime
import torch.nn as nn
import torch.optim as optim
class Neg_Pearson(nn.Module):   
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):       
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])             
            sum_y = torch.sum(labels[i])               
            sum_xy = torch.sum(preds[i]*labels[i])      
            sum_x2 = torch.sum(torch.pow(preds[i],2))  
            sum_y2 = torch.sum(torch.pow(labels[i],2))
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
        if (pearson>=0):
            loss += 1 - pearson
        else:
            loss += 1 - torch.abs(pearson)
        #loss += 1 - pearson 
        loss = loss/preds.shape[0]
        return loss
class Mydata(data.Dataset):
    def __init__(self,x_train_path=True,y_train_path=True):
        x_train = [os.path.join(x_train_path,x) for x in os.listdir(x_train_path)]
        y_train = [os.path.join(y_train_path,y) for y in os.listdir(y_train_path)]
        self.x_train = x_train
        self.y_train = y_train
    def __getitem__(self,index):
        y_path = self.y_train[index]
        label = np.load(y_path)
        
        x_path = self.x_train[index]
        data = np.load(x_path)
        data = data.reshape(3,128,64,64)
        return data, label    
    def __len__(self):
        return len(self.x_train)

path1 = ''
path2 = ''
train_data = Mydata(path1,path2)

data_loader = DataLoader(train_data,batch_size=4,shuffle=True)
model = Net().cuda()
cost = Neg_Pearson()
optimizer = optim.Adam(model.parameters(),lr=0.0002)
start = datetime.datetime.now()
for i in range(30):
    running_loss = 0.0
    print('-----epoch', i, '-----')
    for x, (x_train, y_train) in enumerate(data_loader):
        x_train, y_train = x_train.cuda(), y_train.cuda()
        output1,output2 = model(x_train)
        
        output1 = (output1-torch.mean(output1)) /torch.std(output1)
        output2 = (output2-torch.mean(output2)) /torch.std(output2)
        
        loss1 = cost(output1, y_train)
        loss2 = cost(output2, y_train)
        loss = 0.5*loss1+loss2
#         if x%100==0:
#             print('400个样本花费的时间：',datetime.datetime.now()-start)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data
    print('loss:%.4f'%running_loss.data)
print(datetime.datetime.now()-start)
torch.save(model, 'rppg.pkl')