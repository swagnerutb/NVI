import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys #not really necessary

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def load_MNIST_data():
    train_ = datasets.MNIST('', train=True, download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))
    test_ = datasets.MNIST('', train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))
    trainset_ = torch.utils.data.DataLoader(train_, batch_size=10, shuffle=True)
    testset_ = torch.utils.data.DataLoader(test_, batch_size=10, shuffle=False)
    return trainset_, testset_


class set_dataset(Dataset):
    def __init__(self, data_to_use):
        path_sked_tot = '/Users/sam/Desktop/NVI/proj4/ut1_d10_00h.txt'
        df_sked_tot = pd.read_csv(path_sked_tot,sep='\s+',header=None)
        df_sked_tot.rename(columns={8:'RMS',6:'formal_error',2:"num_obs"},inplace=True)

        path_xydata = '/Users/sam/Desktop/NVI/proj4/sked_data/all_obs_data.csv'
        path_holesdata = '/Users/sam/Desktop/NVI/proj4/sked_data/holes_full_data.csv'
        path_intervaldata = '/Users/sam/Desktop/NVI/proj4/sked_data/interval_obs.csv'
        path_var_intervaldata = '/Users/sam/Desktop/NVI/proj4/sked_data/var_of_interval_obs.csv'
        path_formatted_skeds = '/Users/sam/Desktop/NVI/proj4/sked_data/sk_d10_00h_formatted'

        if('xydata' in data_to_use): # xy data
            print("...using x, y data...")
            xy = np.loadtxt(path_xydata,delimiter=',',dtype=np.float32,skiprows=1)
            print("\n\nlen(xy):\n",len(xy))
            x_ = xy[:,3:]
            y_ = xy[:,[2]].astype(int)

        
        if('holesdata' in data_to_use): # holes data
            print("...using holes data...")
            xy = np.loadtxt(path_holesdata,delimiter=',',dtype=np.float32,skiprows=1,usecols=(2,3,4,5))
            # print("======= xy :\n",pd.read_csv(csv_file,index_col=0).columns.tolist())

            ### Use raw holes data
            if(True):
                print(len(xy[:,1]))
                try:
                    x_ = np.append(x_,xy[:,1:],axis=1)
                except:
                    x_ = xy[:,1:]
                try:
                    y_ = np.append(y_,xy[:,[0]].astype(int),axis=1)
                except:
                    y_ = xy[:,[0]].astype(int)
                print(x_)


            # ### Use var of hole size
            if(False):
                rad = np.zeros(shape=(501,2))

                for i in range(501):
                    rad[i,0] = np.var(xy[i:i+51,-1]) # var of hole radius
                    rad[i,1] = xy[51*i,0]
                
                try:
                    x_ = np.append(x_,rad[:,[0]],axis=1)
                except:
                    x_ = rad[:,[0]]
                
                y_ = rad[:,[1]].astype(int)
        
        if('intervaldata' in data_to_use):
            print("...using interval data...")
            ### Use variance of observations in x and y direction
            if(False):
                # df_xy = pd.read_csv(path_var_intervaldata,sep=',',index_col=0)
                xy_var = np.loadtxt(path_var_intervaldata,delimiter=',',dtype=np.float32,skiprows=1,usecols=(2,3,4))

                try:
                    x_ = np.append(x_,xy_var[:,1:],axis=1)
                except:
                    x_ = xy_var[:,1:]
                
                y_ = xy_var[:,[0]].astype(int)

            ### Use grid of observation in intervals
            if(True):
                df_xy = pd.read_csv(path_intervaldata,sep=',',index_col=0)
                
                xy = np.loadtxt(path_intervaldata,delimiter=',',dtype=np.float32,skiprows=1,usecols=(2,3,4))

                try:
                    x_ = np.append(x_,xy[:,1:],axis=1)
                except:
                    x_ = xy[:,1:]
                try:
                    if(len(y_) == len(xy[:,0])):
                        y_ = xy[:,[0]].astype(int)
                    else:
                        print('y_ lengths do not agree')
                        sys.exit()
                except:
                    y_ = xy[:,[0]].astype(int)
            
        if('griddata' in data_to_use):
            ## We want to get an amount of observations per interval on the form [x1,...,xn,y1,...,yn]
            ## i.e. in the end we get 501*51 arrays of [x1,...,xn,y1,...,yn], sort of like a grid.
            ## This will give us data similar to images.

            ## Maybe the values x1,...,xn,y1,...,yn needs to be rounded to e.g. 1 digit -> they are
            ## In a later stage we could include adjacent observations in x and y values, as to
            ## include scattering over time...
            ## Nu kör vi grid_obs på varje rad, där varje rad är ''oberoende av varandra''
            ## Kanske borde ta hänsyn till att vissa grejer kommer efter varandra eller dess
            ## förhållande respektive varandra.

            ## Limitation of this approach: Excludes data and only uses data of same length as the
            ## smallest observation in a given 10 min interval...

            ## Måste nog köra en grid_approach, för annars kommer den inte fatta att de olika punkternas
            ## ordning inte spelar någon roll..
            
            sked_list = list(range(501))
            sked_list[-1] = 'cov'
            time_list = list(range(51))
            # time_list = np.linspace(0,50,6)

            x_min = -85
            x_max = 88
            x_grid_res = 60 #x_max - x_min
            y_min = 10
            y_max = 90
            y_grid_res = 40 #y_max - y_min

            len_time = len(time_list)

            x_ = np.zeros(shape=(501*len_time,(x_grid_res+1)*(y_grid_res+1)))
            y_ = np.zeros(shape=(501*len_time,1))

            for sked in sked_list:
                try:
                    if(sked%100==0):
                        print("sked:",sked)
                except: print("sked: 500")
                df = pd.read_csv(path_formatted_skeds+f'/sked_{sked}.csv',sep=',',index_col=0)
                # print(df)
                if(sked == 'cov'):
                    sked = 500
                
                for time in time_list:
                    current_time = time + 5
                    df3 = df[np.abs(np.subtract(df['epoch_sec_tot'].astype(float),current_time)) < 60*10/2] #get 10 min interval
                    
                    idx_list = df3['plot_1'].index.tolist()

                    grid = np.zeros(shape=(x_grid_res+1, y_grid_res+1))
                    # print("grid shape:",grid.shape)
                    

                    for ind in idx_list:
                        x = df3.loc[ind,'plot_1']
                        y = df3.loc[ind,'plot_2']

                        x = -85
                        x = 88
                        y = 10
                        y = 90

                        # print(f"(x,y) = ({x},{y})")
                        
                        ix = math.floor(x_grid_res*(x - x_min)/(x_max-x_min))
                        iy = math.floor(y_grid_res*(y - y_min)/(y_max-y_min))

                        # print(f"(ix,iy) = ({ix},{iy})")
                        # print("x_grid_res:",x_grid_res)
                        # print("y_grid_res:",y_grid_res)

                        grid[ix,iy] += 1
                    
                    x_[int(len_time*sked+time),:] = grid.flatten()
                    y_[int(len_time*sked+time),0] = np.round(df_sked_tot.loc[sked,'RMS']).astype(int) #round to nearest int

        print("\nx_.shape[0]:", x_.shape[0])
        print("x_.shape[1]:", x_.shape[1])
        print("y_.shape[0]:", y_.shape[0],"\n")
        self.x = torch.from_numpy(x_)
        self.y = torch.from_numpy(y_).type(torch.LongTensor)
        self.n_samples = x_.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
    def get_nbrofvariables(self):
        return self.x.shape[1]


####### READ DATA #######
data_to_use = ['griddata'] #['xydata','holesdata','intervaldata','griddata']

dataset = set_dataset(data_to_use)

## Length of input variables
input_len = dataset.get_nbrofvariables()
## Batch size
batch_size = 64
## Accept adjacent values in evaluation
pm_1_output = True
## Proportion of data used for training
prop_train = 0.8
nbr_train = int(len(dataset)*prop_train)

trainset, testset = torch.utils.data.random_split(dataset, (nbr_train,len(dataset)-nbr_train))
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_len, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 13) #SE TILL ATT DENNA OUTPUT 1, så att den kollar på continuous outputs #rms_max-rms_min+1 = 8 (i.e. 8 possible RMSs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

EPOCHS = 3

for epoch in range(EPOCHS):
    print(f'=== starting epoch {epoch+1} ===')
    for data in trainset:
        # data is a batch of featuresets and Labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1,input_len).float()) #network output
        # loss = the inaccuracy of the prediction, or rather in the resulting prob. distr.
        loss = F.nll_loss(output, y) #use nll_loss if y is scalar, use MSE if y is vector
        loss.backward() #backpropagate loss
        optimizer.step()
    print("loss:",loss)

correct = 0
total = 0

#For every prediction, does it match the target value?
with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1,input_len).float())
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            if(pm_1_output):
                try:
                    if torch.argmax(i) == y[idx-1]:
                        correct += 1
                except:
                    pass
                try:
                    if torch.argmax(i) == y[idx+1]:
                        correct += 1
                except:
                    pass
            total += 1

print("\ncorrect/total = ", round(correct/total,3))


# for i in range(10):
#     plt.imshow(X[i].view(28,28))
#     plt.title(f"Prediction: {torch.argmax(net(X[i].view(-1,28*28))[0])}, with real value {y[i]}")
#     plt.show()


