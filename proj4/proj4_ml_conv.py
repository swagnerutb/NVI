import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys #not necessary

REBUILD_DATA_dogsvscats = False
LOAD_DATA_dogsvscats = False
REBUILD_DATA = False

########## Run on GPU if possible ##########
if(torch.cuda.is_available()):
    device = torch.device("cuda:0")
    print("...running on GPU...")
else:
    device = torch.device("cpu")
    print("...running on CPU...")
############################################


class DogsVSCats():
    IMG_SIZE = 50 #makes images 50*50
    
    CATS = '/Users/sam/Desktop/NVI/proj4/PetImages/Cat'
    DOGS = '/Users/sam/Desktop/NVI/proj4/PetImages/Dog'
    LABELS = {CATS:0, DOGS:1}

    training_data = []

    catcount = 0
    dogcount = 0
    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label,f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass # print(str(e))
        np.random.shuffle(self.training_data)
        np.save('training_data_dogsvscats.npy', self.training_data)
        print("Cats:", self.catcount)
        print("Dogs:", self.dogcount)

if REBUILD_DATA_dogsvscats:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

if(LOAD_DATA_dogsvscats):
    training_data_dogsvscats = np.load('training_data_dogsvscats.npy', allow_pickle=True)
    print("training_data_dogsvscats.shape:",training_data_dogsvscats.shape)
    print("training_data_dogsvscats:",training_data_dogsvscats)


class proj4data():
    GRID_SIZE = 50

    training_data = []

    def make_training_data(self):
        path_sked_tot = '/Users/sam/Desktop/NVI/proj4/ut1_d10_00h.txt'
        df_sked_tot = pd.read_csv(path_sked_tot,sep='\s+',header=None)
        df_sked_tot.rename(columns={8:'RMS',6:'formal_error',2:"num_obs"},inplace=True)
        
        path_format_skeds = '/Users/sam/Desktop/NVI/proj4/sked_data/sk_d10_00h_formatted'
        
        sked_list = list(range(501))
        sked_list[-1] = 'cov'
        time_list = list(range(51))
        # time_list = np.linspace(0,50,6)

        x_min, x_max = -85, 88
        x_grid_res = 49 #grid.shape[0] = x_grid_res + 1
        y_min, y_max = 10, 90
        y_grid_res = 49 #grid.shape[1] = y_grid_res + 1

        len_time = len(time_list)

        x_ = np.zeros(shape=(501*len_time,(x_grid_res+1),(y_grid_res+1)))
        y_ = np.zeros(shape=(501*len_time,1))

        for sked in sked_list:
            try:
                if(sked%100==0):
                    print("sked:", sked)
            except: pass
            df = pd.read_csv(path_format_skeds+f'/sked_{sked}.csv',sep=',',index_col=0)
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

                    # print(f"(x,y) = ({x},{y})")
                    
                    ix = math.floor(x_grid_res*(x - x_min)/(x_max-x_min))
                    iy = math.floor(y_grid_res*(y - y_min)/(y_max-y_min))

                    # print(f"(ix,iy) = ({ix},{iy})")
                    # print("x_grid_res:",x_grid_res)
                    # print("y_grid_res:",y_grid_res)

                    grid[ix,iy] += 1

                # x_[int(len_time*sked+time),:,:] = grid
                # y_[int(len_time*sked+time),0] = np.round(df_sked_tot.loc[sked,'RMS']).astype(int) #round to nearest int
                
                #As of right now, rms_ is discretised
                rms_ = np.round(df_sked_tot.loc[sked,'RMS']).astype(int)
                self.training_data.append([grid, np.eye(8)[rms_-5]])

        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data)

if REBUILD_DATA:
    proj4 = proj4data()
    proj4.make_training_data()

training_data = np.load('training_data.npy', allow_pickle=True)

# print("training_data.shape:",training_data.shape)
# print("training_data:",training_data)

# training_data = np.load('training_data.npy', allow_pickle=True)

if(False):
    plt.imshow(training_data[1][0],cmap='gray')
    plt.show()

###################################################################

class Net(nn.Module):
    def __init__(self):
        super().__init__() #TODO: can be done in 3 dim as well.. maybe that could be done for an entire sked!
        self.conv1 = nn.Conv2d(1, 32, 5) #input 1, outputs 32 conv features, kernell size 5
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x) # we need to flatten this stuff before passing to fully conn layer

        self.fc1 = nn.Linear(self._to_linear,512) #_to_linear will have been set by calling convs(x) in prev. line
        self.fc2 = nn.Linear(512, 8) #output is 8 due to 8 categories of rms_

    def convs(self, x): #like a forwards method
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) #(2,2) is shape of pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x) #pass through all conv layers
        x = x.view(-1, self._to_linear) #flatten
        x = F.relu(self.fc1(x)) #pass through first fully conn layer
        x = self.fc2(x)
        return F.softmax(x, dim=1)

###################################################################

# net = Net().to(device) #create network to GPU

# optimizer = optim.Adam(net.parameters(), lr=1e-3) #optim step 1e-3
# loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0 #scaling pixel values to [0,1]
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1 #prop of data -> test
val_size = int(len(X)*VAL_PCT)
# print(val_size)

##### Splitting training and testing data
train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

###################################################################

def train_old(net):
    optimizer = optim.Adam(net.parameters(), lr=1e-3) #optim step 1e-3
    loss_function = nn.MSELoss()
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
            batch_y = train_y[i:i+BATCH_SIZE]

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss}")

def test_old(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1,1,50,50).to(device))[0]
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1
    
    print("Accuracy:", round(correct/total,3))

# train_old(net)
# test_old(net)

def fwd_pass(X, y, train=False): #train=False not to mix train and test data
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs,y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs,y) #loss_function previously defined
    
    if train:
        loss.backward()
        optimizer.step()
    return acc, loss

def test(size=32):
    random_start = np.random.randint(len(test_X)-size)
    X, y = test_X[random_start:random_start+size], test_y[random_start:random_start+size]
    
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1,1,50,50).to(device),y.to(device))
    
    return val_acc, val_loss

net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3) #optim step 1e-3
loss_function = nn.MSELoss()

val_acc, val_loss = test(size=100)
print(val_acc, val_loss)

MODEL_NAME = f'model-{int(time.time())}'
print("MODEL_NAME:",MODEL_NAME)

def train():
    BATCH_SIZE = 100 #if we get memory error, change batch size
    EPOCHS = 8
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50).to(device)
                batch_y = train_y[i:i+BATCH_SIZE].to(device)
                
                acc, loss = fwd_pass(batch_X, batch_y, train=True)
                if i%50 == 0: #check every 50 steps
                    val_acc, val_loss = test(size=100)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")

train()








