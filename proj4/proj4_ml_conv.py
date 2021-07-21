import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import sys

REBUILD_DATA = False

class DogsVSCats():
    IMG_SIZE = 50 #makes images 50*50
    CATS = '/Users/sam/Desktop/NVI/proj4/PetImages/Cat'
    DOGS = '/Users/sam/Desktop/NVI/proj4/PetImages/Dog'
    # CATS = '/Users/sam/Desktop/NVI/proj4/PetImages/Cats'
    # DOGS = '/Users/sam/Desktop/NVI/proj4/PetImages/Dogs'
    LABELS = {CATS:0, DOGS: 1}

    training_data = []

    catcount = 0
    dogcount = 0
    def make_training_data(self):
        k = 0
        for label in self.LABELS:
            # print("label:",label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label,f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE)) # här är felet
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass # print(str(e))
        print(f"k = {k}")
        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data)
        print("Cats:", self.catcount)
        print("Dogs:", self.dogcount)

if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

training_data = np.load('training_data.npy', allow_pickle=True)

if(False):
    plt.imshow(training_data[1][0],cmap='gray')
    plt.show()

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
        self.fc2 = nn.Linear(512, 2) #output 2: dogs or cats

    def convs(self, x): #like a forwards method
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) #(2,2) is shape of pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        # print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x) #pass through all conv layers
        x = x.view(-1, self._to_linear) #flatten
        x = F.relu(self.fc1(x)) #pass through first fully conn layer
        x = self.fc2(x)
        return F.softmax(x, dim=1)

net = Net()

optimizer = optim.Adam(net.parameters(), lr=1e-3) #optim step 1e-3
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0 #scaling pixel values to [0,1]
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1 #test against 10% of data set
val_size = int(len(X)*VAL_PCT)
# print(val_size)

### Splitting training and testing data
train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

# print(len(train_X))
# print(len(test_X))

BATCH_SIZE = 100 #if we memory error, change batch size

EPOCHS = 1

for epoch in range(EPOCHS):
    #tqdm for progress bar
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

print(loss)

correct = 0
total = 0

with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1,1,50,50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1

print("Accuracy:", round(correct/total,3))









