# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os
import pandas as pd 
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):  
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        self.basic_transform = transforms.Compose([
            transforms.Resize(225), # resize image
            transforms.ToTensor() # Convert a PIL Image or numpy.ndarray to tensor.
        ])

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)
        else: 
            x = self.basic_transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


# +
data_transform = transforms.Compose([
    transforms.Resize(225), # resize image
    transforms.ColorJitter(), # Randomly change the brightness, contrast and saturation of an image.
    transforms.RandomCrop(224), # Crop the given PIL Image at a random location.
    transforms.RandomHorizontalFlip(p=0.5), # Horizontally flip the given image randomly with a given probability
    # transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3, fill=0), # Performs Perspective transformation of the given PIL Image randomly with a given probability.
    transforms.RandomRotation(degrees=10), # Randomly rotate image by (-10, 10) degree
    transforms.RandomResizedCrop(225, scale=(0.8, 1.0), ratio=(0.75, 1.33), interpolation=2),
    # transforms.RandomAffine(degrees=10, translate=None, scale=None, shear=None, resample=False, fillcolor=0)# Random affine transformation of the image keeping center invariant
    transforms.ToTensor() # Convert a PIL Image or numpy.ndarray to tensor.

])

# -

path = "../data/dogs-vs-cats/train/"
image_paths = [path + f for f in os.listdir(path)]

# extract labels
labels = []
for x in image_paths[0:100]: 
    if "dog." in x: 
        labels.append(1)
    else: 
        labels.append(0)


# extract images
images = []
for im in image_paths[0:100]: 
    im_temp = Image.open(im) #/255
    # im_temp = im_temp.transpose(2,0,1)
    images.append(im_temp)


# +
# define transformaiton in pytorch
        
data_transform = transforms.Compose([
    transforms.Resize(256), # resize image
    transforms.ColorJitter(), # Randomly change the brightness, contrast and saturation of an image.
    transforms.RandomCrop(224), # Crop the given PIL Image at a random location.
    transforms.RandomHorizontalFlip(p=0.5), # Horizontally flip the given image randomly with a given probability
    # transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3, fill=0), # Performs Perspective transformation of the given PIL Image randomly with a given probability.
    transforms.RandomRotation(degrees=10), # Randomly rotate image by (-10, 10) degree
    transforms.RandomResizedCrop(256, scale=(0.9, 1.0), ratio=(0.8, 1.2), interpolation=2),
    # transforms.RandomAffine(degrees=10, translate=None, scale=None, shear=None, resample=False, fillcolor=0)# Random affine transformation of the image keeping center invariant
    transforms.ToTensor() # Convert a PIL Image or numpy.ndarray to tensor.

])

# -

2**8

# create training set 
CleanTrainData = MyDataset(images, labels, transform=False) #,transform=None
TrainData = MyDataset(images, labels, data_transform) #,transform=None

dataloader = DataLoader(TrainData, batch_size = 32, shuffle=True, num_workers=4)

np.shape(TrainData[1][0])


def plot_idx(idx, Dataset): 
    image = np.transpose(Dataset[idx][0], (1,2,0))
    label = Dataset[idx][1]
    if label == 1: label = "dog"
    else: label = "cat"
    plt.imshow(image)
    plt.title(label)
    # plt.show()


plot_idx(5, CleanTrainData)

print('Each input image has the shape', TrainData[0][0].shape)
plt.figure(figsize=(15,15))
for i in range(0, 9): 
    plt.subplot(4,3, i + 1)
    plot_idx(i, TrainData)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False, width=0)

128*2*2

256/2/2/2/2/2

256*8*8


# A rather simple NN with 3 Convolutional Kernels, 2D Maxpooling and 2 Fully Connected layers
class NeuronalNetwork(nn.Module):

    def __init__(self):
        super(NeuronalNetwork, self).__init__()
        # 5 convolutional kernels
        self.conv1 = nn.Conv2d(3,  32, 5, padding=2)   #input dim, output dim, conv dim (5x5) , padding to reserve img size
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256,3, padding=1)

        self.fc1 = nn.Linear(256*8*8 , 120)  # 8*8 from image dimension after 5 Max_pooling and input size 256x256
        self.out_layer = nn.Linear(120, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, (2, 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.out_layer(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


256*64*64


# +
def train(use_gpu=False): # if torch.cuda.is_available(), use gpu to speed up training
    
    # Here we instanciate our model. The weights of the model are automatically
    # initialized by pytorch
    P = NeuronalNetwork().float()
    criterion = nn.Sigmoid()
    
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(P.parameters(), lr=0.0001) #Dl, lr=0.0001
    
    Data = DataLoader(TrainData, batch_size=5)
    #Dltest= 
    if use_gpu:
        P.cuda()
        criterion.cuda()
    print("start epoch")
    for epoch in tqdm(range(5), desc='Epoch'):
        for step, [example, label] in enumerate(tqdm(Data, desc='Batch')):
            if use_gpu:
                example = example.cuda()
                label = label.cuda()
            
            optimizer.zero_grad()
            
            prediction = P(example.float()) #[4]
            print(prediction, label)
            loss = criterion(prediction, label)
            
            # Here pytorch applies backpropagation for us completely
            # automatically!!! That is quite awesome!
            #loss.backward()

            # The step method now adds the gradients onto the model parameters
            # as specified by the optimizer and the learning rate.
            #optimizer.step()

            #if (step % 375) == 0:
                # Your code here
             #   acc = batch_accuracy(class_label(prediction), label)
             #   tqdm.write('Batch Accuracy: {}%, Loss: {}'.format(acc, loss))
            break

train()
# -

plt.imshow()

dataloader.__dir__

samples, labels = iter(dataloader).next()
plt.figure(figsize=(16,24))
grid_imgs = torchvision.utils.make_grid(samples[:24])
# np_grid_imgs = grid_imgs.numpy()
# in tensor, image is (batch, width, height), so you have to transpose it to (width, height, batch) in numpy to show it.
# plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))



TrainData.targets

len(images)

# +

plt.imshow(images[6])
# -



image_paths[0:10], labels[0:10]

image_paths[0]

image_paths[0]

Image.open(image_paths[1])


