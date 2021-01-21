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

# +
import os 
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader #, ConcatDataset ,
import torch
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import random
from torchsummary import summary

from CNN_morelayers import NeuronalNetwork_deeper
# -

torch.cuda.empty_cache()
#torch.cuda.memory_allocated()
#torch.cuda.max_memory_allocated()
#torch.cuda.memory_reserved(device=None)
#torch.cuda.reset_max_memory_cached(device=None)
#torch.cuda.memory_cached(device=None)

# +
import win32file

win32file._setmaxstdio(8192)


# -

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):  
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        self.basic_transform = transforms.Compose([
            transforms.Resize(64), # resize image #256
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
    transforms.Resize(64), # resize image
    transforms.ColorJitter(), # Randomly change the brightness, contrast and saturation of an image.
    #transforms.RandomCrop(224), # Crop the given PIL Image at a random location.
    transforms.RandomHorizontalFlip(p=0.5), # Horizontally flip the given image randomly with a given probability
    # transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3, fill=0), # Performs Perspective transformation of the given PIL Image randomly with a given probability.
    transforms.RandomRotation(degrees=10), # Randomly rotate image by (-10, 10) degree
    transforms.RandomResizedCrop((64), scale=(0.8, 1.0), ratio=(0.75, 1.33), interpolation=2),
    # transforms.RandomAffine(degrees=10, translate=None, scale=None, shear=None, resample=False, fillcolor=0)# Random affine transformation of the image keeping center invariant
    transforms.ToTensor() # Convert a PIL Image or numpy.ndarray to tensor.

])

# -

path = "../data/dogs-vs-cats/train/"
image_paths = [path + f for f in os.listdir(path)]

# +
#import win32file

#win32file._setmaxstdio(8192)


# +
# extract labels
# extract images

images = []
labels = []

for x in image_paths:  #when more data it somehow just stop running (no error just stops) #6500
    im_temp = Image.open(x)
    keep = im_temp.copy()     #copy and append than close otherwise ran into too many fiels open error
    images.append(keep)
    #images.append(im_temp)
    
    if "dog." in x: 
        labels.append(1)
    else: 
        labels.append(0)
    im_temp.close()
        
    
        
#for x in image_paths[-4000:]: 
#    im_temp = Image.open(x)
#    keep = im_temp.copy()     #copy and append than close otherwise ran into too many fiels open error
#    images.append(keep)
#    
#    if "dog." in x: 
#        labels.append(1)
#    else: 
#        labels.append(0)
#    im_temp.close()






# +

# create training set 
#CleanTrainData = MyDataset(images, labels, transform=False) #,transform=None

TrainData = MyDataset(images, labels, data_transform) #,transform=None

print('Our Training set included', len(images) ,'images')
print('/n')
print('Each input image has the shape', TrainData[0][0].shape)

# +
#from torchsummary import summary
#model=NeuronalNetwork_deeper()
#summary(model.cuda(), input_size=(3, 64,64))
#del model
# -



# +
def batch_accuracy(prediction, label):
    #whats input dimension prediction, label
    #torch.sum(prediction == label)/
    k=np.sum(list(label.size()))
    return torch.sum(prediction == label)*100/k  #()[0]  /label.size()

def class_label(prediction):
    #whats input dimension prediction
    #so here we have values between 0 and 1 for each label and want to set the prediction to 100% label where the prediction has 
    #higest prob?
    #eg if prediction is [0,0,0,0.2,0.2,0.6,0,0,0,0] than this function should return 5 as label
    #but is input prediction array like or tensor?
    
    return torch.max(prediction, 1)[1]




# +
# specify training process

P =  NeuronalNetwork_deeper().float()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(P.parameters(), lr=0.001) #actually 0.0001 #Dl, lr=0.0001 #0.0005 :maybe try #0.00005  #or 0.0001
num_epochs = 40

train_set, val_set = torch.utils.data.random_split(TrainData, [np.int(0.8*len(TrainData)),np.int(0.2*len(TrainData))])

Data_train = DataLoader(train_set, batch_size=16, shuffle=True)   #next try large batch size and small one #35 was maximal possible value for =40 not enough memory?
Data_val= DataLoader(val_set, batch_size=16, shuffle=True)  #16

use_gpu = torch.cuda.is_available()

# -

def train(model, criterion , optimizer, num_epochs , Data):
    #l=[80,30,10]
    #tqdm.write('Test: {}%'.format(statistics.mean(l)))
    model.train()
    
    #for step, [example, label] in enumerate(tqdm(Data, desc='Batch')):
    for step, [example, label] in enumerate(Data):
        if use_gpu:
            example = example.cuda()
            label = label.cuda()
            
        
        
        optimizer.zero_grad()

        prediction = model(example.float()) #[4]
        #loss = criterion(prediction.float(), label.float())  #when use bceloss
        loss = criterion(prediction, label)
                # Here pytorch applies backpropagation for us completely
                # automatically!!! That is quite awesome!

                #need to define validation data?
        loss.backward()

                # The step method now adds the gradients onto the model parameters
                # as specified by the optimizer and the learning rate.
        optimizer.step()

        if (step % 200) == 0:
                    
            acc = batch_accuracy(class_label(prediction), label)
            tqdm.write('Batch Accuracy: {}%, Loss: {}'.format(acc, loss))
            
    tqdm.write('Batch Accuracy: {}%, Loss: {}'.format(acc, loss))


def val(model, criterion , Data):

    
    accuracies=[]   
    
    model.eval()  
    
    #for step, [val_example, val_label] in enumerate(tqdm(Data, desc='Vali')):
    for step, [val_example, val_label] in enumerate(Data):
        if use_gpu:
            val_example = val_example.cuda()
            val_label = val_label.cuda()

        prediction = model(val_example.float()) 
        #loss = criterion(prediction.float(), label.float())  #when use bceloss
        loss = criterion(prediction, val_label)
        accuracies.append(batch_accuracy(class_label(prediction), val_label))
    mean = sum(accuracies)/len(accuracies)  
    print('Validation Accuracy: {}%, Loss: {}'.format(mean,loss))        
    val_ac.append(mean)
    val_loss.append(loss)
    #tqdm.write('Validation Accuracy: {}%'.format(torch.mean(accuracies)))

torch.cuda.empty_cache()

val_ac=[]
val_loss=[]
def train_whole( model, criterion , optimizer, num_epochs , Data_train, Data_val):

    if use_gpu:
        model.cuda()
        criterion.cuda()
        print('Is using the GPU')
        
    print('/n')
    
    model.zero_grad()
    
    print("start epoch")
    for epoch in tqdm(range(num_epochs), desc='Epoch'):
        #DO training 
        train(model, criterion , optimizer, num_epochs , Data_train)
        
        #DO validation after each epoch --> does not woooork     
        
        val(model, criterion , Data_val )
        
        torch.save(model.state_dict(), 'DeeperNN-64-nowtrue-smallerbatch-all-withmygpu-lr0.001.ckpt')
        
        #torch.save(P.state_dict(), 'perceptron_{}.ckpt'.format(epoch))
        
    #val(model, criterion , Data_val )


train_whole(P, criterion , optimizer, num_epochs , Data_train, Data_val)

 #load and test on data: 'DeeperNN-withmygpu-lr0.001.ckpt'


import matplotlib.pyplot as plt


# +
#plt.plot(val_ac)
#val_ac=[]
#val_loss=[]

plt.plot(val_ac, 'b-.')
plt.xlabel('epoch')
plt.ylabel('Validation accuracy')
#plt.plot(validation_acc_resnet18,'g', label='ResNet18')
plt.legend()
plt.savefig('Val_acc_deepNN-smallerbatch.more.png')
plt.show()

plt.plot(val_loss, 'g')
plt.xlabel('epoch')
plt.ylabel('Validation accuracy')
#plt.plot(validation_acc_resnet18,'g', label='ResNet18')
plt.legend()
plt.savefig('Val_loss_deepNN-smallerbatch.more.png')
plt.show()

# +
import matplotlib.pyplot as plt
import numpy as np
#plot with batch acc and val ac
val_ac=[69,72,78,80,81,80,84,84,84,85,86,86,86,82,87]
batch_ac=[56,75,81,93,81,81,93,87,93,81,81,93,93,87,81]
x=np.linspace(1,15,15)
plt.xlabel('Epochs')
plt.ylabel('Accuracy [%]')
plt.plot(x,val_ac, 'g', label='Validation Accurcay')
plt.plot(x,batch_ac,'b--',label='Single Batch Accuracy')
plt.legend()

plt.savefig('Val+Batchacc_6464_smallerbatch_deepermodel.png')
# -

del model
del TestData
del Data_test


torch.cuda.memory_allocated()

torch.cuda.empty_cache() 

# +
## testing

# +
# extract labels
# extract images

test_images = []
test_labels = []

for x in image_paths[10000:10500]:  #when more data it somehow just stop running (no error just stops) #6500
    im_temp = Image.open(x)
    #keep = im_temp.copy()     #copy and append than close otherwise ran into too many fiels open error
    test_images.append(im_temp)
    #images.append(im_temp)
    
    if "dog." in x: 
        test_labels.append(1)
    else: 
        test_labels.append(0)

        
    
        
for x in image_paths[-10500:-10000]: 
    im_temp = Image.open(x)
    #keep = im_temp.copy()     #copy and append than close otherwise ran into too many fiels open error
    test_images.append(im_temp)
    
    if "dog." in x: 
        test_labels.append(1)
    else: 
        test_labels.append(0)
    
# -

TestData = MyDataset(test_images, test_labels, data_transform) #,transform=None
Data_test = DataLoader(TestData, batch_size=10, shuffle=True)
use_gpu = torch.cuda.is_available()


model =  NeuronalNetwork_deeper().float()
model.load_state_dict(torch.load('DeeperNN-withmygpu-lr0.001.ckpt'))
criterion = nn.CrossEntropyLoss()
model.eval()
#del model

torch.cuda.memory_allocated()


#test on never before seen data
def test(model, criterion , Data):    
    if use_gpu:
        model.cuda()
        criterion.cuda()
        print('Is using the GPU')
        
    accuracies=[]   

    
    #for step, [val_example, val_label] in enumerate(tqdm(Data, desc='Vali')):
    for step, [test_example, test_label] in enumerate(Data):
        if use_gpu:
            test_example = test_example.cuda()
            test_label = test_label.cuda()

        prediction = model(test_example.float()) 
        #loss = criterion(prediction.float(), label.float())  #when use bceloss
        loss = criterion(prediction, test_label)
        accuracies.append(batch_accuracy(class_label(prediction), test_label))
    mean = sum(accuracies)/len(accuracies)  
    print('Test Accuracy on 1000 never seen images: {}%, Loss: {}'.format(mean,loss)) 

test(model, criterion,Data_test )

# ## net=NeuronalNetwork()
# net.load_state_dict(torch.load('DeepVisionProject/SimpleNN-withbn_1621_020720.ckpt'))
