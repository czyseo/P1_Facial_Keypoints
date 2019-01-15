## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        #110X110X32
        self.pool1= nn.MaxPool2d(2,2)
        #(W-F)/S +1 = (110-4)/1 +1 = 108
        self.conv2 = nn.Conv2d(32,  64, 3)
     
        #54X54X64
        self.pool2= nn.MaxPool2d(2,2)
        #(W-F)/S +1 = (54-3)/1 +1 = 52
        self.conv3 = nn.Conv2d(64,  128, 3)
        #torch.nn.init.xavier_normal_( self.conv3.weight)
        #26X26X128
        self.pool3 = nn.MaxPool2d(2,2)
        #(W-F)/S +1 = (26-3)/1 +1 = 24
        self.conv4 = nn.Conv2d(128,  256, 3)
        #12X12X256
        self.pool4 = nn.MaxPool2d(2,2)
        #(W-F)/S +1 = (12-1)/1 +1 = 12
        self.conv5 = nn.Conv2d(256,  512, 1)
        #6X6X512
        self.pool5 = nn.MaxPool2d(2,2)
        
        
        self.fc1= nn.Linear(6*6*512,1024)
        self.fc2= nn.Linear(1024,136)
        
        self.dropout = nn.Dropout(p=0.2)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x= self.pool1(F.selu(self.conv1(x)))
        x= self.dropout(x)
        x= self.pool2(F.selu(self.conv2(x)))
        x= self.dropout(x)
        x= self.pool3(F.selu(self.conv3(x)))
        x= self.dropout(x)
        x =self.pool4(F.selu(self.conv4(x)))
        x= self.dropout(x)
        x =self.pool5(F.selu(self.conv5(x)))
        x= self.dropout(x)
        x = x.view(x.size(0), -1)
        x= F.selu(self.fc1(x))
        x= self.dropout(x)
        x= self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x