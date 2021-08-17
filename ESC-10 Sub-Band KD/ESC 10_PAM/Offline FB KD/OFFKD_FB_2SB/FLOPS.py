import torch.optim as optim # Import optim 
import torchvision # Import torchvision 
import h5py #import h5py
from torch.utils.data import dataset # import dataset
#from ESC10Customdataset_Train_2SB1 import LAEData_Train # Call Customdataloader to read Training Data
#from ESC10Customdataset_Test_2SB1 import LAEData_Test # Call Customdataloader to read Test Data
from torch.utils.data import DataLoader # Import Dataloader 
import torchvision.transforms as transforms # Import Transform 
import pandas as pd # Import Pnadas 
import torch # Import Torch 
import torch.nn as nn # Import NN module from Torch 
from torchvision.transforms import transforms# Import transform module from torchvision 
from torch.utils.data import DataLoader # Import dataloader from torch 
from torch.optim import Adam # import optimizer module from torch 
from torch.autograd import Variable # Import autograd from torch 
import numpy as np # Import numpy module 
import torchvision.datasets as datasets #Import dataset from torch 
from Attention import PAM_Module # import position attention module 
from Attention import CAM_Module # import channel attention module
from Attention import SA_Module # Import Self attention module
from torch import optim, cuda # import optimizer  and CUDA
import random # import random 
import torch.nn.functional as F # Import nn.functional 
import time # import time 
import sys # Import System 
import os # Import OS
from pytorchtools import EarlyStopping
from torchvision import models
import warnings
SEED = 1234 # Initialize seed 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda') # Define device type 
num_classes=10 # Define Number of classes 
in_channel=1   # Define Number of Input Channels 
learning_rate=2e-5 # Define Learning rate 
batch_size=16 # Define Batch Size 
EPOCHS =1000   # Define maximum Number of Epochs
FC_Size=512
SFC_Size=512
Temprature=10
alpha=0.3
N_models=3
warnings.filterwarnings("ignore")
class FB_Model(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(FB_Model, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2])
        self.features=Pre_Trained_Layers
        #self.PAM=PAM_Module(512)
        #self.CAM=CAM_Module(512)
        #self.SA=SA_Module(512)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(512,num_classes)  # Set output layer as an one output
    def forward(self,image):
        x = self.features(image)
        #x1=self.SA(x)
        #x1=self.PAM(x)
        #x1=self.CAM(x)
        x4=self.avgpool(x)
        x4=x4.view(x4.size(0),-1)
        x5=self.fc(x4)
        return x5
FB_Model=FB_Model()
FB_Model=FB_Model.to(device)
from flopth import flopth
#sum_flops = flopth(FB_Model, input_Image_size=[[1,64,16]])
#print("Number of FLOPS ResNet-34=",sum_flops)
#pytorch_total_params = sum(p.numel() for p in FB_Model.parameters() if p.requires_grad)
#print("Total Number of Parameters=",pytorch_total_params/1000000)
sum_flops1 = flopth(FB_Model, in_size=[[1,64,431]])
sum_flops2 = flopth(FB_Model, in_size=[[1,64,431]])
sum_flops3 = flopth(FB_Model, in_size=[[1,42,431]])
sum_flops4 = flopth(FB_Model, in_size=[[1,40,431]])
sum_flops5 = flopth(FB_Model, in_size=[[1,46,431]])
sum_flops6 = flopth(FB_Model, in_size=[[1,32,431]])
sum_flops7 = flopth(FB_Model, in_size=[[1,32,431]])
sum_flops8 = flopth(FB_Model, in_size=[[1,32,431]])
sum_flops9 = flopth(FB_Model, in_size=[[1,32,431]])
sum_flops10 = flopth(FB_Model, in_size=[[1,128,431]])
print("Number of FLOPS M1=",sum_flops1)
print("Number of FLOPS M2=",sum_flops2)
print("Number of FLOPS M3=",sum_flops3)
print("Number of FLOPS M4=",sum_flops4)
print("Number of FLOPS M5=",sum_flops5)
print("Number of FLOPS M6=",sum_flops6)
print("Number of FLOPS M7=",sum_flops7)
print("Number of FLOPS M8=",sum_flops8)
print("Number of FLOPS M9=",sum_flops9)
print("Number of FLOPS M10=",sum_flops10)
