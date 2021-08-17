import torch.optim as optim # Import optim 
import torchvision # Import torchvision 
import h5py #import h5py
from torch.utils.data import dataset # import dataset
from ESC10Customdataset_Train_4SB1 import LAEData_Train # Call Customdataloader to read Training Data
from ESC10Customdataset_Test_4SB1 import LAEData_Test # Call Customdataloader to read Test Data
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
from Attention import CAM_Module # import position attention module 
#from Attention import CAM_Module # import channel attention module
#from Attention import SA_Module # Import Self attention module
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
learning_rate=2e-4 # Define Learning rate 
batch_size=16 # Define Batch Size 
EPOCHS =1000   # Define maximum Number of Epochs
FC_Size=512
SFC_Size=512
Temprature=10
alpha=0.3
N_models=5
from Pathname import FB_CAM_Path, SB41_CAM_Path,SB42_CAM_Path,SB43_CAM_Path,SB44_CAM_Path,S3_CAM_AV_Path
warnings.filterwarnings("ignore")
train_transformations = transforms.Compose([ # Training Transform 
    #transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])
test_transformations = transforms.Compose([ # Test Transform 
    #transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])
train_dataset=LAEData_Train(transform=train_transformations) # Create tensor of training data 
Test_Dataset=LAEData_Test(transform=test_transformations)# Create tensor of test dataset 
train_size = int(0.7 * len(train_dataset)) # Compute size of training data using (70% As Training and 30% As Validation)
valid_size = len(train_dataset) - train_size # Compute size of validation data using (70% As Training and 30% As Validation)
Train_Dataset,Valid_Dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size]) # Training and Validation Data After (70%-30%)Data Split 
#train_set,test_set=torch.utils.data.random_split(dataset,[6000,2639])
#Labels=pd.read_csv("Devlopment.csv")
train_loader=DataLoader(dataset=Train_Dataset,batch_size=batch_size,shuffle=True) # Create Training Dataloader 
valid_loader=DataLoader(dataset=Valid_Dataset,batch_size=batch_size,shuffle=False)# Create Validation Dataloader 
test_loader=DataLoader(dataset=Test_Dataset,batch_size=batch_size,shuffle=False) # Create Test Dataloader 
class FB_Model(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(FB_Model, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2])
        self.features=Pre_Trained_Layers
        self.CAM=CAM_Module(512)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(512,num_classes)  # Set output layer as an one output
    def forward(self,image):
        x = self.features(image)
        x1=self.CAM(x)
        x4=self.avgpool(x1)
        x4=x4.view(x4.size(0),-1)
        x5=self.fc(x4)
        return x5
FB_Model=FB_Model()
FB_Path=FB_CAM_Path
FB_Model.load_state_dict(torch.load(FB_Path))
FB_Model=FB_Model.to(device)
class SB1_Model(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(SB1_Model, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.features=Pre_Trained_Layers
        self.CAM=CAM_Module(512)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(512,num_classes)  # Set output layer as an one output
    def forward(self,image):
        x = self.features(image)
        x1=self.CAM(x)
        x4=self.avgpool(x1)
        x4=x4.view(x4.size(0),-1)
        x5=self.fc(x4)
        return x5
SB1_Model=SB1_Model()
SB1_Path=SB41_CAM_Path
SB1_Model.load_state_dict(torch.load(SB1_Path))
SB1_Model=SB1_Model.to(device)
class SB2_Model(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(SB2_Model, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.features=Pre_Trained_Layers
        self.CAM=CAM_Module(512)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(512,num_classes)  # Set output layer as an one output
    def forward(self,image):
        x = self.features(image)
        x1=self.CAM(x)
        x4=self.avgpool(x1)
        x4=x4.view(x4.size(0),-1)
        x5=self.fc(x4)
        return x5
SB2_Model=SB2_Model()
SB2_Path=SB42_CAM_Path
SB2_Model.load_state_dict(torch.load(SB2_Path))
SB2_Model=SB2_Model.to(device)
class SB3_Model(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(SB3_Model, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.features=Pre_Trained_Layers
        self.CAM=CAM_Module(512)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(512,num_classes)  # Set output layer as an one output
    def forward(self,image):
        x = self.features(image)
        x1=self.CAM(x)
        x4=self.avgpool(x1)
        x4=x4.view(x4.size(0),-1)
        x5=self.fc(x4)
        return x5
SB3_Model=SB3_Model()
SB3_Path=SB43_CAM_Path
SB3_Model.load_state_dict(torch.load(SB3_Path))
SB3_Model=SB3_Model.to(device)
class SB4_Model(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(SB4_Model, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.features=Pre_Trained_Layers
        self.CAM=CAM_Module(512)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(512,num_classes)  # Set output layer as an one output
    def forward(self,image):
        x = self.features(image)
        x1=self.CAM(x)
        x4=self.avgpool(x1)
        x4=x4.view(x4.size(0),-1)
        x5=self.fc(x4)
        return x5
SB4_Model=SB4_Model()
SB4_Path=SB44_CAM_Path
SB4_Model.load_state_dict(torch.load(SB4_Path))
SB4_Model=SB4_Model.to(device)
class Student_Model(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(Student_Model, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.features=Pre_Trained_Layers
        self.CAM=CAM_Module(512)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(512,num_classes)  # Set output layer as an one output
    def forward(self,image):
        x = self.features(image)
        x1=self.CAM(x)
        x4=self.avgpool(x1)
        x4=x4.view(x4.size(0),-1)
        x5=self.fc(x4)
        return x5
Student_Model=Student_Model()
Student_Model=Student_Model.to(device)
Student_optimizer = optim.Adam(Student_Model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function 
def AVGEN_Loss(Predicted_FB_Label,Predicted_SB1_Label, Predicted_SB2_Label,Predicted_SB3_Label,Predicted_SB4_Label,Predicted_Student_Label,T,N_models,y,device):
        FB_Prob=F.softmax(Predicted_FB_Label/T,dim=1)
        SB1_Prob=F.softmax(Predicted_SB1_Label/T,dim=1)
        SB2_Prob=F.softmax(Predicted_SB2_Label/T,dim=1)
        SB3_Prob=F.softmax(Predicted_SB3_Label/T,dim=1)
        SB4_Porb=F.softmax(Predicted_SB4_Label/T,dim=1)
        Student_Prob=F.log_softmax(Predicted_Student_Label/T,dim=1)
        AVG_Prob=(FB_Prob+SB1_Prob+SB2_Prob+SB3_Prob+SB4_Porb)/N_models
        Student_Loss=F.kl_div(Student_Prob,AVG_Prob)
        #print(Total_E_Loss)
        return Student_Loss
def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc
def train(FB_Model, SB1_Model, SB2_Model,SB3_Model,SB4_Model,Student_Model,device,iterator, optimizer, criterion,alpha,Temp): # Define Training Function 
    early_stopping = EarlyStopping(patience=7, verbose=True)
    #print("Training Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    Student_Model.train() # call model object for training 
    #print(Temp)
    for (a,b,c,d,e,y) in iterator:
        a=a.float()
        b=b.float()
        c=c.float()
        d=d.float()
        e=e.float()
        a=a.to(device)
        b=b.to(device)
        c=c.to(device)
        d=d.to(device)
        e=e.to(device)
        y=y.to(device)# Transfer label  to device
        optimizer.zero_grad() # Initialize gredients as zeros 
        count=count+1
        #print(x.shape)
        Predicted_FB_Label=FB_Model(a)
        Predicted_SB1_Label=SB1_Model(b)
        Predicted_SB2_Label=SB2_Model(c)
        Predicted_SB3_Label=SB3_Model(d)
        Predicted_SB4_Label=SB3_Model(e)
        Predicted_Train_Label=Student_Model(a)
        CEL_Loss = criterion(Predicted_Train_Label, y) # training loss
        #Temp=torch.from_numpy(Temp)
        #Temp=Temp.to(device)
        #print((1-alpha)*(CEL_Loss))
        EN_AVG_Loss=AVGEN_Loss(Predicted_FB_Label,Predicted_SB1_Label, Predicted_SB2_Label,Predicted_SB3_Label,Predicted_SB4_Label,Predicted_Train_Label,Temp,N_models,y,device)
        #print(alpha*(Temp*Temp)*(EN_AVG_Loss))
        loss=((1-alpha)*(CEL_Loss))+(alpha*(Temp*Temp)*(EN_AVG_Loss))
        acc = calculate_accuracy(Predicted_Train_Label, y) # training accuracy 
        #print("Training Iteration Number=",count)
        loss.backward() # backpropogation 
        optimizer.step() # optimize the model weights using an optimizer 
        epoch_loss += loss.item() # sum of training loss
        epoch_acc += acc.item() # sum of training accuracy  
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model,device,iterator, criterion): # Evaluate Validation accuracy 
    #print("Validation Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    model.eval() # call model object for evaluation 
    
    with torch.no_grad(): # Without computation of gredient 
        for (x,b,c,d,e,y) in iterator:
            x=x.float()
            x=x.to(device) # Transfer data to device 
            y=y.to(device) # Transfer label  to device 
            count=count+1
            Predicted_Label = model(x) # Predict claa label 
            loss = criterion(Predicted_Label, y) # Compute Loss 
            acc = calculate_accuracy(Predicted_Label, y) # compute Accuracy 
            #print("Validation Iteration Number=",count)
            epoch_loss += loss.item() # Compute Sum of  Loss 
            epoch_acc += acc.item() # Compute  Sum of Accuracy 
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator) 
MODEL_SAVE_PATH = S3_CAM_AV_Path
best_valid_loss = float('inf')
Temp=np.zeros([EPOCHS,6]) # Temp Matrix to Store all model accuracy, loss and time parameters 
print("ESC 10 CNN Model is in Training Mode") 
print("---------------------------------------------------------------------------------------------------------------------")   
early_stopping = EarlyStopping(patience=7, verbose=True) # early Stopping Criteria
for epoch in range(EPOCHS):
    start_time=time.time() # Compute Start Time 
    train_loss, train_acc = train(FB_Model, SB1_Model, SB2_Model,SB3_Model,SB4_Model,Student_Model,device,train_loader,Student_optimizer,criterion,alpha,Temprature) # Call Training Process 
    train_loss=round(train_loss,2) # Round training loss 
    train_acc=round(train_acc,2) # Round training accuracy 
    valid_loss, valid_acc = evaluate(Student_Model,device,valid_loader,criterion) # Call Validation Process 
    valid_loss=round(valid_loss,2) # Round validation loss
    valid_acc=round(valid_acc,2) # Round accuracy 
    end_time=(time.time()-start_time) # Compute End time 
    end_time=round(end_time,2)  # Round End Time 
    print(" | Epoch=",epoch," | Training Accuracy=",train_acc*100," | Validation Accuracy=",valid_acc*100," | Training Loss=",train_loss," | Validation_Loss=",valid_loss,"Time Taken(Seconds)=",end_time,"|")
    print("---------------------------------------------------------------------------------------------------------------------")
    Temp[epoch,0]=epoch # Store Epoch Number 
    Temp[epoch,1]=train_acc # Store Training Accuracy 
    Temp[epoch,2]=valid_acc # Store Validation Accuracy 
    Temp[epoch,3]=train_loss # Store Training Loss 
    Temp[epoch,4]=valid_loss # Store Validation Loss 
    Temp[epoch,5]=end_time # Store Running Time of One Epoch 
    early_stopping(valid_loss,Student_Model,MODEL_SAVE_PATH) # call Early Stopping to Prevent Overfitting 
    if early_stopping.early_stop:
        print("Early stopping")
        break
    Student_Model.load_state_dict(torch.load(MODEL_SAVE_PATH))
np.save(' S3_CAM_AV_ESC10_Model_Parameters',Temp) # Save Temp Array as numpy array 
Student_Model.load_state_dict(torch.load(MODEL_SAVE_PATH)) # load the trained model 
test_loss, test_acc = evaluate(Student_Model, device, test_loader, criterion) # Compute Test Accuracy on Unseen Signals 
test_loss=round(test_loss,2)# Round test loss
test_acc=round(test_acc,2) # Round test accuracy 
print("|Test Loss=",test_loss,"Test Accuracy=",test_acc*100) # print test accuracy     

