# Acoustic Event Detection Using Knowledge Distillation from Attention-Based Subband Specilist Deep Model 
import torch.optim as optim # Import optim 
import torchvision # Import torchvision 
import h5py #import h5py
from torch.utils.data import dataset # import dataset
from ESC10Customdataset_Train_FB import LAEData_Train # Call Customdataloader to read Training Data
from ESC10Customdataset_Test_FB import LAEData_Test # Call Customdataloader to read Test Data
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
from sklearn.metrics import confusion_matrix
from Pathname import FB_CAM_Path
from Pathname import FB_CAM_Path,SB21_CAM_Path,SB22_CAM_Path,S2_CAM_MI_Path
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
batch_size=64 # Define Batch Size 
EPOCHS =1000   # Define maximum Number of Epochs
FC_Size=512
SFC_Size=512
Temp=3
alpha=0.7
N_models=6
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
class Teacher(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(Teacher, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        #Pre_Trained_Layers = list(models.resnet34(pretrained=True).children())[:-4]
        #Pre_Trained_Layers = models.resnet34(pretrained=True) # Initialize model layers and weights
        #print(Pre_Trained_Layers)
        self.features=Pre_Trained_Layers
        self.CAM=CAM_Module(512)
        #self.CAM=CAM_Module(512)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))

        #self.features.Flat=nn.Flatten()
        self.fc=nn.Linear(512,10)  # Set output layer as an one output

    def forward(self,image):
        x = self.features(image)
        x1=self.CAM(x)
        #x2=self.CAM(x1)
        #x3=x1+x2
        x4=self.avgpool(x1)
        #x4=x3.view(x3.shape[0],-1)
        x4=x4.view(x4.size(0),-1)
        #x4=torch.flatten(x4)
        #print(x4.shape)
        #x4=torch.unsqueeze(x4,-1)
        #print(x4.shape)
        x5=self.fc(x4)
        return x5
Teacher_Model=Teacher()
#print(Teacher_Model)
Teacher_Model=Teacher_Model.to(device)
Teacher_optimizer = optim.Adam(Teacher_Model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function
def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    #T=3
    #LL=nn.KLDivLoss()((F.log_softmax(fx/T,dim=1)),(F.softmax(fx/T,dim=1)))
    #print(fx.shape)
    return acc
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc
def evaluate(model,device,iterator, criterion): # Evaluate Validation accuracy 
    #print("Validation Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    model.eval() # call model object for evaluation 
    
    with torch.no_grad(): # Without computation of gredient 
        for (x, y) in iterator:
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
MODEL_SAVE_PATH = S2_CAM_MI_Path
Teacher_Model.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_loss, test_acc = evaluate(Teacher_Model, device, test_loader, criterion)
print("|Test Loss=",test_loss,"Test Accuracy=",test_acc*100)
def get_all_preds(model,loader):
    all_preds = torch.tensor([])
    all_preds=all_preds.to(device)
    all_actual=torch.tensor([])
    all_actual=all_actual.to(device)
    for (images,labels) in loader:
        #images, labels = batch
        images=images.to(device)
        labels=labels.to(device)
        labels=labels.float()
        images=images.float()
        #print(labels) 
        #print(model(images))
        preds = (nn.functional.softmax(model(images),dim=1)).max(1,keepdim=True)[1]
        #fx.max(1, keepdim=True)[1]
        #print(preds)
        #print(labels)
        #dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        all_preds = torch.cat((all_preds, preds.float()),dim=0)
        #print(all_preds)
        all_actual = torch.cat((all_actual,labels),dim=0)
    return all_preds,all_actual
with torch.no_grad():
    #train_preds,train_actual = get_all_preds(Teacher_Model,train_loader)
    test_preds,test_actual = get_all_preds(Teacher_Model,test_loader)
#Train_CM = confusion_matrix(train_actual.cpu().numpy(),train_preds.cpu().numpy())
Test_CM = confusion_matrix(test_actual.cpu().numpy(),test_preds.cpu().numpy())
#print(Train_CM)
#print(Test_CM)
import itertools
import numpy as np
import matplotlib.pyplot as plt
classes= ["0","1","2","3","4","5","6","7","8","9"]
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    print(cm.diagonal())
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90,size=12)
    plt.yticks(tick_marks, classes,rotation=45,size=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',size=12)
    plt.xlabel('Predicted label',size=12)
plt.figure(figsize=(7,7))
plt.rcParams['font.size'] = 12
plot_confusion_matrix(Test_CM,classes) 
plt.tight_layout()
plt.savefig('MI3SB_CAM_MNIST_CM.pdf')
plt.show()  
from sklearn.metrics import precision_recall_fscore_support
Precision=precision_recall_fscore_support(test_actual.cpu(),test_preds.cpu(),average='macro')
print(Precision)
