import os
import torch
import numpy as np 
import h5py
from torch.utils.data import dataset
import pandas as pd 
from PIL import Image
from Pathname import ESC10_Train_Label_Path,ESC10_Test_Label_Path,ESC10_Train_Data_Path,ESC10_Test_Data_Path,ESC10_H5py_Path
class LAEData_Train():
	def __init__(self,transform=None):
		self.annotations=np.load(ESC10_Train_Data_Path,allow_pickle=True) # Read The names of Test Signals 
		self.Label=np.load(ESC10_Train_Label_Path,allow_pickle=True)
		self.Label=np.array(self.Label)
		self.transform=transform
	def __len__(self):
		return len(self.annotations)
	def __getitem__(self,index):
		key=self.annotations[index]
		with h5py.File(ESC10_H5py_Path, 'r') as f: # Database for Left Channel Test Spectrogram 
			SG_Data = f[key][()]
			SG_Data=np.array(SG_Data)
			
			#SG_Data=np.log((SG_Data + 1e-10))
			#for i in range(len(SG_Data)):
				#SG_Data[i,:]=SG_Data[i,:]/np.sum(SG_Data[i,:])
			#SGmax,SGmin=SG_Data.max(),SG_Data.min()
			#SG_Data=(SG_Data-SGmin)/(SGmax-SGmin)
			SG_Label= torch.from_numpy(np.array((self.Label[index])))
			#print(np.shape(KS_Data),np.shape(SG_Data))
			#ES_Data=np.dstack((SG_Data,KS_Data))
			r1=0
			r2=31
			r3=32
			r4=63
			r5=64
			r6=95
			r7=96
			r8=127
			FB_Data=SG_Data
			SB41_Data=SG_Data[r1:r2,:]
			SB42_Data=SG_Data[r3:r4,:]
			SB43_Data=SG_Data[r5:r6,:]
			SB44_Data=SG_Data[r7:r8,:]
			FB_Data=Image.fromarray(FB_Data)
			SB41_Data=Image.fromarray(SB41_Data)
			SB42_Data=Image.fromarray(SB42_Data)
			SB43_Data=Image.fromarray(SB43_Data)
			SB44_Data=Image.fromarray(SB44_Data)
			if self.transform:
				FB_Data=self.transform(FB_Data)
				SB41_Data=self.transform(SB41_Data)
				SB42_Data=self.transform(SB42_Data)
				SB43_Data=self.transform(SB43_Data)
				SB44_Data=self.transform(SB44_Data)
		return (FB_Data,SB41_Data,SB42_Data,SB43_Data,SB44_Data,SG_Label)
