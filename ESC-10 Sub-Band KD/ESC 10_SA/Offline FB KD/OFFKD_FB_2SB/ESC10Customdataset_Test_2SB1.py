import os
import torch
import numpy as np 
import h5py
from torch.utils.data import dataset
import pandas as pd 
from PIL import Image
from Pathname import ESC10_Train_Label_Path,ESC10_Test_Label_Path,ESC10_Train_Data_Path,ESC10_Test_Data_Path,ESC10_H5py_Path
class LAEData_Test():
	def __init__(self,transform=None):
		self.annotations=np.load(ESC10_Test_Data_Path,allow_pickle=True) # Read The names of Test Signals 
		self.Label=np.load(ESC10_Test_Label_Path,allow_pickle=True)
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
			r2=63
			r3=64
			r4=127
			FB_Data=SG_Data
			SB21_Data=SG_Data[r1:r2,:]
			SB22_Data=SG_Data[r3:r4,:]
			FB_Data=Image.fromarray(FB_Data)
			SB21_Data=Image.fromarray(SB21_Data)
			SB22_Data=Image.fromarray(SB22_Data)
			if self.transform:
				FB_Data=self.transform(FB_Data)
				SB21_Data=self.transform(SB21_Data)
				SB22_Data=self.transform(SB22_Data)
		return (FB_Data,SB21_Data,SB22_Data,SG_Label)
