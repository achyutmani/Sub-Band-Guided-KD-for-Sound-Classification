import os
import torch
import numpy as np 
import h5py
from torch.utils.data import dataset
from Pathname import ESC10_Train_Label_Path,ESC10_Test_Label_Path,ESC10_Train_Data_Path,ESC10_Test_Data_Path,ESC10_H5py_Path
import pandas as pd 
from PIL import Image
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
			SG_Label= torch.from_numpy(np.array((self.Label[index])))
			r1=41
			r2=81
			ES_Data=SG_Data[r1:r2,:]
			ES_Data=Image.fromarray(ES_Data)
			if self.transform:
				ES_Data=self.transform(ES_Data)
		return (ES_Data,SG_Label)
