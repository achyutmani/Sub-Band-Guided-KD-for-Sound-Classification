import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
Data=pd.read_csv("MNIST.csv")
Data=np.array(Data)
#print(Data)
X=Data[:,0]
Y=Data[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
X_train=np.asarray(X_train)
y_train=np.asarray(y_train)
X_test=np.asarray(X_test)
y_test=np.asarray(y_test)
print("Train Data=\n",X_train[0])
print("Train Label=\n",y_train[0])
print("Test Data=\n",X_test[0])
print("Test Label=\n",y_test[0])
np.save('MNISTTrainData',X_train)
np.save('MNISTTrainLabel',y_train)
np.save('MNISTTestData',X_test)
np.save('MNISTTestLabel',y_test)
A1=np.load('MNISTTrainData.npy',allow_pickle=True)
A2=np.load('MNISTTrainLabel.npy',allow_pickle=True)
print(A1[0:100])
print(A2[0:100])
