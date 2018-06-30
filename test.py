import numpy as np
import flexNet as flex

m = int(input('number of training examples: '))

Xtrain1 = np.random.randint(0,2,size=(1,m))
Xtrain2 = np.random.randint(0,2,size=(1,m))
Xtrain3 = np.random.randint(0,2,size=(1,m))
Xtrain4 = np.random.randint(0,2,size=(1,m))
Xtrain = np.concatenate((Xtrain1,Xtrain2),axis=0)
Xtrain = np.concatenate((Xtrain,Xtrain3),axis=0)
Xtrain = np.concatenate((Xtrain,Xtrain4),axis=0)
print(Xtrain1)
print(Xtrain2)
print(Xtrain3)
print(Xtrain4)
print(Xtrain)

