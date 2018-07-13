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

Ytrain = Xtrain4
print(Ytrain)



net = flex.net([4,2,1],binaryClassification=True)

trainingIterations = int(input("number of training iterations: "))

for i in range(trainingIterations):
	net.forwardPropogate(Xtrain)
	net.backPropogate(Ytrain)
	print("\n",net.W[1])


Xtest1 = np.random.randint(0,2,size=(1,m))

Xtest2 = np.random.randint(0,2,size=(1,m))

Xtest3 = np.random.randint(0,2,size=(1,m))

Xtest4 = np.random.randint(0,2,size=(1,m))

Xtest = np.concatenate((Xtest1,Xtest2),axis=0)
Xtest = np.concatenate((Xtest,Xtest3),axis=0)
Xtest = np.concatenate((Xtest,Xtest4),axis=0)
Ytest = Xtest4

print(Xtest)

print(net.forwardPropogate(Xtest,round=True))

