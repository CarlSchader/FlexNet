import numpy as np
import flexNet as flex

m = int(input("number of training examples: "))

Xtrain = np.random.randint(0,10,size=(2,m))
Xtest = np.random.randint(0,10,size=(2,m))
Ytrain = np.zeros((2,m))
Ytest = np.zeros((2,m))

for i in range(m):
	Ytrain[0][i] = int((Xtrain[0][i] + Xtrain[1][i])/10)
	Ytrain[1][i] = (Xtrain[0][i] + Xtrain[1][i])%10
	Ytest[0][i] = int((Xtest[0][i] + Xtest[1][i])/10)
	Ytest[1][i] = (Xtest[0][i] + Xtest[1][i])%10


print(Xtrain)
print(Ytrain)


net = flex.net([2,8,16,8,2],learningRate=0.1, regularized=True)

trainingIterations = int(input("number of traing iterations: "))

for i in range(trainingIterations):
	net.forwardPropogate(Xtrain)
	net.backPropogate(Ytrain)

print(Xtest)
print(net.forwardPropogate(Xtest))