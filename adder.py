import numpy as np
import flexNet as flex

m = int(input("number of training examples: "))

Xtrain = np.random.randint(0,10,size=(2,m))
Xtest = np.random.randint(0,10,size=(2,m))
Ytrain = np.zeros((1,m))
Ytest = np.zeros((1,m))

for i in range(m):
	Ytrain[0][i] = Xtrain[0][i] + Xtrain[1][i]
	Ytest[0][i] = Xtest[0][i] + Xtest[1][i]


print(Xtrain)
print(Ytrain)


net = flex.net([2,5,5,2,1])

trainingIterations = int(input("number of traing iterations: "))

for i in range(trainingIterations):
	net.forwardPropogate(Xtrain)
	net.backPropogate(Ytrain)

print(Xtest)
print(net.forwardPropogate(Xtest))