import numpy as np
import flexNet as flex

m = int(input("number of training examples: "))

Xtrain = np.random.randint(0,2,size=(3,m))
Xtest = np.random.randint(0,2,size=(3,m))

Ytrain = np.zeros((m,2))
Ytest = np.zeros((m,2))
for i in range(m):
	if np.sum(Xtrain.T[i]) == 0:
		Ytrain[i] = [0,0]
	elif np.sum(Xtrain.T[i]) == 1:
		Ytrain[i] = [0,1]
	elif np.sum(Xtrain.T[i]) == 2:
		Ytrain[i] = [1,0]
	elif np.sum(Xtrain.T[i]) == 3:
		Ytrain[i] = [1,1]

	if np.sum(Xtest.T[i]) == 0:
		Ytest[i] = [0,0]
	elif np.sum(Xtest.T[i]) == 1:
		Ytest[i] = [0,1]
	elif np.sum(Xtest.T[i]) == 2:
		Ytest[i] = [1,0]
	elif np.sum(Xtest.T[i]) == 3:
		Ytest[i] = [1,1]

Ytrain = Ytrain.T
Ytest = Ytest.T

# print(Xtrain)
# print(Ytrain)
# print()
# print(Xtest)
# print(Ytest)

net = flex.net([3,3,2],binaryClassification=True)

trainingIterations = int(input("number of training iterations: "))

for i in range(trainingIterations):
	net.forwardPropogate(Xtrain)
	net.backPropogate(Ytrain)

print(Xtest)
outcome = net.forwardPropogate(Xtest,round=True)

print(outcome)
correct = 0
for i in range(m):
	if outcome.T[i][0] == Ytest.T[i][0] and outcome.T[i][1] == Ytest.T[i][1]:
		correct += 1

print(float(correct)/float(m) * 100,'%')
