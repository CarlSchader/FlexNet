import numpy as np
import os

def sigmoid(z,derivative=False):
	if derivative:
		return sigmoid(z) * (1.0 - sigmoid(z))
	else:
		return 1.0 / (1.0 + np.exp(-z))

def tanh(z,derivative=False):
	if derivative:
		return 1 - (tanh(z)**2)
	else:
		return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

def ReLU(z,derivative=False):
	if derivative:
		return np.where(z > 0, 1.0, 0.0)
	else:
		return np.maximum(z,0)

# def leakyReLU(z,rate=0.01,derivative=False):
	# Unfinished

class logisticRegression:
	def __init__(self,nx,learningRate=1.0):
		self.nx = nx
		self.w = np.zeros((nx,1))
		self.b = 0.0
		self.A = np.zeros((1,nx))
		self.X
		self.learningRate = learningRate

	def forward(self,x,round=False):
		if round:
			return np.around(sigmoid(np.dot(self.w.T,x) + self.b)[0][0]).astype(int)
		else:
			return sigmoid(np.dot(self.w.T,x) + self.b)[0][0]

	def forwardPropogate(self,X,round=False):
		self.X = X
		self.A = sigmoid(np.dot(self.w.T,X) + self.b)
		if round:
			return np.around(self.A)[0].astype(int)
		else:
			return self.A[0]

	def backPropogate(self,Y):
		dldZ = self.A - Y
		dldw = np.zeros((self.nx,1))
		dldb = 0.0
		dldw = (1.0/Y.size)*np.dot(self.X,dldZ.T)
		dldb = (1.0/Y.size)*np.sum(dldZ)
		self.w = self.w - (self.learningRate*dldw)
		self.b = self.b - (self.learningRate*dldb)

class net:
	def __init__(self,layers,learningRate=1.0,binaryClassification=False,leakRate=0.01):
		self.layerCount = len(layers)
		self.learningRate = learningRate
		self.binaryClassification = binaryClassification
		self.leakRate = leakRate
		self.W = [None]*self.layerCount
		self.b = [None]*self.layerCount
		self.Z = [None]*self.layerCount
		self.A = [None]*self.layerCount

		for i in range(self.layerCount):
			if i != 0:
				self.W[i] = np.random.sample((layers[i],layers[i-1]))
				self.b[i] = np.random.sample((layers[i],1))

	def forwardPropogate(self,X,round=False):
		self.A[0] = X
		for i in range(1,self.layerCount):
			self.Z[i] = np.dot(self.W[i],self.A[i-1]) + self.b[i]
			if i == self.layerCount-1:
				if self.binaryClassification:
					self.A[i] = sigmoid(self.Z[i])
				else:
					self.A[i] = ReLU(self.Z[i])
			else:
				self.A[i] = ReLU(self.Z[i])
		if round:
			return np.around(self.A[self.layerCount-1])
		else:
			return self.A[self.layerCount-1]

	def backPropogate(self,Y):
		dZ = [None]*self.layerCount
		dW = [None]*self.layerCount
		db = [None]*self.layerCount
		for i in range(self.layerCount-1,0,-1):
			if i == self.layerCount-1:
				dZ[i] = np.subtract(self.A[i],Y)
			else:
				dZ[i] = np.dot(self.W[i+1].T,dZ[i+1])*ReLU(self.Z[i],derivative=True)
			dW[i] = (1.0/len(self.A[0].T))*np.dot(dZ[i],self.A[i-1].T)
			db[i] = (1.0/len(self.A[0].T))*np.sum(dZ[i],axis=1,keepdims=True)
		for i in range(1,self.layerCount):
			self.W[i] = self.W[i] - (self.learningRate*dW[i])
			self.b[i] = self.b[i] - (self.learningRate*db[i])

	def save(self,directory):
		if not(os.path.exists(directory)):
			os.mkdir(directory)
		for i in range(self.layerCount):
			filename = directory+"/W"+str(i)
			if os.path.exists(filename):
				os.remove(filename)
			np.save(filename,self.W[i])
			filename = directory+"/b"+str(i)
			if os.path.exists(filename):
				os.remove(filename)
			np.save(filename,self.b[i])

	def load(self,directory):
		for i in range(self.layerCount):
			filename = directory+"/W"+str(i)+".npy"
			self.W[i] = np.load(filename)
			filename = directory+"/b"+str(i)+".npy"
			self.b[i] = np.load(filename)





