#! /Users/mimac/miniconda3/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math

class Model():
	def __init__(self, data):
		# data
		self.data = data
		self.generate_data()

		# hyperParam
		self.neuronsInHidden = 10
		self.inputBits = 2
		self.outputBits = 1
		self.lr = 0.1

		# weights
		self.w1 = np.random.randn(self.inputBits, self.neuronsInHidden)
		self.w2 = np.random.randn(self.neuronsInHidden, self.neuronsInHidden)
		self.w3 = np.random.randn(self.neuronsInHidden, self.outputBits)
		self.b1 = np.zeros((self.neuronsInHidden, 1))
		self.b2 = np.zeros((self.neuronsInHidden, 1))
		self.b3 = np.zeros((self.outputBits, 1))

		self.lossList = []
	
	def softmax(self, x):
		soft_x = np.exp(x) / np.sum(np.exp(x))
		return soft_x

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
	
	def derivative_sigmoid(self, x):
		return np.multiply(x, 1-x)

	def mseloss(self, out, y):
		return np.mean((out - y)**2)

	def dloss(self, out, y): 
		return out.flatten() - y.flatten()
	
	def stepFunction(self, out):
		return np.array((out > 0.5) * 1)

	def generate_linear(self, n=100):
		pts = np.random.uniform(0, 1 , (n,2))
		inputs = []
		labels = []
		for pt in pts:
			inputs.append([pt[0], pt[1]])
			distance = (pt[0]-pt[1])/1.414
			if pt[0] > pt[1]:
				labels.append(0)
			else:
				labels.append(1)
		return np.array(inputs), np.array(labels).reshape(n, 1)

	def generate_XOR_easy(self):
		inputs = []
		labels = []
		for i in range(11):
			inputs.append([0.1*i, 0.1*i])
			labels.append(0)
			if 0.1*i == 0.5:
				continue
			inputs.append([0.1*i, 1-0.1*i])
			labels.append(1)
		return np.array(inputs), np.array(labels).reshape(21, 1)

	def generate_data(self):
		if self.data.lower() == 'xor':
			self.x, self.y = self.generate_XOR_easy()
		if self.data.lower() == 'linear':
			self.x, self.y = self.generate_linear()
		
	def forwardPropagation(self):
		self.z1 = np.dot(self.w1.T, self.x.T) + self.b1
		self.a1 = self.sigmoid(self.z1)
		self.z2 = np.dot(self.w2, self.a1) + self.b2
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.w3.T, self.a2) + self.b3
		self.out = self.sigmoid(self.z3)
	
	def backwardPropagation(self):
		self.dout = self.dloss(self.out, self.y)
		self.dz3 = np.multiply(self.dout, self.derivative_sigmoid(self.out))
		self.dw3 = np.dot(self.dz3, self.a2.T) 
		self.db3 = np.sum(self.dz3, axis = 1, keepdims = True)

		self.da2 = np.dot(self.dz3.T, self.w3.T)
		self.dz2 = np.multiply(self.da2.T, self.derivative_sigmoid(self.a2))
		self.dw2 = np.dot(self.dz2, self.a1.T) 
		self.db2 = np.sum(self.dz2, axis = 1, keepdims = True) 

		self.da1 = np.dot(self.w2.T, self.dz2)
		self.dz1 = np.multiply(self.da1, self.derivative_sigmoid(self.a1))
		self.dw1 = np.dot(self.dz1, self.x) 
		self.db1 = np.sum(self.dz1, axis = 1, keepdims = True)

	def updateParameters(self):
		self.w1 = self.w1 - self.lr * self.dw1.T
		self.w2 = self.w2 - self.lr * self.dw2.T
		self.w3 = self.w3 - self.lr * self.dw3.T

		self.b1 = self.b1 - self.lr * self.db1
		self.b2 = self.b2 - self.lr * self.db2
		self.b3 = self.b3 - self.lr * self.db3

	def train(self, epoch):
		for i in range(epoch):
			self.forwardPropagation()
			self.backwardPropagation()
			self.updateParameters()
			# print(self.stepFunction(self.out), self.y)
			loss = self.mseloss(self.out.flatten(), self.y.flatten())
			self.lossList.append(loss)
			print(f'epoch {i+1}, loss{loss}')

	def show_result(self):
		plt.subplot(1,2,1)
		plt.title("Ground truth", fontsize=18)
		for i in range(self.x.shape[0]):
			if(self.y[i] == 0):
				plt.plot(self.x[i][0], self.x[i][1], "ro")
			else:
				plt.plot(self.x[i][0], self.x[i][1], "bo")

		plt.subplot(1,2,2)
		plt.title("Predict result", fontsize=18)
		for i in range(self.x.shape[0]):
			if self.out.flatten()[i] < 0.5:
				plt.plot(self.x[i][0], self.x[i][1], "ro")
			else:
				plt.plot(self.x[i][0], self.x[i][1], "bo")
		plt.show()
if __name__ == '__main__':
	# Hyper Parameters
	EPOCH = 100000

	xorModel = Model('xor')
	xorModel.train(EPOCH)
	xorModel.show_result()
	
	linearModel = Model('linear')
	linearModel.train(EPOCH)
	linearModel.show_result()

