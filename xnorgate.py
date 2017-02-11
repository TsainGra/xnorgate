import numpy as np
import random

# sigmoid function
def sigmoid(x, derivative=False):
	if(derivative == True):
		return x * (1-x)
	return 1/(1+np.exp(-x))

# input dataset
x = np.array([[0,0], [0,1], [1,0], [1,1]])

# transposing y
y = np.array([[1,0,0,1]]).T

# for consistency of random seed
# np.random.seed(12345)

# initializing random weights
synapses_0 = 2*np.random.random((2,4))-1
synapses_1 = 2*np.random.random((4,1))-1

# number of epochs
n_epoch = 1000000

# gradient descent
for iter in xrange(n_epoch):
	l0 = x
	l1 = sigmoid(np.dot(l0,synapses_0))
	l2 = sigmoid(np.dot(l1,synapses_1))
	l2_error = y - l2
	error = np.mean(np.abs(l2_error))
	print 'error=',error
	# calculating weights difference using gradient
	l2_delta = l2_error * sigmoid(l2, derivative=True)
	l1_error = l2_delta.dot(synapses_1.T)
	l1_delta = l1_error * sigmoid(l1, derivative=True)
	# update weights
	synapses_1 += l1.T.dot(l2_delta)
	synapses_0 += l0.T.dot(l1_delta)

print 'Output after training'
Y = []
for row in l2:
	if row[0] >= 0.75:
		Y.append(1)
	else:
		Y.append(0)

print Y