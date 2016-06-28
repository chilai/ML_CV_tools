#!/usr/bin/env python 
from __future__  import print_function
import numpy as np
from training import *
#Python script defining a class for a neural network classifier

#------------------------------------------------------------------------
#Implementation of the forward and backward propagation for the 
#different operations in a layer of a neural network

#1)Linear part 
# x:= input of layer
#w, b := paramters of linear function

def linear_forward(x,w,b): #forward propagation
	out = np.dot(x,w) + b 
	cache = [x,w,b]  #storing values for back propagation
	return out, cache
	
def linear_backward(dout,cache): #backward propagation
    x,w,b = cache
    dx = np.dot(dout, w.T)
    dw = np.dot(x.T,dout)
    db = np.sum(dout, axis= 0)
    return dx, dw, db #gradients
    
#2)RELU activation function
def RELU_forward(x):
	out = np.fmax(x,0)
	cache = x
	return out, cache
	
def RELU_backward(dout, cache):
	x = cache
	dx = dout*(x > 0)
	return dx

#3)Composite functions
def linear_RELU_forward(x, w, b): #linear function followed by RELU activation
  a, af_cache = linear_forward(x, w, b)
  out, relu_cache = RELU_forward(a)
  cache = (af_cache, relu_cache)
  return out, cache

def linear_RELU_backward(dout, cache):
  af_cache, relu_cache = cache
  da = RELU_backward(dout, relu_cache)
  dx, dw, db = linear_backward(da, af_cache)
  return dx, dw, db
  
#output and gradient of loss function corresponding to application of
#softmax function for estimating class probabilities
#x := numpy array of scores
#y:= numpy array of assigned class labels
def softmax_loss(x, y):
  probs = np.exp(x)
  probs /= np.sum(probs, axis=1, keepdims=True) #softmax 
  N = len(x)
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx #loss value and gradient

#----------------------------------------------------------------------- 
"""Implementation of a neural network class"""
class NeuralNetwork(object):
	#1) Constructor
	def __init__(self, input_dims, hidden_dims, output_dims, reg, dtype=np.float32):
		"""hidden_dims := list consisting of the dimension of each hidden layer
		output_dims := number of classes"""
		self.input_dims = input_dims #dimensions of input features
		self.reg = reg  #regularization hyperparamter
		self.num_layers = 1 + len(hidden_dims) #total number of layers of network
		self.dtype = dtype
		self.params = {} #initializing dictionary to store all the parameters of model
		self.model_dims = hidden_dims[:]
		self.model_dims.append(output_dims)
		self.model_dims.insert(0, input_dims)
		model_dims = self.model_dims #list consisting of all dimensions 
		                             # including input and output dimensions
		#Initializing the parameters
		for i in range(1, len(model_dims)):
			num = str(i)
			#parameters for linear function of each layer
			self.params['b'+num] = np.zeros(model_dims[i])
			self.params['W'+ num] = np.random.randn(model_dims[i-1], model_dims[i])*np.sqrt(2.0/(model_dims[i-1] + model_dims[i]))
						
		#Setting the dtype for the paramater values
		for k, v in self.params.items():
			self.params[k] = v.astype(self.dtype)
		
	#2)Loss function given input data
	def loss(self,X, y = None):
		"""X:= input features
		y:= array of class labels if provided"""
		model_dims = self.model_dims
		X = X.astype(self.dtype)
		#Checking if input dimensions are correct
		if X.shape[1] != self.input_dims:
			raise TypeError('Input has wrong size. Not equal to input dimensions for network')
		mode = 'test' if y is None else 'train'
		scores = None
		#Implementing scores
		cache_storage = []
		xin = X
		for i in range(1,len(model_dims)):
			num = str(i)
			w = self.params['W'+ num]
			b = self.params['b'+ num]
			if i == len(model_dims)-1:
				xout, cache = linear_forward(xin, w, b)
			else:
				xout, cache = linear_RELU_forward(xin, w, b)
			#storing caches											
			cache_storage.append(cache)
			xin = xout
			
		scores = xin #final layer output 
		
		if mode == 'test': #no response variable y
			return scores  
			
		loss, grads = 0.0, {}
		#Finding the loss and gradients
		loss, dscores = softmax_loss(scores,y)
		loss_reg = 0.0
		for k, v in self.params.items():
			if k.startswith('W'):
				loss_reg += 0.5*self.reg*np.sum(v**2)
		loss += loss_reg
		#Getting the gradients
		dout = dscores
		for i in range(len(model_dims)-1,0,-1):
			cache = cache_storage.pop()
			num = str(i)
			if i == len(model_dims)-1:
				dx, dw, db = linear_backward(dout, cache)
			else:
				dx, dw, db = linear_RELU_backward(dout, cache)
								
			grads['W'+num] = dw + self.reg*self.params['W'+num]
			grads['b'+num] = db
			dout = dx
		
		return loss, grads
		
	
	#3)Function to predict the class label
	def class_predict(self,X):
		return predict(X,self.loss) #invoving function from training.py 
	
	#4)Function to estimate accuracy of prediction	
	def model_accuracy(self,X,Y):
		return accuracy(X,Y, self.loss)
		
	#5)Function for stochastic training of neural network using Training class from training.py
	def stoch_train(self,data_train, data_val,**extrargs):
		"""data_train, data_val and extrargs are of the same type used
		in Training class constructor."""
		model = self
		Trainer = Training(model,data_train, data_val,**extrargs) #
		Trainer.train()


		
			
				
			
			
		
		
		
		
	
			
		
			

		
		
	

		
		  
        
	


	

	
	
	
	
