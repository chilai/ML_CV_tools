#!/usr/bin/env python 
from __future__ import print_function
import numpy as np
from training import *

#score function
def score(Xag,W):
	"""Input:
	Xag : augmented numpy array of features
	W : aumented coefficients
	the bias coefficients correspond to the last row of the matrix W
	Output:
	out : numpy array of scores"""
	out = np.dot(Xag, W)
	return out

#Loss function and its gradient with respect to the parameters
def loss_gradient(Xag,Y,W, reg, Delta =1):
    N = Xag.shape[0]
    S = score(Xag,W)
    
    label_score = S[np.arange(N),Y] #scores corresponding to correct label
    #defining loss function
    loss = np.sum(np.maximum(0,S-label_score[:, np.newaxis] + Delta))/N + 0.5*reg*np.sum(W[:-1,:]**2)/N -Delta
    #computing the gradients
    dmax = 1.0*((S-label_score[:,np.newaxis]+ Delta > 0))
    grads= np.dot(Xag.T,dmax) 
    B = np.arange(W.shape[1])[np.newaxis,:] == Y[:,np.newaxis]
    A = Xag*(np.sum(dmax, axis =1)[:,np.newaxis])
    grads -= np.dot(A.T,B)
    grads[:-1,:] += reg*W[:-1,:]
    grads /= N
    return loss,grads
    
#-----------------------------------------------------------------------
    
class SVM(object):
	
	def __init__(self, input_dims, num_classes, reg):
		self.num_classes = num_classes
		self.params = {}
		self.params['w'] = np.random.randn((input_dims +1),self.num_classes)
		self.reg = reg
				
			
	def loss(self,X,y=None):
		Xaug = np.append(X,np.ones((X.shape[0],1)), axis = 1)
		reg = self.reg
		
		W = self.params['w']
		if y is None:
			return score(Xaug,W)
		Loss, grad = loss_gradient(Xaug,y,W,reg)
		grads={}
		grads['w'] = grad
		return Loss, grads
		
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
		
	
		
			
		
	
	
		
				
				
        
        
		
		
		
		 
		
		
		
