#!/usr/bin/env python 
from __future__  import print_function
import numpy as np
import sys
#Python scripts that implements class for training machine learning 
#classification models using stochastic update methods


current_module = sys.modules[__name__]
#-----------------------------------------------------------------------
#Collection of update functions for training model parameters
# hyperparams : = dictionary of hyperparameters values used for training model

def gradient_descent(w,grad,hyperparams={}):  #regular gradient descent
	rate = hyperparams.setdefault('learning_rate',0.01)
	out = w-rate*grad
	return out, hyperparams
	
def momentum(w,grad, hyperparams={}): #gradient descent with momentum
	rate = hyperparams.setdefault('learning_rate',0.01)
	momentum = hyperparams.setdefault('learning_rate',0.9) #momentum
	v    = hyperparams.setdefault('velocity', np.zeros_like(w))
	v = momentum * v - rate*grad  #update velocity
	out  = w + v #update w
	hyperparams['velocity'] = v #store updated velocity
	return out, hyperparams
	
def adam(w, grad, hyperparams ={}):  #Adam update rule
	rate = hyperparams.setdefault('learning_rate',0.01)
	beta1 = hyperparams.setdefault('beta1',0.9) # exponential decay rate for 1st moment 
	beta2 = hyperparams.setdefault('beta2',0.999) #exponential decay rate for 2nd moment
	epsilon = hyperparams.setdefault('epsilon',1e-8) #constant for smoothing purposes
	m   = hyperparams.setdefault('m', np.zeros_like(w))  #first moment 
	v   = hyperparams.setdefault('v', np.zeros_like(w))  #second moment
	t  =  hyperparams.setdefault('t',0) #iteration number
	t += 1
	m  = beta1*m + (1-beta1)*grad #update first moment
	v = beta2*v + (1-beta2)*grad**2    #update second moment
	m_hat = m/(1-beta1**t)   #compute corrected first moment estimate
	v_hat = v/(1-beta2**t)   #compute corrected second moment estimate
	out = w - rate*m_hat/(np.sqrt(v_hat) + epsilon)
	hyperparams['m'] = m 
	hyperparams['v'] = v
	hyperparams['t'] = t
	return out,  hyperparams
	
#-----------------------------------------------------------------------
def predict(X,loss_function):
	S = loss_function(X)
	prediction = np.argmax(S, axis =1)
	return prediction
	
def accuracy(X,y,loss_function, batch_size = 400):
	N = len(X)
	num_batches = max(len(X)//batch_size,1)
	num_correct = 0                     #number of correct predictions
	for i in range(num_batches+1):
		X_batch = X[i*num_batches:min((i+1)*num_batches,N),:]
		y_batch =y[i*num_batches:min((i+1)*num_batches,N)]
		batch_prediction = predict(X_batch,loss_function)
		num_correct += np.sum(batch_prediction == y_batch)
	return num_correct*100.0/N

#------------------------------------------------------------------------
class Training(object):
	
	def __init__(self,model,data_train, data_val,**extrargs):
		self.model = model
		#Storing training and validation data
		self.Xtrain = data_train['Xtrain']
		self.Ytrain = data_train['Ytrain']
		self.Xval = data_val['Xval']
		self.Yval = data_val['Yval']
		#Optional arguments
		self.update_method = extrargs.pop('update_method', 'gradient_descent') #update method used for training 
		self.rate_decay  = extrargs.pop('rate_decay',1.0)  #decay rate of learning rate parameter
		self.batch_size = extrargs.pop('batch_size',400)  #size of batch
		self.num_epochs = extrargs.pop('num_epochs',10) #number of epochs to run training
		self.print_rate = extrargs.pop('print_rate',10)  #how often to print
		self.printing   = extrargs.pop('printing',True) #option to print
		self.update_hyperparams = extrargs.pop('update_hyperparams', {})  #hyperparameters for update method
		#Checking that the update_method is included in the training.py script and retrieve it 
		try:
			self.update_method = getattr(current_module, self.update_method)
		except AttributeError:
			print('Invalid update_method "%s"' % self.update_method)
				
		#Throw error if there are other keyword arguments
		if len(extrargs) > 0:
			left = ', '.join('"%s"' % k for k in extrargs.keys())
			raise ValueError('Unrecognized arguments %s' % left)
		#Reset tracking variables
		self._start()
		
	def _start(self):
		#Reset some tracking and storage variables
		self.epoch  = 0        #current epoch
		self.bestval_accuracy = 0  #best validation accuracy
		self.related_trainacc = 0 #corresponding training accuracy
		self.trainingaccuracy_hist =[] #storage of training accuracy values
		self.valaccuracy_hist = []  #storage of validation accuracy values
		self.bestparams = {}   #best model paramaters
		self.loss_hist = []  #storage of training loss values
		#Copying update method hyperparameters for each parameter of the model
		self.training_hyperparams = {}
		for k in self.model.params:
			values = {k: v for k, v in self.update_hyperparams.items()}
			self.training_hyperparams[k] = values
			
	
	def _update(self):
		
		#selecting a batch for update 
		batch_indices = np.random.choice(len(self.Xtrain), self.batch_size)
		Xbatch = self.Xtrain[batch_indices,:]
		Ybatch = self.Ytrain[batch_indices]
		#Computing loss and gradient and storing loss value
		loss, grads = self.model.loss(Xbatch,Ybatch)
		self.loss_hist.append(loss)
		#Perform update of parameters
		for k, v in self.model.params.items():
			grad = grads[k]
			hyperparams = self.training_hyperparams[k]
			w, new_hyperparams = self.update_method(v,grad, hyperparams)
			self.model.params[k] = w
			self.training_hyperparams[k] = new_hyperparams
	
	def train(self):
		"""Training the model."""
		self._start()
		iterations_per_epoch = max(len(self.Xtrain)// self.batch_size, 1)
		num_iterations = self.num_epochs * iterations_per_epoch
		for i in range(num_iterations):
			self._update()
			#print training loss (optional)
			if self.printing and ((i+1) % self.print_rate == 0):
				print('Iteration %s / %s) loss: %s' % (i + 1, num_iterations, self.loss_hist[-1]))
			#Increment epoch counter at the end of every epoch and decay learning rate
			if (i+1) % iterations_per_epoch == 0 :
				self.epoch += 1
				for k in self.training_hyperparams:
					self.training_hyperparams[k]['learning_rate'] *=  self.rate_decay
			#Checking training accuracy and validation accuracy at the
			#first and last iterations and the end of each epoch
			checking = (i == 0) or ((i+1) % iterations_per_epoch == 0) or (i+1 == num_iterations)
			if checking:
				training_acc = accuracy(self.Xtrain,self.Ytrain,self.model.loss,self.batch_size)
				val_acc = accuracy(self.Xval,self.Yval,self.model.loss,self.batch_size)
				self.trainingaccuracy_hist.append(training_acc)
				self.valaccuracy_hist.append(val_acc)
				if self.printing:
					print('(Epoch %s / %s) training accuracy: %s; validation accuracy: %s' 
					% (self.epoch, self.num_epochs, training_acc, val_acc))
				#keep track of best validation accuracy
				if val_acc > self.bestval_accuracy:
					self.bestval_accuracy = val_acc
					self.related_trainacc = training_acc
					self.bestparams = {}
				for k, v in self.model.params.items():
					self.bestparams[k] = v.copy()
		#At the end of training, set model parameters to best parameters and print bestval_accuracy
		#and related_trainacc
		self.model.params = self.bestparams
		print('The best validation accuracy: ',self.bestval_accuracy)
		print('The corresponding training accuracy: ', self.related_trainacc)
		
				
			  

			
		
			
			
		
		
		
		
		
		
		
		
		
		
		
			
		
		
			
		
	
	
		
			
		
		
		
		
		
		
		
	
	
	
	
	
	
	
