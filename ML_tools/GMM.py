#!/usr/bin/env python 
from __future__  import print_function
import numpy as np
import math
import  matplotlib.pyplot as plt
from Kmeans import *

#Python script defining a class for Gaussian mixture models

#-----------------------------------------------------------------------
def normal_pdf(x,mu,sigma):
	"""Function to compute normal pdf
	Input:
	x := numpy data array
	mu := numpy array of mean of normal pdf
	sigma := numpy array of variance of normal pdf
	Output:
	out := numpy array corresponding to normal pdf values for each data row
	"""
	D = x.shape[1]
	precision = np.linalg.inv(sigma)
	det = np.linalg.det(sigma)
	out = np.exp(-np.sum(np.dot(x-mu,precision/2)*(x-mu), axis =1))/math.sqrt((2*math.pi)**D*det)
	
	return out
	
def responsibilities(x,params,mixing):
	"""Function to update the responsibilities
	Input:
	x := numpy array of data
	params:= list storing the paramaters for each component
	where each entry stores a list [mu,sigma] corresponding to the mean,
	the variance and  the mixing component for a Gaussin component
	mixing := numpy array of mixing coefficients
	Output:
	R := numpy array of responsibilities
	with Rij denoting responsibility of jth component for ith data """
	K = len(mixing)     #number of Gaussian components
	P = np.zeros((len(x),K))  
	for k in range(K):
		mu  = params[k][0]
		sigma = params[k][1]
		P[:,k] = normal_pdf(x,mu,sigma)
	R = np.dot(P,np.diag(mixing))
	Rn = np.sum(R,axis =1, keepdims = True)
	R /= Rn  
	return R  
	
		
def update_parameters(x,R):
	"""Function to update the responsibilities
	Input:
	x := numpy array of data
	params:= list storing the paramaters for each component
	where each entry stores a list [mu,sigma] corresponding to the mean,
	the variance and  the mixing component for a Gaussin component
	mixing := numpy array of mixing coefficients
	R := numpy array of responsibilities
	with Rij denoting responsibility of jth component for ith data
	Output:
	new_params:= list storing updated parameters
	new_mixing  = numpy array of updated mixing coefficients"""
	N = len(x)
	#update mixing coefficients
	mixing_freq = np.sum(R,axis=0)
	new_mixing = mixing_freq/N
	#update parameters
	Mu = np.dot(x.T,R)/mixing_freq
	new_params = []
	for k in range(R.shape[1]):
		mu  = Mu[:,k]
		resp = R[:,k]
		sigma = np.dot(((x-mu).T)*resp, x-mu)/mixing_freq[k]
		new_params.append([mu,sigma])
	return new_mixing, new_params
	
	
class GaussianMM(object):
	
	#constructor
	def __init__(self,num_comps,params = None, mixing = None):
		self.num_comps = num_comps #number of Gaussian components
		self.params = params
		self.mixing  = mixing 
		
	def train(self,X, max_iterations = 1000, plotting = False):
		"""Function to train the model using the EM-algorithm"""
		K  = self.num_comps
		#Initializing the parameters if necessary
		if self.mixing  is None:
			self.mixing  = (1.0/K)*np.ones(K)
		#Using Kmeans to initialize centroids and 
		if self.params is None:
			indices_choice = np.random.choice(len(X),K)
			centroids = []
			for k in range(K):
				mu =  X[indices_choice[k],:] 
				centroids.append(mu)
			#Running Kmeans for 4 iterations
			for i in range(4):
				indices = closest_centroid(X,centroids)
				centroids,variances = update_centroids(X,indices,K)
			self.params = []
			for k in range(K):
				self.params.append([centroids[k],variances[k]])
											
		rel_diff = 1
		log_lkh = [] #list to store data log-likelihood during training
		#Starting training
		i = 0 #iteration counter
		while (i < max_iterations) and (rel_diff > 1e-8):
			i += 1
			#E-step
			Resp = responsibilities(X,self.params,self.mixing)
			#M-step
			self.mixing, self.params = update_parameters(X,Resp)
			newLogLkh =  self.log_likelihood(X) #new log likelihood value
			if i > 1:
				rel_diff = abs(newLogLkh -log_lkh[-1])/max(abs(newLogLkh),abs(log_lkh[-1]))
			log_lkh.append(newLogLkh)
		if plotting  == True:
			#Plotting the log_likelihood vs iteration and saving the figure
			x = np.arange(len(log_lkh))
			fig = plt.figure()
			plt.plot(x,log_lkh)
			plt.xlabel('iterations')
			plt.ylabel('Log likelihood')
			plt.savefig('GMMtraining.jpg')
			plt.close(fig)
			
	def log_likelihood(self,X):
		K  = self.num_comps
		P = np.zeros((len(X),K)) 
		for k in range(K):
			mu  = self.params[k][0]
			sigma = self.params[k][1]
			P[:,k] = normal_pdf(X,mu,sigma)
		return np.sum(np.log(np.dot(P,self.mixing)))
			
	def model_responsibility(self,X):
		return responsibilities(X,self.params,self.mixing)
		
	def hard_clustering(self,X):
		#Cluster assignmnet based on maximum of responsibilities
		R = self.model_responsibility(X)
		return np.argmax(R,axis=1)
		
	def get_centroids(self):
		"""List to store means of Gaussian components."""
		Centroids = []
		for k in range(self.num_comps):
			Centroids.append(self.params[k][0])
		return Centroids
	
		
	
			
			
		
			
			
			
			
	
				
		
						 
		
		
     
		
	
	
	
	

	
	 
	
	
	
	
	
	
	
	
	
