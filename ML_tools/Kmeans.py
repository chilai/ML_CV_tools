#!/usr/bin/env python 
from __future__  import print_function
import numpy as np
import math

#Module containing functions used for Kmeans clustering

def closest_centroid(x,centroids):
	"""Function for finding the closest closest centroid
	Input :
	x := numpy array of data
	centroids := list of centroids
	Output:
	out := numpy array of index of closest centroids"""
	K =len(centroids)
	N = len(x)
	Distance = np.zeros((N,K))
	for j in range(K):
		mu = centroids[j]
		Distance[:,j] = np.linalg.norm(x-mu,axis=1)
	out = np.argmin(Distance,axis=1) 
	return out
	
def update_centroids(x,indices,K):
	"""Function for updating centroids
	Input:
	x := numpy array of data
	K:= number of centroids
	indices := numpy array of indices of closest centroids
	Output :
	centroids, variance := updated list of centroids and list of variances
	for each cluster"""
	centroids = [] 
	variances = []
	for j in range(K):
		x_closest  = x[indices == j,:]
		mu = np.mean(x_closest, axis = 0)
		variance = np.dot((x_closest-mu).T,x_closest-mu)/len(x_closest)
		centroids.append(mu)
		variances.append(variance)
	return centroids, variances  
