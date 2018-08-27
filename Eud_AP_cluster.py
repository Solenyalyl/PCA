#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,sys,re
import math
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
import numpy as np
from fastdtw import fastdtw
from  sklearn.cluster  import  AffinityPropagation
from sklearn import preprocessing
import matplotlib.pyplot as plt
from itertools import cycle
'''
def Gauss():
	sigma = 0.5
	mu = 34
	weight_array = np.zeros(58)
	for i in range(58):
		weight_array[i] = (1/(2*math.pi*sigma**2)) * math.e**(-((i-mu)**2)/(2*sigma**2))
	weight_array /= weight_array.sum()
	return weight_array

'''
def EU_D(list2):
	Distance = []
	matrix_list2 = list2
	for i in range(len(matrix_list2)):
		subDistance = []
		for j in range(len(matrix_list2)):
			euDistance = 0
			for k in range(len(matrix_list2[i])):
				euDistance += (matrix_list2[i][k][0] - matrix_list2[j][k][0])**2
			subDistance.append(euDistance)
		Distance.append(subDistance)
	return Distance
'''
def DTW(list2):
	similarityMatrix = []
	for i in range(len(list2)):
		subSimilarityMatrix = []
		for j in range(len(list2)):
			X = np.array(list2[i])
			Y = np.array(list2[j])					
			distance, path = fastdtw(X,Y, dist = euclidean)
			subSimilarityMatrix.append(-distance)
		similarityMatrix.append(subSimilarityMatrix)
	similarityMatrix = np.array(similarityMatrix)
#	return similarityMatrix
	return np.min(similarityMatrix), np.median(similarityMatrix),np.max(similarityMatrix)
'''	
def AP(list2):
	X = EU_D(list2)
	p = np.median(X)
	af = AffinityPropagation(damping = 0.5, preference = p, affinity='precomputed').fit(X)
	label = af.fit_predict(X)
	cluster_centers_indices = af.cluster_centers_indices_.tolist()
	n_clusters_ = len(af.cluster_centers_indices_)

	return label, cluster_centers_indices, n_clusters_


def clasify(labels, list2):
	clusters, centers, n_clusters_ = AP(list2)
	classes = []
	for i in range(len(centers)):
		T = []
		for j in range(len(clusters)):
			if i ==clusters[j]:
				T.append(labels[j])
		classes.append(T)
	return classes
	 
if __name__=='__main__':
	infile1 = sys.argv[1]
	infile2 = sys.argv[2]
#	outfile = sys.argv[3]
	IN1 = open(infile1,'r')
	IN2 = open(infile2,'r')
#	OUT = open(outfile,'w')
	list1 = []

	for line in IN1:
		line = line.strip("\n")
		line = line.split("\t")
#		list1.append([math.log10(float(line[0]) + 1e-99),math.log10(float(line[1]))])
		list1.append([math.log10(float(line[1]))])
	list2 = []
	list1 = np.array(list1)
	list1_scalled = preprocessing.scale(list1)

	list3 = []
	for i in range(int(len(list1_scalled)/58)):
		list3 = list1_scalled[(i) * 58:(i + 1) * 58].tolist()
		list2.append(list3)
	list2 = np.array(list2)
#	print(list2)
	labels = []
	for line2 in IN2:
		line2 = line2.strip("\n")
		labels.append(line2)

	IN1.close()
	IN2.close()

#	print(len(list2))

#	main_element = Pca(list2)
#	print(main_element)
#	simiMatrix=EU_D(list2)
#	print(simiMatrix)


	cluster, centers, k = AP(list2)
	print(cluster, centers, k)
#	print(labels)
	classes = clasify(labels, list2)
	for i in range(len(classes)):
		print(classes[i])
		
#	num = cos_sim(list2)
#	print(num)


	
