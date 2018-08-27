#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,sys,re
import math
from sklearn.decomposition import PCA
#from scipy.spatial.distance import euclidean
import numpy as np
#from fastdtw import fastdtw
#from  sklearn.cluster  import  AffinityPropagation
from sklearn import preprocessing
import matplotlib.pyplot as plt
#from itertools import cycle
import scipy
import scipy.cluster.hierarchy as sch


def Pca(list2):
	list4 = []
	for i in range(len(list2)):
		pca=PCA(n_components= 1)
		newData=pca.fit_transform(list2[i]).T.tolist()
		list4.append(newData)
	list4 = np.array(list4).reshape(-1,58)
	return list4

def hierarchy(list2, label):
	X = list2
	Y = label
	disMat = scipy.spatial.distance.pdist(X, 'euclidean')
	Z = sch.linkage(disMat, method = 'average')
	plt.figure(figsize=(30, 30))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('sample index')
	plt.ylabel('distance')
	P = sch.dendrogram(Z, leaf_font_size = 20)
	plt.savefig('euclidean__Hierarchical.png')
	cluster= sch.fcluster(Z, t = 1, criterion = 'inconsistent', depth = 2)
	cluster = np.array(cluster)
	clustCenter = np.max(cluster)
	classes = []
	indexes = []
	for i in range(clustCenter):
		j = i + 1
		T = []
		subIndex = []	
		for k in range(len(cluster)):
			if j == cluster[k]:
				T.append(Y[k])
				subIndex.append(k)
		classes.append(T)
		indexes.append(subIndex)
	return classes, indexes



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
#		list1.append([math.log10(float(line[0]) + 1e-99), math.log10(float(line[1]))])
		list1.append(float(line[1]))
	list2 = []
#	list1 = np.array(list1)
#	list1_scalled = preprocessing.scale(list1)

	list3 = []
	for i in range(int(len(list1)/58)):
		list3 = list1[(i) * 58:(i + 1) * 58]
		list3 = np.array(list3).reshape(58)
		list3_scalled = preprocessing.scale(list3)
		list3_scalled = list3_scalled.tolist()
		list2.append(list3_scalled)
	list2 = np.array(list2)
#	print(list2)
#	print(list1)
	label = []
	for line2 in IN2:
		line2 = line2.strip("\n")
		label.append(line2)
	label = np.array(label)

	IN1.close()
	IN2.close()
#	print(list2.shape)
#	print(len(list2))
	classes, indexes = hierarchy(list2, label)
	for i in range(len(classes)):
		print(i+1, classes[i])
	print(indexes)
#	for i in range(len(indexes)):
#		print(i+1, indexes[i])
	print(label)


