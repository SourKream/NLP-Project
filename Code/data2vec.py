import numpy as np
import sys
from myutils import *
from process import *

DIM_SIZE = 300
WINDOW = 4

def data2vec(data,labels,RMat,dictionary):
	flag = 0
	X_train = []
	label = []
	for p,h,l in data:
		if(l not in labels):
			continue
		label.append(labels[l])
		p = tokenize(p)
		h = tokenize(h)
		if(len(p) > 40 or len(h) > 40):
			continue
		for i in xrange(40 - len(p)):
			p = ['unk'] + p
		for i in xrange(40 - len(h)):
			h = h + ['unk']
		lst = p + h
		accum = np.empty((300,1))
		for idx in xrange(len(lst)):
			wrd = lst[idx]
			if (wrd in dictionary):
				accum = np.hstack((accum,RMat[dictionary[wrd]].transpose().reshape(300,1)))
			else:
				#averaging
				jdx = idx - 1
				tmat = np.zeros((1,300))
				#TODO: Change this to windowsz
				cnt = 4 
				count = 0
				while(cnt > 0 and jdx >= 0):
					if(lst[jdx] in dictionary):
						tmat += RMat[dictionary[lst[jdx]]]
						count += 1
						cnt -=1
					jdx -= 1
				cnt = 4
				jdx = idx + 1
				while(cnt > 0 and jdx < len(lst)):
					if(lst[jdx] in dictionary):
						tmat += RMat[dictionary[lst[jdx]]]
						count += 1
						cnt -=1
					jdx += 1
				tmat /= count
				accum = np.hstack((accum,tmat.transpose()))
		X_train += [accum]
	lsz = len(label)
	X_train = np.array(X_train)
	return X_train,np.array([label]).reshape((lsz,1))

f = open(sys.argv[1])
train = [l.strip().split('\t') for l in f]
vocab = get_vocab(train)

labels = {'neutral':0, 'contradiction':-1,'entailment':1}
RMatrix = np.load('VocabMat.npy')
with open('Dictionary.txt','r') as inf:
    dictionary = eval(inf.read())

X_train,y_train = data2vec(train,labels,RMatrix,dictionary)
print X_train.shape


