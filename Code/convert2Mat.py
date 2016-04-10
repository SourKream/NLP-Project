import numpy as np
import sys
from myutils import *
from process import *

DIM_SIZE = 300
WINDOW = 4
f = open(sys.argv[1])
train = [l.strip().split('\t') for l in f]
vocab = get_vocab(train)
g = open('Dictionary.txt','w')
g.write(str(vocab))

labels = {'neutral':0, 'contradiction':-1,'entailment':1}
RMatrix = np.zeros((len(vocab), DIM_SIZE))

vocab.pop('unk',None)
vocab.pop('delimiter',None)

inGlove = set()



print 'Loading GloVe. This takes time'
f = open(sys.argv[2])
for line in f:
	line = line.split()
	if (line[0] in vocab):
		inGlove.add(line[0])
		RMatrix[vocab[line[0]]] = np.array([map(lambda x: eval(x), line[1:])])
print 'Done loading GloVe. Yay :)'
for p,h,l in train:
	if(l not in labels):
		continue
	p = tokenize(p)
	h = tokenize(h)
	#Combine the vocab lists

	p = p + h
	for idx in xrange(len(p)):
		wrd = p[idx]
		if(wrd == 'unk' or wrd == 'delimiter'):
			continue
		elif(wrd in inGlove):
			continue
		else:
			jdx = idx - 1
			cnt = WINDOW
			counter = 0
			tmat = np.zeros((1,DIM_SIZE))
			while(cnt > 0 and jdx >= 0):
				if(p[jdx] in inGlove):
					tmat += RMatrix[vocab[p[jdx]]]
					counter += 1
					cnt -= 1
				jdx -=1
			cnt = WINDOW
			jdx = idx + 1
			while(cnt > 0 and jdx < len(p)):
				if(p[jdx] in inGlove):
					tmat += RMatrix[vocab[p[jdx]]]
					counter += 1
					cnt -= 1
				jdx +=1
			tmat /= counter
			#Running average
			RMatrix[vocab[wrd]] = (RMatrix[vocab[wrd]] + tmat) / 2

print RMatrix.shape
np.save('VocabMat.npy',RMatrix)



