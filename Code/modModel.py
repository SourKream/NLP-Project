# from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility
import os
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2, activity_l2
from keras.callbacks import *
# from visualizer import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical,accuracy
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
#from keras.utils.visualize_util import plot, to_graph # THIS IS BAD
# from data_reader import *
from reader import *
from myutils import *
import logging
from datetime import datetime
# from myconfig import DATAPATH,MYPATH
#Some # Defines 

def get_params():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lstm', action="store", default=150, dest="lstm_units", type=int)
    parser.add_argument('-epochs', action="store", default=20, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=32, dest="batch_size", type=int)
    parser.add_argument('-emb', action="store", default=100, dest="emb", type=int)
    parser.add_argument('-xmaxlen', action="store", default=20, dest="xmaxlen", type=int)
    parser.add_argument('-ymaxlen', action="store", default=20, dest="ymaxlen", type=int)
    parser.add_argument('-maxfeat', action="store", default=35000, dest="max_features", type=int)
    parser.add_argument('-classes', action="store", default=351, dest="num_classes", type=int)
    parser.add_argument('-sample', action="store", default=1, dest="samples", type=int)
    parser.add_argument('-nopad', action="store", default=False, dest="no_padding", type=bool)
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-load', action="store", default=False, dest="load_save", type=bool)
    parser.add_argument('-verbose', action="store", default=False, dest="verbose", type=bool)
    parser.add_argument('-embed_dim', action="store", default=300, dest="embedding_size", type=int)
    parser.add_argument('-train',action="store",default='../Data/Train.txt',dest="train_file",type=str) 
    parser.add_argument('-dev',action="store",default='../Data/Dev.txt',dest="dev_file",type=str) 
    parser.add_argument('-test',action="store",default='../Data/Test.txt',dest="test_file",type=str) 
    opts = parser.parse_args(sys.argv[1:])
    print "lstm_units", opts.lstm_units
    print "epochs", opts.epochs
    print "batch_size", opts.batch_size
    print "emb", opts.emb
    print "samples", opts.samples
    print "xmaxlen", opts.xmaxlen
    print "ymaxlen", opts.ymaxlen
    print "max_features", opts.max_features
    print "no_padding", opts.no_padding
    print "embedding_size",opts.embedding_size
    print "Training file",opts.train_file
    print "Dev file",opts.dev_file
    print "Test file",opts.test_file
    return opts

class AccCallBack(Callback):
    def __init__(self, xtrain, ytrain, xdev, ydev, xtest, ytest, vocab, opts):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xdev = xdev
        self.ydev = ydev
        self.xtest = xtest
        self.ytest = ytest
        self.vocab=vocab
        self.opts = opts


    def on_epoch_end(self, epoch, logs={}):
        train_acc=compute_acc(self.xtrain, self.ytrain, self.vocab, self.model, self.opts)
        dev_acc=compute_acc(self.xdev, self.ydev, self.vocab, self.model, self.opts)
        test_acc=compute_acc(self.xtest, self.ytest, self.vocab, self.model, self.opts)
        logging.info('----------------------------------')
        logging.info('Epoch ' + str(epoch) + ' train loss:'+str(logs.get('loss'))+' - Validation loss: ' + str(logs.get('val_loss')) + ' train acc: ' + str(train_acc[0])+'/'+str(train_acc[1]) + ' dev acc: ' + str(dev_acc[0])+'/'+str(dev_acc[1]) + ' test acc: ' + str(test_acc[0])+'/'+str(test_acc[1]))
        logging.info('----------------------------------')

class MyEmbedding(Embedding):
    def __init__(self, input_dim, output_dim, use_mask=True, **kwargs):
        self.use_mask = use_mask
        super(MyEmbedding, self).__init__(input_dim, output_dim, **kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.use_mask:
            m = np.ones((self.input_dim, self.output_dim))
            m[0] = [0]*self.output_dim
            mask = K.variable(m, dtype=self.W.dtype)
            outW = K.gather(self.W, X)
            outM = K.gather(mask, X)
            return outW*outM
        else:
            return K.gather(self.W, X)

def get_H_n(X):
    ans=X[:, -1, :]  # get last element from time dim
    return ans

def get_H_hypo(X):
    xmaxlen=K.params['xmaxlen']
    return X[:, xmaxlen:, :]  # get elements L+1 to N

def get_WH_Lpi(i):  # get element i
    def get_X_i(X):
        return X[:,i,:];
    return get_X_i

def get_Y(X):
    xmaxlen=K.params['xmaxlen']
    return X[:, :xmaxlen, :]  # get first xmaxlen elem from time dim

def get_R(X):
    Y, alpha = X.values()  # Y should be (None,L,k) and alpha should be (None,L,1) and ans should be (None, k,1)
    tmp=K.permute_dimensions(Y,(0,)+(2,1))  # copied from permute layer, Now Y is (None,k,L) and alpha is always (None,L,1)
    ans=K.T.batched_dot(tmp,alpha)
    return ans


def build_model(opts, verbose=False):
    model = Graph()
    k = opts.lstm_units
    L = opts.xmaxlen
    N = opts.xmaxlen + opts.ymaxlen + 1  # for delim
    print "x len", L, "total len", N
    # model.add_input(name='inputx', input_shape=(opts.xmaxlen,), dtype=int)
    # model.add_input(name='inputy', input_shape=(opts.ymaxlen,), dtype=int)
    # model.add_node(Embedding(opts.max_features, opts.wx_emb, input_length=opts.xmaxlen), name='x_emb',
    #                input='inputx')
    # model.add_node(Embedding(opts.max_features, opts.wy_emb, input_length=opts.ymaxlen), name='y_emb',
    #                input='inputy')
    # model.add_node(LSTM(opts.lstm_units, return_sequences=True), name='forward', inputs=['x_emb', 'y_emb'],
    #                concat_axis=1)
    # model.add_node(LSTM(opts.lstm_units, return_sequences=True, go_backwards=True), name='backward',
    #                inputs=['x_emb', 'y_emb'], concat_axis=1)

    model.add_input(name='input', input_shape=(options.embedding_size,N), dtype=float)
    model.add_node(GRU(opts.lstm_units, return_sequences=True), name='forward', input='input')
#    model.add_node(GRU(opts.lstm_units, return_sequences=True, go_backwards=True), name='backward', input='d_emb')

#    model.add_node(Dropout(0.1), name='dropout', inputs=['forward','backward'])
    model.add_node(Dropout(0.1), name='dropout', input='forward')
    model.add_node(Lambda(get_H_n, output_shape=(k,)), name='h_n', input='dropout')

    # model.add_node(Lambda(XMaxLen(10), output_shape=(L, k)), name='Y', input='dropout')

    model.add_node(Lambda(get_Y, output_shape=(L, k)), name='Y', input='dropout')
#   model.add_node(SliceAtLength((None,N,k),L), name='Y', input='dropout')
#    model.add_node(Dense(k,W_regularizer=l2(0.01)),name='Wh_n', input='h_n')
#    model.add_node(RepeatVector(L), name='Wh_n_cross_e', input='Wh_n')
    model.add_node(TimeDistributedDense(k,W_regularizer=l2(0.01)), name='WY', input='Y')

    ###########
    # "dropout" layer contains all h vectors from h_1 to h_N

    model.add_node(Lambda(get_H_hypo, output_shape=(N-L, k)), name='h_hypo', input='dropout')
    model.add_node(TimeDistributedDense(k,W_regularizer=l2(0.01)), name='Wh_hypo', input='h_hypo')

    # GET R1

    f = get_WH_Lpi(0)
    model.add_node(Lambda(f, output_shape=(k,)), name='Wh_lp1', input='Wh_hypo')
    model.add_node(RepeatVector(L), name='Wh_lp1_cross_e', input='Wh_lp1')
    model.add_node(Activation('tanh'), name='M1', inputs=['Wh_lp1_cross_e', 'WY'], merge_mode='sum')
    model.add_node(TimeDistributedDense(1,activation='softmax'), name='alpha1', input='M1')
    model.add_node(Lambda(get_R, output_shape=(k,1)), name='_r1', inputs=['Y','alpha1'], merge_mode='join')
    model.add_node(Reshape((k,)),name='r1', input='_r1')
    model.add_node(Dense(k,W_regularizer=l2(0.01),activation='tanh'), name='Tan_Wr1', input='r1')
    model.add_node(Dense(k,W_regularizer=l2(0.01)), name='Wr1', input='r1')
    model.add_node(RepeatVector(L), name='Wr1_cross_e', input='Wr1')

    # GET R2, R3, .. R_N
    for i in xrange(2,N-L):
        f = get_WH_Lpi(i-1)
        model.add_node(Lambda(f, output_shape=(k,)), name='Wh_lp'+str(i), input='Wh_hypo')
        model.add_node(RepeatVector(L), name='Wh_lp'+str(i)+'_cross_e', input='Wh_lp'+str(i))
        model.add_node(Activation('tanh'), name='M'+str(i), inputs=['Wh_lp'+str(i)+'_cross_e', 'WY', 'Wr'+str(i-1)+'_cross_e'], merge_mode='sum')
        model.add_node(TimeDistributedDense(1,activation='softmax'), name='alpha'+str(i), input='M'+str(i))
        model.add_node(Lambda(get_R, output_shape=(k,1)), name='_r'+str(i), inputs=['Y','alpha'+str(i)], merge_mode='join')
        model.add_node(Reshape((k,)),name='*r'+str(i), input='_r'+str(i))
        model.add_node(Layer(), merge_mode='sum', inputs=['*r'+str(i),'Tan_Wr'+str(i-1)], name='r'+str(i))
        if i != (N-L-1):
            model.add_node(Dense(k,W_regularizer=l2(0.01),activation='tanh'), name='Tan_Wr'+str(i), input='r'+str(i))
            model.add_node(Dense(k,W_regularizer=l2(0.01)), name='Wr'+str(i), input='r'+str(i))
            model.add_node(RepeatVector(L), name='Wr'+str(i)+'_cross_e', input='Wr'+str(i))


# THIS WORKS FOR GETTING R2
#    model.add_node(Lambda(get_WH_Lp2, output_shape=(k,)), name='Wh_lp2', input='Wh_hypo')
#    model.add_node(RepeatVector(L), name='Wh_lp2_cross_e', input='Wh_lp2')
#    model.add_node(Activation('tanh'), name='M2', inputs=['Wh_lp2_cross_e', 'WY', 'r1_cross_e'], merge_mode='sum')
#    model.add_node(TimeDistributedDense(1,activation='softmax'), name='alpha2', input='M2')
#    model.add_node(Lambda(get_R, output_shape=(k,1)), name='_r2', inputs=['Y','alpha2'], merge_mode='join')
#    model.add_node(Reshape((k,)),name='*r2', input='_r2')
#    model.add_node(Layer(), merge_mode='sum', inputs=['*r2','Tan_Wr1'], name='r2')

#    model.add_node(Dense(k,W_regularizer=l2(0.01),activation='tanh'), name='Tan_Wr2', input='r2')
#    model.add_node(RepeatVector(L), name='r2_cross_e', input='r2')



    ###########

#    model.add_node(Activation('tanh'), name='M', inputs=['Wh_n_cross_e', 'WY'], merge_mode='sum')
#    model.add_node(TimeDistributedDense(1,activation='softmax'), name='alpha', input='M')
#    model.add_node(Lambda(get_R, output_shape=(k,1)), name='_r', inputs=['Y','alpha'], merge_mode='join')
#    model.add_node(Reshape((k,)),name='r', input='_r')
#    model.add_node(Dense(k,W_regularizer=l2(0.01)), name='Wr', input='r')

    model.add_node(Dense(k,W_regularizer=l2(0.01)), name='Wr', input='r'+str(N-L-1)) ##### ADDED
    model.add_node(Dense(k,W_regularizer=l2(0.01)), name='Wh', input='h_n')
    model.add_node(Activation('tanh'), name='h_star', inputs=['Wr', 'Wh'], merge_mode='sum')

    model.add_node(Dense(3, activation='softmax'), name='out', input='h_star')
    model.add_output(name='output', input='out')
    model.summary()


    if verbose:
        model.summary()
#        graph = to_graph(model, show_shape=True)
#        graph.write_png("model2.png")

    # model.compile(loss={'output':'binary_crossentropy'}, optimizer=Adam())
    model.compile(loss={'output':'categorical_crossentropy'}, optimizer=Adam(options.lr))
    return model


def compute_acc(X, Y, vocab, model, opts):
    scores=model.predict({'input': X},batch_size=options.batch_size)['output']
    prediction=np.zeros(scores.shape)
    for i in range(scores.shape[0]):
        l=np.argmax(scores[i])
        prediction[i][l]=1.0
    assert np.array_equal(np.ones(prediction.shape[0]),np.sum(prediction,axis=1))
    plabels=np.argmax(prediction,axis=1)
    tlabels=np.argmax(Y,axis=1)
    acc = accuracy(tlabels,plabels)
    return acc,acc

def getConfig(opts):
    conf=[opts.xmaxlen,
          opts.ymaxlen,
          opts.batch_size,
          opts.emb,
          opts.lr,
          opts.samples,
          opts.lstm_units,
          opts.epochs]
    if opts.no_padding:
        conf.append("no-pad")
    return "_".join(map(lambda x: str(x), conf))


def save_model(model,wtpath,archpath,mode='yaml'):
    if mode=='yaml':
        yaml_string = model.to_yaml()
        open(archpath, 'w').write(yaml_string)
    else:
        with open(archpath, 'w') as f:
            f.write(model.to_json())
    model.save_weights(wtpath)


def load_model(wtpath,archpath,mode='yaml'):
    if mode=='yaml':
        model = model_from_yaml(open(archpath).read())#,custom_objects={"MyEmbedding": MyEmbedding})
    else:
        with open(archpath) as f:
            model = model_from_json(f.read())#, custom_objects={"MyEmbedding": MyEmbedding})
    model.load_weights(wtpath)
    return model


def concat_in_out(X,Y,vocab):
    numex = X.shape[0] # num examples
    glue=vocab["delimiter"]*np.ones(numex).reshape(numex,1)
    inp_train = np.concatenate((X,glue,Y),axis=1)
    return inp_train


def setup_logger(config_str):
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=datetime.now().strftime('mylogfile_%H_%M_%d_%m_%Y.log'),
                    filemode='w')

def data2vec(data,labels,RMat,options,dictionary):
    flag = 0
    X_train = []
    label = []
    for p,h,l in data:
        if(l not in labels):
            continue
        label.append(labels[l])
        p = tokenize(p)
        h = tokenize(h)
        if(len(p) > options.xmaxlen or len(h) > options.xmaxlen):
            continue
        for i in xrange(options.xmaxlen - len(p)):
            p = ['unk'] + p
        for i in xrange(options.ymaxlen - len(h)):
            h = h + ['unk']
        lst = p + h
        accum = np.empty((options.embedding_sizes,1))
        for idx in xrange(len(lst)):
            wrd = lst[idx]
            if (wrd in dictionary):
                accum = np.hstack((accum,RMat[dictionary[wrd]].transpose().reshape(options.embedding_size,1)))
            else:
                #averaging
                jdx = idx - 1
                tmat = np.zeros((1,options.embedding_size))
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





if __name__ == "__main__":
    options=get_params()
    train=[l.strip().split('\t') for l in open(options.train_file)]
    dev=[l.strip().split('\t') for l in open(options.dev_file)]
    test=[l.strip().split('\t') for l in open(options.test_file)]
    vocab=get_vocab(train)
    labels = {'neutral':0,'contradiction':-1,'entailment':1}
    RMatrix = np.load('VocabMat.npy')
    with open('Dictionary.txt','r') as inf:
        dictionary = eval(inf.read())

    print "vocab (incr. maxfeatures accordingly):",len(vocab)
    model = build_model(options)
    ep_n = 0
    while( ep_n < options.epochs):
        for idx in xrange(0,len(train),options.batch_size,options):
            X_train,Y_train = data2vec(train[idx:idx+options.batch_size],labels,RMatrix)
            train_dict = {'input':X_train,'output':Y_train}
            loss, accuracy = model.train_on_batch(train_dict,accuracy =True)
        ep_n += 1



#   X_train,Y_train,Z_train=load_data(train,vocab)
#   X_dev,Y_dev,Z_dev=load_data(dev,vocab)
#   X_test,Y_test,Z_test=load_data(test,vocab)
   
#   params={'xmaxlen':options.xmaxlen}
#   setattr(K,'params',params)

#   config_str = getConfig(options)
#   MODEL_ARCH = "arch_att" + config_str + ".yaml"
#   MODEL_WGHT = "weights_att" + config_str + ".weights"
   
#   MAXLEN=options.xmaxlen
#   X_train = pad_sequences(X_train, maxlen=MAXLEN,value=vocab["unk"],padding='pre')
#   X_dev = pad_sequences(X_dev, maxlen=MAXLEN,value=vocab["unk"],padding='pre')
#   X_test = pad_sequences(X_test, maxlen=MAXLEN,value=vocab["unk"],padding='pre')
#   Y_train = pad_sequences(Y_train, maxlen=MAXLEN,value=vocab["unk"],padding='post')
#   Y_dev = pad_sequences(Y_dev, maxlen=MAXLEN,value=vocab["unk"],padding='post')
#   Y_test = pad_sequences(Y_test, maxlen=MAXLEN,value=vocab["unk"],padding='post')
   
#   net_train=concat_in_out(X_train,Y_train,vocab)
#   net_dev=concat_in_out(X_dev,Y_dev,vocab)
#   net_test=concat_in_out(X_test,Y_test,vocab)

#   Z_train=to_categorical(Z_train, nb_classes=3)
#   Z_dev=to_categorical(Z_dev, nb_classes=3)
#   Z_test=to_categorical(Z_test, nb_classes=3)

#   print X_train.shape,Y_train.shape,net_train.shape
#   print map_to_txt(net_train[0],vocab),Z_train[0]
#   print map_to_txt(net_train[1],vocab),Z_train[1]
#   setup_logger(config_str)

#   assert net_train[0][options.xmaxlen] == 1
#   train_dict = {'input': net_train, 'output': Z_train}
#   dev_dict = {'input': net_dev, 'output': Z_dev}
#   print 'Build model...'
#   model = build_model(options)
#   
#   logging.info(vars(options))
#   logging.info("train size: "+str(len(net_train))+" dev size: "+str(len(net_dev))+" test size: "+str(len(net_test)))
#   if options.load_save and os.path.exists(MODEL_ARCH) and os.path.exists(MODEL_WGHT):
#      
#       print("Loading pre-trained model from", MODEL_WGHT)
#       load_model(MODEL_WGHT,MODEL_ARCH,'json')
#       train_acc=compute_acc(net_train, Z_train, vocab, model, options)
#       dev_acc=compute_acc(net_dev, Z_dev, vocab, model, options)
#       test_acc=compute_acc(net_test, Z_test, vocab, model, options)
#       print train_acc,dev_acc,test_acc

#   else:
#      
#       history = model.fit(train_dict,
#                       batch_size=options.batch_size,
#                       nb_epoch=options.epochs,
#                       validation_data=dev_dict,
#                       callbacks=[AccCallBack(net_train,Z_train,net_dev,Z_dev,net_test,Z_test,vocab,options)]
#   )

#       train_acc=compute_acc(net_train, Z_train, vocab, model, options)
#       dev_acc=compute_acc(net_dev, Z_dev, vocab, model, options)
#       test_acc=compute_acc(net_test, Z_test, vocab, model, options)
#       print "Training Accuracy: ", train_acc
#       print "Dev Accuracy: ", dev_acc
#       print "Testing Accuracy: ", test_acc
#       save_model(model,MODEL_WGHT,MODEL_ARCH)
