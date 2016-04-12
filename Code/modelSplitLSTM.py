# from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility
import os
import sys
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
    return opts

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
#    k = opts.lstm_units
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

    model.add_input(name='input', input_shape=(N,), dtype=int)
#    model.add_node(GRU(opts.lstm_units, return_sequences=True), name='forward', input='input')

#    InitWeights = np.load('/home/ee/btech/ee1130798/Code/VocabMat.npy')
    InitWeights = np.load('VocabMat.npy')
    model.add_node(Embedding(InitWeights.shape[0], InitWeights.shape[1], input_length=N, weights=[InitWeights]), name='emb',
                   input='input')
    model.add_node(Dropout(0.1), name='d_emb', input='emb')


    model.add_node(Lambda(get_Y, output_shape=(L,300)), name='premise', input='d_emb')
    model.add_node(Lambda(get_H_hypo, output_shape=(N-L,300)), name='hypothesis', input='d_emb')

    model.add_node(LSTM(opts.lstm_units, return_sequences=True), name='h_premise', input='premise')
    model.add_node(LSTM(opts.lstm_units, return_sequences=True), name='h_hypothesis', input='hypothesis')


    model.add_node(Dropout(0.1), name='dropout', inputs=['h_premise','h_hypothesis'], merge_mode='concat', concat_axis=1) #CORREECT MERGE MODE

    model.add_node(Lambda(get_H_n, output_shape=(k,)), name='h_n', input='dropout')

    # model.add_node(Lambda(XMaxLen(10), output_shape=(L, k)), name='Y', input='dropout')

    model.add_node(Lambda(get_Y, output_shape=(L, k)), name='Y', input='dropout')
    # model.add_node(SliceAtLength((None,N,k),L), name='Y', input='dropout')
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
    for i in range(2,N-L+1):
        f = get_WH_Lpi(i-1)
        model.add_node(Lambda(f, output_shape=(k,)), name='Wh_lp'+str(i), input='Wh_hypo')
        model.add_node(RepeatVector(L), name='Wh_lp'+str(i)+'_cross_e', input='Wh_lp'+str(i))
        model.add_node(Activation('tanh'), name='M'+str(i), inputs=['Wh_lp'+str(i)+'_cross_e', 'WY', 'Wr'+str(i-1)+'_cross_e'], merge_mode='sum')
        model.add_node(TimeDistributedDense(1,activation='softmax'), name='alpha'+str(i), input='M'+str(i))
        model.add_node(Lambda(get_R, output_shape=(k,1)), name='_r'+str(i), inputs=['Y','alpha'+str(i)], merge_mode='join')
        model.add_node(Reshape((k,)),name='*r'+str(i), input='_r'+str(i))
        model.add_node(Layer(), merge_mode='sum', inputs=['*r'+str(i),'Tan_Wr'+str(i-1)], name='r'+str(i))
        if i != (N-L):
            model.add_node(Dense(k,W_regularizer=l2(0.01),activation='tanh'), name='Tan_Wr'+str(i), input='r'+str(i))
            model.add_node(Dense(k,W_regularizer=l2(0.01)), name='Wr'+str(i), input='r'+str(i))
            model.add_node(RepeatVector(L), name='Wr'+str(i)+'_cross_e', input='Wr'+str(i))

#    model.add_node(Activation('tanh'), name='M', inputs=['Wh_n_cross_e', 'WY'], merge_mode='sum')
#    model.add_node(TimeDistributedDense(1,activation='softmax'), name='alpha', input='M')
#    model.add_node(Lambda(get_R, output_shape=(k,1)), name='_r', inputs=['Y','alpha'], merge_mode='join')
#    model.add_node(Reshape((k,)),name='r', input='_r')
#    model.add_node(Dense(k,W_regularizer=l2(0.01)), name='Wr', input='r')

    model.add_node(Dense(k,W_regularizer=l2(0.01)), name='Wr', input='r'+str(N-L)) ##### ADDED
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

class WeightSharing(Callback):
    def __init__(self, shared):
        self.shared = shared

    def on_batch_end(self, batch, logs={}):
        weights = np.mean([self.model.nodes[n].get_weights()[0] for n in self.shared],axis=0)
        biases = np.mean([self.model.nodes[n].get_weights()[1] for n in self.shared],axis=0)
        for n in self.shared:
            self.model.nodes[n].set_weights([weights, biases])

if __name__ == "__main__":
#    train=[l.strip().split('\t') for l in open('/home/ee/btech/ee1130798/Code/train.txt')]
#    dev=[l.strip().split('\t') for l in open('/home/ee/btech/ee1130798/Code/dev.txt')]
#    test=[l.strip().split('\t') for l in open('/home/ee/btech/ee1130798/Code/test.txt')]

    train=[l.strip().split('\t') for l in open('train.txt')]
    dev=[l.strip().split('\t') for l in open('dev.txt')]
    test=[l.strip().split('\t') for l in open('test.txt')]

#    vocab=get_vocab(train)
#    with open('/home/ee/btech/ee1130798/Code/Dictionary.txt','r') as inf:
    with open('Dictionary.txt','r') as inf:
        vocab = eval(inf.read())


    print "vocab (incr. maxfeatures accordingly):",len(vocab)
    X_train,Y_train,Z_train=load_data(train,vocab)
    X_dev,Y_dev,Z_dev=load_data(dev,vocab)
    X_test,Y_test,Z_test=load_data(test,vocab)
    options=get_params()
   
    params={'xmaxlen':options.xmaxlen}
    setattr(K,'params',params)

    config_str = getConfig(options)
    MODEL_ARCH = "/home/ee/btech/ee1130798/Code/Models/SplitLSTM/arch_att" + config_str + ".yaml"
    MODEL_WGHT = "/home/ee/btech/ee1130798/Code/Models/SplitLSTM/weights_att" + config_str + ".weights"
   
    MAXLEN=options.xmaxlen
    X_train = pad_sequences(X_train, maxlen=MAXLEN,value=vocab["unk"],padding='pre')
    X_dev = pad_sequences(X_dev, maxlen=MAXLEN,value=vocab["unk"],padding='pre')
    X_test = pad_sequences(X_test, maxlen=MAXLEN,value=vocab["unk"],padding='pre')
    Y_train = pad_sequences(Y_train, maxlen=MAXLEN,value=vocab["unk"],padding='post')
    Y_dev = pad_sequences(Y_dev, maxlen=MAXLEN,value=vocab["unk"],padding='post')
    Y_test = pad_sequences(Y_test, maxlen=MAXLEN,value=vocab["unk"],padding='post')
   
    net_train=concat_in_out(X_train,Y_train,vocab)
    net_dev=concat_in_out(X_dev,Y_dev,vocab)
    net_test=concat_in_out(X_test,Y_test,vocab)

    Z_train=to_categorical(Z_train, nb_classes=3)
    Z_dev=to_categorical(Z_dev, nb_classes=3)
    Z_test=to_categorical(Z_test, nb_classes=3)

    print X_train.shape,Y_train.shape,net_train.shape
    print map_to_txt(net_train[0],vocab),Z_train[0]
    print map_to_txt(net_train[1],vocab),Z_train[1]
#    setup_logger(config_str)

    assert net_train[0][options.xmaxlen] == 1
    train_dict = {'input': net_train, 'output': Z_train}
    dev_dict = {'input': net_dev, 'output': Z_dev}
    print 'Build model...'
    model = build_model(options)
    print '#BRK 0'
#    logging.info(vars(options))
#    logging.info("train size: "+str(len(net_train))+" dev size: "+str(len(net_dev))+" test size: "+str(len(net_test)))

#    def data2vec(data, RMatrix):
#        X = np.empty((300,len(data[0])))
#        for sample in data:
#            rep = np.empty((300,1))
#            for word in sample:
#                rep = np.hstack((rep, RMatrix[word].reshape(300,1)))
#            rep = rep[:,1:]
#            X = np.dstack((X, rep))
#        X = X.swapaxes(0,2)
#        return X[1:,:,:]

#    def generate_GloVe_embedding_samples(net_train, Z_train, batch_size):
#        RMatrix = np.load('VocabMat.npy')
#        num_batches = len(net_train)/batch_size
#        while 1:
#            for idx in xrange(0, num_batches*batch_size, batch_size):
#                X_train = data2vec(net_train[idx:idx+batch_size], RMatrix)
#                yield {'input': X_train, 'output': Z_train}

    if options.load_save and os.path.exists(MODEL_ARCH) and os.path.exists(MODEL_WGHT):
        print '#BRK 1'
        print("Loading pre-trained model from", MODEL_WGHT)
        load_model(MODEL_WGHT,MODEL_ARCH,'json')
        train_acc=compute_acc(net_train, Z_train, vocab, model, options)
        dev_acc=compute_acc(net_dev, Z_dev, vocab, model, options)
        test_acc=compute_acc(net_test, Z_test, vocab, model, options)
        print train_acc,dev_acc,test_acc

    else:
        print '#BRK 2'
        group1 = []
        group2 = []
        group3 = []
        for i in range(1,options.ymaxlen+1):
            group1.append('Tan_Wr'+str(i))
            group2.append('Wr'+str(i))
            group3.append('alpha'+str(i))
        group3.append('alpha'+str(options.ymaxlen+1))

#        history = model.fit_generator(generate_GloVe_embedding_samples(net_train, Z_train, options.batch_size), 
#                        len(net_train), 
#                        options.epochs, 
#                        show_accuracy=True, 
#                        callbacks=[WeightSharing(group1), WeightSharing(group2), WeightSharing(group3)],
#                        validation_data=generate_GloVe_embedding_samples(net_dev, Z_dev, len(Z_dev)), 
#                        nb_val_samples=len(Z_dev))

        history = model.fit(train_dict,
                        batch_size=options.batch_size,
                        nb_epoch=options.epochs,
                        validation_data=dev_dict,
                        show_accuracy=True,
                        callbacks=[WeightSharing(group1), WeightSharing(group2), WeightSharing(group3)])

        train_acc=compute_acc(net_train, Z_train, vocab, model, options)
        dev_acc=compute_acc(net_dev, Z_dev, vocab, model, options)
        test_acc=compute_acc(net_test, Z_test, vocab, model, options)
        print "Training Accuracy: ", train_acc
        print "Dev Accuracy: ", dev_acc
        print "Testing Accuracy: ", test_acc

        save_model(model,MODEL_WGHT,MODEL_ARCH)
