import numpy as np

np.random.seed(1337)  # for reproducibility
import os
import sys
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2, activity_l2
from keras.callbacks import *
from theano import tensor as T
# from visualizer import *
from keras.layers import *
from keras.models import Model

from keras.optimizers import *
from keras.utils.np_utils import to_categorical,accuracy
from keras.layers.core import *


#from keras.utils.visualize_util import plot, to_graph # THIS IS BAD
# from data_reader import *
from reader import *
from myutils import *
import logging
from datetime import datetime
def time_distributed_dense(x, w,
                           input_dim=None, output_dim=None, timesteps=None,repeat_len = None):
    '''Apply y.w + b for every temporal slice y of x.
    '''

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))

    x = K.dot(x, w)
    
    x = K.repeat_elements( x.dimshuffle((0,'x',1)) , repeat_len, axis = 1)
    # reshape to 4D tensor
    x = K.reshape(x, (-1, timesteps,repeat_len,output_dim))
    return x

class mLSTM(Recurrent):
    '''
    Word by Word attention model 

    # Arguments
        output_dim: output_dimensions
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
    # Comments:
        Takes in as input a concatenation of the vectors YH, where Y is the vectors being attended on and
        H are the vectors on which attention is being applied
    # References
        - [REASONING ABOUT ENTAILMENT WITH NEURAL ATTENTION](http://arxiv.org/abs/1509.06664v2)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 W_regularizer=None, U_regularizer=None,b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.W1_regularizer = regularizers.get(W_regularizer)
        self.W2_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(mLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None,None]
        
        self.W_y = self.init((input_dim, self.output_dim),
                           name='{}_W_y'.format(self.name))

        self.W_h = self.init((input_dim, self.output_dim),
                           name='{}_W_h'.format(self.name))

        self.W = self.init((self.output_dim, 1),
                           name='{}_W'.format(self.name))

        self.U_r = self.inner_init((self.output_dim, self.output_dim),
                                 name='{}_U_r'.format(self.name))

        # mLSTM Initializations
        self.W_i = self.init((self.output_dim + self.output_dim,self.output_dim),
                            name='{}_W_i'.format(self.name))
        self.W_f = self.init((self.output_dim + self.output_dim,self.output_dim),
                            name='{}_W_f'.format(self.name))
        self.W_o = self.init((self.output_dim + self.output_dim,self.output_dim),
                            name='{}_W_o'.format(self.name))
        self.W_c = self.init((self.output_dim + self.output_dim,self.output_dim),
                            name='{}_W_c'.format(self.name))  

        self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                 name='{}_U_i'.format(self.name))
        self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                 name='{}_U_f'.format(self.name))
        self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                 name='{}_U_o'.format(self.name))
        self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                 name='{}_U_c'.format(self.name)) 
        
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))
        self.b_f = K.zeros((self.output_dim,), name='{}_b_f'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))                                                                                                                                                                                        

        self.regularizers = []
        if self.W1_regularizer:
            self.W1_regularizer.set_param(K.concatenate([self.W_y,
                                                        self.W_h,
                                                        self.W
                                                        ]))
            self.regularizers.append(self.W1_regularizer)
            self.W2_regularizer.set_param(K.concatenate([self.W_i,
                                                        self.W_f,
                                                        self.W_o,
                                                        self.W_c]))
            self.regularizers.append(self.W2_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(K.concatenate([self.U_r,                                                        
                                                        self.U_i,
                                                        self.U_f,
                                                        self.U_o,
                                                        self.U_c]))
            self.regularizers.append(self.U_regularizer)
        if(self.b_regularizer):
            self.b_regularizer.set_param([self.b_i,
                                          self.b_f,
                                          self.b_o,
                                          self.b_c])
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W_y, self.W_h, self.W,self.W_i,self.W_f,self.W_o,self.W_c,
                                  self.U_r, self.U_i,self.U_f,self.U_o,self.U_c,
                                  self.b_i,self.b_f,self.b_o,self.b_c]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        self.Y = x[:,:K.params['xmaxlen'],:]
        x = x[:,K.params['xmaxlen']:,:]
        self.precompute_W_y_y = K.dot(self.Y,self.W_y)
        input_dim = x.shape[2]
        timesteps = x.shape[1]
        repeat_len = self.Y.shape[1]
        return time_distributed_dense(x,self.W_h,input_dim,self.output_dim,timesteps,repeat_len)
        # return x

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]

        h_tilde = x[:,0,:]
        L = K.params['xmaxlen']
        
        M = K.tanh(self.precompute_W_y_y + x + K.repeat_elements(K.dot(h_tm1, self.U_r).dimshuffle((0,'x',1)),L, axis=1))
        alpha = K.dot(M, self.W)
        alpha = K.softmax(alpha[:,:,0]) 
        alpha = alpha.dimshuffle((0,'x',1))
        
        output = T.batched_dot(alpha,self.Y) 
        output = output[:,0,:]
        
        xt = K.concatenate([h_tilde,output],axis = 1)

        it = K.sigmoid( K.dot(xt,self.W_i) + K.dot(h_tilde,self.U_i) + self.b_i )
        ft = K.sigmoid(K.dot(xt,self.W_f) + K.dot(h_tilde,self.U_f) + self.b_f)
        ot = K.sigmoid(K.dot(xt,self.W_o) + K.dot(h_tilde,self.U_o) + self.b_o)
        c_tilde_t = K.dot(xt,self.W_c) + K.dot(h_tilde,self.U_c) + self.b_c
        c_t = ft * c_tm1 + it*K.tanh( c_tilde_t )

        h_t = ot * K.tanh(c_t)

        return h_t, [h_t,c_t]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.dropout(ones, self.dropout_U) for _ in range(2)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(2)])

        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.dropout(ones, self.dropout_W) for _ in range(3)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
