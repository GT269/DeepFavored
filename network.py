# ========================================
# Codes for static computational graph
# ========================================
import re, os, sys
import numpy as np
import pandas as pd
import tensorflow as tf

class DeepFavored():
    def __init__(self, featuresNum, model_hyparams, training_params):
        #super.__init__(featuresNum, model_hyparams, training_params)
        self.featuresNum = featuresNum
        self.model_hyparams = model_hyparams
        self.training_params = training_params

    def WrapperActFunc(self, actFuncName, Input, nodeNum=None):
        if actFuncName in ['relu', 'sigmoid', 'tanh', 'softmax']:
            Output = eval('tf.keras.activations.'+actFuncName)(Input)
        if actFuncName in ['LeakyReLU']:
            Output = eval('tf.keras.layers.'+actFuncName)().call(Input)
        if actFuncName in ['PReLU']:
            ActObj = eval('tf.keras.layers.'+actFuncName)()
            ActObj.build(input_shape=(None, nodeNum))
            Output = ActObj.call(Input)
        return Output

    def build(self):
        tf.compat.v1.set_random_seed(1)
        initializer = tf.compat.v1.keras.initializers.glorot_normal()
        X = tf.compat.v1.placeholder("float", [None, self.featuresNum], name='X')

        ### Construct shared Layers
        sharedLayer_nodeNum = self.model_hyparams['sharedLayer_nodeNum']
        if sharedLayer_nodeNum in [[0], None]:
            shared_layer = X
            lastShareLayerNodeNum = self.featuresNum
        else:
            actFuncName = self.model_hyparams['hiddenLayer_actiFunc']
            for idx in range(len(sharedLayer_nodeNum)):
                currLayerNodeNum = sharedLayer_nodeNum[idx]
                initial_bias = initializer(shape=(1, currLayerNodeNum))#initialize bias
                bias = tf.Variable(initial_bias, name="share_b_"+str(idx), dtype="float32")
                if idx == 0:
                    initial_weights = initializer(shape=(self.featuresNum, currLayerNodeNum))#initialize Weights
                    weights = tf.Variable(initial_weights, name="share_W_"+str(idx), dtype="float32")
                    Input = tf.matmul(X, weights)+bias
                    shared_layer = self.WrapperActFunc(actFuncName, Input, currLayerNodeNum)
                else:
                    initial_weights = initializer(shape=(preLayerNodeNum, currLayerNodeNum))#initialize Weights
                    weights = tf.Variable(initial_weights, name="share_W_"+str(idx), dtype="float32")
                    Input = tf.matmul(shared_layer, weights)+bias
                    shared_layer = self.WrapperActFunc(actFuncName, Input, currLayerNodeNum)
                if self.training_params['hidden_batchnorm']:
                    shared_layer = tf.keras.layers.BatchNormalization(center=True, scale=False)(shared_layer)
                preLayerNodeNum = currLayerNodeNum
            lastShareLayerNodeNum = sharedLayer_nodeNum[-1]

        ### Construct hidden layers of 'H' classifier
        H_hiddenLayer_nodeNum = self.model_hyparams['H_hiddenLayer_nodeNum']
        if H_hiddenLayer_nodeNum:
            actFuncName = self.model_hyparams['hiddenLayer_actiFunc']
            for idx in range(len(H_hiddenLayer_nodeNum)):
                currLayerNodeNum = H_hiddenLayer_nodeNum[idx]
                initial_bias = initializer(shape=(1, currLayerNodeNum))
                bias = tf.Variable(initial_bias, name="H_hiddenLayer_b_"+str(idx), dtype="float32")
                if idx == 0:
                    initial_weights = initializer(shape=(lastShareLayerNodeNum, currLayerNodeNum))
                    weights = tf.Variable(initial_weights, name="H_haddenLayer_W_"+str(idx), dtype="float32")
                    Input = tf.matmul(shared_layer, weights)+bias
                    H_hiddenLayer = self.WrapperActFunc(actFuncName, Input, currLayerNodeNum)
                else:
                    initial_weights = initializer(shape=(preLayerNodeNum, currLayerNodeNum))
                    weights = tf.Variable(initial_weights, name="H_haddenLayer_W_"+str(idx), dtype="float32")
                    Input = tf.matmul(H_hiddenLayer, weights)+bias
                    H_hiddenLayer = self.WrapperActFunc(actFuncName, Input, currLayerNodeNum)
                if self.training_params['hidden_batchnorm']:
                    H_hiddenLayer = tf.keras.layers.BatchNormalization(center=True, scale=False)(H_hiddenLayer)
                preLayerNodeNum = currLayerNodeNum

        ### Construct hidden layers of 'O' classifier
        O_hiddenLayer_nodeNum = self.model_hyparams['O_hiddenLayer_nodeNum']
        if O_hiddenLayer_nodeNum:
            actFuncName = self.model_hyparams['hiddenLayer_actiFunc']
            for idx in range(len(O_hiddenLayer_nodeNum)):
                currLayerNodeNum = O_hiddenLayer_nodeNum[idx]
                initial_bias = initializer(shape=(1, currLayerNodeNum))
                bias = tf.Variable(initial_bias, name="O_hiddenLayer_b_"+str(idx), dtype="float32")
                if idx == 0:
                    initial_weights = initializer(shape=(lastShareLayerNodeNum, currLayerNodeNum))
                    weights = tf.Variable(initial_weights, name="O_haddenLayer_W_"+str(idx), dtype="float32")
                    Input = tf.matmul(shared_layer, weights)+bias
                    O_hiddenLayer = self.WrapperActFunc(actFuncName, Input, currLayerNodeNum)
                else:
                    initial_weights = initializer(shape=(preLayerNodeNum, currLayerNodeNum))
                    weights = tf.Variable(initial_weights, name="O_haddenLayer_W_"+str(idx), dtype="float32")
                    Input = tf.matmul(O_hiddenLayer, weights)+bias
                    O_hiddenLayer = self.WrapperActFunc(actFuncName, Input, currLayerNodeNum)
                if self.training_params['hidden_batchnorm']:
                    O_hiddenLayer = tf.keras.layers.BatchNormalization(center=True, scale=False)(O_hiddenLayer)
                preLayerNodeNum = currLayerNodeNum
        
        ### Construct out layer of 'H' classifier
        H_outLayer_nodeNum = self.model_hyparams['H_outLayer_nodeNum']
        actFuncName = self.model_hyparams['H_outLayer_actiFunc']
        initial_bias = initializer(shape=(1, H_outLayer_nodeNum))
        bias = tf.Variable(initial_bias, name="H_outLayer_b", dtype="float32")
        if H_hiddenLayer_nodeNum:
            initial_weights = initializer(shape=(H_hiddenLayer_nodeNum[-1], H_outLayer_nodeNum))
            weights = tf.Variable(initial_weights, name="H_outLayer_W", dtype="float32")
            Input = tf.matmul(H_hiddenLayer, weights)+bias
            H = self.WrapperActFunc(actFuncName, Input)
        else:
            initial_weights = initializer(shape=(lastShareLayerNodeNum, H_outLayer_nodeNum))
            weights = tf.Variable(initial_weights, name="H_outLayer_W", dtype="float32")
            Input = tf.matmul(shared_layer, weights)+bias
            H = self.WrapperActFunc(actFuncName, Input)
        
        ### Construct out layer of 'O' classifier
        O_outLayer_nodeNum = self.model_hyparams['O_outLayer_nodeNum']
        actFuncName = self.model_hyparams['O_outLayer_actiFunc']
        initial_bias = initializer(shape=(1, O_outLayer_nodeNum))
        O_outLayer_bias = tf.Variable(initial_bias, name="O_outLayer_b", dtype="float32")
        if O_hiddenLayer_nodeNum:
            initial_weights = initializer(shape=(O_hiddenLayer_nodeNum[-1], O_outLayer_nodeNum))
            weights = tf.Variable(initial_weights, name="O_outLayer_W", dtype="float32")
            Input = tf.matmul(O_hiddenLayer, weights)+O_outLayer_bias
            O = self.WrapperActFunc(actFuncName, Input)
        else:
            initial_weights = initializer(shape=(lastShareLayerNodeNum, O_outLayer_nodeNum))
            weights = tf.Variable(initial_weights, name="O_outLayer_W", dtype="float32")
            Input = tf.matmul(shared_layer, weights)+O_outLayer_bias
            O = self.WrapperActFunc(actFuncName, Input)
        
        H_pred = tf.compat.v1.identity(H, name='Y1_pred')
        O_pred = tf.compat.v1.identity(O, name='Y2_pred')
        return X, H_pred, O_pred
