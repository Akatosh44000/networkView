'''
@author: j.langlois
'''

import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys
import batch_functions
from pandas.util.testing import network
import threading
import euler

sys.setrecursionlimit(50000)
      
class Network():
    
    def __init__(self,dimChannels,dimFeatures,dimOutput,name="default_network",paramsImport=False):
        self.name=name
        self.dimChannels=dimChannels
        self.dimFeatures=dimFeatures
        self.dimOutput=dimOutput
        self.loss=0
        self._observers = []

        input_var = T.tensor4('inputs')
        target_var = T.matrix('targets')
        learning_rate=T.scalar('learning rate')

        l_input = lasagne.layers.InputLayer(shape=(None, dimChannels, dimFeatures, dimFeatures),input_var=input_var)

        l_conv1 = lasagne.layers.Conv2DLayer(l_input, num_filters=16, filter_size=(8, 8),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
        l_conv2 = lasagne.layers.Conv2DLayer(l_conv1, num_filters=7, filter_size=(5, 5),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    
        l_fc1 = lasagne.layers.DenseLayer(l_conv2,num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.Orthogonal())
        l_fc2 = lasagne.layers.DenseLayer(l_fc1,num_units=dimOutput,nonlinearity=lasagne.nonlinearities.tanh,W=lasagne.init.Orthogonal())
        
        if paramsImport:
            lasagne.layers.set_all_param_values(l_fc2, paramsImport)
        
        prediction = lasagne.layers.get_output(l_fc2)
        loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = loss.mean()
 
        params = lasagne.layers.get_all_params(l_fc2, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=0.9)
        
        test_prediction = lasagne.layers.get_output(l_fc2, deterministic=True)
        test_loss = lasagne.objectives.squared_error(test_prediction,target_var)
        test_loss=test_loss.mean()
        
        self.last_layer=l_fc2
        self.f_train=theano.function([input_var, target_var,learning_rate], loss, updates=updates)
        self.f_predict=theano.function([input_var], test_prediction)
        self.f_accuracy=theano.function([input_var,target_var],test_loss)
        self.f_export_params=theano.function([],params)
        
    def createTrainingThread(self,train_data,train_labels,valid_data,valid_labels,trigger):
        self.trainingThread = threading.Thread(None, self.trainNetwork, None,(),
                                               {'train_data':train_data,
                                                'train_labels':train_labels,
                                                'valid_data':valid_data,
                                                'valid_labels':valid_labels,
                                                'trigger':trigger})
    
    def trainThread(self,name):
        print('test Thread',name)
    
    def set_loss(self, value):
        self._loss = value
        for callback in self._observers:
            callback(self._loss)
            
    def bind_to(self, callback):
        self._observers.append(callback)
                
    def trainNetwork(self,train_data,train_labels,valid_data,valid_labels,trigger,batchSize=20,epochs=100,learningRate=0.001,penalty=0.001):
        
        epoch_kept=0
        min_valid_error=10000
        valid_data=batch_functions.normalize_batch(valid_data)
        
        for e in range(0,epochs):
            train_err = 0
            train_batches = 0
            valid_error=0
            for batch in batch_functions.iterate_minibatches(train_data, train_labels, 20, shuffle=True):
                inputs, targets = batch
                inputs=batch_functions.normalize_batch(inputs)
                train_err += self.f_train(inputs, targets, learningRate)
                train_batches += 1
            valid_error=self.f_accuracy(valid_data,valid_labels)
            learningRate=learningRate/1.01
            
            if valid_error<min_valid_error:
                min_valid_error=valid_error
                epoch_kept=e
                params=lasagne.layers.get_all_param_values(self.last_layer)
                #self.exportNetwork()
            self.set_loss(train_err)
            print("EPOCH : "+str(e)+'    LOSS: '+str(train_err)+'    ERROR: '+str(valid_error)+'    RATE: '+str(learningRate))
        print('END - EPOCH KEPT : '+str(epoch_kept))   
        return Network(self.dimChannels,self.dimFeatures,self.dimOutput,self.name,paramsImport=params)
    
    def getParamsValues(self):
        return lasagne.layers.get_all_param_values(self.last_layer)
    
    def predict(self,test_data):
        batch_functions.normalize_batch(test_data)
        return self.f_predict(test_data)
    
    def testNetwork(self,test_data,test_labels):
        test_data=batch_functions.normalize_batch(test_data)
        prediction=self.f_predict(test_data)
        #print(test_labels,prediction)
        
        moy=0
        for i in range(prediction.shape[0]):
            q=[test_labels[i,0],test_labels[i,1],test_labels[i,2],test_labels[i,3]]
            anglesR=np.array(euler.quat2euler(prediction[i,:]))
            anglesV=np.array(euler.quat2euler(q))
            for j in range(3):
                anglesR[j]=anglesR[j]*180/np.pi
                if anglesR[j]<0:
                    anglesR[j]=anglesR[j]+360.0
                anglesV[j]=anglesV[j]*180/np.pi
                if anglesV[j]<0:
                    anglesV[j]=anglesV[j]+360.0
            #print(anglesR,anglesV)
            anglesD=[]
            for j in range(3):
                angle = np.abs(anglesR[j] - anglesV[j]) % 360;  
                if angle > 180 :
                    anglesD.append(360-angle)
                else: 
                    anglesD.append(angle)
            #print(anglesD)
            moy+=np.mean(anglesD)
        moy/=prediction.shape[0]
        return moy
                
        
