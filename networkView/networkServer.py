'''
@author: j.langlois
'''
import numpy as np
import threading
import time
import requests
import http.server
import web
import os
from socketIO_client import SocketIO,BaseNamespace
import networkInstance
import import2D

class networkServer:
    
    def __init__(self):
        print('INFO:: STARTING NETWORK CLIENT')

        #BYPASS PROXY RULES FOR LOCALHOST DOMAIN
        os.environ['NO_PROXY'] = 'localhost'
        
        #NETWORK PARAMS
        self.id=0
        self.name='reseau_1'
        payload = {'network_id': self.id, 'network_name': self.name}
        '''
        NETWORK CREATION...
        '''
        self.buildNetwork()
        
        #HANDLE PROBE RESPONSE FROM SERVER
        def handle_probe_response(args):
            print(args)
            print('INFO:: NETWORK GOT THE ID '+args[0]['network_id']+' FROM THE SERVER.')
            if self.id==0:
                print('INFO:: NETWORK GOT THE ID '+args[0]['network_id']+' FROM THE SERVER.')
                self.id=int(args[0]['network_id'])
                
        #HANDLE MESSAGE FROM SERVER
        def handle_message(args):
            message=args[0]['message']
            self.handleMessage(message)
            
        class MessageNamespace(BaseNamespace):
            def on_connect(self):
                print('[Connected]')
            def on_disconnect(self):
                print('[Disconnected]')
            def on_PROBE_FROM_SERVER(self,*args):
                handle_probe_response(args)
            def on_MESSAGE_FROM_SERVER(self,*args):
                handle_message(args)

        #SOCKET CREATION
        self.socketio=SocketIO('localhost', 8080,MessageNamespace)
        #HANDLING CALLBACKS
        self.socketio.emit('PROBE_FROM_NETWORK',payload)
        self.socketio.wait()
    
    def buildNetwork(self):
        print('INFO:wait: BUILDING NEURAL NETWORK...')
        self.network=networkInstance.Network(1,64,2)
        print('INFO:success: NETWORK BUILT !')
        
    def handleMessage(self,message):
        print('INFO:: HANDLING MESSAGE FROM SERVER '+message)
        if message=='getInfo':
            self.getInfo()
            network=self.getNetwork()
            self.trainNetwork()
            print(network.name)
        if message=='getNetwork':
            network=self.getNetwork()
            print(network.name)
    
    def getInfo(self):
        print('SENDING INFO')
        payload = {'message':'coucou serveur'}
        self.socketio.emit('MESSAGE_FROM_NETWORK',payload)
        self.socketio.wait(seconds=0.1)
    
    def getNetwork(self):    
        return self.network
    
    def trainNetwork(self):
        print('INFO:wait: IMPORTING DATASETS...')
        MODEL_PATH='/home/akatosh/DATASETS'
        DATASET='MULTITUDE'
        MODEL_OBJECT='BREATHER'
        DIMENSION='SAMPLES_3D'
        DEPTH=False
        [train_data,train_labels]=import2D.datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,DIMENSION,'TRAIN',DEPTH)
        [valid_data,valid_labels]=import2D.datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,DIMENSION,'VALIDATION',DEPTH)
        print('INFO:success: DATASETS IMPORTED !')
        print('INFO:wait: CREATING GPU TRAINING THREAD...')
        self.network.createTrainingThread(train_data,train_labels,valid_data,valid_labels)
        print('INFO:wait: STARTING GPU TRAINING THREAD...')   
        self.network.trainingThread.start()          
        print('INFO:success: TRAINING STARTED !')
        
                