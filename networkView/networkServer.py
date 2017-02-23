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
        print('WAIT:: STARTING NETWORK CLIENT')

        #BYPASS PROXY RULES FOR LOCALHOST DOMAIN
        os.environ['NO_PROXY'] = 'localhost'
        
        #NETWORK PARAMS
        self.id=0
        self.name='reseau_1'
        '''
        NETWORK CREATION...
        '''
        
        self.buildNetwork()
        #HANDLE MESSAGE FROM SERVER
        def handle_message_from_server(args):
            message=args[0]
            self.handle_message_from_server(message)
        #HANDLE MESSAGE FROM CLIENT
        def handle_message_from_client(args):
            message=args[0]
            self.handle_message_from_client(message)
        #HANDLE REQUEST FROM CLIENT
        def handle_request_from_client(args):
            request=args[0]
            self.handle_request_from_client(request)
            
        class MessageNamespace(BaseNamespace):
            def on_connect(self):
                print('[Connected]')
            def on_disconnect(self):
                print('[Disconnected]')
            def on_MESSAGE_FROM_SERVER_TO_NETWORK(self,*args):
                handle_message_from_server(args)
            def on_MESSAGE_FROM_CLIENT_TO_NETWORK(self,*args):
                handle_message_from_client(args)
            def on_REQUEST_FROM_CLIENT_TO_NETWORK(self,*args):
                handle_request_from_client(args)

        self.socketio=SocketIO('localhost', 8080,MessageNamespace)
        print('SUCCESS:: NETWORK CLIENT STARTED')
        
        self.sendRequestToServer('getNewId',{'network_id': self.id,'network_name': self.name})
        
        
        
        
        self.socketio.wait()
    def sendMessageToserver(self,name,data):
        self.socketio.emit('MESSAGE_FROM_NETWORK_TO_SERVER',
                           {'name':name,'data':data})
    def sendRequestToServer(self,name,params):
        self.socketio.emit('REQUEST_FROM_NETWORK_TO_SERVER',
                           {'name':name,'params':params})
    def sendMessageToClient(self,name,data,client_socket_id):
        self.socketio.emit('MESSAGE_FROM_NETWORK_TO_CLIENT',
                           {'client_socket_id':client_socket_id,'name':name,'data':data})        
    def buildNetwork(self):
        print('WAIT:: BUILDING NEURAL NETWORK...')
        self.network=networkInstance.Network(1,64,4)
        print('SUCCESS:: NETWORK BUILT !')
        
    def handle_message_from_server(self,message):   
        if self.id==0:
            print('INFO:: NETWORK GOT THE ID '+message['network_id']+' FROM THE SERVER.')
            self.id=int(message['network_id']) 
            
    def handle_message_from_client(self,request):
        return 1
    
    def handle_request_from_client(self,request):
        print('INFO:: HANDLING REQUEST FROM CLIENT ',request['name'])
        if request['name']=='setTrain':
            self.trainNetwork()
        if request['name']=='getNetworkArchitecture':
            self.sendMessageToClient('architecture',{'architecture':self.getNetworkArchitecture()},request['client_socket_id']);
        if request['name']=='getLoss':
            payload = {'client_socket_id':request['client_socket_id'],'message':{'loss':self.getLoss()}}
            self.socketio.emit('MESSAGE_FROM_NETWORK_TO_CLIENT',payload)
        if request['name']=='getParams':
            self.sendMessageToClient('params',{'params':self.getFakeParams()},request['client_socket_id']);

        
    def getNetwork(self):    
        return self.network
    
    def trainNetwork(self):
        print('WAIT:: IMPORTING DATASETS...')
        MODEL_PATH='/home/akatosh/DATASETS'
        DATASET='MULTITUDE'
        MODEL_OBJECT='BREATHER'
        DIMENSION='SAMPLES'
        DEPTH=False
        [train_data,train_labels]=import2D.datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,DIMENSION,'TRAIN',DEPTH)
        [valid_data,valid_labels]=import2D.datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,DIMENSION,'VALIDATION',DEPTH)
        print('SUCCESS:: DATASETS IMPORTED !')
        print('WAIT:: CREATING GPU TRAINING THREAD...')
        self.network.bind_to(self.newEpoch)
        trigger = threading.Event()
        self.network.createTrainingThread(train_data,train_labels,valid_data,valid_labels,trigger)
        print('WAIT:: STARTING GPU TRAINING THREAD...')
        self.network.trainingThread.start()
        print('SUCCESS:: TRAINING STARTED !')
        
    def getLoss(self):
        return self.network.loss
    
    def newEpoch(self,loss):
        print("NEW EPOCH !!!",str(loss))
        self.sendMessageToserver('newEpoch','')
        
    def getNetworkArchitecture(self):
        params=np.asarray(self.network.getParamsValues())
        architecture=[]
        for i in range(0,params.shape[0],2):
            if len(params[i].shape)>2:
                #CONVOLUTION LAYER
                architecture.append(['CONV',params[i].shape])
            else:
                architecture.append(['FC',params[i].shape])
        return architecture
    
    def getFakeParams(self):
        params=[]
        firstKernel=np.asarray(self.network.getParamsValues())[0][0,0,:,:]
        for i in range(8):
            for j in range(8):
                params.append(str(firstKernel[i,j]))
        return params
        
        
                