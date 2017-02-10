'''
@author: j.langlois
'''
import numpy as np
import threading
import time
import requests
import http.server
import web

class MyApplication(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))


class networkServer:
    
    def __init__(self):
        
        print('hello')
        self.id=0
        self.port=0
        self.name='reseau1'
        payload = {'network_id':str(self.id),'network_port':str(self.port),'network_name': self.name}
        r=requests.post('http://localhost:8080/networkProbe',data=payload)
        if self.id==0:
            self.id=int(r.text)
        
        t = threading.Thread(target=self.threadPooling)
        t.setDaemon(True)
        t.start()
        
        self.port = 8080+int(self.id)
        self.urls = (
        '/build_network', 'construct'
        )
        app=MyApplication(self.urls,globals())
        app.run(port=self.port)
        return
    
    def threadPooling(self):
        t = threading.currentThread()
        t.setName('NETWORK_SERVER_POOL')
        max=10
        s=0
        while s<max:
            time.sleep(1)
            s+=1
            payload = {'network_id':str(self.id),'network_port':str(self.port),'network_name': self.name}
            r=requests.post('http://localhost:8080/networkProbe',data=payload)
            print(r.text)
        print('ending')
        
    def construct(self):
        print('NETWORK OK')
        
    