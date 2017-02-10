#!/usr/bin/env python
import web
import numpy as np
import xml.etree.ElementTree as ET
import threading
import time 
tree = ET.parse('user_data.xml')
root = tree.getroot()

urls = (
    '/list_networks', 'list_networks',
    '/network_probe', 'network_probe',
    '/new_network', 'create_network'
)

app = web.application(urls, globals())


def f():
    t = threading.currentThread()
    t.setName('NETWORK')
    r = np.random.randint(1,0)
    print('sleeping %s', r)
    time.sleep(r)
    print('ending')
    return

networksList=[]

class list_networks:
    def GET(self):
        return str(networksList)

class create_network:
    def GET(self):
        for _ in range(3):
            t = threading.Thread(target=f)
            t.setDaemon(True)
            t.start()

class network_probe:
    def POST(self):
        receive=web.input()
        id=receive['network_id']
        if id=='0':
            print('NEW NETWORK DETECTED !')
            proposal_id=1
            fineID=False
            while not fineID:
                fineID=True
                for network in networksList:
                    if proposal_id==network[0]:
                        proposal_id+=1
                        fineID=False
                        break
            print(proposal_id)
            networksList.append([proposal_id,str(8080+proposal_id),receive['network_name']])
            return str(proposal_id)
        else:
            print('PROBE FROM NETWORK #',id)
        



if __name__ == "__main__":
    app.run()