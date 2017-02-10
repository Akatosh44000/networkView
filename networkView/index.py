'''
@author: j.langlois
'''

#!/usr/bin/python3
# -*- coding: utf-8 -*
import numpy as np
import threading
import sys
import time

def f():
    t = threading.currentThread()
    t.setName('NETWORK2')
    r = np.random.randint(1,0)
    print('sleeping', r)
    time.sleep(r)
    print('ending')
    return

for _ in range(3):
    t = threading.Thread(target=f)
    t.setDaemon(True)
    t.start()