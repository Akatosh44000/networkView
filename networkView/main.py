'''
@author: j.langlois
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import getchCustom
import threading, time
import sys
from PyQt4 import QtGui
import networkWindow

from msvcrt import getch

key = "lol"

def thread1():
    global key
    lock = threading.Lock()
    while True:
        with lock:
            key = getchCustom.getch()

threading.Thread(target = thread1).start()

app=QtGui.QApplication(sys.argv)
mainwindow=networkWindow.MainWindow()

plt.ion()
while True:
    time.sleep(1)
    print(key)
    DATA=[]
    for i in range(4):
        DATA.append(np.zeros([3,10,10]))
        for j in range(DATA[i].shape[0]):
            DATA[i][j,:,:]=np.random.random([10,10])
    index=0
    nbFiltres=DATA[index].shape[0]
    square=int(np.sqrt(nbFiltres))+1
    

sys.exit(app.exec_())
    
'''
for i in range(nbFiltres):
    plt.subplot(square,square,i+1)
    plt.imshow(DATA[0][i],cmap=plt.get_cmap('gray'))
plt.pause(0.001)
plt.show()
'''