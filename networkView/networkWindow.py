'''
@author: j.langlois
'''

import sys
from PyQt4 import QtGui


class MainWindow(QtGui.QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
        
    def initUI(self):
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('NETWORK VIEWER')
        self.setWindowIcon(QtGui.QIcon('NeuralNetwork.PNG'))        
        self.show()


