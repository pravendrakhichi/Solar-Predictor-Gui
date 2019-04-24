# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'final1.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QStatusBar, QAction,QMessageBox,QCheckBox,QFileDialog,QProgressBar,QPushButton,QApplication, QComboBox, QDialog,
        QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
        QLabel, QLineEdit, QMenu, QMenuBar,QSpinBox, QTextEdit,
        QVBoxLayout)
from PyQt5.QtGui import *
from table import *
from PyQt5.QtCore import *
import os

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense,Flatten,Reshape
from keras.layers import LSTM
from keras import optimizers
from keras.models import load_model
from pandas import DataFrame
from pandas import concat
import pytz
from datetime import datetime
print("dependencies done\n")
      
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow,self).__init__()
        Dialog=self
        Dialog.setObjectName("Dialog")
        Dialog.resize(755, 576)
        Dialog.setAutoFillBackground(False)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(280, 210, 161, 21))
        self.label.setStyleSheet("font: 75 11pt \"Rockwell\";")
        self.label.setObjectName("label")
        self.spinBox = QtWidgets.QSpinBox(Dialog)
        self.spinBox.setGeometry(QtCore.QRect(500, 210, 41, 20))
        self.spinBox.setObjectName("spinBox")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(280, 250, 181, 21))
        self.label_2.setStyleSheet("font: 75 11pt \"Rockwell\";")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(280, 290, 181, 21))
        self.label_3.setStyleSheet("font: 75 11pt \"Rockwell\";")
        self.label_3.setObjectName("label_3")
        self.line = QtWidgets.QFrame(Dialog)
        self.line.setGeometry(QtCore.QRect(240, 160, 20, 221))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(60, 180, 121, 16))
        self.label_4.setStyleSheet("font: 75 11pt \"Rockwell\";")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(60, 230, 181, 21))
        self.label_5.setStyleSheet("font: 75 11pt \"Rockwell\";")
        self.label_5.setObjectName("label_5")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(110, 200, 101, 23))
        self.pushButton.setStyleSheet("font: 75 11pt \"Rockwell\";")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(110, 260, 101, 21))
        self.pushButton_2.setStyleSheet("font: 75 11pt \"Rockwell\";")
        self.pushButton_2.setObjectName("pushButton_2")
        self.line_2 = QtWidgets.QFrame(Dialog)
        self.line_2.setGeometry(QtCore.QRect(50, 130, 651, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(50, 80, 371, 51))
        self.label_6.setStyleSheet("font: 75 36pt \"Rockwell\";")
        self.label_6.setObjectName("label_6")
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(390, 400, 221, 61))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Dialog)
        self.pushButton_4.setGeometry(QtCore.QRect(390, 400, 221, 61))
        self.pushButton_4.setObjectName("pushButton_4")
        self.toolButton_2 = QtWidgets.QToolButton(Dialog)
        self.toolButton_2.setGeometry(QtCore.QRect(500, 290, 41, 20))
        self.toolButton_2.setObjectName("toolButton_2")
        self.toolButton = QtWidgets.QToolButton(Dialog)
        self.toolButton.setGeometry(QtCore.QRect(500, 250, 41, 19))
        self.toolButton.setObjectName("toolButton")
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(260, 170, 341, 181))
        self.groupBox.setAutoFillBackground(True)
        self.groupBox.setStyleSheet("font: 75 11pt \"Rockwell\";\n"
"text-decoration: underline;")
        self.groupBox.setObjectName("groupBox")
        self.groupBox.raise_()
        self.label.raise_()
        self.spinBox.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.line.raise_()
        self.label_4.raise_()
        self.label_5.raise_()
        self.pushButton.raise_()
        self.pushButton_2.raise_()
        self.line_2.raise_()
        self.label_6.raise_()
        self.pushButton_3.raise_()
        self.pushButton_4.raise_()
        self.toolButton_2.raise_()
        self.toolButton.raise_()

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.predict_tab()
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Solar Predictor"))
        self.label.setText(_translate("Dialog", "No of days to Predict :"))
        self.label_2.setText(_translate("Dialog", "Select DataSet (in CSV) :"))
        self.label_3.setText(_translate("Dialog", "Select Trained Model :"))
        self.label_4.setText(_translate("Dialog", "Train Model :"))
        self.label_5.setText(_translate("Dialog", "Predict Data from Model :"))
        self.pushButton.setText(_translate("Dialog", "Train Model "))
        self.pushButton_2.setText(_translate("Dialog", "Predict Data"))
        self.label_6.setText(_translate("Dialog", "Solar Predictor"))
        self.pushButton_3.setText(_translate("Dialog", "Predict"))
        self.pushButton_4.setText(_translate("Dialog", "Retrain"))
        self.toolButton_2.setText(_translate("Dialog", "..."))
        self.toolButton.setText(_translate("Dialog", "..."))
        self.groupBox.setTitle(_translate("Dialog", "Predict DataSet"))

    def predict_tab(self):

        self.toolButton.clicked.connect(self.file_open)
        self.toolButton_2.clicked.connect(self.file_open1)
        self.pushButton_2.clicked.connect(self.predict_data_button)
        self.pushButton.clicked.connect(self.retrain_model_button)
        self.pushButton_3.hide()
        self.pushButton_4.hide()

    def retrain_model_button(self):
        self.groupBox.setTitle("Retrain Model")
        self.label.hide()
        self.spinBox.hide()
        self.pushButton_2.show()
        self.pushButton.hide()
        self.pushButton_4.clicked.connect(self.retrain_model)
        self.pushButton_4.show()
        self.pushButton_3.hide()
    def predict_data_button(self):
        self.groupBox.setTitle("Predict DataSet")
        self.label.show()
        self.spinBox.show()
        self.pushButton.show()
        self.pushButton_2.hide()
        self.pushButton_3.clicked.connect(self.predict_model)
        self.pushButton_4.hide()
        self.pushButton_3.show()

    def predict_table_window(self,pre):
        print('function running')
        self.predict_table.setRowCount(0)
        for row_no,row_data in enumerate(pre):
            self.predict_table.insertRow(row_no)
            print(row_no)
            for col_no,col_data in enumerate(row_data):
                self.predict_table.setItem(row_no,col_no,QtWidgets.QTableWidgetItem(str(col_data)))
        self.predict_table.setGeometry(QtCore.QRect(1000, 5, 1300, 700))


    def file_open(self):
        name=QFileDialog.getOpenFileName(self,'Select CSV')
        self.csv_file=str(name[0])
        print(self.csv_file)
    def file_open1(self):
        name=QFileDialog.getOpenFileName(self,'Select Model')
        self.csv_file1=str(name[0])        

    def predict_model(self):
        print('starting prediction')
        self.spinbox_value = self.spinBox.value()
        print(self.spinbox_value)



        clean=pd.read_csv(self.csv_file)

        model2 = load_model(self.csv_file1)
        clean=clean.drop('Unnamed: 0',1)
        clean=clean.drop('GHI',1)
        clean=clean.drop('DIF',1)
        clean=clean.drop('Minutes',1)
        clean=clean.drop('Month',1)
        clean=clean.drop('Year',1)

        clean=clean[72500:]
        print(clean.shape)

        train_scaler = MinMaxScaler(feature_range=(0, 1))
        clean_sc = train_scaler.fit_transform(clean)
        crow=clean.values
        X_train,X_test,y_train,y_test=train_test_split(clean_sc,crow,shuffle=False)
        ast=train_scaler.transform(clean)
        p=X_test[:-30]

        print('scaling\n')
        global new_data
        new_data=X_test[:-60]
        day_to_pre=self.spinbox_value
        print(day_to_pre)
        data_of_a_day=20
        pre=[]
        #print(*[p[i] for i in range(60)] ,sep='\n')
        print('predicting')
        look_back=30
        prediction_seqs = []
        curr_frame = new_data[-31:]

        print('time series checked')
        predict=[]
        l=0
        for i in range(int(day_to_pre*data_of_a_day/4)):
            cow_data=new_data[-31:]
            print(*cow_data,sep='\n')
            pre_gen = TimeseriesGenerator(cow_data, np.array([i for i in range(len(cow_data))]),
            length=look_back, sampling_rate=1,stride=1,
            batch_size=1)
            x,y=pre_gen[0]
            print('time series checked')
            print(*x,sep='\n')
            predict=model2.predict_generator(iter(pre_gen),1)
            print("\n model2 checked \n")
            print(*predict,sep='\n')
            predict=predict.reshape(4,8)
            print('reshaped')
            for j in range(4):
                pre.append(predict[j])
            print('appended')

            predict=train_scaler.transform(predict)
            new_data = np.vstack([new_data, predict])
            #print(new_data.shape)
            print('stacked')
            l+=1
        p=0
        print('saving data')
        print('%d   %f'%(day_to_pre,len(pre)))
        pred=np.array([pre[i][p] for i in range(20*day_to_pre)])

        # not aloud print("%d \n"%[pre[i][p] for i in range(40)])
        #print(*pred,sep="\n")
        #plt.plot(pred,'orange')
        #plt.plot(original,'blue')
        #plt.show()

        print(*pred,sep='\n')

        
        pred=pred.reshape(-1,1)
        t=datetime.now(pytz.utc)
        t=str(t)
        t=t[:10]
        g='Predicted Data '+t+'.csv'
        #self.predict_table_window(pred)
        np.savetxt(g,pred , delimiter=",")
        print('all done')
        print('%s %d'%(l,len(pre)))

#retraining

    def retrain_model(self):
        print('retrain')
        clean=pd.read_csv(self.csv_file)
        clean=clean.drop('Unnamed: 0',1)
        clean=clean.drop('DNI',1)
        clean=clean.drop('DIF',1)
        clean=clean.drop('Minutes',1)
        clean=clean.drop('Month',1)
        clean=clean.drop('Year',1)
        print('series to supervised')
        self.series_to_supervised(clean,n_in=4,n_out=0)
        shifted_frame=self.agg
        
        print('normalizing')
        print(shifted_frame)
        clean=clean[:-4]
        # normalize the dataset
        train_scaler = MinMaxScaler(feature_range=(-1 , 1))
        X_train = train_scaler.fit_transform(clean)
        # split into train and test sets
        
        y_train=shifted_frame.values
        print('scaled \n')
        print(y_train)
        print(X_train)
        print("%s %d" %(len(X_train),len(y_train)))
        look_back=30
        train_data_gen = TimeseriesGenerator(X_train, y_train,
                length=look_back, sampling_rate=1,stride=1,
                batch_size=1)
        print('generated train data')

        print('starting retraining ')
	#Model name must be given.
        model2 = load_model(self.csv_file1)
        model2.compile(loss='mean_squared_error', optimizer='RMSProp')
        t=datetime.now(pytz.utc)
        t=str(t)
        t=t[:10]
        g='Retrained Model '+t+'.h5'
        model2.save(g)
        history = model2.fit_generator(train_data_gen,epochs=10).history


        t=datetime.now(pytz.utc)
        t=str(t)
        t=t[:10]
        g='Retrained Model '+t+'.h5'
        model2.save(g)
    def series_to_supervised(self,data, n_in=1, n_out=1, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
            Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        print('started')
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        self.agg = concat(cols, axis=1)
        self.agg.columns = names
        # drop rows with NaN values
        if dropnan:
            self.agg.dropna(inplace=True)
	

    

def run():
    app = QtWidgets.QApplication(sys.argv)
    Gui = Ui_MainWindow()
    Gui.show()
    sys.exit(app.exec_())

run()
