#import PyQt5
import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier 
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras.models  import Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
print(tf.__version__)


# load dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("diabetes.csv", header=None, names=col_names)

feature_cols = ['glucose','bp','bmi','pedigree','age']
X = pima[feature_cols].iloc[:,:].values
y = pima.iloc[:,-1].values
#Arrange
X= np.delete(X, (0), axis=0)
y = np.delete(y, (0), axis=0)
X=X.astype(float)
y=y.astype(float)

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15, random_state=42)

#Train the model
LogReg = LogisticRegression(penalty="l2", C=1.800,class_weight='balanced')
LogReg.fit(X_train, y_train)

#Predict 
y_pred = LogReg.predict(X_test)
#Metrics
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix

print(classification_report(y_test, y_pred))

print("Validation Accuracy Logistic Regression:",metrics.accuracy_score(y_test, y_pred)*100)

DTreeModel = DecisionTreeClassifier(criterion = 'entropy',max_depth = 3)
DTreeModel.fit(X_train,y_train)
yPredDTree = DTreeModel.predict(X_test)
cDTree = 0
for indx in range(len(yPredDTree)):
    if yPredDTree[indx] == y_test[indx]:
        cDTree +=1
print('Validation Accuracy Descision Tree :{}'.format((cDTree/len(X_test))*100))


# Random Forest Classifier


rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
#print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Validation Accuracy Random Forest: {}".format((rf.score(X_test, y_test))*100))



# ---------------------------------------- linear svc -------------------------------------- 
from sklearn.svm import SVC

# Create the Linear SVC model
LinearSVCModel = SVC(kernel='linear', random_state=42)

# Train the model
LinearSVCModel.fit(X_train, y_train)

# Predict using the Linear SVC model
yPredSVC = LinearSVCModel.predict(X_test)

# Calculate validation accuracy
cSVC = 0
for indx in range(len(yPredSVC)):
    if yPredSVC[indx] == y_test[indx]:
        cSVC += 1

print('Validation Accuracy Linear SVC: {:.2f}%'.format((cSVC / len(X_test)) * 100))

#------------------------------------------- Gradient Boosting Classifier (XGBoost)---------------------
from xgboost import XGBClassifier

XGBModel = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, random_state=42)
XGBModel.fit(X_train, y_train)
yPredXGB = XGBModel.predict(X_test)

cXGB = 0
for indx in range(len(yPredXGB)):
    if yPredXGB[indx] == y_test[indx]:
        cXGB += 1
print('Validation Accuracy XGBoost: {:.2f}%'.format((cXGB / len(X_test)) * 100))
# -------------------------------------------- SGD Classifier ---------------------------------------
from sklearn.linear_model import SGDClassifier

# Create and train the model
SGDModel = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=42)  # 'hinge' is for linear SVM
SGDModel.fit(X_train, y_train)

# Predict on the test set
yPredSGD = SGDModel.predict(X_test)

# Calculate accuracy manually
cSGD = 0
for indx in range(len(yPredSGD)):
    if yPredSGD[indx] == y_test[indx]:
        cSGD += 1
print('Validation Accuracy SGD Classifier: {:.2f}%'.format((cSGD / len(X_test)) * 100))
# ---------------------------------------------- SVC -------------------------------------------
from sklearn.svm import SVC

SVMModel = SVC(kernel='linear', C=1, random_state=42)  # You can try other kernels like 'rbf' or 'poly'
SVMModel.fit(X_train, y_train)
yPredSVM = SVMModel.predict(X_test)

cSVM = 0
for indx in range(len(yPredSVM)):
    if yPredSVM[indx] == y_test[indx]:
        cSVM += 1
print('Validation Accuracy SVM: {:.2f}%'.format((cSVM / len(X_test)) * 100))


#Optimizer

#Normalizing the Data

normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)

DNNModel = keras.models.Sequential([
        Dense(16,activation = 'relu',input_shape= (5,)),
        Dense(8,activation = 'relu'),
        Dense(4,activation = 'relu'),
        Dense(1,activation = 'sigmoid') #single neuron with sigmoid activation function to predict +ve or -ve result
        ])
    
DNNModel.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['acc'])

print('Training Neural Network Model for 1000 Epochs......')

NNHist = DNNModel.fit(X_train_norm,y_train,validation_data = (X_test_norm,y_test),epochs = 100,verbose = 0)


print('Validation Accuracy Neural Network:',(NNHist.history['val_acc'][-1])*100)

fig, ax = plt.subplots()
ax.plot(NNHist.history["loss"],'r', marker='.', label="Train Loss")
ax.plot(NNHist.history["val_loss"],'b', marker='.', label="Validation Loss")
ax.legend()

from PyQt5 import QtWidgets,uic
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap


def Result():
	glu= float(dlg.Iglu.text())
	bpp= float(dlg.Ibp.text())
	bmii=float(dlg.Ibmi.text())
	pedi=float(dlg.Iped.text())
	agee=float(dlg.Iage.text())
	if(pedi==0):
		pedi=0.4726
	[pred1]=LogReg.predict([[glu,bpp,bmii,pedi,agee]])
	
	if (pred1==0):
		dlg.output.setText("No! Our Diagnostics suggest you \ndo not suffer from Diabetes \n ٩(◕‿◕)۶")
	else:
		dlg.output.setText("Yes. Our Diagnostics suggest you \ndo suffer from Diabetes. \nPlease visit a Doctor. \n (｡•́︿•̀｡)	 ")
	
def Clfn():
    dlg.Iglu.clear()
    dlg.Iped.clear()
    dlg.Ibp.clear()
    dlg.Ibmi.clear()
    dlg.Iage.clear()
    dlg.output.clear()

app= QtWidgets.QApplication([])
dlg=uic.loadUi("gui1.ui")
pixmap=QPixmap('img1.png')

dlg.img.setPixmap(pixmap)
dlg.subbtn.clicked.connect(Result)
dlg.clrbtn.clicked.connect(Clfn)

dlg.show()
app.exec()
