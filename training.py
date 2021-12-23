import numpy as np
import cv2
import os
import tensorflow
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import keras.models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import pickle

path='old_data'
testRatio=0.1
valRatio=0.1



count=0
image=[]
classNo=[]
myList=os.listdir(path)
noOfClasses=10

for x in range(0,10):
    myPicList =os.listdir(path+'/'+str(x))
    for y in myPicList:
        curImg=cv2.imread(path+'/'+str(x)+'/'+y)
        curImg=cv2.resize(curImg,(32,32))
        image.append(curImg)
        classNo.append(x)
    print(x,end=" ")

print()
print(len(image))
print(len(classNo))

image=np.array(image)
classNo=np.array(classNo)

x_train,x_test,y_train,y_test=train_test_split(image,classNo,test_size=testRatio)
x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=valRatio)

numofSample=[]
for x in range(0,10):
    numofSample.append(len(np.where(y_train==0)[0]))
print(numofSample)

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses),numofSample)
plt.title("No of Image for each class")
plt.xlabel("Class Id")
plt.ylabel("Number of Image")
plt.show()

def preProcessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img


x_train=np.array(list(map(preProcessing,x_train)))
x_test=np.array(list(map(preProcessing,x_test)))
x_validation=np.array(list(map(preProcessing,x_validation)))

print(x_train.shape)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation=x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_train.shape[2],1)

dataGen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.2,
                           shear_range=0.1,
                           rotation_range=10)

dataGen.fit(x_train)

y_train =to_categorical(y_train,noOfClasses)
y_test =to_categorical(y_test,noOfClasses)
y_validation =to_categorical(y_validation,noOfClasses)

# lenet model
def myModel():
    noOfFilters=60
    sizeOfFilter1=(5,5)
    sizeOfFilter2=(3,3)
    sizeOfPool=(2,2)
    noOfNode=500

    model=Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(32,32,1),activation="relu")))
    model.add((Conv2D(noOfFilters, sizeOfFilter1,  activation="relu")))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation="softmax"))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

    return model

model=myModel()
print(model.summary())

batchSizeVal=50
epochVal=100
stepsPerEpochVal =len(x_train)//batchSizeVal

history=model.fit_generator(dataGen.flow(x_train,y_train,batch_size=batchSizeVal),
                            steps_per_epoch=stepsPerEpochVal,epochs=epochVal,
                             validation_data=(x_validation,y_validation),shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

plt.show()
score=model.evaluate(x_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy = ',score[1])




pickle_out=open("A/model_trained.p",'wb')
pickle.dump(model,pickle_out)
pickle_out.close()