import numpy as np
import random
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import tensorflow as tf

# モデルを構築 --- (※2)
#input_shape = (128, 14, 1)
def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(70, 3, 3,
	border_mode='same',
	input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(90, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(90, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
	optimizer=Adam(),
	metrics=['accuracy'])

    return model

# モデルを訓練する --- (※3)
def model_train(X, y,model):
    model.fit(X, y, batch_size=100, epochs=100)
    # モデルを保存する --- (※4)
    # hdf5_file = "./after/brain.hdf5"
    # model.save_weights(hdf5_file)
    return model

# モデルを評価する --- (※5)
def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])

data1 = np.loadtxt("after/left/sekileft1.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))
data2 = np.loadtxt("after/left/satoleft3.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))
data3 = np.loadtxt("after/left/satoleft2.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))
data4 = np.loadtxt("after/left/satoleft1.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))
data5 = np.loadtxt("after/left/iwaleft4.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))
data6 = np.loadtxt("after/left/iwaleft3.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))
data7 = np.loadtxt("after/left/iwaleft2.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))
data8 = np.loadtxt("after/left/iwaleft1.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))

data9 = np.loadtxt("after/neutral/iwaneutral.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))
data10 = np.loadtxt("after/neutral/satoneutral.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))
data11 = np.loadtxt("after/neutral/sekineutral.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))

data12 = np.loadtxt("after/right/iwaright.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))
data13 = np.loadtxt("after/right/satoright.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))
data14 = np.loadtxt("after/right/sekiright1.csv",delimiter=",", skiprows=1,usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15))


left = np.vstack((data1,data2,data3,data4,data5,data6,data7,data8))
neutral = np.vstack((data9,data10,data11))
right = np.vstack((data12,data13,data14))

print(left.shape)
print(neutral.shape)
print(right.shape)


long = 128
dataleft = []
labelleft = []
for i in range(0, left.shape[0] - long, long):
    a = left[i:i + long]
    b = [1,0,0]
    dataleft.append(a)
    labelleft.append(b)
dataleft = np.array(dataleft)
labelleft = np.array(labelleft)

dataneutral = []
labelneutral = []
for i in range(neutral.shape[0] - long):
    a = neutral[i:i + long]
    b = [0,1,0]
    dataneutral.append(a)
    labelneutral.append(b)
dataneutral = np.array(dataneutral)
labelneutral = np.array(labelneutral).reshape(-1,1)

dataright = []
labelright = []
for i in range(right.shape[0] - long):
    a = right[i:i + long]
    b = [0,0,1]
    dataright.append(a)
    labelright.append(b)
dataright = np.array(dataright)
labelright = np.array(labelright).reshape(-1,1)

print(labelneutral.shape)
print(labelright.shape)
print(labelneutral.shape)

alldata = np.vstack((dataleft,dataneutral,dataright))
alllabel = np.hstack((labelleft,labelneutral,labelright))


randomall = random.sample(range(alldata.shape[0]), k = alldata.shape[0])

x = []
e = []
f = []
for x in randomall:
    c = alldata[x]
    d = alllabel[x]
    e.append(c)
    f.append(d)

e = np.array(e)
f = np.array(f)

x_train = round(e.shape[0]*0.7)
train_x = e[:x_train]
test_x = e[x_train:]
train_y = f[:x_train]
test_y = f[x_train:]



#正規化
train_x = train_x.astype("float") / 32
test_x  = test_x.astype("float")  / 32

X_train =train_x.reshape(train_x.shape[0], 128, 14, 1)
X_test = test_x.reshape(test_x.shape[0], 128, 14, 1)
input_shape = (128, 14, 1)


model = build_model(input_shape)
model2 = model_train(X_train,train_y,model)
model_eval(model2 ,X_test, test_y)

