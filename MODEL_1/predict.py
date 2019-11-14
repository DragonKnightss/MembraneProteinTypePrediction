import numpy as np
from keras.utils.np_utils import to_categorical
import pandas as pd
from keras.layers import Dense, LSTM, Activation, Input, Dropout, BatchNormalization, Conv1D,MaxPool1D, Merge
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.optimizers import Adam, SGD
from keras.utils import normalize
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from keras.layers import Bidirectional
from keras.layers import Masking
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import random

def file2str(filename):
    fr = open(filename)
    numline = fr.readlines()
    m = len(numline)
    index = -1
    A = []
    F = []
    for eachline in numline:
        index += 1
        if '>' in eachline:
            A.append(index)
    B = []
    for eachline in numline:
        line = eachline.strip()
        listfoemline = line.split()
        B.append(listfoemline)

    for i in range(len(A) - 1):
        K = A[i]
        input_sequence = B[K + 1]
        input_sequence = str(input_sequence)
        input_sequence = input_sequence[1:-1]
        for j in range(A[i + 1] - A[i]):
            if K < A[i + 1] - 2:
                C = str(B[K + 2])
                input_sequence = input_sequence + C[1:-1]
                K += 1
        input_sequence = input_sequence.replace('\'', '')
        F.append(input_sequence)
    return F

def str2dic(input_sequence):
    char = sorted(['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H'])
    char_to_index = {}
    index = 0
    result_index = []
    for c in char:
        char_to_index[c] = index
        index = index + 1
    for word in input_sequence:
        result_index.append(char_to_index[word])
    return result_index

def vec_to_onehot(mat,m,n,pc,k):
    return_mat = np.zeros((m, 5000, k))

    for i in range(len(mat)):
        metrix = np.zeros((5000, k))
        for j in range(len(mat[i])):
            metrix[j] = pc[mat[i][j]]
        return_mat[i,:,:] = metrix
    return return_mat[:,:n,:]

def create_model(input_shape,unit,filter):
    model = Sequential()

    model.add(Conv1D(input_shape=input_shape,strides = 10, kernel_size = 15,filters = filter, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Masking(input_shape=input_shape,mask_value = 0))

    model.add(Bidirectional(LSTM(unit,return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(unit,return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add((Dense(8)))
    model.add(Activation('softmax'))

    return model

def fasta2num(filename):
    input_sequence = file2str(filename)

    X = []
    for i in range(len(input_sequence)):
        result_index = str2dic(input_sequence[i])
        X.append(result_index)
    return X

X_train = fasta2num('\DATASET\DATASET_1\Sequence\Train.fasta')
X_test = fasta2num('\DATASET\DATASET_1\Sequence\Test.fasta')

y_train = np.loadtxt('\DATASET\DATASET_1\Sequence/trainLabel') - 1
y_test = np.loadtxt('\DATASET\DATASET_1\Sequence/testLabel') - 1

pc = np.loadtxt('Auto_enconder_19')
pc = pc.reshape(20,19)

X_train = vec_to_onehot(X_train,3249,2000,pc,19)

X, X_val, Y, Y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=20, stratify=y_train)

Y = to_categorical(Y)
Y_val = to_categorical(Y_val)

X_test = vec_to_onehot(X_test,4333,2000,pc,19)
y_test = to_categorical(y_test)

print('Preprocess Finished')

model = create_model([2000, 19],128,256)
model.load_weights('weights.best___19')
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
predict = model.predict(X_val)

label = np.array([np.argmax(predict[i]) for i in range(len(X_val))])
y_test__ = np.array([np.argmax(Y_val[i]) for i in range(len(X_val))])

predict = np.sum(label == y_test__) / len(y_test__)
print(np.sum(label == y_test__) / len(y_test__))
print(recall_score(y_test__, label, average=None))
print(classification_report(y_test__, label))

model = create_model([2000, 19],128,256)
model.load_weights('weights.best___19')
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

predict = model.predict(X_test)

label = np.array([np.argmax(predict[i]) for i in range(len(X_test))])
y_test__ = np.array([np.argmax(y_test[i]) for i in range(len(X_test))])

predict = np.sum(label == y_test__)/len(y_test__)
print(np.sum(label == y_test__)/len(y_test__))
