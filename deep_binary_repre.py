'''Trains a simple deep hash based image retrieval.

Author: Xuchao, Lu
School: Shanghai Jiaotong University
Mentor: Li, Song
'''

from __future__ import print_function
import numpy as np
from argparse import ArgumentParser
from random import sample
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import time
import os
from generate_code import CodeGenerator, generate_code_74
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def parse_parameters():
    ### Parse the command line parameters.

    parser = ArgumentParser()
    parser.add_argument('-l', type=int, default=36, help='hash length')
    parser.add_argument('-q', type=int, default=20, help='query num')
    parser.add_argument('-e', type=int, default=10, help='number of epoch')
    parser.add_argument('-s', type=int, default=2, help='0=silent, 1=log')
    parser.add_argument('-t', type=int, default=1000, help='retrieve top n')
    parser.add_argument('-o', type=int, default=1, help='optimizer: 1.adadelta 2.sgd')
    parser.add_argument('-a', type=int, default=2, help='additional AC layer: 1.no 2.yes')
    parser.add_argument('-c', type=int, default=1, help='hash code type: 1.origin 2.74')
    parser.add_argument('-n', type=int, default=50000, help='number of training images')

    args = parser.parse_args()

    print ('Image Retrieval Experiment')
    print ('-------------------------------------')
    print ('Dataset:             Cifar10')
    print ('Hash Length:         %s' % args.l)
    print ('Query Number:        %s' % args.q)
    print ('Epoch Number:        %s' % args.e)
    if (args.o == 1):
        opt = 'Adadelta'
    elif (args.o == 2):
        opt = 'SGD'
    print ('Optimizer:           %s' % opt)
    if (args.a == 1):
        opt = 'No'
    elif (args.a == 2):
        opt = 'Yes'
    print ('Additional AC layer: %s' % opt)
    print ('-------------------------------------')

    return args

def data_preparation(args):

    def generate_code(hash_len=12, nb_classes=10):
        if (hash_len == 32):
            coder = CodeGenerator(8,4)
            code = coder.get_hash(nb_classes)
            return [c * 4 for c in code]
        coder = CodeGenerator(12, 6)
        code = coder.get_hash(nb_classes)
        return [c * (hash_len / 12) for c in code]

    ### Prepare the training data

    batch_size = 32
    nb_classes = 10
    nb_epoch = args.e
    hash_len = args.l
    top = args.t
    query_num = args.q

    # input image dimensions
    img_rows, img_cols = 32, 32
    # the CIFAR10 images are RGB
    img_channels = 3

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train2 = []
    y_train2 = []
    X_test2 = []
    y_test2 = []
    if (args.n != 50000):
        print ('Reshape training datasets!')
        num = args.n / 10
        num_test = num / 5
        count = [0 for i in range(10)]
        count_test = [0 for i in range(10)]
        for i,y in enumerate(y_train):
            if (count[int(y)] < num):
                count[int(y)] += 1
                X_train2.append(X_train[i])
                y_train2.append(y_train[i])
        for i,y in enumerate(y_test):
            if (count_test[int(y)] < num_test):
                count_test[int(y)] += 1
                X_test2.append(X_test[i])
                y_test2.append(y_test[i])
        X_train = np.array(X_train2)
        y_train = np.array(y_train2)
        X_test = np.array(X_test2)
        y_test = np.array(y_test2)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to hash matrices
    if (args.c == 1):
        code = generate_code(hash_len)
        print ('Use new hash code.')
    if (args.c == 2):
        code = generate_code_74(hash_len)
        print ('Use 74 hamming code.')
    Y_train = np.array([code[y[0]] for y in y_train]).astype(float)
    Y_test = np.array([code[y[0]] for y in y_test]).astype(float)

    return (X_train, X_test, y_train, y_test, Y_train, Y_test)

def deepbin_train(args, data):

    X_train, X_test, y_train, y_test, Y_train, Y_test = data

    ### Build the model
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    if (args.a == 2):
        model.add(Dense(512))
        model.add(Activation('relu'))
        print ('Add another AC layer!')
    model.add(Dense(args.l))
    model.add(Activation('sigmoid'))

    # Let's train the model using RMSprop
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    if (args.o == 1):
        optimizer = 'adadelta'
    elif (args.o == 2):
        optimizer = sgd
    model.compile(loss='mean_squared_error',
                  # optimizer='adadelta',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')

    model.fit(X_train, Y_train,
              batch_size=32,
              nb_epoch=args.e,
              verbose=args.s,
              validation_data=(X_test, Y_test),
              shuffle=True)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    timenow = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
    if not os.path.exists('model'):
        os.mkdir('model', 0755)
    model.save('model/%s' % timenow)
    print ('-------------------------------------')
    print ('Saved: model/%s' % timenow)
    print ('-------------------------------------')
    # model = load_model('test.h5')

    return model

def deepbin_retrieve(args, data, model):

    X_train, X_test, y_train, y_test, Y_train, Y_test = data

    # a: [[1,2],[3,4]], b: [[5,6],[7,8]]
    # c: [[[1,5],[2,6]],[[3,7],[4,8]]]
    def converge_two_list(a, b):
        c = [[[0,0] for i in range(len(a[0]))] for j in range(len(a))]
        for i in range(len(a)):
            for j in range(len(a[0])):
                c[i][j][0] = a[i][j]
                c[i][j][1] = b[i][j]
        return c

    # hs: (hamming dist, similarity) pair of query results set
    # top_num: the number of ranking
    # return the mAP of the retrieval
    def calculate_map(hs, top_num):
        query_num = len(hs)
        ap_sum = 0.0
        for p in hs:
            ap = 0.0
            count = 0.0
            for i in range(top_num):
                if p[i][1] == 1:
                    count += 1
                    ap += count / (i + 1)
            ap = 0 if count == 0.0 else ap / count
            ap_sum += ap
        mAP = ap_sum / query_num
        return mAP

    # return the mAC of the retrieval
    def calculate_mac(hs, top_num):
        query_num = len(hs)
        correct = 0.0
        for p in hs:
            count = 0
            for i in range(top_num):
                if p[i][1] == 1:
                    count += 1
                    correct += 1
            # print count, y_test[m]
        mAC = correct / (query_num * top_num)
        return mAC

    ### Start the retrieval

    train_feature = model.predict(X_train)
    query_feature = model.predict(X_test)


    train_hash = train_feature > 0.5
    query_hash = query_feature > 0.5

    # Fetch n random samples from the query set
    index = sample(range(y_test.shape[0]), args.q)
    query_hash_sample = [query_hash[i] for i in index]
    hamming = [[sum(t ^ q) for t in train_hash] for q in query_hash_sample]

    # similarity matrix
    y_test_sample = [y_test[i] for i in index]
    similarity = [[1 if q == t else 0 for t in y_train] for q in y_test_sample]

    # sort the hamming,similarity pair
    hs_pair = converge_two_list(hamming, similarity)
    hs_pair_sort = [sorted(p) for p in hs_pair]

    # calculate the mAP and mAC
    mAP = calculate_map(hs_pair_sort, args.t)
    mAC = calculate_mac(hs_pair_sort, args.t)
    print ('mAP = %.2f%%' % (mAP*100))
    print ('mAC = %.2f%%' % (mAC*100))

    return (mAP, mAC) 

if __name__ == '__main__':
    args = parse_parameters()
    data = data_preparation(args)
    model = deepbin_train(args, data)
    result = deepbin_retrieve(args, data, model)


