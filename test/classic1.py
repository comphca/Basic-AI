import numpy as np
import os
import math

zeroclass = []
label_zeroclass = []

oneclass = []
label_oneclass = []

def get_files(file_dir,ratio):
    for file in os.listdir(file_dir+'/dolphin'):
        zeroclass.append(file_dir + '/dolphin' + '/' + file)
        label_zeroclass.append(0)

    for file in os.listdir(file_dir+'/seahorse'):
        zeroclass.append(file_dir + '/seahorse' + '/' + file)
        label_zeroclass.append(1)

    image_list = np.hstack((zeroclass,oneclass))
    label_list = np.hstack((label_zeroclass,label_oneclass))

    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = list(temp[:,0])
    all_label_list = list(temp[:,1])

    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))
    n_train = n_sample - n_val

    train_image = all_image_list[0:n_train]
    train_label = all_label_list[0:n_train]
    train_label = [int(float(i)) for i in train_label]
    return train_image,train_label

