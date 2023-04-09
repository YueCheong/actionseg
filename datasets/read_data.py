# import h5py
import numpy as np
# import os
# import datetime

dir_save = "/temp/liuf/hoi4d/action_depth_npy/"
data = np.load(dir_save + "depth_clip2.npy")
print(type(data))
print(data.shape)
print(data)


# train_data1 = h5py.File("/home/dh/alisa/temp/actionseg/train1.h5","r")
# print("the keys of train data1")
# print([key for key in train_data1.keys()])
# print("1.  get values of center:", train_data1['center'][:].shape)
# # print(train_data['center'][:])
# print("2. get values of pcd:", train_data1['pcd'][:].shape)
# # print(train_data['pcd'][:])
# print("3. get values of label:", train_data1['label'][:].shape)
# # print(train_data['label'][:])


# train_data2 = h5py.File("/home/dh/alisa/temp/actionseg/train2.h5","r")
# print("the keys of train data2")
# print([key for key in train_data2.keys()])
# print("1.  get values of center:", train_data2['center'][:].shape)
# # print(train_data['center'][:])
# print("2. get values of pcd:", train_data2['pcd'][:].shape)
# # print(train_data['pcd'][:])
# print("3. get values of label:", train_data2['label'][:].shape)
# # print(train_data['label'][:])

# train_data3 = h5py.File("/home/dh/alisa/temp/actionseg/train3.h5","r")
# print("the keys of train data3")
# print([key for key in train_data3.keys()])
# print("1.  get values of center:", train_data3['center'][:].shape)
# # print(train_data['center'][:])
# print("2. get values of pcd:", train_data3['pcd'][:].shape)
# # print(train_data['pcd'][:])
# print("3. get values of label:", train_data3['label'][:].shape)
# # print(train_data['label'][:])


# train_data4 = h5py.File("/home/dh/alisa/temp/actionseg/train4.h5","r")
# print("the keys of train data4")
# print([key for key in train_data4.keys()])
# print("1.  get values of center:", train_data4['center'][:].shape)
# # print(train_data['center'][:])
# print("2. get values of pcd:", train_data4['pcd'][:].shape)
# # print(train_data['pcd'][:])
# print("3. get values of label:", train_data4['label'][:].shape)
# # print(train_data['label'][:])




# test_data = h5py.File("/home/dh/alisa/temp/actionseg/test_wolabel.h5","r")
# print("the keys of test data")
# print([key for key in test_data.keys()])
# print("1.  get values of center:", train_data4['center'][:].shape)
# # print(train_data['center'][:])
# print("2. get values of pcd:", train_data4['pcd'][:].shape)
# # print(train_data['pcd'][:])
# # print("3. get values of label:", train_data4['label'][:].shape)
# # # print(train_data['label'][:])
