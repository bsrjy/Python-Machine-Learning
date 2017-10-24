# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:39:04 2017

@author: dell
"""

#数据预处理

import numpy as np
from sklearn import preprocessing
data = np.array([[3,-1.5,2,-5.4],[0,4,-0.3,2.1],[1,3.3,-1.9,-4.3]])
data_standardized = preprocessing.scale(data)
print('\nMean = ',data_standardized.mean(axis=0))
print('Std deviation = ',data_standardized.std(axis=0))
data_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
data_scaled = data_scaler.fit_transform(data)
print('\nMin max scaled dataa = ',data_scaled)
data_normalized = preprocessing.normalize(data,norm = 'l1')
print('\nL1 normalized data = ',data_normalized)
data_binarized = preprocessing.Binarizer(threshold = 1.4).transform(data)
print('\nBinarized data = ',data_binarized)
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0,2,1,12],[1,3,5,3],[2,3,2,12],[1,2,4,3]])
encoded_vector = encoder.transform([[2,3,5,3]]).toarray()
print('\nEncoded vector = ',encoded_vector)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
input_classes = ['audi','ford','audi','toyota','ford','bmw']
label_encoder.fit(input_classes)
print('\nClass mapping:')
for i,item in enumerate(label_encoder.classes_):
    print(item,'-->',i)
labels=['toyota','ford','audi']
encoded_labels=label_encoder.transform(labels)
print('\nLabels=',labels)
print('Encoded labels=',encoded_labels)
encoded_labels=[2,1,0,3,1]
decoded_labels=label_encoder.inverse_transform(encoded_labels)
print("\nEncoded labels=",encoded_labels)
print("Decoded labels=",decoded_labels)






