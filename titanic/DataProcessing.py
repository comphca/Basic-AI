import urllib.request
import os
import numpy
import pandas as pd
from sklearn import preprocessing

url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"

filepath = "./data/titanic3.xls"

'''
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url,filepath)
    print('download:',result)
'''

all_df = pd.read_excel(filepath)
print(all_df[:5])


cols = ['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']

all_df = all_df[cols]
print(all_df[:5])


#axis???
df = all_df.drop(['name'],axis=1)

#survived      0
#name          0
#pclass        0
#sex           0
#age         263
#sibsp         0
#parch         0
#fare          1
#embarked      2
#dtype: int64
print(all_df.isnull().sum())

age_mean = df['age'].mean()
df['age'] = df['age'].fillna(age_mean)

fare_mean = df['fare'].mean()
df['fare'] = df['fare'].fillna(fare_mean)


df['sex'] = df['sex'].map({'female':0,'male':1}).astype(int)



#   survived  pclass  sex     ...      embarked_C  embarked_Q  embarked_S
#0         1       1    0     ...               0           0           1
#1         1       1    1     ...               0           0           1
x_OneHot_df = pd.get_dummies(data=df,columns=['embarked'])

print(x_OneHot_df[:2])



(1309, 10)
ndarray = x_OneHot_df.values
print(ndarray.shape)

#[[  1.       1.       0.      29.       0.       0.     211.3375   0.
#   0.       1.    ]
# [  1.       1.       1.       0.9167   1.       2.     151.55     0.
#    0.       1.    ]]
print(ndarray[:2])


Label = ndarray[:,0]
Features = ndarray[:,1:]

# [1. 1.]
print(Label[:2])

#[[  1.       0.      29.       0.       0.     211.3375   0.       0.
#    1.    ]
# [  1.       1.       0.9167   1.       2.     151.55     0.       0.
#    1.    ]]
print(Features[:2])


minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))

scaledFeature = minmax_scale.fit_transform(Features)


#[[0.         0.         0.36116884 0.         0.         0.41250333
#  0.         0.         1.        ]
# [0.         1.         0.00939458 0.125      0.22222222 0.2958059#
#  0.         0.         1.        ]]
print(scaledFeature[:2])

msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]

print('total:',len(all_df),
      'train:',len(train_df),
      'test:',len(test_df))


def PreprocessData(raw_df):
    df = raw_df.drop(['name'],axis=1)

    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)

    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)

    df['sex'] = df['sex'].map({'female':0,'male':1}).astype(int)

    x_OneHot_df = pd.get_dummies(data=df,columns=["embarked"])

    ndarray = x_OneHot_df.values

    Features = ndarray[:,1:]
    Label = ndarray[:,0]


    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))

    scaledFeature = minmax_scale.fit_transform(Features)

    return scaledFeature,Label


train_Feature , train_Label = PreprocessData(train_df)
test_Feature, test_Label = PreprocessData(test_df)

#[[0.         0.         0.36116884 0.         0.         0.41250333
#  0.         0.         1.        ]
# [0.         1.         0.37369494 0.125      0.22222222 0.2958059
#  0.         0.         1.        ]]
#[1. 0.]
print(train_Feature[:2])
print(train_Label[:2])