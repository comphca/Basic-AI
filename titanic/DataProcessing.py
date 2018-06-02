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
#加载数据
#使用dataframe 读取数据并进行预处理
all_df = pd.read_excel(filepath)
print(all_df[:5])

#因为数据中有很多特征是训练不需要的，先剔除掉
cols = ['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']
#重新组织数据
all_df = all_df[cols]
print(all_df[:5])


#使用Panas DataFrame 进行数据处理
#axis???   name属性在训练阶段不需要，但预测阶段需要，先剔除  再将整个数据集赋值给 df
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
#查看筛选后，整个数据集里面还有多少数据的值是null
print(all_df.isnull().sum())

#age有空值 ，处理方式不是将空值置为0，不合理，而是计算整个age的平均值，然后再赋到空值中去
age_mean = df['age'].mean()
df['age'] = df['age'].fillna(age_mean)

fare_mean = df['fare'].mean()
df['fare'] = df['fare'].fillna(fare_mean)

#性别再数据集里面是文字形式，转换为数字0，1
df['sex'] = df['sex'].map({'female':0,'male':1}).astype(int)



#   survived  pclass  sex     ...      embarked_C  embarked_Q  embarked_S
#0         1       1    0     ...               0           0           1
#1         1       1    1     ...               0           0           1
#登机口里面有三个值为字母   进行有效一位编码   
x_OneHot_df = pd.get_dummies(data=df,columns=['embarked'])
print(x_OneHot_df[:2])


#将DataFrame转化为array
#(1309, 10)    数据共1309项10个字段
ndarray = x_OneHot_df.values
print(ndarray.shape)

#[[  1.       1.       0.      29.       0.       0.     211.3375   0.
#   0.       1.    ]
# [  1.       1.       1.       0.9167   1.       2.     151.55     0.
#    0.       1.    ]]
#第0个字段是label  第1及之后的字段是features
print(ndarray[:2])


#切片操作 提取features 和 label
Label = ndarray[:,0]
Features = ndarray[:,1:]

# [1. 1.]
print(Label[:2])

#[[  1.       0.      29.       0.       0.     211.3375   0.       0.
#    1.    ]
# [  1.       1.       0.9167   1.       2.     151.55     0.       0.
#    1.    ]]
print(Features[:2])

#sklearn模块里面提供了preprocessing数据预处理模块进行标准化  输入参数feature_range设置标准化后的范围在0到1之间
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
#.fit_transform函数传入要标准化的数据集  进行标准化
scaledFeature = minmax_scale.fit_transform(Features)


#[[0.         0.         0.36116884 0.         0.         0.41250333
#  0.         0.         1.        ]
# [0.         1.         0.00939458 0.125      0.22222222 0.2958059#
#  0.         0.         1.        ]]
print(scaledFeature[:2])


#以随机分布方式将数据集分为训练数据和测试数据
# numpy.random.rand  按照8：2的比例产生msk
msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]

print('total:',len(all_df),
      'train:',len(train_df),
      'test:',len(test_df))



#将上面的数据处理流程写在函数里面
#0.剔除name属性  1.将null填上  2.将sex对应成数字  3.embarked一位有效编码   4.转化为array   5.分离label和features   6.标准化数据  
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

#使用上面函数  原始数据需要建立剔除不需要的列表DataFrame  8：2分配数据  再处理