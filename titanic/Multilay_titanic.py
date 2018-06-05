import pandas as pd
import numpy
from keras.models import Sequential
from keras.layers import Dense,Dropout


from util.titanic.dataprocess import PreprocessData

all_df = pd.read_excel("data/titanic3.xls")

#
cols = ['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']
all_df = all_df[cols]

#
msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]

print('total:',len(all_df),
      'train:',len(train_df),
      'test:',len(test_df))


#函数里面做了 0.剔除name属性  1.将null填上  2.将sex对应成数字  3.embarked一位有效编码
# 4.转化为array   5.分离label和features   6.标准化数据
train_Features,train_Label = PreprocessData(train_df)
test_Features,test_Label = PreprocessData(test_df)

model = Sequential()

model.add(Dense(units=40,input_dim=9,
                kernel_initializer='uniform',
                activation='relu'))

model.add(Dense(units=30,
                kernel_initializer='uniform',
                activation='relu'))

model.add(Dense(units=1,
                kernel_initializer='uniform',
                activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',metrics=['accuracy'])

train_history = model.fit(x=train_Features,
                          y = train_Label,
                          validation_split=0.1,
                          epochs=30,
                          verbose=1)

scores = model.evaluate(x = test_Features,
                        y = test_Label)
print("\n")
#正确率0.8178571428571428
print(scores[1])

Jack = pd.Series([0,'jack',3,'male',23,1,0,5.0000,'S'])
Rose = pd.Series([1,'jack',1,'female',20,1,0,100.0000,'S'])

JR_df = pd.DataFrame([list(Jack),list(Rose)],
                     columns=['survived','name','pclass','sex','age','sibsp',
                              'parch','fare','embarked'])

all_df = pd.concat([all_df,JR_df])
#   survived  name  pclass     sex   age  sibsp  parch   fare embarked
#0         0  jack       3    male  23.0      1      0    5.0        S
#1         1  jack       1  female  20.0      1      0  100.0        S
print(all_df[-2:])



all_Features,Label = PreprocessData(all_df)

all_probability = model.predict(all_Features)
#[[0.96628034]
 #[0.42929074]
 #[0.96210974]
 #[0.292653  ]
 #[0.95798945]
 #[0.24008392]
 #[0.9311328 ]
 #[0.27213326]
 #[0.9298086 ]
 #[0.26397398]]
print(all_probability[:10])

pd=all_df
pd.insert(len(all_df.columns),
          'probability',all_probability)
#   survived  name  pclass     ...        fare  embarked  probability
#0         0  jack       3     ...         5.0         S     0.140175
#1         1  jack       1     ...       100.0         S     0.965212
print(pd[-2:])


#     survived     ...     probability
#2           0     ...        0.983349
#4           0     ...        0.976059
#105         0     ...        0.968682
#169         0     ...        0.961033
#286         0     ...        0.952543
print(pd[(pd['survived']==0) &  (pd['probability']>0.9) ])