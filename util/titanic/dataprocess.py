import pandas as pd
from sklearn import preprocessing

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