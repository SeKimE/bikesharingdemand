import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
import seaborn as sns
import os
from scipy import stats


train_df = pd.read_csv("F:/01_WORK/98_STUDY/kaggle/bikesharingdemand/data/train.csv")
test_df = pd.read_csv("F:/01_WORK/98_STUDY/kaggle/bikesharingdemand/data/test.csv")

# 1. 데이터 형식 확인 (사용가능한 형태로)
train_df.info()
train_df['datetime'] = pd.to_datetime(train_df['datetime'])
test_df['datetime'] = pd.to_datetime(test_df['datetime'])
train_df.info()

# 2. 중복, 결측치 확인
sum(train_df.duplicated())
sum(test_df.duplicated())
train_df.isnull().sum()
test_df.isnull().sum()

# 3. 분포, 이상치 확인 (EDA)
train_df['year'] = train_df['datetime'].dt.year
train_df['month'] = train_df['datetime'].dt.month
train_df['day'] = train_df['datetime'].dt.day
train_df['hour'] = train_df['datetime'].dt.hour
train_df['min'] = train_df['datetime'].dt.minute
train_df['sec'] = train_df['datetime'].dt.second
train_df['dayofweek'] = train_df['datetime'].dt.dayofweek
del train_df['min'], train_df['sec']

test_df['year'] = test_df['datetime'].dt.year
test_df['month'] = test_df['datetime'].dt.month
test_df['day'] = test_df['datetime'].dt.day
test_df['hour'] = test_df['datetime'].dt.hour
test_df['min'] = test_df['datetime'].dt.minute
test_df['sec'] = test_df['datetime'].dt.second
test_df['dayofweek'] = test_df['datetime'].dt.dayofweek
del test_df['min'], test_df['sec']


# train_df['year'].value_counts()
# train_df['month'].value_counts()
# train_df['day'].value_counts()
# train_df['hour'].value_counts()
# train_df['dayofweek'].value_counts()



figure, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows = 2, ncols=3)
figure.set_size_inches(18,10)

sns.barplot(data=train_df, x='year', y='count', ax =ax1)
sns.barplot(data=train_df, x='month', y='count', ax =ax2)
sns.barplot(data=train_df, x='day', y='count', ax =ax3)
sns.barplot(data=train_df, x='hour', y='count', ax =ax4)
sns.barplot(data=train_df, x='dayofweek', y='count', ax =ax5)


figure, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(18,10)
sns.boxplot(data=train_df, x='season',y='count',ax=ax1)
sns.boxplot(data=train_df, x='workingday',y='count',ax=ax2)
sns.boxplot(data=train_df, x='holiday',y='count',ax=ax3)
sns.boxplot(data=train_df,y='count',ax=ax4)


figure, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4)
figure.set_size_inches(18,30)

sns.pointplot(data=train_df, x='hour',y='count',ax=ax1)
sns.pointplot(data=train_df, x='hour',y='count',hue='workingday',ax=ax2)
sns.pointplot(data=train_df, x='hour',y='count',hue='holiday',ax=ax3)
sns.pointplot(data=train_df, x='hour',y='count',hue='weather',ax=ax4)



corr_data = train_df[["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"]]
colormap = plt.cm.PuBu
plt.title('Correlation of Numeric Features with Rental Count',y=1,size=18)
sns.heatmap(corr_data.corr(), vmax=.8, linewidths=0.1,square=True,annot=True,cmap=colormap, linecolor="white",annot_kws = {'size':14})



#이상치 제거
from collections import Counter

def detect_outliers(df, n, features):
    outlier_indices=[]
    for i in features:
        q1 = np.percentile(df[i], 25)
        q3 = np.percentile(df[i],75)
        iqr = q3-q1
        
        outlier_step = 1.5 *iqr

        outlier_list_col = df[(df[i]<q1-outlier_step)|(df[i]>q3+outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v>n)

    return multiple_outliers    

outlierstodrop = detect_outliers(train_df, 2, ["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"])

train_df = train_df.drop(outlierstodrop, axis=0).reset_index(drop=True)


#정규화
f, ax = plt.subplots(1, 1, figsize=(10,6))
g = sns.distplot(train_df['count'], color='r', label="Skewness:{:2f}".format(train_df['count'].skew()), ax=ax)
g = g.legend(loc='best')

print("skew : %f"%train_df['count'].skew())
print("kurtosis: %f"%train_df['count'].kurt())
#왜도 0 첨도 3

train_df['count_log'] = train_df['count'].map(lambda i : np.log(i) if i>0 else 0)
f, ax = plt.subplots(1,1,figsize=(10,6))
g = sns.distplot(train_df['count_log'], color='b', label="skewness:{:2f}".format(train_df['count_log'].skew()), ax=ax)
g= g.legend(loc = 'best')

print('skew:%f'%train_df['count_log'].skew())
print('kurt:%f'%train_df['count_log'].kurt())


#windspeed 0 제거 안한경우

#onehotencoding

train_df = pd.get_dummies(train_df, columns=['weather'], prefix='weather')
test_df = pd.get_dummies(test_df, columns=['weather'], prefix='weather')

train_df = pd.get_dummies(train_df, columns=['season'], prefix='season')
test_df = pd.get_dummies(test_df, columns=['season'], prefix='season')

datetime_test = test_df['datetime']
train_df.drop(['datetime','registered','casual','holiday','year','month','count'],axis=1, inplace=True)
test_df.drop(['datetime','holiday','year','month'],axis=1,inplace=True)


#학습
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train = train_df.drop('count_log', axis=1).values
target_label = train_df['count_log'].values
x_test = test_df.values
x_tr, x_vld, y_tr, y_vld = train_test_split(x_train, target_label, test_size=0.3, random_state=2000)
#70프로 학습.

#그래디언트부스트
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05, max_depth=4, min_samples_leaf=15, min_samples_split=10, random_state=42)
regressor.fit(x_tr, y_tr)

y_hat = regressor.predict(x_tr)
plt.scatter(y_tr, y_hat, alpha=0.2)
plt.xlabel('Targets (y_tr)', size = 16)
plt.ylabel('Predictions (y_hat)',size=18)
plt.show()


y_hat_test = regressor.predict(x_vld)
plt.scatter(y_vld, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_vld)', size = 16)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.show()

use_logvals = 1
pred_gb = regressor.predict(x_test)

sub_gb = pd.DataFrame()
sub_gb['datetime'] = datetime_test
sub_gb['count'] = pred_gb
if use_logvals ==1:
    sub_gb['count'] = np.exp(sub_gb['count'])

sub_gb.to_csv('../result/gb.csv',index=False)