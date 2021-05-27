import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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







