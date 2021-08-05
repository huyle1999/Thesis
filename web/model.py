# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:58:39 2020

@author: HP PC
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#rd_data=pd.read_csv("C:\\Users\\Administrator\\Desktop\\New folder (9)\\web\\AB_NYC_2019.csv")
df=pd.read_csv("C:\\Users\\Administrator\\Desktop\\Khoaluan\\web\\united2.csv")

df['Rank'] = df.USD.copy()
df_1 = df[df['USD'] < 50000]
df_2 = df[(df['USD'] >= 50000) & (df['USD'] < 100000)]
df_3 = df[(df['USD'] >= 100000) & (df['USD'] < 150000)]
df_4 = df[df['USD'] >= 150000]
## Tìm những trường dữ liệu có giá trị NaN
nan_values = df.isna()
nan_columns = nan_values.any()
columns_with_nan = df.columns[nan_columns].tolist()
columns_with_nan
## Tỉ lệ NaN của các trường dữ liệu
df.isna().sum()/df.shape[0]*100
per = 0.5 # Chọn xóa những cột dữ liệu có trên 50% data là NaN
df_dropped = df.dropna(axis=1,thresh=int(df.shape[0]*per))
df_dropped_2 = df_dropped.dropna(how='any')
df_dropped_2.to_csv('united3.csv',encoding="utf-8-sig",index=False)
df_dropped_2=pd.read_csv("united3.csv")
#df_dropped_2 = df_dropped.dropna(axis=0,thresh=int(df.shape[1]*0.5))
df_dropped_2
for i in range(0,len(df_dropped_2)):
    if df_dropped_2['Quan'][i]==' Quận 1':
        df_dropped_2['Quan'][i]=1
    elif df_dropped_2['Quan'][i]==' Quận 2':
        df_dropped_2['Quan'][i]=2
    elif df_dropped_2['Quan'][i]==' Quận 3':
        df_dropped_2['Quan'][i]=3
    elif df_dropped_2['Quan'][i]==' Quận 4':
        df_dropped_2['Quan'][i]=4
    elif df_dropped_2['Quan'][i]==' Quận 5':
        df_dropped_2['Quan'][i]=5
    elif df_dropped_2['Quan'][i]==' Quận 6':
        df_dropped_2['Quan'][i]=6
    elif df_dropped_2['Quan'][i]==' Quận 7':
        df_dropped_2['Quan'][i]=7
    elif df_dropped_2['Quan'][i]==' Quận 8':
        df_dropped_2['Quan'][i]=8
    elif df_dropped_2['Quan'][i]==' Quận 9':
        df_dropped_2['Quan'][i]=9
    elif df_dropped_2['Quan'][i]==' Quận 10':
        df_dropped_2['Quan'][i]=10
    elif df_dropped_2['Quan'][i]==' Quận 11':
        df_dropped_2['Quan'][i]=11
    elif df_dropped_2['Quan'][i]==' Quận 12':
        df_dropped_2['Quan'][i]=12
    elif df_dropped_2['Quan'][i]==' Quận Bình Tân':
        df_dropped_2['Quan'][i]=13 
    elif df_dropped_2['Quan'][i]==' Quận Thủ Đức':
        df_dropped_2['Quan'][i]=14
    elif df_dropped_2['Quan'][i]==' Quận Bình Thạnh':
        df_dropped_2['Quan'][i]=15 
    elif df_dropped_2['Quan'][i]==' Huyện Bình Chánh':
        df_dropped_2['Quan'][i]=16
    elif df_dropped_2['Quan'][i]==' Quận Tân Bình':
        df_dropped_2['Quan'][i]=17  
    elif df_dropped_2['Quan'][i]==' Huyện Nhà Bè':
        df_dropped_2['Quan'][i]=18
    elif df_dropped_2['Quan'][i]==' Quận Tân Phú':
        df_dropped_2['Quan'][i]=19
    elif df_dropped_2['Quan'][i]==' Quận Gò Vấp':
        df_dropped_2['Quan'][i]=20
    elif df_dropped_2['Quan'][i]==' Quận Phú Nhuận':
        df_dropped_2['Quan'][i]=21
    elif df_dropped_2['Quan'][i]==' Huyện Hóc Môn':
        df_dropped_2['Quan'][i]=22
    else:
        df_dropped_2['Quan'][i]=23  
for i in range(0,len(df_dropped_2)):
    if df_dropped_2['TinhTrangBDS'][i]=='Đã bàn giao':
        df_dropped_2['TinhTrangBDS'][i]=1
    else:
        df_dropped_2['TinhTrangBDS'][i]=2
for i in range(0,len(df_dropped_2)):
    if df_dropped_2['Loai'][i]=='Chung cư':
        df_dropped_2['Loai'][i]=1
    elif df_dropped_2['Loai'][i]=='Căn hộ dịch vụ':
        df_dropped_2['Loai'][i]=2
    elif df_dropped_2['Loai'][i]=='Duplex':
        df_dropped_2['Loai'][i]=3
    elif df_dropped_2['Loai'][i]=='Officetel':
        df_dropped_2['Loai'][i]=4
    elif df_dropped_2['Loai'][i]=='Penthouse':
        df_dropped_2['Loai'][i]=5
    elif df_dropped_2['Loai'][i]=='Căn hộ dịch vụ, mini':
        df_dropped_2['Loai'][i]=6
    else:
        df_dropped_2['Loai'][i]=7
for i in range(0,len(df_dropped_2)):
    if df_dropped_2['TinhTrangNoiThat'][i]=='Hoàn thiện cơ bản':
        df_dropped_2['TinhTrangNoiThat'][i]=1
    elif df_dropped_2['TinhTrangNoiThat'][i]=='Nội thất đầy đủ':
        df_dropped_2['TinhTrangNoiThat'][i]=2
    elif df_dropped_2['TinhTrangNoiThat'][i]=='Nội thất cao cấp':
        df_dropped_2['TinhTrangNoiThat'][i]=3
    else:
        df_dropped_2['TinhTrangNoiThat'][i]=4
for i in range(0,len(df_dropped_2)):
    if df_dropped_2['GiayTo'][i]=='Đã có sổ':
        df_dropped_2['GiayTo'][i]=1
    elif df_dropped_2['GiayTo'][i]=='Đang chờ sổ':
        df_dropped_2['GiayTo'][i]=2
    else:
        df_dropped_2['GiayTo'][i]=3
df_x = df_dropped_2.iloc[:, 1:9]
df_y = df_dropped_2.iloc[:, 10]
scaler = StandardScaler()
scaler.fit(df_x)
data_preprocessed = scaler.transform(df_x)
X,X_test,Y,Y_test = train_test_split(data_preprocessed,df_y,test_size=0.2,random_state=365)
reg = LinearRegression().fit(X, Y)

# total null valves
# rd_data.isnull().sum()

#mean imputation for missing values in reviews per month section
#rd_data['reviews_per_month'].fillna(rd_data['reviews_per_month'].mean(), inplace=True)

#converting character values to numerical
# for i in range(0,len(rd_data)):
    # if rd_data['room_type'][i]=='Private room':
        # rd_data['room_type'][i]=1
    # elif rd_data['room_type'][i]=='Entire home/apt':
        # rd_data['room_type'][i]=2
    # else:
        # rd_data['room_type'][i]=3      

# for i in range(0,len(rd_data)):
    # if rd_data['neighbourhood_group'][i]=='Brooklyn':
        # rd_data['neighbourhood_group'][i]=1
    # else:
        # rd_data['neighbourhood_group'][i]=2         
        
#droppping columns which have no impact on prediction such as name,id ,host id
# housing_data=rd_data.drop(columns=['id','name','host_id','host_name','last_review','latitude','longitude','reviews_per_month','neighbourhood'])
# housing_data=housing_data.dropna()

#khong co tac dung chi de ve
# import seaborn as sns
# f, ax = plt.subplots(figsize=(10, 8))
# corr = housing_data.corr()
# sns.heatmap(corr, annot=True, annot_kws={"size": 9}, cmap = sns.color_palette("PuOr_r", 50), 
                     # vmin = -1, vmax = 1)


#creating dumy variables for Neighbourhood_group and Neighbourhood
#housing_data=pd.get_dummies(housing_data,columns=['neighbourhood_group'])
#housing_data=pd.get_dummies(housing_data,columns=['neighbourhood'])
#housing_data=pd.get_dummies(housing_data,columns=['room_type'])
#housing_data.head()

# housing_data.dtypes
# housing_data=housing_data.apply(pd.to_numeric)

# from sklearn.model_selection import train_test_split
# y=housing_data['price']
# x=housing_data.drop('price',axis=1)
# X = x.apply(pd.to_numeric, errors='coerce')
# Y = y.apply(pd.to_numeric, errors='coerce')
# xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.3, random_state = 42)

# from sklearn.linear_model import LinearRegression as lm

# regressor=lm().fit(xTrain,yTrain)
# predictions=regressor.predict(xTest)

# from sklearn.metrics import mean_squared_error, r2_score
# print("Mean squared error: %.2f"% mean_squared_error(predictions,yTest))
# print("R-square: %.2f" % r2_score(yTest,predictions))



# #Using Ridge
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV

# ridge=Ridge()
# parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
# ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
# ridge_regressor.fit(xTrain,yTrain)
# predictions_ridge=ridge_regressor.predict(xTest)

# print("Mean squared error: %.2f"% mean_squared_error(predictions_ridge,yTest))
# print("R-square: %.2f" % r2_score(yTest,predictions_ridge))


# #Using Lasso
# from sklearn.linear_model import Lasso
# from sklearn.model_selection import GridSearchCV
# lasso=Lasso()
# parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
# lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
# lasso_regressor.fit(xTrain,yTrain)
# predictions_lasso=ridge_regressor.predict(xTest)

# print("Mean squared error: %.2f"% mean_squared_error(predictions_lasso,yTest))
# print("R-square: %.2f" % r2_score(yTest,predictions_lasso))


import pickle
# Saving model to disk
pickle.dump(reg, open('model.pkl','wb'))

# Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# model.score(xTest,yTest)
# print(model.predict([[2,9,6,4,5,6]]))

