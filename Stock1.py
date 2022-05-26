# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:57:58 2022

@author: HOANG
"""

import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import investpy
import pandas as pd
import datetime as dt

from sklearn.preprocessing import MinMaxScaler

import numpy as np
from pandas_datareader import data as pdr
import yfinance as yfin
import vnquant.DataLoader as dl

import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import pandas as pd
import matplotlib.dates as mpl_dates

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense


from sklearn.preprocessing import MinMaxScaler

yfin.pdr_override()
# Load data
start = '2010-01-01'
end = dt.datetime.now().strftime("%Y-%m-%d")
#dates= start + ' to ' + end
dfmack=pd.read_csv("C:\\Users\\HOANG\\Desktop\\Python\\Ma CK Viet Nam.csv")
dfmack=pd.DataFrame(dfmack)
#dfmack=dfmack.loc[dfmack['SAN']=='HNX']
listck=dfmack['Ma CK'].values.tolist()

stock_final = pd.DataFrame()

# download the stock price 
stock = []
stock = dl.DataLoader("HPG", start,end, data_source='VND', minimal=True)
data = stock.download()
data.columns = data.columns.droplevel(1)
#data=data.drop(columns=['avg','volume'])
data['Date']=data.index
data = data.reset_index(drop=True)
ohlc = data.loc[:, ['Date', 'open', 'high', 'low', 'close']]

#print(data.head())

plt.style.use('ggplot')
ohlc['Date'] = ohlc['Date'].apply(mpl_dates.date2num)
ohlc =ohlc.astype(float)
fig, ax = plt.subplots()
#print(ohlc.head())
candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

# Setting labels & titles
ax.set_xlabel('Date')
ax.set_ylabel('Price')
fig.suptitle('HPG Plots')

# Formatting Date
date_format = mpl_dates.DateFormatter("%d-%m-%Y")
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()

fig.tight_layout()

plt.show()        

data=data.loc[:, ['Date', 'open', 'high', 'low', 'close','volume']]
#data.to_csv("C:\\Users\\HOANG\\Desktop\\Python\\all\\"f"HPG.csv",index=False)
#stock_final=pd.DataFrame(stock_final) 
#stock_final=stock_final.groupby(['name']).mean()
#stock_final['day']=dates
#stock_final=stock_final.sort_values(by='volume', ascending=False)

#stock_final.to_csv("C:\\Users\\HOANG\\Desktop\\Python\\all\\"f"HPG.csv")
data["Date"]=pd.to_datetime(data.Date,format="%Y-%m-%d")
data.index=data['Date']
data=data.sort_index(ascending=True,axis=0)
print(data)
new_dataset=pd.DataFrame(index=range(0,len(data)),columns=['Date','close'])

print(new_dataset)
print(range(0,len(data)))
for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["close"][i]=data["close"][i]
    
new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)


final_dataset=new_dataset.values

train_data=final_dataset[0:1000,:]
valid_data=final_dataset[1000:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)

x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
    
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))




lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)


X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)
print(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=lstm_model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

lstm_model.save("saved_lstm_model.h5")

train_data=new_dataset[:1000]
valid_data=new_dataset[1000:]
valid_data['Predictions']=closing_price
print(valid_data)
valid_data.to_csv("Predictions HPG.csv")
plt.plot(train_data["close"])
plt.plot(valid_data[['close',"Predictions"]])