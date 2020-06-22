# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:45:37 2020

@author: vinay
"""

import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statistics
import statsmodels.api as sm

ticker='ZEEL.NS'
end=dt.datetime.today()
start=dt.datetime.today()-dt.timedelta(3650)

ohclv_data=pd.DataFrame()
ohclv_data=yf.download(ticker,start,end)
ohclv_data=ohclv_data.dropna()

# Technical data collection

def MACD(DF,a,b,c):
    df1=DF.copy()
    df1['fma']=df1['Close'].ewm(span=a,min_periods=a).mean()
    df1['sma']=df1['Close'].ewm(span=b,min_periods=b).mean()
    df1['MACD']=df1['fma']-df1['sma']
    df1['signal_line']=df1['MACD'].ewm(span=c,min_periods=c).mean()
    df1=df1.dropna()
    return df1

df1=MACD(ohclv_data,12,26,9)    
df1.iloc[-100:,[-1,-2]].plot()


def ATR(DF,n):
    df=DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-Close']=abs(df['High']-df['Close'].shift(1))
    df['L-Close']=abs(df['Low']-df['Close'].shift(1))
    df['TR']=df[['H-L','H-Close','L-Close']].max(axis=1,skipna=False)
    df['ATR']=df['TR'].rolling(n).mean()
    df2=df.copy()
    return df2

df2=ATR(ohclv_data,14)     
df2.dropna()
df2.iloc[-100:,[-1]].plot()

def BBS(DF,n):
    df=DF.copy()
    df['MA']=DF['Close'].rolling(n).mean()
    df['BB_up']=df['MA']+(2*df['MA'].rolling(n).std())
    df['BB_down']=df['MA']-(2*df['MA'].rolling(n).std())
    df['BB_range']=df['BB_up']-df['BB_down']
    df3=df.copy()
    df3.dropna(inplace=True)
    return df3

df3=BBS(ohclv_data,14)
df3.iloc[-100:,[0,-4,-3,-2]].plot()

def RSI(DF,n):
    "function to calculate RSI"
    df = DF.copy()
    df['delta']=df['Close'] - df['Close'].shift(1)
    df['gain']=np.where(df['delta']>=0,df['delta'],0)
    df['loss']=np.where(df['delta']<0,abs(df['delta']),0)
    avg_gain = []
    avg_loss = []
    gain = df['gain'].tolist()
    loss = df['loss'].tolist()
    for i in range(len(df)):
        if i < n:
            avg_gain.append(np.NaN)
            avg_loss.append(np.NaN)
        elif i == n:
            avg_gain.append(df['gain'].rolling(n).mean().tolist()[n])
            avg_loss.append(df['loss'].rolling(n).mean().tolist()[n])
        elif i > n:
            avg_gain.append(((n-1)*avg_gain[i-1] + gain[i])/n)
            avg_loss.append(((n-1)*avg_loss[i-1] + loss[i])/n)
    df['avg_gain']=np.array(avg_gain)
    df['avg_loss']=np.array(avg_loss)
    df['RS'] = df['avg_gain']/df['avg_loss']
    df['RSI'] = 100 - (100/(1+df['RS']))
    df4=df.copy()
    return df4

df4=RSI(ohclv_data,14)
df4=df4.dropna()
df4.iloc[-100:,[-1]].plot()
    
def ADX(DF,n):
    "function to calculate ADX"
    df2 = DF.copy()
    df2['TR'] = ATR(df2,n)['TR'] 
    df2['DMplus']=np.where((df2['High']-df2['High'].shift(1))>(df2['Low'].shift(1)-df2['Low']),df2['High']-df2['High'].shift(1),0)
    df2['DMplus']=np.where(df2['DMplus']<0,0,df2['DMplus'])
    df2['DMminus']=np.where((df2['Low'].shift(1)-df2['Low'])>(df2['High']-df2['High'].shift(1)),df2['Low'].shift(1)-df2['Low'],0)
    df2['DMminus']=np.where(df2['DMminus']<0,0,df2['DMminus'])
    TRn = []
    DMplusN = []
    DMminusN = []
    TR = df2['TR'].tolist()
    DMplus = df2['DMplus'].tolist()
    DMminus = df2['DMminus'].tolist()
    for i in range(len(df2)):
        if i < n:
            TRn.append(np.NaN)
            DMplusN.append(np.NaN)
            DMminusN.append(np.NaN)
        elif i == n:
            TRn.append(df2['TR'].rolling(n).sum().tolist()[n])
            DMplusN.append(df2['DMplus'].rolling(n).sum().tolist()[n])
            DMminusN.append(df2['DMminus'].rolling(n).sum().tolist()[n])
        elif i > n:
            TRn.append(TRn[i-1] - (TRn[i-1]/n) + TR[i])
            DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/n) + DMplus[i])
            DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/n) + DMminus[i])
    df2['TRn'] = np.array(TRn)
    df2['DMplusN'] = np.array(DMplusN)
    df2['DMminusN'] = np.array(DMminusN)
    df2['DIplusN']=100*(df2['DMplusN']/df2['TRn'])
    df2['DIminusN']=100*(df2['DMminusN']/df2['TRn'])
    df2['DIdiff']=abs(df2['DIplusN']-df2['DIminusN'])
    df2['DIsum']=df2['DIplusN']+df2['DIminusN']
    df2['DX']=100*(df2['DIdiff']/df2['DIsum'])
    ADX = []
    DX = df2['DX'].tolist()
    for j in range(len(df2)):
        if j < 2*n-1:
            ADX.append(np.NaN)
        elif j == 2*n-1:
            ADX.append(df2['DX'][j-n+1:j+1].mean())
        elif j > 2*n-1:
            ADX.append(((n-1)*ADX[j-1] + DX[j])/n)
    df2['ADX']=np.array(ADX)
    df5=df2.copy()
    return df5
   
df5=ADX(ohclv_data,14)
df5=df5.dropna()
df5.iloc[-100:,[-1]].plot()
   
def OBV(DF):
    df6=DF.copy()
    df6['change']=df6['Close'].pct_change()
    df6['direction']=np.where(df6['change']>0,1,-1)
    df6['direction'][0]=0
    df6['vol adj']=df6['Volume']*df6['direction']
    df6['obv']=df6['vol adj'].cumsum()
    return df6

df6=OBV(ohclv_data)
df6.iloc[-100:,[-1]].plot()

def slope(DF,n) :
    df7=DF.copy()
    close_price=df7['Close']
    open_price=df7['Open']
    slope_close=[i*0 for i in range(0,n-1)]
    for i in range(n,len(close_price)+1):
         y=close_price[i-n:i]
         x=np.array(range(0,n))
         x_scaled=(x-x.min())/(x.max()-x.min())
         y_scaled=(y-y.min())/(y.max()-y.min())
         model=sm.OLS(y_scaled,x_scaled)
         result=model.fit()
         slope_close.append(result.params[-1])
    df7['slope_close']=np.rad2deg(np.arctan(np.array(slope_close)))
    slope_open=[i*0 for i in range(0,n-1)]
    for i in range(n,len(open_price)+1):
         y=open_price[i-n:i]
         x=np.array(range(0,n))
         x_scaled=(x-x.min())/(x.max()-x.min())
         y_scaled=(y-y.min())/(y.max()-y.min())
         model=sm.OLS(y_scaled,x_scaled)
         result=model.fit()
         slope_open.append(result.params[-1])
    df7['slope_open']=np.rad2deg(np.arctan(np.array(slope_open)))
    return df7

df7=slope(ohclv_data,10)
(df7.iloc[-100:,[-1,-2]]).plot()
ohclv_data.iloc[-100:,[0,3]].plot()

from stocktrends import Renko

def Renko_trend(DF,n):
    df8=DF.copy()
    df8=df8.reset_index()
    df8=df8.iloc[:,[0,1,2,3,4]]
    df8.columns=['date','open','high','low','close']
    renko_df=Renko(df8)
    x=statistics.mean(((ATR(ohclv_data,200)).iloc[:,-1].dropna()))
    renko_df.brick_size=x
    df8=renko_df.get_ohlc_data()
    return df8

df8=Renko_trend(ohclv_data,200)


import talib 
ohlc_pattern=ohclv_data.copy()
ohlc_pattern['pattern_bullish']=talib.CDL3LINESTRIKE(ohlc_pattern.iloc[:,0],ohlc_pattern.iloc[:,1],ohlc_pattern.iloc[:,2],ohlc_pattern.iloc[:,3])
ohlc_pattern['pattern_bearish']=talib.CDL3BLACKCROWS(ohlc_pattern.iloc[:,0],ohlc_pattern.iloc[:,1],ohlc_pattern.iloc[:,2],ohlc_pattern.iloc[:,3])

def PSAR(DF):
    df9=DF.copy()
    opening_price=df9.iloc[:,0].tolist()
    low=df9.iloc[:,2].tolist()
    high=df9.iloc[:,1].tolist()
    AF=0.02
    psar=np.zeros(len(opening_price),dtype='float')
    count=0
    ep_up=0
    ep_down=0
    for i in range(0,len(opening_price)):
        
        if i==0:
            
            if opening_price[i+10]>opening_price[i]:
                e=[]
                f=[]
                for t in range(0,10):
                    e.append(low[t])
                    f.append(high[t])                     
                psar[10]=min(e)
                ep_up=max(f)
                psar[11]=psar[10]+abs((AF*(ep_up-psar[10])))
                if high[11]>ep_up:
                    count=count+1
                    AF=0.04
                    ep_up=high[11]
            else:
                e=[]
                f=[]
                for t in range(0,10):
                    e.append(high[t])
                    f.append(low[t])
                psar[10]=max(e)
                ep_down=min(f)
                psar[11]=(psar[10]-abs((AF*(ep_down-psar[10]))))
                if low[10]<ep_down:
                    ep_down=low[10]
                    count=count+1
                    AF=0.04
                       
        elif (i>=12 and psar[i-1]<=opening_price[i-1]):
            if (count<=9 and psar[i-2]>opening_price[i-2]) :
                count=0
                AF=0.02
                ep_up=high[i]
            
            psar[i]=psar[i-1]+abs((AF*(ep_up-psar[i-1])))
                
            if(high[i]>ep_up and count<9):
                ep_up=high[i]
                count=count+1
                AF=AF+ (0.02*count)
                
            if(high[i]>ep_up):
                ep_up=high[i]
            
            if count==9:
                AF=0.2
    
                    
        elif (i>=12 and psar[i-1]>opening_price[i-1]):
            if (count<=9 and psar[i-2]<opening_price[i-2] ):
                count=0
                AF=0.02
                ep_down=low[i]
                
            psar[i]=psar[i-1]-abs((AF*(ep_down-psar[i-1])))
                
            if(low[i]<ep_down and count<9):
                ep_down=low[i]
                count=count+1
                AF=AF+ (0.02*count)
                
            if (low[i]<ep_down):

                ep_down=low[i]
            
            if count==9:
                AF=0.2
            
    psar=np.asarray(psar).reshape(len(psar),1)
    df9['PSAR']=psar
    return df9

df9=PSAR(ohclv_data)

    
def CAGR(DF):
    df10=DF.copy()
    df10['change']=df10['Close'].pct_change()
    df10['cum_growth']=(df10['change']+1).cumprod()
    CAGR=(((df10['cum_growth'][-1])**(1/5))-1)*100
    return CAGR

cagr=CAGR(ohclv_data)

def VIX(DF):
    
    df10=DF.copy()
    df10['change']=df10['Close'].pct_change()
    vol=(df10['change'].std())*np.sqrt(252)
    return vol*100

volitility=VIX(ohclv_data)


# Machine learning Predictions
x=pd.DataFrame()
x=ohclv_data.iloc[-694:,[1,2,3,5]].copy()
x['PSAR']=df9.iloc[-694:,-1].values
x['OBV']=df6.iloc[-694:,-1].values
x['ADX']=df5.iloc[-694:,-1].values
x['MACD']=df1.iloc[-694:,-1].values
x['RSI']=df4.iloc[-694:,-1].values
x=x.iloc[0:693,:].values
y=ohclv_data.iloc[-694:,[0]].values
y=y[1:694]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform((x_test))

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


new=ohclv_data.iloc[-1,[1,2,3,5]].tolist()
a=float(df9.iloc[-1,-1])
new.append(a)
a=float(df6.iloc[-1,-1])
new.append(a)
a=float(df5.iloc[-1,-1])
new.append(a)
a=float(df1.iloc[-1,-1])
new.append(a)
a=float(df4.iloc[-1,-1])
new.append(a)
new=np.asarray(new).reshape(1,9)
new=sc_x.transform(new)
z=regressor.predict(new)
print(z)

x=pd.DataFrame()
x=ohclv_data.iloc[-694:,[0,1,2,5]].copy()
x['PSAR']=(np.asarray(df9.iloc[-694:,-1].values).reshape(694,1))
x['OBV']=df6.iloc[-694:,-1].values
x['ADX']=df5.iloc[-694:,-1].values
x['MACD']=df1.iloc[0:694,-1].values
x['RSI']=df4.iloc[-694:,-1].values
x=x.iloc[0:693,:].values
y=ohclv_data.iloc[-694:,[3]].values
y=y[1:694]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform((x_test))

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

new=ohclv_data.iloc[-1,[0,1,2,5]].tolist()
a=float(df9.iloc[-1,-1])
new.append(a)
a=float(df6.iloc[-1,-1])
new.append(a)
a=float(df5.iloc[-1,-1])
new.append(a)
a=float(df1.iloc[-1,-1])
new.append(a)
a=float(df4.iloc[-1,-1])
new.append(a)
new=np.asarray(new).reshape(1,9)
new=sc_x.transform(new)
b=regressor.predict(new)
print(b)

x=pd.DataFrame()
x=ohclv_data.iloc[-694:,[0,2,3,5]].copy()
x['OBV']=df6.iloc[-694:,-1].values
x['PSAR']=df9.iloc[-694:,-1].values
x['ADX']=df5.iloc[-694:,-1].values
x['MACD']=df1.iloc[0:694,-1].values
x['RSI']=df4.iloc[-694:,-1].values
x=x.iloc[0:693,:].values
y=ohclv_data.iloc[-694:,[1]].values
y=y[1:694]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform((x_test))

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

new=ohclv_data.iloc[-1,[0,2,3,5]].tolist()
a=float(df6.iloc[-1,-1])
new.append(a)
a=float(df9.iloc[-1,-1])
new.append(a)
a=float(df5.iloc[-1,-1])
new.append(a)
a=float(df1.iloc[-1,-1])
new.append(a)
a=float(df4.iloc[-1,-1])
new.append(a)
new=np.asarray(new).reshape(1,9)
new=sc_x.transform(new)
c=regressor.predict(new)
print(c)

df_plot=df9.copy()
df_plot=df_plot.reset_index()
plt.scatter(df_plot.iloc[-50:,0].values,df_plot.iloc[-50:,-1].values,color='red')
plt.plot(df_plot.iloc[-50:,0].values,df_plot.iloc[-50:,1].values,color='blue')
plt.show()
    
length=len(df1.iloc[:,0])
x=pd.DataFrame()
x=ohclv_data.iloc[-length:,[0,2,3,5]].copy()
x['OBV']=df6.iloc[-length:,-1].values
x['PSAR']=df9.iloc[-length:,-1].values
x['ADX']=df5.iloc[-length:,-1].values
x['MACD']=df1.iloc[-length:,-1].values
x['RSI']=df4.iloc[-length:,-1].values

y=[]
for i in range(0,length-5):
    if x.iloc[i+5,0]>x.iloc[i,0]:
        y.append(1)
    else:
        y.append(0)

y=np.asarray(y).reshape(length-5,1) 

x=x.iloc[-(length-5):,:].values
x=x[0:len(x)-1,:]
y=y[1:len(y)]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform((x_test))

from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))

# Adding the second hidden layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
       
classifier.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

classifier.fit(x_train,y_train,batch_size=10,epochs=100)

y_pred=classifier.predict(x_test)>0.5

new=ohclv_data.iloc[-1,[0,2,3,5]].tolist()
a=float(df6.iloc[-1,-1])
new.append(a)
a=float(df9.iloc[-1,-1])
new.append(a)
a=float(df5.iloc[-1,-1])
new.append(a)
a=float(df1.iloc[-1,-1])
new.append(a)
a=float(df4.iloc[-1,-1])
new.append(a)
new=np.asarray(new).reshape(1,9)
new=sc_x.transform(new)

print(classifier.predict(new)>0.5)




