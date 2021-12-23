import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator, MonthLocator, YearLocator
from matplotlib.ticker import FixedFormatter
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

import datetime

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")


def plot_timeseries(sales, date=pd.Series([])):
 
    fig, ax = plt.subplots(figsize=(12, 6))

    sales_series = pd.Series(sales)
    
    if not date.empty:
        ax=sns.lineplot(x=date, y=sales_series)
        
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(DateFormatter("%b"))
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('\n\n%Y'))

        ax.xaxis.remove_overlapping_locs = False
    else:
        ax=sns.lineplot(x=sales_series.index, y=sales_series)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Weekly Sales")

    plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)
    plt.show()



def plot_timeseries_complex(group, value, df, time):

    fig, ax = plt.subplots(figsize=(12, 6))

    ax=sns.lineplot(x=time, y=value, hue=group, data=df, 
    legend="full", palette=sns.color_palette("Paired", df[group].nunique() ))
    ax.set_xlabel("Time")
    ax.set_ylabel("Weekly Sales")
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_minor_formatter(DateFormatter("%b"))
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('\n\n%Y'))

    ax.xaxis.remove_overlapping_locs = False

    plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)
    plt.show()

def plot_timeseries_analysis(sales):
    df = pd.DataFrame({"sales": sales})
    fig, axes= plt.subplots(3, 1, figsize=(12,8))
    fig.tight_layout(pad=3.0)
    # ax1 = fig.add_subplot(211)
    sns.lineplot(df.index, df["sales"], ax=axes[0])
    # ax2 = fig.add_subplot(212)
    fig = plot_acf(df['sales'].dropna(),lags=40,ax=axes[1])
    # ax3 = fig.add_subplot(213)
    fig = plot_pacf(df['sales'].dropna(),lags=40,ax=axes[2])


# def plot_forecast(train, forecast, test=pd.Series([]), full_series = False):
#     if not full_series:
#         train = pd.Series(train)
#         index_start = train.index[-1] + 1
#         pred_series = pd.Series(forecast, index=np.arange(start=index_start, stop=index_start + len(forecast)))
#     else:
#         pred_series = pd.Series(forecast)

#     plt.figure(figsize=(12,5), dpi=100)
#     plt.plot(train, label='training')
#     if test.any():
#         test_series = pd.Series(test, index=np.arange(start=index_start, stop=index_start + len(test)))
#         plt.plot(test_series, label='actual')
        
    
#     plt.plot(pred_series, label='forecast')
#     # plt.fill_between(lower_series.index, lower_series, upper_series, 
#     #                  color='k', alpha=.15)
#     plt.title('Forecast vs Actuals')
#     plt.legend(loc='upper left', fontsize=8)
#     plt.show()

def plot_forecast(train, pred, test=pd.Series([]), full_series = False, 
time= pd.Series([])):
    if len(time) > 0 :
        df_train = pd.DataFrame({"value": train, "Date": time[:len(train)]})
        df_test = pd.DataFrame({"value": test, "Date": time[-len(test):]})
        df_pred = pd.DataFrame({"value": pred, "Date": time[-len(pred):]})
        
        df_train["Source"] = "train"
        df_test["Source"] = "test"
        df_pred["Source"] = "pred"
        
        temp = pd.concat([df_train, df_test, df_pred], axis = 0).reset_index()

        fig, ax = plt.subplots(figsize=(12, 6))

        ax=sns.lineplot(x="Date", y="value", hue="Source", data=temp, 
        legend="full", palette="Set2")
        ax.set_xlabel("Time")
        ax.set_ylabel("Weekly Sales")
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(DateFormatter("%b"))
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('\n\n%Y'))

        ax.xaxis.remove_overlapping_locs = False

        plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)
        plt.show()

        
    else: 
        if not full_series:
            train = pd.Series(train)
            index_start = train.index[-1] + 1
            pred_series = pd.Series(pred, index=np.arange(start=index_start, stop=index_start + len(pred)))
        else:
            pred_series = pd.Series(pred)

        plt.figure(figsize=(12,5), dpi=100)
        plt.plot(train, label='training')
        if test.any():
            test_series = pd.Series(test, index=np.arange(start=index_start, stop=index_start + len(test)))
            plt.plot(test_series, label='actual')
            
        
        plt.plot(pred_series, label='forecast')
        # plt.fill_between(lower_series.index, lower_series, upper_series, 
        #                  color='k', alpha=.15)
        plt.title('Forecast vs Actuals')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()