import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from display_grahs import *
from collect_and_clean import *
from evaluate import *
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from scipy.stats import probplot
from typing import Union
from itertools import product
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL

alpha_vantage_api_key = "CYU1Y2YO3CDE5ZI7"

ts_data, ts_metadata = collect_stock_data(alpha_vantage_api_key, ticker_name = "TGT") 
ts_data = ts_data.rename(columns=lambda x: x.lstrip('0123456789. '))
plot_data(df = ts_data, 
          x_variable = "date_time", 
          y_variable = "close", 
          title ="High Values, Daily Data")

#ADF Test

forecast=pd.Series(np.diff(ts_data['high'], n=0))
forecast.dropna(inplace=True)
forecast.name='closing values'
train_data,test_data=split_data(forecast,0.8)

ADF_RESULT = adfuller(forecast)
print('ADF Statistic: %f' % ADF_RESULT[0])
print('p-value: %f' % ADF_RESULT[1])

forecast=pd.Series(np.diff(ts_data['high'], n=12))
forecast.dropna(inplace=True)
forecast.name='closing values'
train_data,test_data=split_data(forecast,0.8)

ADF_RESULT = adfuller(forecast)
print('ADF Statistic: %f' % ADF_RESULT[0])
print('p-value: %f' % ADF_RESULT[1])

#Test for seasonality
train_data,test_data=split_data(ts_data['close'],0.8)
decomposition = STL(ts_data['close'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(16,8))
ax1.plot(decomposition.observed)
ax1.set_ylabel('Observed')

ax2.plot(decomposition.trend)
ax2.set_ylabel('Trend')

ax3.plot(decomposition.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(decomposition.resid)
ax4.set_ylabel('Residuals')

plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

#implementing the SARIMA model
def optimize_SARIMA(endog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int) -> pd.DataFrame:
    
    results = []
    
    for order in tqdm_notebook(order_list):
        try: 
            model = SARIMAX(
                endog, 
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False).fit(disp=False)
        except:
            continue
            
        aic = model.aic
        results.append([order, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q,P,Q)', 'AIC']
    
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df
#choosing (p,d,q)(P,D,Q)
ps = range(0, 4, 1)
qs = range(0, 4, 1)
Ps = range(0, 4, 1)
Qs = range(0, 4, 1)

SARIMA_order_list = list(product(ps, qs, Ps, Qs))

d = 0
D = 0
s = 12

SARIMA_result_df = optimize_SARIMA(train_data, SARIMA_order_list, d, D, s)
SARIMA_result_df
'''
ARIMA_model = SARIMAX(train_data, order=(11,2,3), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)

print(ARIMA_model_fit.summary())
ARIMA_model_fit.plot_diagnostics(figsize=(10,8))
plt.show()

residuals = ARIMA_model_fit.resid

lbvalue, pvalue = acorr_ljungbox(residuals, np.arange(1, 11, 1))

print(pvalue)'''
