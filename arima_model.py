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

alpha_vantage_api_key = "CYU1Y2YO3CDE5ZI7"

ts_data, ts_metadata = collect_stock_data(alpha_vantage_api_key, ticker_name = "JNJ", output_size = "full") 
ts_data = ts_data.rename(columns=lambda x: x.lstrip('0123456789. '))
# ts_data.set_index('date_time', inplace=True)

#Plot the high prices
plot_data(df = ts_data, 
          x_variable = "date_time", 
          y_variable = "high", 
          title ="High Values, Daily Data")
#differencing--------------------------------------------------------------------------------

forecast=pd.Series(np.diff(ts_data['close'], n=2))
forecast.dropna(inplace=True)
forecast.name='closing values'
train_data,test_data=split_data(forecast,0.8)

ADF_RESULT = adfuller(forecast)
print('ADF Statistic: %f' % ADF_RESULT[0])
print('p-value: %f' % ADF_RESULT[1])

def optimize_ARIMA(endog: Union[pd.Series, list], order_list: list, d: int) -> pd.DataFrame:#finds p d q which have least aic value
    
    results = []
    
    for order in tqdm_notebook(order_list):
        try: 
            model = SARIMAX(endog, order=(order[0], d, order[1]), simple_differencing=False).fit(disp=False)
        except:
            continue
        aic = model.aic
        results.append([order, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'AIC']
    
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df

ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 2

order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train_data, order_list, d)
print(result_df)

model = SARIMAX(train_data, order=(1,1,3), simple_differencing=False)#order is filled from output of SARIMAX
model_fit = model.fit(disp=False)
residuals = model_fit.resid
#qq plots -----------------------------------------------------------------------------------
qqplot(residuals, line='45')

print(model_fit.summary())
model_fit.plot_diagnostics(figsize=(10,8))
plt.show()

predictions = model_fit.get_predict(start=test_data.index[0], end=test_data.index[-1])
residuals = model_fit.resid

lbvalue, pvalue = acorr_ljungbox(residuals, np.arange(1, 11, 1))#checking if residuals till lag 10 are correlated?
print(pvalue)#if p value >0.05 we cannot reject null hypothesis and thy are not correlated hence they correspond to white noise


plt.plot(residuals, label='Residuals')
plt.legend()
mape_arima=mape(test_data,predictions)
mse = np.mean((predictions - test_data) ** 2)
rmse = np.sqrt(mse)

print('MAPE: %.3f' % mape_arima)
print('RMSE: %.3f' % rmse)

plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Testing Data')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()



#---------------------------SARIMA-----------------------------------------------