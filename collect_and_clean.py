from alpha_vantage.timeseries import TimeSeries

#overloaded function to collect data from alpha vantage API
def collect_stock_data(alpha_vantage_api_key, ticker_name, output_size = "compact"):
    ts = TimeSeries(key = alpha_vantage_api_key, output_format = 'pandas')
    data, meta_data = ts.get_daily_adjusted(ticker_name, outputsize = output_size)
    data['date_time'] = data.index
    return data, meta_data

def stockdata_of_interval(alpha_vantage_api_key,ticker, start_date, end_date):
    ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')
    data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
    df = data.loc[start_date:end_date].sort_index(ascending=True).reset_index()
    return df

#splitting the data
def split_data(df,per):
    # Handle missing values as of now dropped but we can interpolate
    df = df.dropna()
    '''
    threshold = 3 # Adjust this value based on your dataset
    z_scores = (df[closing_column] - df[closing_column].mean()) / df[closing_column].std()
    df = df[(z_scores.abs() < threshold)]
    '''
    train_size = int(len(df) * per)  # 80% for training, 20% for testing
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    return train_data, test_data