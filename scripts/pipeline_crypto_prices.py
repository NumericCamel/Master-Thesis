from datetime import datetime, date
import pandas as pd
import requests
import time

def construct_url(ticker, period_1, period_2, interval='daily'):
    """
    Construct the Yahoo Finance URL for data download.
    
    Parameters:
    ----------
    ticker : str
        Ticker symbol
    period_1 : str
        Start date in 'YYYY-MM-DD' format
    period_2 : str
        End date in 'YYYY-MM-DD' format
    interval : str
        Time interval, one of 'daily', 'weekly', 'monthly'
    """
    def convert_to_seconds(period):
        datetime_value = datetime.strptime(period, '%Y-%m-%d')
        total_seconds = int(time.mktime(datetime_value.timetuple()))
        return total_seconds

    try:
        # Handle special tickers
        ticker_mapping = {'Vix': '%5EVIX', 'SNP': '%5EGSPC', 'Dow': '%5EDJI', 'Gold': 'GC=F', 'Oil':'CL=F'}
        ticker = ticker_mapping.get(ticker, ticker)
        
        interval_dic = {'daily': '1d', 'weekly': '1wk', 'monthly': '1mo'}
        _interval = interval_dic.get(interval)
        if _interval is None:  
            print('Interval code is incorrect')
            return None
        p1 = convert_to_seconds(period_1)
        p2 = convert_to_seconds(period_2)
        url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={p1}&period2={p2}&interval={_interval}&events=history'
        return url
    
    except Exception as e:
        print(f"Error constructing URL: {e}")
        return None

def download_data(url, retries=5, backoff_factor=0.3):
    """
    Download the data with retries.
    
    Parameters:
    ----------
    url : str
        URL to download from
    retries : int
        Number of retries
    backoff_factor : float
        Factor to increase wait time between retries
    
    Returns:
    -------
    pandas.DataFrame or None
        DataFrame containing the downloaded data, or None if download failed
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    for i in range(retries):
        try:
            # Use pandas to read directly from the URL
            return pd.read_csv(url, header=0)  # Changed 'headers' to 'header'
        except requests.exceptions.HTTPError as e:
            if getattr(e.response, 'status_code', None) == 429:  # Too many requests
                wait = backoff_factor * (2 ** i)
                print(f"Too many requests. Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"HTTP error occurred: {e}")
                return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    print("Failed to download data after several retries")
    return None

def fill_missing_dates(df, date_column='Date'):
    """
    Fill in the missing dates in the DataFrame by adding rows for the missing dates
    and using the last available values to fill them in.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing date and price information.
    date_column : str
        The name of the date column in the DataFrame.
    
    Returns:
    -------
    pd.DataFrame
        DataFrame with missing dates filled in.
    """
    # Ensure the date column is of datetime type
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Set the date column as the index
    df.set_index(date_column, inplace=True)
    
    # Create a complete date range from the min to the max date in the DataFrame
    complete_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    
    # Reindex the DataFrame to the complete date range
    df = df.reindex(complete_date_range, method='ffill')
    
    # Reset the index to bring the date column back as a regular column
    df.reset_index(inplace=True)
    
    # Rename the index column back to the original date column name
    df.rename(columns={'index': date_column}, inplace=True)
    
    return df

def get_prices(ticker="BTC-USD", start_date=None, end_date=None, interval="daily", percent_change=False, index=False):
    """
    Main function to get Yahoo Finance data.
    
    Parameters:
    ----------
    ticker : str
        Ticker symbol (default is "BTC-USD")
    start_date : str
        Start date in 'YYYY-MM-DD' format (default is January 1 of current year)
    end_date : str
        End date in 'YYYY-MM-DD' format (default is current date)
    
    Returns:
    -------
    pandas.DataFrame or None
        DataFrame containing the price data, or None if download failed
    """
    if end_date is None:
        end_date = date.today().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date = f"{date.today().year}-01-01"
    
    query_url = construct_url(ticker, start_date, end_date, interval=interval)
                       # Check if the ticker is one of the special tickers and fill missing dates if true

    
    if query_url:
        df = download_data(query_url)
        if df is not None:
            df['Date'] = pd.to_datetime(df['Date'])

            special_tickers = {'%5EVIX', '%5EGSPC', '%5EDJI','Vix', 'SNP', 'Dow', 'Gold', 'GC=F', 'Oil', 'CL=F','NVDA'}
            if ticker in special_tickers:
                df = fill_missing_dates(df) 

            df['Returns'] = df['Close'].pct_change()

            if percent_change:
                df = df.apply(lambda x: x.pct_change() if x.name not in ['Date','Returns'] else x)
            
            if index:
                df = df[['Date', 'Returns', 'Volume']].rename(columns={'Returns': f'{ticker}_Returns', 'Volume': f'{ticker}_Volume'})


            print(f"Data for {ticker} from {start_date} to {end_date} has been downloaded successfully")



        
            return df
    else:
        print("Failed to construct URL.")
    
 
        
    return None

# Example usage
if __name__ == "__main__":
    btc_data = get_prices("BTC-USD")
    if btc_data is not None:
        print(btc_data.tail())
    
    # Uncomment and modify the line below to use custom parameters
    # aapl_data = get_prices("AAPL", start_date="2023-01-01", end_date="2023-12-31")
