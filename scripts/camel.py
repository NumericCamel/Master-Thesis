import pandas as pd
import numpy as np

def data_statistics(df):
    """
    Generate diagnostics for a given DataFrame and return them as a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing various diagnostics.
    """
    # Number of rows and columns
    num_rows = df.shape[0]
    num_columns = df.shape[1]

    # Number of missing values (NaNs) per column
    num_missing = df.isna().sum()

    # Number of infinite values per column (only for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number])
    num_infinite = numeric_cols.apply(lambda x: np.isinf(x).sum())

    # Data types of columns
    data_types = df.dtypes

    # Basic statistical summaries
    mean_values = numeric_cols.mean()
    median_values = numeric_cols.median()
    std_values = numeric_cols.std()
    min_values = numeric_cols.min()
    max_values = numeric_cols.max()

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Data Type': data_types,
        'Number of Missing Values': num_missing,
        'Number of Infinite Values': num_infinite,
        'Mean': mean_values,
        'Median': median_values,
        'Standard Deviation': std_values,
        'Min': min_values,
        'Max': max_values
    })

    # Fill NaN values in summary statistics with N/A for non-numeric columns
    summary_df = summary_df.fillna('N/A')

    # Add overall summary
    overall_summary = pd.DataFrame({
        'Data Type': 'Overall',
        'Number of Missing Values': num_missing.sum(),
        'Number of Infinite Values': num_infinite.sum(),
        'Mean': num_rows,  # Total rows
        'Median': num_columns,  # Total columns
        'Standard Deviation': 'N/A',
        'Min': 'N/A',
        'Max': 'N/A'
    }, index=['Overall'])

    # Combine the overall summary with the detailed summary
    summary_df = pd.concat([summary_df, overall_summary])

    return summary_df

def convert_to_percentage_change(df, blacklist_columns=None):
    # Make a copy of the dataframe to avoid modifying the original
    df_pct = df.copy()
    
    # Ensure blacklist_columns is a list, even if a single column is provided
    if blacklist_columns is None:
        blacklist_columns = []
    elif isinstance(blacklist_columns, str):
        blacklist_columns = [blacklist_columns]
    elif not isinstance(blacklist_columns, list):
        raise ValueError("blacklist_columns should be a list, string, or None")
    
    # Automatically add 'Date' column to the blacklist if it exists
    if 'Date' in df_pct.columns and 'Date' not in blacklist_columns:
        blacklist_columns.append('Date')
    
    # Get all columns except the blacklisted columns
    columns_to_convert = [col for col in df_pct.columns if col not in blacklist_columns]
    
    # Calculate percentage change for each column
    for col in columns_to_convert:
        df_pct[col] = df_pct[col].pct_change()
    
    # Replace infinity values with NaN
    df_pct = df_pct.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN values in the columns_to_convert
    df_pct = df_pct.dropna(subset=columns_to_convert)
    
    return df_pct


#### Data Science Modules
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def pca(data, n_components=2, blacklist=None):
    """
    Perform PCA on the given time series data, excluding specified columns, and plot the explained variance and principal components.

    Parameters:
    data (pd.DataFrame): DataFrame containing time series data with rows as time points and columns as different indexes.
    n_components (int): Number of principal components to calculate. Default is 2.
    blacklist (list): List of columns to exclude from the PCA computation. Default is None.

    Returns:
    pd.DataFrame: DataFrame containing the principal components.
    """
    if blacklist is None:
        blacklist = []

    # Exclude the blacklisted columns
    data_1 = data.copy()
    data_to_pca = data_1.drop(columns=blacklist)

    # Standardizing the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_pca)

    # Applying PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)

    # Creating a DataFrame with the principal components
    pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

    # Plotting the explained variance
    plt.figure(figsize=(4, 3))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Explained')
    plt.title('Explained Variance by Principal Components')
    plt.show()

    # Biplot of the first two principal components
    if n_components >= 2:
        fig, ax = plt.subplots(figsize=(6, 3))

        # Scatter plot of the principal components
        sns.scatterplot(x=pc_df['PC1'], y=pc_df['PC2'], ax=ax)

        # Plotting the loadings (arrows) and feature names
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        for i, feature in enumerate(data_to_pca.columns):
            ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5)
            ax.text(loadings[i, 0], loadings[i, 1], feature, color='g', ha='center', va='center')

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('PCA Biplot')
        plt.grid()
        plt.show()

    return pc_df

from pca import pca
import matplotlib.pyplot as plt

def pca_v2(data, n_components=2, blacklist=None, n_feat=15, show_biplot=True, show_explained_variance=True):
    """
    Perform PCA on the given time series data, excluding specified columns, and plot the explained variance and principal components.

    Parameters:
    data (pd.DataFrame): DataFrame containing time series data with rows as time points and columns as different indexes.
    n_components (int): Number of principal components to calculate. Default is 2.
    blacklist (list): List of columns to exclude from the PCA computation. Default is None.
    n_feat (int): Number of features to show in the biplot. Default is 15.

    Returns:
    pd.DataFrame: DataFrame containing the principal components.
    """
    if blacklist is None:
        blacklist = []

    # Exclude the blacklisted columns
    data_to_pca = data.drop(columns=blacklist)

    # Standardizing the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_pca)

    # Applying PCA
    model = pca(n_components=n_components)
    results = model.fit_transform(scaled_data)

    if show_explained_variance:
     # Plot the explained variance
        fig, ax = model.plot()
        plt.show()

    if show_biplot:
        fix, ax = model.biplot(n_feat=n_feat)
    

    # Creating a DataFrame with the principal components
    pc_df = pd.DataFrame(data=results['PC'], columns=[f'PC{i+1}' for i in range(n_components)], index=data.index)
    
    # Concatenate the blacklisted columns with the principal components
    result_df = pd.concat([data[blacklist], pc_df], axis=1).reset_index(drop=True)
    
    return result_df

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca_v4(data, n_components=4, date_columns=None):
    """
    Perform PCA on the given time series data, excluding date columns, and return the principal components
    along with the original date columns.

    Parameters:
    data (pd.DataFrame): DataFrame containing time series data with rows as time points and columns as different indexes.
    n_components (int): Number of principal components to calculate. Default is 4.
    date_columns (list): List of column names that contain date information. Default is None.

    Returns:
    pd.DataFrame: DataFrame containing the principal components and original date columns.
    """
    if date_columns is None:
        date_columns = []

    # Identify numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove date columns from numeric columns if they were accidentally included
    numeric_columns = [col for col in numeric_columns if col not in date_columns]

    # Separate numeric data for PCA
    data_to_pca = data[numeric_columns]

    # Standardizing the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_pca)

    # Applying PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    # Creating a DataFrame with the principal components
    pc_df = pd.DataFrame(
        data=pca_result,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data.index
    )
    
    # Concatenate the date columns with the principal components
    result_df = pd.concat([data[date_columns], pc_df], axis=1)
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    total_variance = np.sum(pca.explained_variance_ratio_) * 100
    plt.title(f'Explained Variance Ratio by Principal Component\nTotal Variance Explained: {total_variance:.2f}%')
    plt.show()

    return result_df


### Creating Technical Analysis
def technical_analysis(df, price_col='Price', volume_col='Volume', high_col='High', low_col='Low', date_col='Date', only_td=False, hold_strat=False):
    # Create a copy of the input DataFrame
    ta = df.copy()
    ta = ta.sort_values(by=date_col)
    ta['return'] = ta[price_col].pct_change()
    
    # 10D MA
    ta['MA'] = ta[price_col].rolling(window=10).mean()
    ta['MA_td'] = (ta[price_col] > ta.MA).astype(int)
    
    # 30D MA
    ta['3MA'] = ta[price_col].rolling(window=30).mean()
    ta['3MA_td'] = (ta[price_col] > ta['3MA']).astype(int)
    
    # %K
    lowest_low = ta[low_col].rolling(window=10).min()
    highest_high = ta[high_col].rolling(window=10).max()
    ta['%K'] = (ta[price_col] - lowest_low) / (highest_high - lowest_low) * 100
    ta['%K_td'] = (ta['%K'] > ta['%K'].shift(1)).astype(int)
    
    # Calculate %D
    ta['%D'] = ta['%K'].rolling(window=3).mean()
    ta['%D_td'] = (ta['%D'] > ta['%D'].shift(1)).astype(int)
    
    # RSI 
    delta = ta[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    ta['RSI'] = 100 - (100 / (1 + rs))
    
    def RSI_td(rsi_values):
        if hold_strat:
            if rsi_values >= 70:
                return -1
            elif rsi_values <= 30:
                return 1
            else:
                return 0
        else:
            return 1 if rsi_values <= 30 else 0
    
    ta['RSI_td'] = ta.RSI.apply(RSI_td)
    
    # Momentum 
    momentum_window = 10 
    ta['Momentum'] = ta[price_col] - ta[price_col].shift(momentum_window)
    ta['Momentum_td'] = (ta.Momentum > 1).astype(int)
    
    # MACD 12,26,9
    ta['EMA12'] = ta[price_col].ewm(span=12, adjust=False).mean()
    ta['EMA26'] = ta[price_col].ewm(span=26, adjust=False).mean()
    ta['MACD'] = ta['EMA12'] - ta['EMA26']
    ta['Signal'] = ta['MACD'].ewm(span=9, adjust=False).mean()
    ta['MACD_td'] = (ta['MACD'] > ta['MACD'].shift(1)).astype(int)
    
    # CCI
    ta['TP'] = (ta[high_col] + ta[low_col] + ta[price_col]) / 3
    ta['SMA_TP'] = ta['TP'].rolling(window=20).mean()
    
    def calculate_md(series):
        return abs(series - series.mean()).mean()
    
    ta['MD'] = ta['TP'].rolling(window=20).apply(calculate_md)
    ta['CCI'] = (ta['TP'] - ta['SMA_TP']) / (0.015 * ta['MD'])
    
    def CCI_td(CCI_values):
        if hold_strat:
            if CCI_values >= 100:
                return -1
            elif CCI_values <= -100:
                return 1
            else:
                return 0
        else:
            return 1 if CCI_values <= -100 else 0
    
    ta['CCI_td'] = ta.CCI.apply(CCI_td)
    
    # Bollinger Bands
    ta['20D_SMA'] = ta[price_col].rolling(window=20).mean()
    ta['20D_STD'] = ta[price_col].rolling(window=20).std()
    ta['Upper_BB'] = ta['20D_SMA'] + (ta['20D_STD'] * 2)
    ta['Lower_BB'] = ta['20D_SMA'] - (ta['20D_STD'] * 2)
    
    def BB_td(row):
        if hold_strat:
            if row[price_col] > row['Upper_BB']:
                return -1
            elif row[price_col] < row['Lower_BB']:
                return 1
            else:
                return 0
        else:
            return 1 if row[price_col] < row['Lower_BB'] else 0
    
    ta['BB_td'] = ta.apply(BB_td, axis=1)
    
    # Average True Range (ATR)
    ta['TR'] = ta[[high_col, low_col]].max(axis=1) - ta[[low_col, high_col]].min(axis=1)
    ta['ATR'] = ta['TR'].rolling(window=14).mean()
    
    def ATR_td(row):
        if row['TR'] > row['ATR']:
            return 1
        else:
            return 0
    
    ta['ATR_td'] = ta.apply(ATR_td, axis=1)
    
    if only_td:
        td_columns = [col for col in ta.columns if col.endswith('_td') or col in [date_col, price_col, volume_col, high_col, low_col]]
        ta = ta[td_columns]
    
    return ta

def thesis_dv(df):
    df['Tomorrow'] = df.Close.shift(-1)
    df['Difference'] = df['Tomorrow'] - df['Close']
    df['in between'] = np.where((df['Tomorrow'] > df['Low']) & (df['Tomorrow'] < df['High']), 1, 0)
    df['Indicator'] = np.where(
    df['in between'] == 1, 
    'Hold', 
    np.where(df['Difference'] > 0, 'Buy', 'Sell')
    )
    df = df.drop(['in between', 'Tomorrow'], axis=1)
    return df

