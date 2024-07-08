import pandas as pd
import numpy as np

def convert_to_percentage_change(df, blacklist_columns=None):
    # Make a copy of the dataframe to avoid modifying the original
    df_pct = df.copy()
    
    # Ensure blacklist_columns is a list, even if a single column is provided
    if blacklist_columns is None:
        blacklist_columns = []
    elif isinstance(blacklist_columns, str):
        blacklist_columns = [blacklist_columns]
    
    # Get all columns except the blacklisted columns
    columns_to_convert = [col for col in df_pct.columns if col not in blacklist_columns]
    
    # Calculate percentage change for each column
    for col in columns_to_convert:
        df_pct[col] = df_pct[col].pct_change() * 100
    
    # Replace infinity values with NaN
    df_pct = df_pct.replace([np.inf, -np.inf], np.nan)
    
    # Drop the first row as it will contain NaN values after pct_change()
    df_pct = df_pct.dropna(subset=columns_to_convert)
    
    return df_pct