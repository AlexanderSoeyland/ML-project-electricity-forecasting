import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# Plot settings
import seaborn as sns



# ============================================================
#  1. Final data-quality assessment
# ============================================================
# Confirming: shape, time range, hourly frequency, 
# duplicates, missing values/counts/percentages,
#  consecutive missing gaps, and basic descriptive statistics.




def data_quality_assessment(df): 

    # ============================================================
    # CHECK RESULT
    # ============================================================
    timestamps = pd.to_datetime(df["timestamp"], utc=True)

    print('Information about the dataset: ')
    print(df.info())

    print('Time range: ')
    print(timestamps.min(), timestamps.max())

    print("\nDataset shape:")
    print(df.shape)

    print("\nTime intervals/hourly frequency:")
    print(
        timestamps
        .diff()
        .value_counts()
        .head()
    )

    print("\nMissing values:")
    print(
        df.isna().sum()
    )

    print("\nConsecutive missing gaps:")
    missing_gap_lengths = {}
    for column in df.columns:
        missing = df[column].isna()
        groups = missing.ne(missing.shift()).cumsum()
        gap_lengths = missing.groupby(groups).sum()
        missing_gap_lengths[column] = int(gap_lengths.max()) if missing.any() else 0
    print(pd.Series(missing_gap_lengths).sort_values(ascending=False).head())

    print("\nBasic descriptive statistics:")
    print(
        df.describe()
    )



# ============================================================
# 2. Target-variable distribution
# ===========================================================
# Analyze day_ahead_price with summary statistics 
# (mean, median, SD, min/max, quartiles, skewness, kurtosis), 
# histogram/KDE, and boxplot. 
# Electricity prices tend to be highly non-normal with spikes,
#  so compare the median and mean and identify extreme-price observations. 
# It would also be useful to report proportions of negative prices and unusually high prices.

def target_variable_distribution(df): 

    df = df.copy()
    # Summary statistics
    print("Summary Statistics for day_ahead_price:")
    print(df['day_ahead_price'].describe())
    print("\nSkewness:", df['day_ahead_price'].skew())
    print("Kurtosis:", df['day_ahead_price'].kurtosis())

    # Histogram and KDE
    plt.figure(figsize=(12, 6))
    sns.histplot(df['day_ahead_price'], bins=50, kde=True)
    plt.title('Histogram and KDE of Day Ahead Price')
    plt.xlabel('Day Ahead Price')
    plt.ylabel('Frequency')
    plt.show()

    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['day_ahead_price'])
    plt.title('Boxplot of Day Ahead Price')
    plt.xlabel('Day Ahead Price')
    plt.show()


    # Proportions of negative and unusually high prices
    negative_proportion = (df['day_ahead_price'] < 0).sum() / len(df)
    high_price_threshold = df['day_ahead_price'].quantile(0.95)
    high_proportion = (df['day_ahead_price'] > high_price_threshold).sum() / len(df)
    print(f"Proportion of negative prices: {negative_proportion*100:.2f}%")
    print(f"Proportion of unusually high prices (above 95th percentile): {high_proportion*100:.2f}%")




# ============================================================
# 3. Price through time
# ===========================================================
# Plot the complete 2019–2025 hourly price series, 
# then monthly or weekly averages to expose longer-term trends. 
# Also plot yearly distributions/boxplots. This will show whether the statistical regime changes 
# substantially across years, which matters for chronological train/test splitting


def plot_price_through_time(df):

    df = df.copy()
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Plot complete hourly price series
    plt.figure(figsize=(15, 6))
    plt.plot(df['timestamp'], df['day_ahead_price'], label='Hourly Price')
    plt.title('Hourly Day Ahead Price (2019-2025)')
    plt.xlabel('Time')
    plt.ylabel('Day Ahead Price')
    plt.legend()
    plt.show()

    # Monthly averages
    df.set_index('timestamp', inplace=True)
    monthly_avg = df['day_ahead_price'].resample('M').mean()
    
    plt.figure(figsize=(15, 6))
    monthly_avg.plot(label='Monthly Average Price', color='orange')
    plt.title('Monthly Average Day Ahead Price (2019-2025)')
    plt.xlabel('Time')
    plt.ylabel('Average Day Ahead Price')
    plt.legend()
    plt.show()

    # Yearly boxplots
    df.reset_index(inplace=True)
    df['year'] = df['timestamp'].dt.year
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='year', y='day_ahead_price', data=df)
    plt.title('Yearly Distribution of Day Ahead Price (2019-2025)')
    plt.xlabel('Year')
    plt.ylabel('Day Ahead Price')
    plt.show()

    # Printig the yearly descriptive statistics
    yearly_stats = df.groupby('year')['day_ahead_price'].describe()
    print("Yearly Descriptive Statistics for Day Ahead Price:")
    print(yearly_stats)




# ============================================================
# 4. Seasonality and calendar patterns
# ===========================================================
# Calculate average/median price by hour of day, day of week, month, season, weekday vs weekend, 
# and year. Particularly useful plots are the 24-hour average price profile and a 
# weekday × hour heatmap. 
# You can do the same for load and temperature where relevant.

def seasonality_and_calendar_patterns(df):
    df = df.copy()

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)

    # Average price by hour of day
    hourly_avg = df['day_ahead_price'].groupby(df.index.hour).mean()
    
    plt.figure(figsize=(12, 6))
    hourly_avg.plot(kind='bar', color='skyblue')
    plt.title('Average Day Ahead Price by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Day Ahead Price')
    plt.xticks(rotation=0)
    plt.show()

    # Average price by day of week
    daily_avg = df['day_ahead_price'].groupby(df.index.dayofweek).mean()
    
    plt.figure(figsize=(12, 6))
    daily_avg.plot(kind='bar', color='lightgreen')
    plt.title('Average Day Ahead Price by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Average Day Ahead Price')
    plt.xticks(rotation=0)
    plt.show()

    # Average price by month
    monthly_avg = df['day_ahead_price'].groupby(df.index.month).mean()
    
    plt.figure(figsize=(12, 6))
    monthly_avg.plot(kind='bar', color='salmon')
    plt.title('Average Day Ahead Price by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Day Ahead Price')
    plt.xticks(rotation=0)
    plt.show()

    # Average price by season
    seasons = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    season = pd.Series(df.index.month, index=df.index).map(seasons)
    season_avg = (
        df['day_ahead_price']
        .groupby(season)
        .mean()
        .reindex(season_order)
    )
    
    plt.figure(figsize=(12, 6))
    season_avg.plot(kind='bar', color='lightcoral')
    plt.title('Average Day Ahead Price by Season')
    plt.xlabel('Season')
    plt.ylabel('Average Day Ahead Price')
    plt.xticks(rotation=0)
    plt.show()




# ============================================================
# 5. Temperature analysis 
# ===========================================================

#Plot mean_temperature_NO1 through time and its distribution. 
# Examine temperature by month/season and validate that the expected seasonal cycle exists.
#  Check the individual stations against one another with correlations and perhaps 
# plot a representative winter/summer period. 
# This validates your constructed NO1 temperature feature.


def temperature_analysis(df):
    df = df.copy()

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df.pop('timestamp'), utc=True)
        df.index = timestamps
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        raise KeyError("DataFrame must contain a 'timestamp' column or have a DatetimeIndex")

    # Plot mean_temperature_NO1 through time
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['mean_temperature_NO1'], label='Mean Temperature NO1', color='blue')
    plt.title('Mean Temperature NO1 Over Time')
    plt.xlabel('Time')
    plt.ylabel('Mean Temperature NO1 (°C)')
    plt.legend()
    plt.show()

    # Distribution of mean_temperature_NO1
    plt.figure(figsize=(12, 6))
    sns.histplot(df['mean_temperature_NO1'], bins=50, kde=True, color='blue')
    plt.title('Distribution of Mean Temperature NO1')
    plt.xlabel('Mean Temperature NO1 (°C)')
    plt.ylabel('Frequency')
    plt.show()

    # Average temperature by season
    seasons = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    season = pd.Series(df.index.month, index=df.index).map(seasons)
    season_avg_temp = (
        df['mean_temperature_NO1']
        .groupby(season)
        .mean()
        .reindex(season_order)
    )

    # Plotting average temperature by season with value labels on top of each bar
    plt.figure(figsize=(12, 6))
    season_avg_temp.plot(kind='bar', color='lightblue')
    for i, (season, temp) in enumerate(season_avg_temp.items()):
        plt.text(i, temp + 0.5, f'{temp:.1f}°C', ha='center', va='bottom')
    plt.title('Average Mean Temperature NO1 by Season')
    plt.xlabel('Season')
    plt.ylabel('Average Mean Temperature NO1 (°C)')
    plt.xticks(rotation=0)
    plt.show()



#============================================================
# 6. Price-temperature relationship and load-temperature relationship
# ===========================================================

# This is especially important for your project. 
# Make a scatterplot of and consider binning temperature into intervals and calculating
# median/mean price within each bin. Don't assume this relationship is linear. 
# You may find something closer to a nonlinear relationship where very cold conditions 
# correspond to higher prices.


def price_temperature_relationship(df):
    df = df.copy()

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df.pop('timestamp'), utc=True)
        df.index = timestamps
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        raise KeyError("DataFrame must contain a 'timestamp' column or have a DatetimeIndex")

    # Scatterplot of day_ahead_price vs mean_temperature_NO1
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='mean_temperature_NO1', y='day_ahead_price', data=df, alpha=0.5)
    plt.title('Day Ahead Price vs Mean Temperature NO1')
    plt.xlabel('Mean Temperature NO1 (°C)')
    plt.ylabel('Day Ahead Price')
    plt.show()

    # Binning temperature into intervals and calculating median price within each bin
    temp_bins = np.arange(df['mean_temperature_NO1'].min(), df['mean_temperature_NO1'].max() + 5, 5)
    df['temp_bin'] = pd.cut(df['mean_temperature_NO1'], bins=temp_bins)
    median_prices = df.groupby('temp_bin')['day_ahead_price'].median()

    # Plotting median price by temperature bins
    plt.figure(figsize=(12, 6))
    median_prices.plot(kind='bar', color='purple')
    plt.title('Median Day Ahead Price by Mean Temperature NO1 Bins')
    plt.xlabel('Mean Temperature NO1 Bins (°C)')
    plt.ylabel('Median Day Ahead Price')
    plt.xticks(rotation=45)
    plt.show()

    correlation = df['day_ahead_price'].corr(df['mean_temperature_NO1'])
    print(f'Correlation between day_ahead_price and mean_temperature_NO1: {correlation:.4f}')


def load_temperature_relationship(df):
    df = df.copy()

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df.pop('timestamp'), utc=True)
        df.index = timestamps
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        raise KeyError("DataFrame must contain a 'timestamp' column or have a DatetimeIndex")

    # Scatterplot of actual_load vs mean_temperature_NO1
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='mean_temperature_NO1', y='actual_load', data=df, alpha=0.5)
    plt.title('Actual Load vs Mean Temperature NO1')
    plt.xlabel('Mean Temperature NO1 (°C)')
    plt.ylabel('Actual Load')
    plt.show()

    # Binning temperature into intervals and calculating median load within each bin
    temp_bins = np.arange(df['mean_temperature_NO1'].min(), df['mean_temperature_NO1'].max() + 5, 5)
    df['temp_bin'] = pd.cut(df['mean_temperature_NO1'], bins=temp_bins)
    median_loads = df.groupby('temp_bin')['actual_load'].median()

    # Plotting median load by temperature bins
    plt.figure(figsize=(12, 6))
    median_loads.plot(kind='bar', color='orange')
    plt.title('Median Actual Load by Mean Temperature NO1 Bins')
    plt.xlabel('Mean Temperature NO1 Bins (°C)')
    plt.ylabel('Median Actual Load')
    plt.xticks(rotation=45)
    plt.show()

    correlation = df['actual_load'].corr(df['mean_temperature_NO1'])
    print(f'Correlation between actual_load and mean_temperature_NO1: {correlation:.4f}')

## ============================================================
# 7. Price relationships with energy variables 
# ===========================================================

# Examine price against load_forecast, actual_load, wind_forecast, hydro_run_of_river, 
# and hydro_reservoir. Use scatterplots, correlations and possibly binned relationships. 


def price_energy_relationships(df):
    df = df.copy()

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df.pop('timestamp'), utc=True)
        df.index = timestamps
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        raise KeyError("DataFrame must contain a 'timestamp' column or have a DatetimeIndex")

    energy_vars = ['load_forecast', 'actual_load', 'wind_forecast', 'hydro_run_of_river', 'hydro_reservoir']

    for var in energy_vars:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=var, y='day_ahead_price', data=df, alpha=0.5)
        plt.title(f'Day Ahead Price vs {var.replace("_", " ").title()}')
        plt.xlabel(var.replace("_", " ").title())
        plt.ylabel('Day Ahead Price')
        plt.show()

        # Calculate and print correlation
        correlation = df['day_ahead_price'].corr(df[var])
        print(f'Correlation between day_ahead_price and {var}: {correlation:.4f}')



#=============================================================
# 8. Correlation matrix and heatmap 
#==========================================================
# Create Pearson and preferably also Spearman correlation matrices for the main numerical variables
# Spearman is valuable because electricity-market relationships can be monotonic
#  without being linear. Include price, load, wind, hydro, temperature and the principal lag 
# variables. Avoid interpreting correlation as causation.

def correlation_matrix_heatmap(df):
    df = df.copy()

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df.pop('timestamp'), utc=True)
        df.index = timestamps
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        raise KeyError("DataFrame must contain a 'timestamp' column or have a DatetimeIndex")

    # Select relevant numerical columns for correlation analysis
    numerical_cols = ['day_ahead_price', 'load_forecast', 'actual_load', 
                      'wind_forecast', 'hydro_run_of_river', 'hydro_reservoir', 
                      'mean_temperature_NO1']

    # Calculate Pearson correlation matrix
    pearson_corr = df[numerical_cols].corr(method='pearson')

    # Plot Pearson correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Pearson Correlation Matrix Heatmap')
    plt.show()

    # Calculate Spearman correlation matrix
    spearman_corr = df[numerical_cols].corr(method='spearman')

    # Plot Spearman correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Spearman Correlation Matrix Heatmap')
    plt.show()




## ============================================================
# 9. Forecast-vs-actual load analysis
# ===========================================================

# Because you have both load_forecast and actual_load, examine e = L_actual − L_forecast
# Calculate MAE/RMSE between them, their correlation, and plot actual vs forecast.
#  This both validates the ENTSO-E data and tells you how accurate the information available to the day-ahead market actually was.

def forecast_vs_actual_load_analysis(df):
    df = df.copy()

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df.pop('timestamp'), utc=True)
        df.index = timestamps
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        raise KeyError("DataFrame must contain a 'timestamp' column or have a DatetimeIndex")

    # Calculate error
    df['load_error'] = df['actual_load'] - df['load_forecast']

    # Calculate MAE and RMSE
    mae = np.mean(np.abs(df['load_error']))
    rmse = np.sqrt(np.mean(df['load_error']**2))
    std_dev = np.std(df['load_error'])
    correlation = df['actual_load'].corr(df['load_forecast'])

    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(f'Standard Deviation of Load Error: {std_dev:.2f}')
    print(f'Correlation between actual load and forecast: {correlation:.4f}')

    # Plot actual vs forecast load
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['actual_load'], label='Actual Load', color='blue')
    plt.plot(df.index, df['load_forecast'], label='Forecast Load', color='orange', alpha=0.7)
    plt.title('Actual vs Forecast Load')
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.legend()
    plt.show()



#============================================================
# 10. Mutual information and feature importance
#==========================================================


from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.multiclass import type_of_target

def mutual_information_table(data, target, continuous=None, discrete=None, categorical=None, binary=None):
    """
    Compute mutual information between features and target variable.

    Parameters:
    - data: DataFrame with all features and target
    - target: Target variable (Series or array)
    - continuous: List of continuous feature names (auto-detected if None)
    - discrete: List of discrete feature names
    - categorical: List of categorical feature names
    - binary: List of binary feature names

    Returns:
    - DataFrame: MI scores for all features, sorted by importance
    """
    # Convert pandas Index to list if necessary
    continuous = list(continuous) if continuous is not None else []
    discrete = list(discrete) if discrete is not None else []
    categorical = list(categorical) if categorical is not None else []
    binary = list(binary) if binary is not None else []

    # Auto-detect feature types if none specified
    if len(continuous) == 0 and len(discrete) == 0 and len(categorical) == 0 and len(binary) == 0:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
        continuous = numeric_cols
        categorical = non_numeric_cols

    feature_names = continuous + discrete + categorical + binary
    if not feature_names:
        raise ValueError("No feature columns were provided.")

    features = data[feature_names].copy()
    if isinstance(target, pd.Series):
        target_series = target.reindex(data.index)
    else:
        target_array = np.asarray(target)
        if len(target_array) != len(data):
            raise ValueError("target must have the same number of rows as data.")
        target_series = pd.Series(target_array, index=data.index)

    valid_rows = target_series.notna() & features.notna().all(axis=1)
    features = features.loc[valid_rows]
    target_series = target_series.loc[valid_rows]
    if features.empty:
        raise ValueError(
            "No rows remain after aligning data and target and removing missing values. "
            "Pass the cleaned feature DataFrame and target with matching indexes, for example "
            "`mutual_information_table(data, target=y)`."
        )

    target_type = type_of_target(target_series.to_numpy())
    mutual_info = (
        mutual_info_regression
        if target_type == 'continuous'
        else mutual_info_classif
    )

    frames = []

    if len(continuous) > 0:
        mi = mutual_info(
            features[continuous],
            target_series,
            random_state=1
        )
        frames.append(pd.DataFrame(mi, index=continuous, columns=['MI']))

    if len(discrete + categorical + binary) > 0:
        encoded_features = OrdinalEncoder().fit_transform(
            features[discrete + categorical + binary]
        )
        mi = mutual_info(
            encoded_features,
            target_series,
            discrete_features=True,
            random_state=1
        )
        frames.append(pd.DataFrame(mi, index=discrete + categorical + binary, columns=['MI']))
    
    mi_results = pd.concat(frames).sort_values('MI', ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=mi_results['MI'], y=mi_results.index, palette='viridis')
    plt.title('Mutual Information Scores for Features')
    plt.xlabel('Mutual Information Score')
    plt.ylabel('Feature')
    plt.show()

    return mi_results



# ============================================================
# 11. Extreme price analysis
# ============================================================
# Define extreme observations using a statistical threshold such as the upper/lower 1%
# rather than an arbitrary price. Compare load, wind, hydro, temperature, hour and month 
# during extreme versus ordinary prices. Electricity price spikes are economically important and
#  RMSE can be heavily influenced by them.

def extreme_price_analysis(df):
    df = df.copy()

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df.pop('timestamp'), utc=True)
        df.index = timestamps
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        raise KeyError("DataFrame must contain a 'timestamp' column or have a DatetimeIndex")

    # Define extreme price thresholds (upper and lower 1%)
    lower_threshold = df['day_ahead_price'].quantile(0.01)
    upper_threshold = df['day_ahead_price'].quantile(0.99)

    # Identify extreme price observations
    extreme_prices = df[(df['day_ahead_price'] < lower_threshold) | (df['day_ahead_price'] > upper_threshold)]
    ordinary_prices = df[(df['day_ahead_price'] >= lower_threshold) & (df['day_ahead_price'] <= upper_threshold)]

    print(f"Number of extreme price observations: {len(extreme_prices)}")
    print(f"Number of ordinary price observations: {len(ordinary_prices)}")

    # Compare load, wind, hydro, temperature, hour and month during extreme vs ordinary prices
    comparison_vars = ['actual_load', 'load_forecast', 'wind_forecast', 'hydro_run_of_river', 
                       'hydro_reservoir', 'mean_temperature_NO1']

    for var in comparison_vars:
        plt.figure(figsize=(12, 6))
        sns.kdeplot(extreme_prices[var], label='Extreme Prices', color='red', fill=True, alpha=0.5)
        sns.kdeplot(ordinary_prices[var], label='Ordinary Prices', color='blue', fill=True, alpha=0.5)
        # labeling the mean extreme and ordinary prices on the plot
        plt.axvline(extreme_prices[var].mean(), color='red', linestyle='--', label=f'Extreme Mean: {extreme_prices[var].mean():.2f}')
        plt.axvline(ordinary_prices[var].mean(), color='blue', linestyle='--', label=f'Ordinary Mean: {ordinary_prices[var].mean():.2f}')
        plt.title(f'Distribution of {var.replace("_", " ").title()} During Extreme vs Ordinary Prices')
        plt.xlabel(var.replace("_", " ").title())
        plt.ylabel('Density')
        plt.legend()
        plt.show()




#============================================================
# 12. Annual changes
#=========================================================
# Compare 2019, 2020, ..., 2025 separately. 
# Calculate annual price mean/median/SD and predictor distributions. 
# This is important because a model trained chronologically assumes historical 
# relationships have some relevance to later periods. If one year has dramatically 
# different price behavior, discuss the resulting distribution shift.


def annual_changes(df):
    df = df.copy()

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df.pop('timestamp'), utc=True)
        df.index = timestamps
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        raise KeyError("DataFrame must contain a 'timestamp' column or have a DatetimeIndex")

    # Extract year from timestamp
    df['year'] = df.index.year

    # Calculate annual statistics for day_ahead_price
    annual_stats = df.groupby('year')['day_ahead_price'].agg(['mean', 'median', 'std', 'min', 'max'])
    print("Annual Statistics for Day Ahead Price:")
    print(annual_stats)

    # Plot annual mean and median prices
    plt.figure(figsize=(12, 6))
    annual_stats[['mean', 'median']].plot(kind='bar')
    plt.title('Annual Mean and Median Day Ahead Prices (2019-2025)')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.xticks(rotation=0)
    plt.legend(['Mean Price', 'Median Price'])
    plt.show()




#============================================================
# 13. Outlier investigation 
#=========================================================
# Don't automatically remove price outliers. 
# Inspect them against load, wind, temperature, hydro and date. 
# A €500/MWh price may be an important genuine market event rather than bad data. 
# Consider the context and potential impact of each outlier before making any decisions.
# If the outlier is due to a specific event, it might be valuable information to include in the model.
# Temperature and load outliers should similarly be checked before removal.



def outlier_investigation(df):
    df = df.copy()

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df.pop('timestamp'), utc=True)
        df.index = timestamps
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        raise KeyError("DataFrame must contain a 'timestamp' column or have a DatetimeIndex")

    # Define outlier thresholds for day_ahead_price (e.g., 1st and 99th percentiles)
    lower_threshold = df['day_ahead_price'].quantile(0.01)
    upper_threshold = df['day_ahead_price'].quantile(0.99)

    # Identify outliers
    outliers = df[(df['day_ahead_price'] < lower_threshold) | (df['day_ahead_price'] > upper_threshold)]
    print(f"Number of price outliers: {len(outliers)}")

    # Inspect outliers against load, wind, temperature, hydro and date
    comparison_vars = ['actual_load', 'load_forecast', 'wind_forecast', 'hydro_run_of_river', 
                       'hydro_reservoir', 'mean_temperature_NO1']

    for var in comparison_vars:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=outliers.index, y=outliers[var], label='Outliers', color='red')
        sns.scatterplot(x=df.index, y=df[var], label='All Data', color='blue', alpha=0.3)
        plt.title(f'{var.replace("_", " ").title()} During Price Outliers vs All Data')
        plt.xlabel('Date')
        plt.ylabel(var.replace("_", " ").title())
        plt.legend()
        plt.show()




#============================================================
# 14. Checking training/validation/test split
#=========================================================
#After examining yearly distributions, 
# compare the target and predictor distributions across your intended chronological split
# Don't randomly shuffle the observations. The EDA should help establish 
# whether 2025 represents a substantially different regime from the training data.




def check_train_val_test_split(df, train_end, val_end):
    df = df.copy()

    # example train_end = '2023-12-31'
    # example val_end = '2024-12-31'

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df.pop('timestamp'), utc=True)
        df.index = timestamps
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        raise KeyError("DataFrame must contain a 'timestamp' column or have a DatetimeIndex")

    # Split the data into training, validation, and test sets based on provided dates
    train_df = df[df.index <= train_end]
    val_df = df[(df.index > train_end) & (df.index <= val_end)]
    test_df = df[df.index > val_end]

    print(f"Training set: {train_df.shape[0]} observations")
    print(f"Validation set: {val_df.shape[0]} observations")
    print(f"Test set: {test_df.shape[0]} observations")

    # Compare distributions of day_ahead_price across the splits
    plt.figure(figsize=(12, 6))
    sns.kdeplot(train_df['day_ahead_price'], label='Training Set', color='blue', fill=True, alpha=0.5)
    # Training set mean and median lines
    plt.axvline(train_df['day_ahead_price'].mean(), color='blue', linestyle='--', label=f'Train Mean: {train_df["day_ahead_price"].mean():.2f}')
    plt.axvline(train_df['day_ahead_price'].median(), color='blue', linestyle=':', label=f'Train Median: {train_df["day_ahead_price"].median():.2f}')
    # Validation set mean and median lines
    sns.kdeplot(val_df['day_ahead_price'], label='Validation Set', color='orange', fill=True, alpha=0.5)
    plt.axvline(val_df['day_ahead_price'].mean(), color='orange', linestyle='--', label=f'Val Mean: {val_df["day_ahead_price"].mean():.2f}')
    plt.axvline(val_df['day_ahead_price'].median(), color='orange', linestyle=':', label=f'Val Median: {val_df["day_ahead_price"].median():.2f}')
    # Test set mean and median lines
    sns.kdeplot(test_df['day_ahead_price'], label='Test Set', color='green', fill=True, alpha=0.5)
    plt.axvline(test_df['day_ahead_price'].mean(), color='green', linestyle='--', label=f'Test Mean: {test_df["day_ahead_price"].mean():.2f}')
    plt.axvline(test_df['day_ahead_price'].median(), color='green', linestyle=':', label=f'Test Median: {test_df["day_ahead_price"].median():.2f}')
    
    plt.title('Distribution of Day Ahead Price Across Train/Val/Test Splits')
    plt.xlabel('Day Ahead Price')
    plt.ylabel('Density')
    plt.legend()
    plt.show()










