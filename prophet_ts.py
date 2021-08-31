

import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from fbprophet import Prophet
from fbprophet.plot import plot_forecast_component, plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics


f = "bike-sharing-dataset/day.csv"
df = pd.read_csv(f)

print(f"Number of null count rows: {int(df.cnt.isnull().any())}")

df['dteday'] = pd.to_datetime(df['dteday'])
df_dt = df.set_index("dteday")
df_dt['day_of_month'] = df_dt.index.day
df_dt['day_of_week'] = df_dt.index.dayofweek
df_dt['weekofyear'] = df_dt.index.weekofyear
df_dt['month'] = df_dt.index.month
df_dt['year'] = df_dt.index.year

# Plot by year
df_grouped = df_dt.groupby(["year", "month"]).sum().reset_index()
fig, ax = plt.subplots(figsize=(16, 5))
colors = {2011:'red', 2012:'blue'}
df1 = df_grouped[df_grouped["year"] == 2011]
df1.plot(ax=ax, kind='line', x='month', y='cnt', label=2011, color='red')
df2 = df_grouped[df_grouped["year"] == 2012]
df2.plot(ax=ax, kind='line', x='month', y='cnt', label=2012, color='blue')
plt.show()

# Plot by month
df_by_month = df_dt.resample('M').sum()
df_by_month["timestamp"] = df_by_month.index
df_by_month.plot(x="timestamp", y="cnt", figsize=(16, 5))

# Plot by day of week
fig,(ax1, ax2)= plt.subplots(nrows=2)
fig.set_size_inches(18, 14)
sns.pointplot(data=df_dt, x='day_of_week', y='cnt', ax=ax1)
sns.pointplot(data=df_dt, x='day_of_week', y='cnt', hue='season', ax=ax2);

# Plot rolling statistics
rol = df["cnt"].rolling(center=False, window=12)
rolmean = rol.mean()
rolstd = rol.std()
df["rolmean"] = rolmean
df["rolstd"] = rolstd
df.plot(
    x="dteday", 
    y=["cnt", "rolstd","rolmean"],
    title="Rolling Mean & Standard Deviation", 
    figsize=(16,6)
)

# stationarity testing, H0: time series is not stationary, so if pval is higher than e.g. 0.05 - it's not stationary (which we see from plot as well).
r = adfuller(df["cnt"], autolag="AIC")
res = dict(stat=r[0], pval=r[1], nlags=r[2], nobs=r[3])
neatoutput = pd.Series(r[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
crit = {}
for key,value in r[4].items():
    crit['crit{0}'.format(key)] = value
    neatoutput['Critical Value ({0})'.format(key)] = value
print(neatoutput)               


# Explore relationship between vars, look for variables correlated with CNT
corrmat = df.corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, annot=True, fmt='.2f', square=True)

# Create prophet-compatible df
df = df[["dteday", "cnt", "temp", "weathersit", "holiday", "hum", "windspeed"]].dropna()  # drop nans

# date column to index
df['dteday'] = pd.to_datetime(df['dteday'])

# Get the column names right
df = df[["dteday", "cnt", "temp", "weathersit", "holiday", "hum", "windspeed"]]
df.columns = ["ds", "y", "temperature_celsius", "weather_condition", "holiday", "humidity", "windspeed"]

# temperature, humidity and windspeed values don't make sense, they seem to be normalized, but I can't find the methodology
# let's just plot
df.plot(x="ds", y="temperature_celsius", figsize=(16,6))
df.plot(x="ds", y="y", figsize=(16,6))

holidays = pd.DataFrame({
  'holiday': 'holiday',
  'ds': df["ds"][df["holiday"] == 1],
  'lower_window': 0,
  'upper_window': 1,
})

m = Prophet(yearly_seasonality=True, holidays=holidays)
m.add_regressor('temperature_celsius')
m.add_regressor('weather_condition')
m.add_regressor('humidity')
m.add_regressor('windspeed')

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))

m.fit(train)

future_data = m.make_future_dataframe(periods=len(test)) 
future_data["temperature_celsius"] = df["temperature_celsius"]
future_data["weather_condition"] = df["weather_condition"]
future_data["humidity"] = df["humidity"]
future_data["windspeed"] = df["windspeed"]
forecast_data = m.predict(future_data)

fig1 = m.plot(forecast_data)
fig = plt.figure(figsize=(20, 6))
plt.plot(df.ds, df.y)
plt.plot(forecast_data.ds, forecast_data.yhat)

fig2 = m.plot_components(forecast_data)

df_cv = cross_validation(m, horizon='30 days')
df_p = performance_metrics(df_cv)
df_p.head(5)
fig3 = plot_cross_validation_metric(df_cv, metric='mape')

m.stan_backend.logger = None  # BUG: https://github.com/facebook/prophet/issues/1361
pkl_path = "forecast_model.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(m, f)

with open('forecast_model.pkl', 'rb') as fin:
    m2 = pickle.load(fin)
