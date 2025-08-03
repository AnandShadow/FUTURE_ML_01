import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("Superstore.csv", encoding="latin1")

df['Order Date'] = pd.to_datetime(df['Order Date'])
df = df.groupby('Order Date')['Sales'].sum().reset_index()
df.columns = ['ds', 'y']

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("forecast.csv", index=False)

model.plot(forecast)
plt.show()

model.plot_components(forecast)
plt.show()
