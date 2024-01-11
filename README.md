# Machine-Learning-Projects

## Forecasting Sales

Project Overview: Sales forecasting model for a painting manufacturing company. Leveraging historical sales data, Python and the Prophet library to build an accurate forecasting model. 

---

```python
import pandas as pd
data = pd.read_csv('sales_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.rename(columns={'timestamp':'ds', 'sales':'y'})

from prophet import Prophet
model = Prophet()
model.fit(data)

# making predictions

future_dates = model.make_future_dataframe(periods=365*2, freq='D')
forecast = model.predict(future_dates)

# visualize

import matplotlib.pyplot as plt
plt.figure()
model.plot(forecast, xlabel='Date', ylabel='Sales Qty')
plt.title('Sales Forecast')
plt.show()

# plot the historical data

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('sales_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
plt.plot(data['timestamp'], data['sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Historical Sales Data')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
```

---
