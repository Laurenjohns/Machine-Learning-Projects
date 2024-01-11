# Machine-Learning-Projects

## Table of Contents

- [Forecasting Sales](#forecasting_sales)
- [Profitability Analysis](#profitability_analysis)

  ---

## Forecasting Sales

Project Overview: Sales forecasting model for a painting manufacturing company. Leveraging historical sales data, Python and the Prophet library to build an accurate forecasting model. 

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

## Profitability Analysis

Project Overview: Python application that can calculate financial metrics and display interactive line charts for better data analysis and decision-making. 

```python
import pandas as pd
import numpy as np
from numpy_financial import npv, irr

df = pd.read_excel('financial_data.xlsx')

rate = 0.1 
cash_flows = df['Cash Flow'].tolist()
npv_result = npv(rate, cash_flows)

irr_result = irr(cash_flows)

initial_investment = cash_flows[0]
payback_period =0
cummulative_cash_flow =0
for period, cash_flow in enumerate(cash_flows):
    cummulative_cash_flow += cash_flow
    if cummulative_cash_flow >= initial_investment:
      payback_period = period + 1
      break

pi = npv_result / abs(initial_investment)

print(f'NPV: ${npv_result: .2f}')
print(f'IRR: {irr_result: .2%}')
print(f'Payback Period: {payback_period} years')
print(f'PI: {pi: .2f}')

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Period"], y=cash_flows, mode="lines", name="Cash Flow"))
```
<img width="609" alt="Screenshot 2024-01-11 163634" src="https://github.com/Laurenjohns/Machine-Learning-Projects/assets/107310914/0250e0db-c6b2-4e4b-95b3-041f217b2255">


```python

fig.update_layout(
    title = "Cash Flow Over Time",
    xaxis_title="Years",
    yaxis_title="Cash Flow",
)
fig.show()

print(f'NPV: ${npv_result: .2f}')
print(f'IRR: {irr_result: .2%}')
print(f'Payback Period: {payback_period} years')
print(f'PI: {pi: .2f}')
```

<img width="829" alt="Screenshot 2024-01-11 164059" src="https://github.com/Laurenjohns/Machine-Learning-Projects/assets/107310914/16abef65-353e-4c90-9e99-518f8cfb57b3">

---
