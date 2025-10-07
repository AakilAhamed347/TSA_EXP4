# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in # Fit-the-ARMA-model-for-any-data-set

### DEVELOPED BY: AAKIL AHAMED S
### REGISTER NO: 212224040002

# AIM:
To implement ARMA model in python.

# ALGORITHM:
1. Import necessary libraries.

2. Set up matplotlib settings for figure size.

3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000 data points using the ArmaProcess class. Plot the generated time series and set the title and x- axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.

5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000 data points using the ArmaProcess class. Plot the generated time series and set the title and x- axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.

# PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("/content/housing_price_dataset.csv")
# Extract the year from the date column by converting to string and slicing
data['year'] = data['date'].astype(str).str[-4:].astype(int)


yearly_scores = data.groupby(data['year'])['Price'].mean().reset_index()
yearly_scores.rename(columns={'Price': 'avg_price'}, inplace=True)

X = yearly_scores['avg_price'].dropna().values
N = 1000

plt.figure(figsize=(12, 6))
plt.plot(yearly_scores['year'], X, marker='o')
plt.title('Yearly Average Prices')
plt.xlabel("Year")
plt.ylabel("Avg Price")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.arparams[0]
theta1_arma11 = arma11_model.maparams[0]

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.figure(figsize=(12, 6))
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(ARMA_1, lags=40, ax=plt.gca())
plt.title("ACF of Simulated ARMA(1,1)")

plt.subplot(2, 1, 2)
plot_pacf(ARMA_1, lags=40, ax=plt.gca())
plt.title("PACF of Simulated ARMA(1,1)")
plt.tight_layout()
plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22, phi2_arma22 = arma22_model.arparams
theta1_arma22, theta2_arma22 = arma22_model.maparams

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.figure(figsize=(12, 6))
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(ARMA_2, lags=40, ax=plt.gca())
plt.title("ACF of Simulated ARMA(2,2)")

plt.subplot(2, 1, 2)
plot_pacf(ARMA_2, lags=40, ax=plt.gca())
plt.title("PACF of Simulated ARMA(2,2)")
plt.tight_layout()
plt.show()
```
# OUTPUT:
<img width="1345" height="674" alt="image" src="https://github.com/user-attachments/assets/dea64e5b-81b0-46f6-880e-f58cff4d16dd" />
<img width="1338" height="336" alt="image" src="https://github.com/user-attachments/assets/55ab93bf-08a9-4e0a-b52f-63e22abfff6e" />
<img width="1363" height="331" alt="image" src="https://github.com/user-attachments/assets/3dc97661-d333-46bd-a3a7-cfa7ad900003" />
<img width="1289" height="660" alt="image" src="https://github.com/user-attachments/assets/6aad10e1-5641-48d8-b037-9f6156c9ecf0" />
<img width="1331" height="327" alt="image" src="https://github.com/user-attachments/assets/bc9dddd4-6bb8-49e0-a366-dbadb0061b57" />
<img width="1344" height="314" alt="image" src="https://github.com/user-attachments/assets/807b5424-c32a-4c62-af1d-f318b7c40928" />
<img width="1337" height="665" alt="image" src="https://github.com/user-attachments/assets/a8d37cf9-dcb5-49bf-8126-69e145917106" />
<img width="1328" height="636" alt="image" src="https://github.com/user-attachments/assets/9749f2aa-bd9e-41e3-be0e-875d49c571b3" />




# RESULT: 
Thus, a python program is created to fir ARMA Model successfully.
