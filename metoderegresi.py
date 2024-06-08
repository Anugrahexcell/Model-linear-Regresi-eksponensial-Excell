import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set up Kaggle API credentials
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)

kaggle_json = '{"username":"anugrahexcel","key":"91ce27d948a643b78a2e7452b29e4045"}'

with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    f.write(kaggle_json)

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset
dataset = 'nikhil7280/student-performance-multiple-linear-regression'
destination_folder = 'Student_Performance'
api.dataset_download_files(dataset, path=destination_folder, unzip=True)

# Load data
file_path = os.path.join(destination_folder, 'Student_Performance.csv')
data = pd.read_csv(file_path)

# 2. Regresi Linear
X = data[['Hours Studied']]
Y = data['Performance Index']

model_linear = LinearRegression()
model_linear.fit(X, Y)

Y_pred_linear = model_linear.predict(X)
rms_linear = np.sqrt(mean_squared_error(Y, Y_pred_linear))

# Plot hasil regresi linear
plt.figure(figsize=(10, 5))
plt.scatter(X, Y, color='blue', label='Data')
plt.plot(X, Y_pred_linear, color='red', label='Regresi Linear')
plt.xlabel('Durasi Waktu Belajar (Hours Studied)')
plt.ylabel('Nilai Ujian (Performance Index)')
plt.title('Regresi Linear')
plt.legend()
plt.show()

# 3. Regresi Eksponensial
X_log = np.log(X)  # Transformasi logaritmik dari variabel independen
model_exp = LinearRegression()
model_exp.fit(X_log, Y)

Y_pred_exp = model_exp.predict(X_log)
rms_exp = np.sqrt(mean_squared_error(Y, Y_pred_exp))

# Plot hasil regresi eksponensial
plt.figure(figsize=(10, 5))
plt.scatter(X, Y, color='blue', label='Data')
plt.plot(X, Y_pred_exp, color='green', label='Regresi Eksponensial')
plt.xlabel('Durasi Waktu Belajar (Hours Studied)')
plt.ylabel('Nilai Ujian (Performance Index)')
plt.title('Regresi Eksponensial')
plt.legend()
plt.show()

# 4. Hasil dan Galat RMS
print(f"Galat RMS Regresi Linear: {rms_linear}")
print(f"Galat RMS Regresi Eksponensial: {rms_exp}")