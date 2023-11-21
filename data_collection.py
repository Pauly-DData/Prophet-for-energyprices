import pandas as pd 
from prophet import Prophet
import matplotlib.pyplot as plt

# The file path to your dataset
file_path = 'C:/Spyder - Python/Clean/FINAL HOURLY MERGED dataset - Price Usagae Temp 20220901 - 20230901'

#read the file as a csv file in the new variable 'data'
data = pd.read_csv(file_path)

#show the first 4 row of the data
print(data.head())

# to get a more common sense of the data we will explore the dataset
print(data.info())
print(data.describe())

#identify if there are missing values in the filepath dataset
print(data.isnull().sum())

#check for outliers
data.boxplot(column=['UsageLDN', 'Price'])
plt.show()

