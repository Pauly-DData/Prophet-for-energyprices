import pandas as pd 
from prophet import Prophet
import matplotlib.pyplot as plt

# The file path to your dataset
file_path = "C:\Spyder - Python\Stichting Park Medisch Verbruik Dataset\Stichting Park Medisch Verbruik Dataset - Elektriciteit\Clean\FINAL HOURLY MERGED dataset - Price Usagae Temp 20220901 - 20230901"

#read the file as a csv file in the new variable 'data'
data = pd.read_csv(file_path)

#show the first 5 rows of the data
print(data.head())

# to get a more common sense of the data we will explore the dataset
print(data.info())
print(data.describe())

#identify if there are missing values in the filepath dataset
print(data.isnull().sum())

#check for outliers in usage and price data
data.boxplot(column=['UsageLDN'])
plt.show()

data.boxplot(column=['Price'])
plt.show()

#we see that price has alot of outliers so we are going to scatter plot the dataset to visualise change over time
#first things first we'll convert the data column to datetime
data['StartDate'] = pd.to_datetime(data['StartDate'])

#now lets create the scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(data['StartDate'], data['Price'], alpha=0.5, s=10)

#add some titles and labels 
plt.title('Price over time')
plt.xlabel('Date')
plt.ylabel('Price')

plt.show()

#we now have a better understanding about our data and we want to delete some columns so we can proceed with building our model
#lets first rename some columns so we can work more easier with the df
data = data.rename(columns={'StartDate': 'Datum', 'UsageLDN': 'Verbruik KwH', 'Price': 'Prijs'})

#usage and price 


