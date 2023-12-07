import pandas as pd 
from prophet import Prophet
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

# The file path to your dataset
file_path = "C:\Spyder - Python\Stichting Park Medisch Verbruik Dataset\Stichting Park Medisch Verbruik Dataset - Elektriciteit\Clean\FINAL HOURLY MERGED dataset - Price Usagae Temp 20220901 - 20230901"

#read the file as a csv file in the new variable 'data'
data = pd.read_csv(file_path)

profile = ProfileReport(data,title="Energy Usage")

profile.to_file("Energy_usgage.html")

#show the first 5 rows of the data
print(data.head())

# to get a more common sense of the data we will explore the dataset
print(data.info())
print(data.describe())

#identify if there are missing values in the filepath dataset
print(data.isnull().sum())

#check for outliers in usage and price data
data.boxplot(column=['UsageLDN'])

#plot show behind # for smoother run
#plt.show()

data.boxplot(column=['Price'])

#plot show behind # for smoother run
#plt.show()

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

#plot show behind # for smoother run
#plt.show()

#we now have a better understanding about our data and we want to delete some columns so we can proceed with building our model
#lets first rename some columns so we can work more easier with the df
data = data.rename(columns={'StartDate': 'Datum', 'UsageLDN': 'Verbruik KwH', 'Price': 'Prijs'})

#usage and price variables need to be transformed because their values need to align with new column names
data['Prijs'] = data['Prijs'] / 1000

#now we need to remove some columns because these are not as intresting for our modelling right now. We will remove 'DD', 'FH', 'FF', 'T', 'SQ', 'Timestamp'
remove_columns = ['DD', 'FH', 'FF', 'T', 'SQ', 'Timestamp']
data = data.drop(columns=remove_columns)

#we have done the data collection, exploration and the cleaning part. For today our job is done, but we will save the new df to a file we'll be using for our model
#import os

#file_path = os.path.join(os.getcwd(), 'prophet_dataset_v1.csv')
#data.to_csv(file_path, index=False)
#print(f"File saved at: {file_path}")


 
