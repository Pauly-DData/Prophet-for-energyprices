import pandas as pd 
from prophet import Prophet

#the file we are now loading is the cleaned file from our previous data collection and cleaning script
file_path = r"C:\Users\PaulDriessens\OneDrive - Energyhouse B.V\Documenten\Paul's map\Project Data24\Prophet-for-energyprices\prophet_dataset_v1.csv"

data_model = pd.read_csv(file_path)

#just checking out if the dataset looks like we want it to be
print(data_model.head())

#for the prophet model we need to rename the columns to 'ds' (date column) and to 'y' that is the target column
data_model.rename(columns={'Verbruik KwH': 'y', 'Datum': 'ds'}, inplace=True)

print(data_model.head())

#now that are df is corrected to something that fitst the model requirements we add a new variable that we will build the model on. we only need the ds en y variables. 
prophet_data = data_model[['ds', 'y']]

#initialize the model - with this action we create a new Prophet model
model = Prophet()

# Fit the model with your dataframe - we then train our model based on our data. We trigger the model that is should start learning from the 'prophet_data'. 
model.fit(prophet_data)

#we create a new dataframe that will contain the predictions. With the 'periods=365' we look a 365 periods into the future and with 'freg=H' we tell the model that every period is one hour. 
#we can adjust periods with the number of periods we want to forecast; so if we want to the model to predict one year we change periods=8760 or a prediction for the next day will be periods=24
future = model.make_future_dataframe(periods=24, freq='H')

#with forecast we instruct the model to do predictions based on the 'future' df. The model learns from the historical data to predict
forecast = model.predict(future)

# Display the forecast based for the next x hours. The ds in the df is equal to the time and time prediction. 'yhat' is the predicted value for the specific time (ds). 
# 'yhat_lower' and 'yhat_upper' represent the upper and lower bound of the predictions. These values present an indication on the uncertainty of the prediction
# '.tail' is used to display the last x rows of the prediction
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24)

#show the results
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])