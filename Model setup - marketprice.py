import pandas as pd 
from prophet import Prophet

#the file we are now loading is the cleaned file from our previous data collection and cleaning script
file_path = r"C:\Users\PaulDriessens\OneDrive - Energyhouse B.V\Documenten\Paul's map\Project Data24\Prophet-for-energyprices\prophet_dataset_v1.csv"

data_model = pd.read_csv(file_path)

#just checking out if the dataset looks like we want it to be
#print(data_model.head())

#for the prophet model we need to rename the columns to 'ds' (date column) and to 'y' that is the target column
data_model.rename(columns={'Prijs': 'y', 'Datum': 'ds'}, inplace=True)
prophet_data = data_model[['ds', 'y']]

print(data_model.head())

#now that are df is corrected to something that fitst the model requirements we add a new variable that we will build the model on. we only need the ds en y variables. 
prophet_data = data_model[['ds', 'y']]

#initialize the model - with this action we create a new Prophet model
#We also add a df with the dutch holidays because our data is from a dutch company
model = Prophet()
model.add_country_holidays(country_name='NL')

# Fit the model with your dataframe - we then train our model based on our data. We trigger the model that is should start learning from the 'prophet_data'. 
model.fit(prophet_data)

#we create a new dataframe that will contain the predictions. With the 'periods=365' we look a 365 periods into the future and with 'freg=H' we tell the model that every period is one hour. 
#we can adjust periods with the number of periods we want to forecast; so if we want to the model to predict one year we change periods=8760 or a prediction for the next day will be periods=24
future = model.make_future_dataframe(periods=720, freq='H')

#with forecast we instruct the model to do predictions based on the 'future' df. The model learns from the historical data to predict
forecast = model.predict(future)

# Display the forecast based for the next x hours. The ds in the df is equal to the time and time prediction. 'yhat' is the predicted value for the specific time (ds). 
# 'yhat_lower' and 'yhat_upper' represent the upper and lower bound of the predictions. These values present an indication on the uncertainty of the prediction
# '.tail' is used to display the last x rows of the prediction
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(720)

#show the results
#print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# intresting, but now lets try to visualize because we can better understand the model's predictions a little better. 
# Firstly we we'll be focusing at the 24 hour period we tried to forecast using Matplotlib to create a custom plot
from prophet.plot import plot_plotly

fig = plot_plotly(model, forecast)
fig.show()

#now that we have a model setup and a first plot using the Prophet model, we are going to check out whether our model performs well or that we are going to need to do some more adjustments.
#we first add some functions from the Prophet and Matplot libary 
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt

# Define the initial training period, period between each cutoff date, and the forecast horizon
initial = '210 days'  # e.g., 6 (180 days) months of data as the initial training period - Set histrorical data to train the model - caputing season but als have some left so we need  6 - 9 months of training data
period = '14 days'    # e.g., 30 days between each cutoff date - Determines how often to make a new forecast. After each period, a new cutoff is created, the model is retrained up to that point and a new forecast is made.
horizon = '7 days'    # e.g., 7 days forecast horizon - set how far in the future each forecast should predict 

# Perform cross-validation - create a new df to store the cross validation
df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)

# Calculate performance metrics (MSE, RMSE, MAE, MAPE, MDAPE, SMAPE, Coverage)
df_performance = performance_metrics(df_cv)
print(df_performance)

# Plotting the performance metric, e.g., MAE
fig = plot_cross_validation_metric(df_cv, metric='coverage')
plt.show()
