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
prophet_data = data_model['ds', 'y']