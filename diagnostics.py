
import pandas as pd
import numpy as np
import timeit
import os
import pickle
import json
import subprocess


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

##################Function to get model predictions
def model_predictions(test_data_file_path):
    #read the deployed model and a test dataset, calculate predictions
    model = pickle.load(open(os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl'), 'rb'))
    df = pd.read_csv(test_data_file_path)
    X_test = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]

    return model.predict(X_test) #return value should be a list containing all predictions


##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(os.path.join(config['output_folder_path'], 'finaldata.csv'))
    df_stats = df.describe().iloc[1:3]
    median_list = []
    for col in df_stats.columns:
        median_list.append(df[col].median(axis=0))
    df_median = pd.DataFrame([median_list], columns=df_stats.columns, index=['median'])
    df_stats = pd.concat([df_stats, df_median])
    
    return df_stats #return value should be a list containing all summary statistics


##################Function to check for missing data
def dataframe_missing():
    #calculate the percentage of na values
    df = pd.read_csv(os.path.join(config['output_folder_path'], 'finaldata.csv'))
    na_list = list(df.isna().sum(axis=0))
    na_percents = [na_list[i]/len(df.index) for i in range(len(na_list))]
    
    return na_percents #return the percentage of na values


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_time = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - start_time
    start_time = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - start_time
    return [ingestion_time, training_time] #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    if not os.path.isfile('requirements.txt'):
        subprocess.check_output(['pip', 'freeze'])

    with open('requirements.txt', 'r') as f:
        modules = f.read().splitlines()
    
    modules_dict = {'module_name': [], 'current_version': [], 'latest_version': []}
    for module in modules:
        module_name, current_version = module.split('==')
        latest_version = subprocess.check_output(['python3 -m pip show {0}'.format(module_name)],shell=True)
        latest_version = latest_version.split(b'Version: ')[1].split(b'\n')[0]
        latest_version = latest_version.decode('ascii')
        modules_dict['module_name'].append(module_name)
        modules_dict['current_version'].append(current_version)
        modules_dict['latest_version'].append(latest_version)
    df_modules = pd.DataFrame(modules_dict)
    
    with open('dependencies.json', 'w') as f:
        json.dump(df_modules.to_dict('records'), f)
        
    return df_modules #return a table of current and latest versions of modules used in this script


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    dataframe_missing()
    execution_time()
    outdated_packages_list()





    
