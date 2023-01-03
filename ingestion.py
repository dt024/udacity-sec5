import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

logging.basicConfig(filename='logging.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')



#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    df_list = []
    file_list = []
    for file_name in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, file_name)
        df = pd.read_csv(file_path)
        df_list.append(df)
        file_list.append(file_path + '\n')
    
    #Creating output folder
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)
    
    #Concatenating and drop duplicate
    final_df = pd.concat(df_list, axis=0, ignore_index=True).drop_duplicates()
    #Save files
    final_data_name = "finaldata.csv"
    ingested_files_name = "ingestedfiles.txt"

    final_data_path = os.path.join(output_folder_path, final_data_name)
    ingested_files_path = os.path.join(output_folder_path, ingested_files_name)
    
    final_df.to_csv(final_data_path, index=False)
    with open(ingested_files_path, 'w') as f:
        f.writelines(file_list)




if __name__ == '__main__':
    merge_multiple_dataframe()
