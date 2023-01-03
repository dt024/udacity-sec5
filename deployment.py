from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import json
import shutil




##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    shutil.copy(os.path.join(config['output_model_path'], 'trainedmodel.pkl'), prod_deployment_path)
    shutil.copy(os.path.join(config['output_model_path'], 'latestscore.txt'), prod_deployment_path)
    shutil.copy(os.path.join(config['output_folder_path'], 'ingestedfiles.txt'), prod_deployment_path)

store_model_into_pickle()
        

