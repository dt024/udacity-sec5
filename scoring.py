from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from sklearn.metrics import f1_score




#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

    
# deployed_model_path = os.path.join(config['output_model_path'],'trainedmodel.pkl')
# test_data_file_path = os.path.join(config['test_data_path'], 'testdata.csv')

#################Function for model scoring
def score_model(deployed_model_path, test_data_file_path):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    model = pickle.load(open(deployed_model_path, 'rb'))
    df = pd.read_csv(test_data_file_path)
    X_test = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_test = df['exited']
    pred = model.predict(X_test)
    f1 = f1_score(y_test, pred)

    with open(os.path.join(config['output_model_path'], 'latestscore.txt'), 'w') as f:
        f.write(str(f1))

    return f1