from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics
import scoring
import json
import os

with open('config.json','r') as f:
    config = json.load(f) 
deployed_model_path = os.path.join(config['output_model_path'],'trainedmodel.pkl')
test_data_file_path = os.path.join(config['test_data_path'], 'testdata.csv')


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    test_data_file_path = request.form.get('path')
    result = diagnostics.model_predictions(test_data_file_path[1:-1])
    return json.dumps([int(item) for item in result]) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    f1_score = scoring.score_model(deployed_model_path, test_data_file_path)
    return json.dumps(f1_score) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary():        
    #check means, medians, and modes for each column
    df_stat = diagnostics.dataframe_summary()
    return df_stat.to_dict() #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():        
    #check timing and percent NA values
    timing = diagnostics.execution_time()
    na_percents = diagnostics.dataframe_missing()
    if os.path.isfile('dependencies.json'):
        dependencies = json.load(open('dependencies.json'))
    else:
        dependencies = diagnostics.dependencies_checking().to_dict('records')
    diagnose_dict = {
        'timing': timing,
        'na_percents': na_percents,
        'dependencies': dependencies
    }
    return json.dumps(diagnose_dict) #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
