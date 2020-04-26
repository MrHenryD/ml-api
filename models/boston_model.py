#from sklearn.externals import joblib
import joblib

import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import xgboost as xgb

class BostonHousingModel(object):
    def __init__(self, hyperparameters=None, transformers=[]):        
        if hyperparameters:
            model = xgb.XGBRegressor(**hyperparameters)
        else:
            model = xgb.XGBRegressor()
        
        self.pipeline = Pipeline([('t_%s' % str(i + 1), transformer) 
                                   for transformer in transformers] + 
                                 [('clf', model)]
                                )
        
    def train(self, data, target):
        self.pipeline = self.pipeline.fit(data, target)
        
    def score(self, data, target, metric='r2'):
        predictions = self.predict(data)
        
        if metric == 'r2':
            return r2_score(target, predictions)
        
        elif metric == 'mse':
            return mean_squared_error(target, predictions)
    
    def save(self, file):
        _ = joblib.dump(self.pipeline, file)
    
    def load(self, file, replace=True):
        pipeline = joblib.load(file)
        
        if replace:
            self.pipeline = pipeline
        
        return pipeline
    
    def predict(self, data):
        predictions = self.pipeline.predict(data)
        return predictions


    def predict_one(self, json_request):
        features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
     'TAX', 'PTRATIO', 'B', 'LSTAT']

        data = np.array([json_request[feature] for feature in features]).reshape(1, -1)
        prediction = self.predict(data)[0]
        return prediction
