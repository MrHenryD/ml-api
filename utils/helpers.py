import pickle
from datetime import datetime

from sklearn.datasets import load_boston

class ModelTracker(object):
    def __init__(self):
        self.training_results = []
        self.best_score = 0
    
    def add_training_result(self, metric, score, record_time=datetime.now()):
        if score > self.best_score:
            self.best_score = score
            
        self.training_results.append((metric, score, record_time))
        
    def compare_score(self, score):
        if self.best_score < score:
            return False
        else:
            return True
        
    @staticmethod
    def save(obj, file):
        with open(file, 'wb') as outfile:
            pickle.dump(obj, outfile)
            
    @staticmethod
    def load(file):
        with open(file, 'rb') as readfile:
            obj = pickle.load(readfile)
            return obj

def load_data():
    """ Load and return the boston house-prices dataset (regression). """
    
    dataset = load_boston()
    data, target, features, description = (dataset['data'], dataset['target'], 
                                           dataset['feature_names'], dataset['DESCR'])
    
    return data, target, features, description


