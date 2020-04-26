from pipeline.boston_pipeline import run_pipeline
from utils.helpers import ModelTracker # need to have access to pickle object to load

def setup():
	_ = run_pipeline()

if __name__ == '__main__':
	setup()