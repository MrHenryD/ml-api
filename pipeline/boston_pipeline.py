import os

from config.settings import BOSTON_MODEL_PATH, MODEL_TRACKER_PATH, HYPER_PARAMETERS, METRIC
from models.boston_model import BostonHousingModel
from utils.helpers import load_data, ModelTracker

def run_pipeline():
    # load model tracker (if exists)
    if os.path.isfile(MODEL_TRACKER_PATH):
        model_tracker = ModelTracker.load(MODEL_TRACKER_PATH)
    else:
        model_tracker = ModelTracker()

    # load model and data
    print('Loading Data.')
    data, target, features, description = load_data()
    model = BostonHousingModel(HYPER_PARAMETERS)

    # train, score and save training iteration
    print('Training Model.')
    _ = model.train(data, target)

    score = model.score(data, target, METRIC)
    model_tracker.add_training_result(METRIC, score)

    print('Model Score: %s - %s' % (METRIC, score))

    # save results if new model is better
    is_better = model_tracker.compare_score(score)

    if is_better:
        print('Saving Model.')
        _ = model.save(BOSTON_MODEL_PATH)

    _ = model_tracker.save(model_tracker, MODEL_TRACKER_PATH)
    print('Completed.')