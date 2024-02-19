import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')

TARGET = 'hazardous'

FEATURES = ['Absolute Magnitude', 'Est Dia in KM(min)', 'Relative Velocity km per sec', 'Miss Dist.(kilometers)', 'Orbit ID', 'Orbit Uncertainity', 'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant', 'Epoch Osculation', 'Eccentricity', 'Inclination', 'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance', 'Perihelion Arg', 'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly', 'Mean Motion']

NUM_FEATURES = ['Absolute Magnitude', 'Est Dia in KM(min)', 'Relative Velocity km per sec', 'Miss Dist.(kilometers)', 'Orbit ID', 'Orbit Uncertainity', 'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant', 'Epoch Osculation', 'Eccentricity', 'Inclination', 'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance', 'Perihelion Arg', 'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly', 'Mean Motion']

CAT_FEATURES = ['Orbiting Body', 'Equinox']  # Add any other categorical columns here if present in dataset

FEATURES_TO_ENCODE = ['Orbiting Body', 'Equinox', 'hazardous']

FEATURE_TO_MODIFY = ['Absolute Magnitude', 'Relative Velocity km per sec']

FEATURE_TO_ADD = ['Distance_From_Earth']

DROP_FEATURES = ['Epoch Osculation', 'Perihelion Distance']

LOG_FEATURES = ['Relative Velocity km per sec', 'Miss Dist.(kilometers)']

