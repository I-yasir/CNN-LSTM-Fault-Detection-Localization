import yaml
import os
from utils.preprocess_data import prepare_dataset
from utils.mat_to_npz import collect_and_save_data


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(PROJECT_ROOT, "configuration", "config.yaml")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

raw_data_path = os.path.join(PROJECT_ROOT,config['path']['raw_data'] )
collected_data_path = os.path.join(PROJECT_ROOT,config['path']['collected_data'])
processed_data_path = os.path.join(PROJECT_ROOT,config['path']['processed_data'])


collect_and_save_data (raw_data_path , collected_data_path )

prepare_dataset(collected_data_path, processed_data_path) 