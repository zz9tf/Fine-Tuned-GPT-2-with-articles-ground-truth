import os
import yaml
from dotenv import load_dotenv

def load_configs():
    """return config, prefix_config"""
    script_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_path, 'config.yaml')
    prefix_config_path = os.path.join(script_path, 'prefix_config.yaml')
    with open(config_path, 'r') as config:
        config = yaml.safe_load(config)
    with open(prefix_config_path, 'r') as prefix_config:
        prefix_config = yaml.safe_load(prefix_config)
    load_dotenv(dotenv_path=os.path.join(script_path, '.env'))
    return config, prefix_config