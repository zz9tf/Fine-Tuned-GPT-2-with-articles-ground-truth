import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from configs.load_config import load_configs

configs, prefix_config = load_configs()