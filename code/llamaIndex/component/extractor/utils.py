import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import json

def parse_obj_to_str(objs):
    if objs is not None:
        return json.dumps([obj.dict() if obj is not None else {} for obj in objs])
    return str([])