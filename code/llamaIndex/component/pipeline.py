from __future__ import annotations
import os
from typing import TYPE_CHECKING
from custom.io import save_nodes_jsonl, load_nodes_jsonl

if TYPE_CHECKING:
    from database import Database

class IndexPipeline:
    def __init__(
            self, 
            index_id: str, 
            index_dir_path:str, 
            database: Database,
            index_pipeline: list,
            delete_cache: bool = True
        ):
        self.steps = [[]]
        self.steps_id = 0
        self.kwargs = {}
        self.kwargs['index_id'] = index_id
        self.database = database
        self.cache_path = os.path.abspath(os.path.join(self.database.root_path, self.database.config['cache']))
        self.delete_cache = delete_cache
        self.kwargs['cache_path'] = self.cache_path
        self.kwargs['index_dir_path'] = index_dir_path
        os.makedirs(self.cache_path, exist_ok=True)
        self.index_pipeline = index_pipeline

        self.add_steps()
    
    def _check_index_pipeline(self, index_pipeline):
        for step_id, step in enumerate(index_pipeline):
            assert len(step) == 1, \
                f"Invalid index pipeline. The length of index step {step_id} is {len(step)}. But the length each step should be 1."

        assert next(iter(index_pipeline[0])) == 'reader', \
            f"Invalid index pipeline. \'reader\' should be the first step of \'index pipeline\'"
        
        assert next(iter(index_pipeline[-1])) == 'storage', \
            f'Invalid index pipeline. \'storage\' should be the final step of \'index pipeline\''
    
    def _get_action_func(self, step_type):
        if step_type == "reader":
            return self.database._load_documents
        elif step_type == "parser":
            return self.database._parser_documents
        elif step_type == "extractor":
            return self.database._extract_metadata
        elif step_type == "storage":
            return self.database._generate_index

    def _save_result(self, cache_name, **kwargs):
        nodes_cache_path = os.path.abspath(os.path.join(self.cache_path, cache_name))
        save_nodes_jsonl(nodes_cache_path, self.kwargs['result'])
        print("[update_database] Cache \'{}\' has been saved at the break point".format(cache_name))

    def _add(self, step_id, step_type, action):
        if step_type == 'break':
            _, prev_kwargs = self.steps[-1][-1]
            cache_name = f'{self.kwargs["index_id"]}_{step_id-1}_{prev_kwargs["step_type"]}_{prev_kwargs["action"]}.jsonl'
            # Rebuild the break point?
            if action == 'force':
                kwargs = {'step_type': step_type, 'action': action, 'cache_name': cache_name, 'step_id': step_id}
                self.steps[-1].append((self._save_result, kwargs))
                self.steps.append([])
            else:
                # Check is the break point cache exist?
                nodes_cache_path = os.path.abspath(os.path.join(self.cache_path, cache_name))
                if os.path.exists(nodes_cache_path):
                    print(f"[update_database] Cache {cache_name} is detected. Now at {cache_name} ...")
                    self.steps_id = len(self.steps)
                    self.steps.append([])
                    self.nodes_cache_path = nodes_cache_path
                else:
                    kwargs = {'step_type': step_type, 'action': action, 'cache_name': cache_name, 'step_id': step_id}
                    self.steps[-1].append((self._save_result, kwargs))
                    self.steps.append([])

        else:
            func = self._get_action_func(step_type)
            config = self.database.prefix_config[step_type][action]
            kwargs = {'step_type': step_type, 'action': action, 'config': config, 'step_id': step_id}
            self.steps[-1].append((func, kwargs))

    def add_steps(self):
        for step_id, step in enumerate(self.index_pipeline):
            step_type, action = next(iter(step.items()))
            self._add(step_id, step_type, action)
    
    def update_kwargs_after_func(self, step_type, result):
        if step_type == 'storage':
            self.kwargs['result'] = result
        elif step_type != 'break':
            self.kwargs['result'] = result
            self.kwargs['nodes'] = result

    def run(self):
        if hasattr(self, 'nodes_cache_path'):
            # Directly use nodes have been generated
            self.kwargs['nodes'] = load_nodes_jsonl(self.nodes_cache_path)
            if self.delete_cache:
                os.remove(self.nodes_cache_path)

        for func, kwargs in self.steps[self.steps_id]:
            # Update kwargs before func
            self.kwargs.update(kwargs)
            # Execute func
            result=func(**self.kwargs)
            # Update kwargs after func
            self.update_kwargs_after_func(self.kwargs['step_type'], result)

        return self.kwargs['step_type'], self.kwargs['action'], self.kwargs['result']
