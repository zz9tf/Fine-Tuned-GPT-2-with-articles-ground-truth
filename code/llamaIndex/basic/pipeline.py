from __future__ import annotations
import os
from typing import TYPE_CHECKING
from utils.get import get_a_store

if TYPE_CHECKING:
    from database import Database

class CreateIndexPipeline:
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
        self.kwargs['index_dir_path'] = index_dir_path
        self.database = database
        self.cache_path = os.path.abspath(os.path.join(self.database.root_path, self.database.config['cache']))
        self.delete_cache = delete_cache
        os.makedirs(self.cache_path, exist_ok=True)
        self.index_pipeline = index_pipeline
        self.isBreak = False

        self.add_steps()
    
    def _get_action_func(self, step_type):
        if step_type == "reader":
            return self.database._load_documents
        elif step_type == "parser":
            return self.database._parser_documents
        elif step_type == "extractor":
            return self.database._extract_metadata
        elif step_type == "storage":
            return self.database._storage

    def _save_result(self, cache_name, **kwargs):
        docstore = get_a_store('SimpleDocumentStore')
        docstore.add_documents(self.kwargs['result'])
        nodes_cache_path = os.path.abspath(os.path.join(self.cache_path, cache_name))
        docstore.persist(persist_path=nodes_cache_path)
        print("[update_database] Cache \'{}\' has been saved at the break point".format(cache_name))

    def _add(self, step_id, step_type, action):
        if step_type == 'break':
            _, prev_kwargs = self.steps[-1][-1]
            cache_name = f'{self.kwargs['index_id']}_{step_id-1}_{prev_kwargs['step_type']}_{prev_kwargs['action']}.json'
            # Rebuild the break point?
            if action == 'force':
                kwargs = {'step_type': step_type, 'action': action, 'cache_name': cache_name}
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
                    kwargs = {'step_type': step_type, 'action': action, 'cache_name': cache_name}
                    self.steps[-1].append((self._save_result, kwargs))
                    self.steps.append([])

        else:
            func = self._get_action_func(step_type)
            config = self.database.prefix_config[step_type][action]
            kwargs = {'step_type': step_type, 'action': action, 'config': config}
            self.steps[-1].append((func, kwargs))

    def add_steps(self):
        for step_id, step in enumerate(self.index_pipeline):
            step_type, action = next(iter(step.items()))
            self._add(step_id, step_type, action)
            if self.isBreak:
                break
    
    def update_kwargs_after_func(self, step_type, result):
        if step_type == 'storage':
            self.kwargs['result'] = result
        elif step_type != 'break':
            self.kwargs['result'] = result
            self.kwargs['nodes'] = result

    def run(self):
        if hasattr(self, 'nodes_cache_path'):
            # Directly use nodes have been generated
            docstore = get_a_store('SimpleDocumentStore').from_persist_path(persist_path=self.nodes_cache_path)
            nodes = [node for _, node in docstore.docs.items()]
            self.kwargs['nodes'] = nodes
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
