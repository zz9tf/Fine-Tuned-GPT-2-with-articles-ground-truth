import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from component.io import load_nodes_jsonl, save_nodes_jsonl

class ManuallyParser():
    """generate cache for mannually parser"""
    def __init__(self, cache_path, cache_name, delete_cache=True, force=False) -> None:
        self.cache_name = cache_name
        self.cache_path = cache_path
        self.delete_cache = delete_cache
        self.force = force

    def get_nodes_from_documents(self, nodes, **kwargs):
        """get nodes from documents"""
        nodes_cache_path = os.path.join(self.cache_path, f"{self.cache_name}_finished.json")
        if os.path.exists(nodes_cache_path) and not self.force:
            nodes = load_nodes_jsonl(nodes_cache_path)
            if self.delete_cache:
                os.remove(nodes_cache_path)
            return nodes
        nodes_cache_path = os.path.join(
            self.cache_path, f"{self.cache_name}_{len(nodes)}_processing.jsonl"
        )
        save_nodes_jsonl(nodes_cache_path, nodes)
        print(
            f"\n[Manually Parser] Cache \'{self.cache_name}\' has been saved." +\
            "Waiting for processing manually..."
        )
        sys.exit(0)
