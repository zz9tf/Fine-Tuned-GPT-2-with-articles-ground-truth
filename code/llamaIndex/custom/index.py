##########################################################################
# Index
from llama_index.core import VectorStoreIndex, PropertyGraphIndex

def get_an_index_generator(index_type):
        if index_type == 'VectorStoreIndex':
            return VectorStoreIndex
        elif index_type == 'PropertyGraphIndex':
            return PropertyGraphIndex
        else:
            raise Exception("Invalid embedding model name. Please provide embedding models {}".format())