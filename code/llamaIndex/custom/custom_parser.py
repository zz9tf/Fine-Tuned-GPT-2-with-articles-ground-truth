import os
import json
from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import BaseNode, Document, NodeRelationship, MetadataMode
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.llms import LLM
from llama_index.core.node_parser.relational.hierarchical import _add_parent_child_relationship
from custom.schema import TemplateSchema
from custom.response_synthesis import TreeSummarize

class CustomHierarchicalNodeParser(NodeParser):
    """Hierarchical node parser.

    Splits a document into a recursive hierarchy Nodes using a NodeParser.

    NOTE: this will return a hierarchy of nodes in a flat list, where there will be
    overlap between parent nodes (e.g. with a bigger chunk size), and child nodes
    per parent (e.g. with a smaller chunk size).
    """
  
    _cache_doc_path: str = PrivateAttr()
    _cache_process_path: str = PrivateAttr()
    _cur_level: int = PrivateAttr()
    _cur_processing_node_id: int = PrivateAttr()
    _cache_docstore: SimpleDocumentStore = PrivateAttr()

    # The chunk level to use when splitting documents: document, section, paragraph, multi-sentences
    _chunk_levels: List[str] = PrivateAttr()

    _doc_id_to_document: Dict[str, Document] = PrivateAttr()

    _sentences_splitter: SentenceSplitter = PrivateAttr()

    _tree_summarizer: TreeSummarize = PrivateAttr()
    
    @classmethod
    def from_defaults(
        cls,
        llm: LLM,
        cache_dir_path: str = None,
        cache_dir_name: str = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None
    ) -> "CustomHierarchicalNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        cls._chunk_levels = ["document", "section", "paragraph", "multi-sentences"]
        
        cls._cur_level = 0
        cls._cur_processing_node_id = 0

        if cache_dir_path is not None:
            if cache_dir_name is not None:
                cache_dir_path = os.path.join(cache_dir_path, cache_dir_name)
            os.makedirs(cache_dir_path, exist_ok=True)
            cls._cache_doc_path = os.path.join(cache_dir_path, 'cache.json')
            cls._cache_process_path =  os.path.join(cache_dir_path, 'process.json')
            if os.path.exists(cls._cache_doc_path):
                cls._cache_docstore = SimpleDocumentStore.from_persist_path(cls._cache_doc_path)
                with open(cls._cache_process_path, 'r') as process_file:
                    process = json.load(process_file)
                    cls._cur_level = process['level']
                    cls._cur_processing_node_id = process['node_id']
            else:
                cls._cache_docstore = SimpleDocumentStore()
                cls._cache_docstore.persist(cls._cache_doc_path)        
                process = {
                    'level': cls._cur_level,
                    'node_id': cls._cur_processing_node_id
                }
                with open(cls._cache_process_path, 'w') as process_file:
                    json.dump(process, process_file, indent=4)
        else:
            cls._cache_docstore = None
            cls._cache_process_path = None

        cls._sentences_splitter = SentenceSplitter(
            chunk_size=128, 
            chunk_overlap=20, 
            include_metadata=False, 
            include_prev_next_rel=False
        )

        cls._tree_summarizer = TreeSummarize.from_defaults(
            query_str=TemplateSchema.tree_summary_section_q_Tmpl,
            summary_str=TemplateSchema.tree_summary_summary_Tmpl,
            qa_prompt=TemplateSchema.tree_summary_qa_Tmpl,
            llm=llm
        )

        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )
    
    @classmethod
    def class_name(cls) -> str:
        return "CustomHierarchicalNodeParser"

    def save_nodes(self, nodes: list[BaseNode]) -> None:
        if self._cache_docstore == None:
            return
        self._cache_docstore.add_documents(nodes)
        self._cache_docstore.persist(self._cache_doc_path)
        process = {
            'level': self._cur_level,
            'node_id': self._cur_processing_node_id
        }
        with open(self._cache_process_path, 'w') as process_file:
            json.dump(process, process_file, indent=4)

    def _get_document_node_from_document(
        self,
        document: Document,
    ) -> List[BaseNode]:
        document_content = document.get_content(metadata_mode=MetadataMode.NONE)

        # build node from document
        all_nodes = build_nodes_from_splits(
            [document_content],
            document,
            id_func=self.id_func,
        )
            
        return all_nodes

    def _summary_content(self, texts):
        summary, _ = self._tree_summarizer.generate_response_hs(
            texts=texts,
            num_children=len(texts)
        )
        return summary

    def _get_section_nodes_from_document_node(
        self, 
        document_node: BaseNode,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        parent_document = self._doc_id_to_document.get(document_node.ref_doc_id, None)
        if parent_document is None:
            raise Exception(f"Parent document is not found with id {document_node.ref_doc_id}")

        titles = []
        sections = []
        summaries = []
        # Get summary of section contents
        abstract_paragraphs = None
        for title, (start, end) in parent_document.metadata['sections'].items():
            section = document_node.get_content()[start: end]
            if title == 'abstract':
                abstract_paragraphs = section.split('\n')
            else:
                titles.append(title)
                sections.append(section)
                # summaries.append(self._summary_content(section.split('\n')))
                summaries.append('')
        
        
        # Get summary of abstract
        # summary_for_document = None
        # if abstract_paragraphs != None:
        #     summary_for_document = self._summary_content(abstract_paragraphs)
        # else:
        #     summary_for_document = self._summary_content(summaries)

        # Update document node
        origin_document_text = document_node.text
        document_node.metadata['original_content'] = origin_document_text
        # document_node.text = summary_for_document
        document_node.text = ''

        # Update section nodes
        all_nodes.extend(
            build_nodes_from_splits(summaries, document_node, id_func=self.id_func)
        )

        for title, node in zip(titles, all_nodes):
            node.metadata['section_title'] = title
            node.metadata['original_content'] = section

        return all_nodes

    def _get_paragraph_nodes_from_section_node(
        self,
        section_node: BaseNode
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        
        splits = section_node.metadata['original_content'].split('\n')

        for i, text in enumerate(splits):
            if i == 0:
                continue
            splits[i] = f"One Paragraph of {splits[0]}: {text}"
        all_nodes.extend(
            build_nodes_from_splits(splits, section_node, id_func=self.id_func)
        )

        return all_nodes

    def _get_multiple_sentences_nodes_from_paragraph_node(
        self,
        paragraph_node: BaseNode
    ) -> List[BaseNode]:
        all_nodes = self._sentences_splitter._parse_nodes([paragraph_node])

        return all_nodes

    def _postprocess_parsed_nodes(
        self, 
        nodes: List[BaseNode], 
        chunk_level: str
    ) -> List[BaseNode]:
        for i, node in enumerate(nodes):
            parent_doc = self._doc_id_to_document.get(node.ref_doc_id, None)
            if parent_doc is not None:
                start_char_idx = parent_doc.text.find(
                        node.get_content(metadata_mode=MetadataMode.NONE)
                    )

                # update start/end char idx
                if start_char_idx >= 0:
                    node.start_char_idx = start_char_idx
                    node.end_char_idx = start_char_idx + len(
                        node.get_content(metadata_mode=MetadataMode.NONE)
                    )

                metadata = {k: v for k, v in parent_doc.metadata.items() if k not in ['sections']}
                metadata['level'] = chunk_level
                node.metadata.update(metadata)

                exclude_keys = list(metadata.keys())
                exclude_keys.remove('title')
                node.excluded_embed_metadata_keys.extend(exclude_keys)
                node.excluded_llm_metadata_keys.extend(exclude_keys)

                self._doc_id_to_document[node.id_] = parent_doc
                    
            if chunk_level != 'document':
                # establish prev/next relationships if nodes share the same source_node
                if (
                    i > 0
                    and node.source_node
                    and nodes[i - 1].source_node
                    and nodes[i - 1].source_node.node_id == node.source_node.node_id
                ):
                    node.relationships[NodeRelationship.PREVIOUS] = nodes[
                        i - 1
                    ].as_related_node_info()
                if (
                    i < len(nodes) - 1
                    and node.source_node
                    and nodes[i + 1].source_node
                    and nodes[i + 1].source_node.node_id == node.source_node.node_id
                ):
                    node.relationships[NodeRelationship.NEXT] = nodes[
                        i + 1
                    ].as_related_node_info()

        return nodes

    def _split_nodes(
            self, chunk_level: str, node: BaseNode
    ) -> List[BaseNode]:
        if chunk_level == 'document':
            nodes =  self._get_document_node_from_document(node)
        elif chunk_level == 'section':
            nodes =  self._get_section_nodes_from_document_node(node)
        elif chunk_level == 'paragraph':
            nodes = self._get_paragraph_nodes_from_section_node(node)
        elif chunk_level == 'multi-sentences':
            nodes = self._get_multiple_sentences_nodes_from_paragraph_node(node)
        else:
            raise Exception(f"Invalid chunk level {chunk_level}")
        
        nodes = self._postprocess_parsed_nodes(nodes, chunk_level)
        return nodes

    def _get_nodes_from_nodes(
        self,
        nodes: List[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Recursively get nodes from nodes."""

        # Is there cache docstore?
        if self._cache_docstore is None:
            all_nodes = []
            prev_level_nodes = nodes
            finished_sub_node = []
        else:
            all_nodes = [node for _, node in self._cache_docstore.docs.items()]
            if self._cur_level == 0:
                # prev_level_nodes
                prev_level_nodes = nodes[self._cur_processing_node_id:]
                # finished current nodes
                finished_sub_node = all_nodes
            else:
                i = 0
                prev_level_nodes = []
                finished_sub_node = []
                for node in all_nodes:
                    # prev_level_nodes
                    if node.metadata['level'] == self._chunk_levels[self._cur_level-1] and i < self._cur_processing_node_id:
                        i += 1
                    else:
                        prev_level_nodes.append(node)

                    # finished current nodes
                    if node.metadata['level'] == self._chunk_levels[self._cur_level]:
                        finished_sub_node.append(node)

        sub_nodes = finished_sub_node

        print(f'cur level: {self._cur_level}')
        print(f'cur node id: {self._cur_processing_node_id}')
        print(f'prev nodes: {len(prev_level_nodes)}')
        print(f'sub nodes: {len(sub_nodes)}')

        cur_level = self._cur_level
        for level in range(cur_level, len(self._chunk_levels)):
            self._cur_level = level
            # first split current nodes into sub-nodes
            nodes_with_progress = get_tqdm_iterable(
                prev_level_nodes, show_progress, f'{self._chunk_levels[level]} level parsing ...'
            )
            
            for node in nodes_with_progress:
                self.save_nodes(all_nodes)
                cur_sub_nodes = self._split_nodes(chunk_level=self._chunk_levels[level], node=node)

                # add parent relationship from sub node to parent node
                # add child relationship from parent node to sub node
                # NOTE: Only add relationships if level > 0, since we don't want to add
                # relationships for the top-level document objects that we are splitting
                if level > 0:
                    for sub_node in cur_sub_nodes:
                        _add_parent_child_relationship(
                            parent_node=node,
                            child_node=sub_node,
                        )
                sub_nodes.extend(cur_sub_nodes)
                self._cur_processing_node_id += 1
            
            prev_level_nodes = sub_nodes
            all_nodes.extend(sub_nodes)
            sub_nodes = []
            self._cur_processing_node_id = 0
            print(len(all_nodes))

        return all_nodes

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse documents into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
            show_progress (bool): whether to show progress bar

        """
        self._doc_id_to_document = {doc.id_: doc for doc in documents}

        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            all_nodes = self._get_nodes_from_nodes(
                nodes=documents,
                show_progress=show_progress
            )

            event.on_end({EventPayload.NODES: all_nodes})

        return all_nodes
    
    # Unused abstract method
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return list(nodes)

import os
from llama_index.core.storage.docstore import SimpleDocumentStore
class ManuallyParser():
    def __init__(self, cache_path, cache_name, delete_cache=True, force=False) -> None:
        self.cache_name = cache_name
        self.cache_path = cache_path
        self.delete_cache = delete_cache
        self.force = force

    def get_nodes_from_documents(self, nodes, **kwargs):
        nodes_cache_path = os.path.join(self.cache_path, f"{self.cache_name}_finished.json")
        if os.path.exists(nodes_cache_path) and not self.force:
            docstore = SimpleDocumentStore().from_persist_path(persist_path=nodes_cache_path)
            nodes = [node for _, node in docstore.docs.items()]
            if self.delete_cache:
                os.remove(nodes_cache_path)
            return nodes
        nodes_cache_path = os.path.join(self.cache_path, f"{self.cache_name}_{len(nodes)}_processing.json")
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        docstore.persist(persist_path=nodes_cache_path)
        print(f"\n[Manually Parser] Cache \'{self.cache_name}\' has been saved. Waiting for processing manually...")
        exit()