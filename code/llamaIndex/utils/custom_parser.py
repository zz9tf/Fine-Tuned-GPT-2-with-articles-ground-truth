from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import BaseNode, Document, NodeRelationship, MetadataMode
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.node_parser.node_utils import build_nodes_from_splits

def _add_parent_child_relationship(parent_node: BaseNode, child_node: BaseNode) -> None:
    """Add parent/child relationship between nodes."""
    child_list = parent_node.relationships.get(NodeRelationship.CHILD, [])
    child_list.append(child_node.as_related_node_info())
    parent_node.relationships[NodeRelationship.CHILD] = child_list

    child_node.relationships[
        NodeRelationship.PARENT
    ] = parent_node.as_related_node_info()

class CustomHierarchicalNodeParser(NodeParser):
    """Hierarchical node parser.

    Splits a document into a recursive hierarchy Nodes using a NodeParser.

    NOTE: this will return a hierarchy of nodes in a flat list, where there will be
    overlap between parent nodes (e.g. with a bigger chunk size), and child nodes
    per parent (e.g. with a smaller chunk size).
    """

    chunk_levels: Optional[List[str]] = Field(
        default=None,
        description=(
            "The chunk level to use when splitting documents: document, section, paragraph, multi-sentences"
        ),
    )

    _doc_id_to_document: Dict[str, Document] = PrivateAttr()

    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "CustomHierarchicalNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        chunk_levels = ["document", "section", "paragraph", "multi-sentences"]

        return cls(
            chunk_levels=chunk_levels,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )
    
    @classmethod
    def class_name(cls) -> str:
        return "CustomHierarchicalNodeParser"

    def _get_document_node_from_document(
        self,
        document: Document,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []

        all_nodes.extend(
            # build node from document
            build_nodes_from_splits(
                [document.get_content(metadata_mode=MetadataMode.NONE)],
                document,
                id_func=self.id_func,
            )
        )
            
        return all_nodes

    def _get_section_nodes_from_document_node(
        self, 
        document_node: BaseNode,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        parent_document = self._doc_id_to_document.get(document_node.ref_doc_id, None)
        if parent_document is None:
            raise Exception(f"Parent document is not found with id {document_node.ref_doc_id}")
        print(parent_document.metadata['sections'])
        exit()
        
        # TODO update split method
        splits = self.split_text(document_node.get_content())

        all_nodes.extend(
            build_nodes_from_splits(splits, document_node, id_func=self.id_func)
        )

        return all_nodes

    def _get_paragraph_nodes_from_section_node(
        self,
        section_node: BaseNode
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        
        # TODO update split method
        splits = self.split_text()

        all_nodes.extend(
            build_nodes_from_splits(splits, section_node, id_func=self.id_func)
        )

        return all_nodes

    def _get_multiple_sentences_nodes_from_paragraph_node(
        self,
        paragraph_node: BaseNode
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        
        # TODO update split method
        splits = self.split_text(paragraph_node.get_content())

        all_nodes.extend(
            build_nodes_from_splits(splits, paragraph_node, id_func=self.id_func)
        )

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

                node.metadata.update(
                    {k: v for k, v in parent_doc.metadata.items() if k not in ['sections']}
                )
                    
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

    def _recursively_get_nodes_from_nodes(
        self,
        nodes: List[BaseNode],
        level: int,
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Recursively get nodes from nodes."""
        if level >= len(self.chunk_levels):
            raise ValueError(
                f"Level {level} is greater than number of text "
                f"splitters ({len(self.chunk_levels)})."
            )

        # first split current nodes into sub-nodes
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, f'{self.chunk_levels[level]} level parsing ...'
        )
        sub_nodes = []
        for node in nodes_with_progress:
            cur_sub_nodes = self._split_nodes(chunk_level=self.chunk_levels[level], node=node)

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

        # now for each sub-node, recursively split into sub-sub-nodes, and add
        if level < len(self.chunk_levels) - 1:
            sub_sub_nodes = self._recursively_get_nodes_from_nodes(
                sub_nodes,
                level + 1,
                show_progress=show_progress,
            )
        else:
            sub_sub_nodes = []

        return sub_nodes + sub_sub_nodes

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
            
            all_nodes = self._recursively_get_nodes_from_nodes(
                nodes=documents,
                level=0,
                show_progress=show_progress
            )

            event.on_end({EventPayload.NODES: all_nodes})

        return all_nodes
    
    # Unused abstract method
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return list(nodes)