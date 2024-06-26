from typing import Any, Dict, List, Sequence
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.extractors import BaseExtractor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.async_utils import run_jobs


TMPL = """\
Here is the context:
{context_str}

Given the contextual information, \
generate {num_questions} questions this context can provide \
specific answers to which are unlikely to be found elsewhere.

Higher-level summaries of surrounding context may be provided \
as well. Try using these summaries to generate better questions \
that this context can answer.

"""


class QAExtractor(BaseExtractor):
    """
    Questions answered extractor. Node-level extractor.
    Extracts `questions_this_excerpt_can_answer` metadata field.

    Args:
        llm (Optional[LLM]): LLM
        questions (int): number of questions to extract
        prompt_template (str): template for question extraction,
        embedding_only (bool): whether to use embedding only
    """

    def __init__(
        self,
        llm: LLM,
        questions: int = 5,
        embedding_only: bool = True,
        num_workers: int = 5,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if questions < 1:
            raise ValueError("questions must be >= 1")

        super().__init__(
            llm=llm,
            questions=questions,
            prompt_template=TMPL,
            embedding_only=embedding_only,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "QAExtractor"

    async def _aextract_questions_from_node(self, node: BaseNode) -> Dict[str, str]:
        """Extract questions from a node and return it's metadata dict."""
        if self.is_text_node_only and not isinstance(node, TextNode):
            return {}

        context_str = node.get_content(metadata_mode=self.metadata_mode)
        prompt = PromptTemplate(template=self.prompt_template)
        questions = await self.llm.apredict(
            prompt, num_questions=self.questions, context_str=context_str
        )

        return {"questions_this_excerpt_can_answer": questions.strip()}

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        questions_jobs = []
        for node in nodes:
            questions_jobs.append(self._aextract_questions_from_node(node))

        metadata_list: List[Dict] = await run_jobs(
            questions_jobs, show_progress=self.show_progress, workers=self.num_workers
        )

        return metadata_list