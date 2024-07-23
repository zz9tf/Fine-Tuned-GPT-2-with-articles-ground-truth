from typing import Dict, List
from llama_index.core.schema import BaseNode
from llama_index.core.llms import LLM
import nest_asyncio
import asyncio

nest_asyncio.apply()

class TreeSummarize():
    def __init__(
            self, 
            texts: List[str], 
            query_str: str, 
            summary_str: str,
            qa_prompt: str, 
            llm: LLM, 
            num_children: int,
            refine_times: int
        ):
        # Normal synchronous initialization
        self.texts = texts
        self.query_str: str = query_str
        self.summary_str: str = summary_str
        self.qa_prompt: str = qa_prompt
        self.llm: LLM = llm
        self.num_children: int = num_children
        self.response_txt: str = None
        self.prompt_records: Dict = dict()
        self.refine_times = refine_times

    @classmethod
    def from_defaults(
        cls,
        texts: List[str], 
        query_str: str, 
        summary_str: str,
        qa_prompt: str,
        llm: LLM, 
        num_children: int=10,
        refine_times: int=10
    ):
        self = cls(texts, query_str, summary_str, qa_prompt, llm, num_children, refine_times)
        return self
    
    def refine_response(self, text):
        i = 0
        while len(text) > 500 and i < self.refine_times:
            fmt_qa_prompt = self.qa_prompt.format(
                context_str=text, query_str=self.summary_str
            )
            new_text = str(self.llm.complete(fmt_qa_prompt)).strip()
            text = text if len(text) < len(new_text) else new_text
            i += 1
        return text

    async def arefine_response(self, text):
        return self.refine_response(text)

    def generate_response_hs(self):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.agenerate_response_hs())
        return result

    async def agenerate_response_hs(self):
        """Generate a response using hierarchical summarization strategy.

        Combine num_children nodes hierarchically until we get one root node.

        """
        self.prompt_records = {0: []}
        text_responses = []

        for text in self.texts:
            fmt_qa_prompt = self.qa_prompt.format(
                context_str=text, query_str=self.query_str
            )
            text_response = str(self.llm.complete(fmt_qa_prompt)).strip()
            text_response = text_response
            text_responses.append(text_response)
            self.prompt_records[0].append(fmt_qa_prompt)

        response_txt = await self.acombine_results([r for r in text_responses], 1)
        response_txt = await self.arefine_response(response_txt)

        return response_txt, self.prompt_records
    
    async def acombine_results(self, texts, level):
        new_texts = []
        cur_prompt_list = []
        self.prompt_records[level] = []
        for idx in range(0, len(texts), self.num_children):
            text_batch = texts[idx : idx + self.num_children]
            context_str = "\n\n".join([t for t in text_batch])
            fmt_qa_prompt = self.qa_prompt.format(
                context_str=context_str, query_str=self.query_str
            )
            combined_response = str(self.llm.complete(fmt_qa_prompt)).strip()
            new_texts.append(str(combined_response))
            self.prompt_records[level].append(fmt_qa_prompt)
            cur_prompt_list.append(fmt_qa_prompt)
        
        tasks = [self.llm.acomplete(p) for p in cur_prompt_list]
        combined_responses = await asyncio.gather(*tasks)
        new_texts = [str(r) for r in combined_responses]

        if len(new_texts) == 1:
            return new_texts[0]
        else:
            return await self.acombine_results(new_texts, level+1)


