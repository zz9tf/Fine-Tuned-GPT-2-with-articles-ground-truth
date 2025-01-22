class TemplateSchema:
    prompt_template_ollama = [
        "questions_this_excerpt_can_answer_and_corresponding_answers_and_reasons",
"""\
You are an expert researcher. Using the following enhanced template, analyze the provided contextual \
information to generate the top {qar_num} critical academic questions that researchers in this field care about. \
For each question, provide a thorough and contextually relevant answer, and explain why these questions \
are significant to researchers and why the provided answers are accurate.

Here is the context:
{context_str}

Here is the QAR (Question, Answer, and Reason) Template:
----------------------------------------------------------------------------------
<Pair number, representing the sequence of QAR, such as 1, 2, 3>
Question:
<Frame a specific and unique question based on the context that addresses a crucial aspect of the topic. \
Ensure the question is detailed and unlikely to be found elsewhere.>
Reason:
<Explain why this question is important to researchers in this field. Discuss the relevance and uniqueness \
of the question and answer, and why such insights are unlikely to be found elsewhere. Highlight the novelty \
and significance of addressing this specific question within the context.>
Answer:
<Provide a detailed and contextually accurate answer to the question, incorporating insights and data from \
the given context. Ensure the answer is comprehensive and directly addresses the question.>
----------------------------------------------------------------------------------
"""
]

    system_prompt = """You are a highly knowledgeable reasearch expert"""

#     prompt_template_openai=[
#         "questions_this_excerpt_can_answer_and_corresponding_answers_and_reasons",
# """\
# Here is the context:
# {context_str}

# Using this context, generate 5 specific questions that this context can uniquely answer. Ensure that these questions:
# 1. Are directly related to the provided context.
# 2. Highlight unique information or insights from the context.
# 3. Cannot be easily answered by general knowledge.
# ----------------------------------------------------------------------------------
# Pair Number of Question, such as 1, 2, or 3.
# Question:<Question content, you should place a specific question which is unlikely to be found elsewhere and is unique comparing with other questions>

# Answer:<Answer content, you should place a specific answer combining with the offered context>

# Reason:<Reason content, you should explain why this question and answer are unlikely to be found elsewhere and are unique comparing with each other>
# ----------------------------------------------------------------------------------
# Higher-level summaries of surrounding context may be provided \
# as well. Try using these summaries to generate better questions \
# that this context can answer.
# """
# ]
    prompt_metadata_key_openai = "questions_this_excerpt_can_answer_and_corresponding_answers_and_reasons"
    prompt_template_openai = """\
Here is the context:
{context_str}

Using this context, generate {qar_num} specific questions that this context can uniquely answer. Ensure that these questions:
1. Are directly related to the provided context.
2. Highlight unique information or insights from the context.
3. Cannot be easily answered by general knowledge.

Higher-level summaries of surrounding context may be provided \
as well. Try using these summaries to generate better questions \
that this context can answer.
"""
    
    DEFAULT_QUESTION_GEN_TMPL="""\
Here is the context:
{context_str}

Given the contextual information, \
generate {num_questions} questions this context can provide \
specific answers to which are unlikely to be found elsewhere.

Higher-level summaries of surrounding context may be provided \
as well. Try using these summaries to generate better questions \
that this context can answer.
"""

    tree_summary_section_q_Tmpl="""\
You are an expert academic researcher. Generate an abstractive summary of the given research content. \
Limit the number of sentences in the summary to a maximum of three.
"""

    tree_summary_summary_Tmpl="""\
You are an expert researcher. Generate an abstractive summary of one or two most important points from the given text. Start with \
a numbered list format, like \'1.\' and \'2.\'.\
"""

    tree_summary_qa_Tmpl="""\
{query_str}
-----------------------------------------------------------------------------------
Here is the content:
{context_str}
-----------------------------------------------------------------------------------
"""

class LLMTemplate:
    tmpl = """\
System: You are an advanced language model designed to provide expert, high-quality responses. Your task \
is to understand the user's input and generate an appropriate response.\n\
User: {query_str}\n\
Response:"""

######################################################################################
# llama_index method
# from pydantic import BaseModel

# class QAR(BaseModel):
#     Question: str
#     Answer: str
#     Reason: str

######################################################################################
# langchain method
from pydantic import BaseModel, Field, validator
class QAR(BaseModel):
    Reason: str = Field(description="The reason for the answer")
    Question: str = Field(description="The question being asked")
    Answer: str = Field(description="The answer to the question")

    # Validator to check if the reason is provided and is not too short
    @validator("Reason")
    def reason_is_provided(cls, value):
        if len(value) < 10:
            raise ValueError("The reason is too short to be meaningful.")
        return value

    # Validator to ensure the question ends with a question mark
    @validator("Question")
    def question_ends_with_question_mark(cls, value):
        if not value.endswith("?"):
            raise ValueError("The question must end with a question mark ('?').")
        return value

    # Validator to check if the answer is not too short
    @validator("Answer")
    def answer_is_meaningful(cls, value):
        if len(value) < 5:
            raise ValueError("The answer is too short to be meaningful.")
        return value
    
class MultipleQARs(BaseModel):
    qars: list[QAR]
    
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

class LCTemp():
    parser = PydanticOutputParser(pydantic_object=MultipleQARs)

    prompt_template = PromptTemplate(
    template="""Based on the provided context, simulate how researchers might generate {qar_num} academic questions \
and find corresponding references to answer them. Each question should be something that can be directly answered by \
the context provided. Answer each question using only the information from the context, limiting answers to three \
sentences. Finally, explain why each question is important and how the context serves as the reference to provide \
the answers.

Context:
{context_str}
----------------------------------------------------------------------------------
{format_instructions}""",
    input_variables=["qar_num", "context_str"],
    partial_variables={"format_instructions": parser.get_format_instructions() + "\nOutput a valid JSON object but do not repeat the schema."},
)

class A(BaseModel):
    Reason: str = Field(description="The reason why answer answers the question")
    Answer: str = Field(description="The answer to the question")

    # Validator to check if the answer is not too short
    @validator("Reason")
    def reason_is_meaningful(cls, value):
        if len(value) < 5:
            raise ValueError("The reason is too short to be meaningful.")
        return value
    
    @validator("Answer")
    def answer_is_meaningful(cls, value):
        if len(value) < 5:
            raise ValueError("The answer is too short to be meaningful.")
        return value

class Gen_Dataset_Temp():
    parser = PydanticOutputParser(pydantic_object=A)
    
    prompt_template = PromptTemplate(
    template="""Using the provided context, provide a complete and concise answer based solely \
on the bullet-point contexts. Limit your response to three sentences. If the contexts are irrelevant, \
reply with "No related contexts found." Also, justify why and how the context provides a sufficient \
answer or clarify why there is a lack of relevance, ensuring the quality of the response.

Question:
{query_str}

Context:
{context_str}
----------------------------------------------------------------------------------
{format_instructions}""",
    input_variables=["context_str", "query_str"],
    partial_variables={'format_instructions': parser.get_format_instructions()+"\nOutput a valid JSON object but do not repeat the schema."}
)
    
from llama_index.core import PromptTemplate

old_gen_temp = PromptTemplate(
    """Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: \n
"""
)