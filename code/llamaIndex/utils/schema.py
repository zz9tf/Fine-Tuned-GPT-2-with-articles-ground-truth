class TemplateSchema:
    prompt_template_ollama = [
        "questions_this_excerpt_can_answer_and_corresponding_answers_and_reasons",
"""\
Here is the context:
{context_str}

Here is the format of question, answer, and reason(QAR) template:
----------------------------------------------------------------------------------
<Pair number, representing which QAR you are at, like 1, 2, 3>
Question:<Question content, you should place a specific question which is unlikely to be found elsewhere and is unique comparing with other questions>

Answer:<Answer content, you should place a specific answer combining with the offered context>

Reason:<Reason content, you should explain why this question and answer are unlikely to be found elsewhere and are unique comparing with each other>
----------------------------------------------------------------------------------

Following by this template, given the contextual information, generate 5 QAR.\
Higher-level summaries of surrounding context may be provided \
as well. Try using these summaries to generate better questions \
that this context can answer.
"""
]

    system_prompt = """\
You are a highly knowledgeable reasearch assistant tasked with generating insightful questions, detailed answers, and \
thorough reasoning based on the provided parts of papers.\
"""

    prompt_template_openai=[
        "questions_this_excerpt_can_answer_and_corresponding_answers_and_reasons",
"""\
Here is the context:
{context_str}

Using this context, generate 5 specific questions that this context can uniquely answer. Ensure that these questions:
1. Are directly related to the provided context.
2. Highlight unique information or insights from the context.
3. Cannot be easily answered by general knowledge.
----------------------------------------------------------------------------------
Pair Number of Question, such as 1, 2, or 3.
Question:<Question content, you should place a specific question which is unlikely to be found elsewhere and is unique comparing with other questions>

Answer:<Answer content, you should place a specific answer combining with the offered context>

Reason:<Reason content, you should explain why this question and answer are unlikely to be found elsewhere and are unique comparing with each other>
----------------------------------------------------------------------------------
Higher-level summaries of surrounding context may be provided \
as well. Try using these summaries to generate better questions \
that this context can answer.
"""
]
    
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
You are an expert researcher. Select the two most important sentences from the given text. Start with \
a numbered list format, like \'1.\' and \'2.\'.\
"""

    tree_summary_document_q_Tmpl="""\
You are an expert researcher. Generate an abstractive summary of the given research paper content. \
Limit the number of sentences in the summary to a maximum of three.
"""

#     tree_summary_document_q_Tmpl="""\
# You are an expert researcher. Generate an abstractive summary of the given research paper content. \
# Limit the number of sentences in the summary to one.
# """

    tree_summary_qa_Tmpl="""\
{query_str}
-----------------------------------------------------------------------------------
Here is the content:
{context_str}
-----------------------------------------------------------------------------------
"""