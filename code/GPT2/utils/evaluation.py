import os
import numpy as np
import evaluate
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from transformers import GenerationConfig
from model.fine_tune_gpt2 import Fine_tune_gpt2

def __load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file = [line.strip() for line in file]
    return file

def generate_text(sequence, max_length=100, 
                  model_path=None, 
                  tokenizer_path=None, 
                  model=None, 
                  verbose=False):
    
    output_dir = os.getenv("output_dir")
    model_path = os.path.join(output_dir, 'final') if model_path==None else model_path
    tokenizer_path = os.path.join(output_dir, 'tokenizer') if tokenizer_path==None else tokenizer_path
    if (model == None):
        model = Fine_tune_gpt2()
        model = model.load_model(model_path)
    tokenizer = model.load_tokenizer(tokenizer_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    generation_config = GenerationConfig(
        max_new_tokens=max_length,
        early_stopping=True,
        num_beams = 3,
        top_k=10,
        top_p=0.80,
        remove_invalid_values = True,
        pad_token_id = model.config.eos_token_id,
        eos_token_id = model.config.eos_token_id,
    )

    final_outputs = model.generate(
        inputs=ids,
        generation_config = generation_config,
        return_dict_in_generate=True, 
        output_scores=True
    )
    
    transition_scores = model.compute_transition_scores(
        final_outputs.sequences, final_outputs.scores, normalize_logits=True
    )[0]

    generated_tokens = final_outputs.sequences[0]
    if verbose:
        for tok, score in zip(generated_tokens, transition_scores):
            # | token | token string | log probability | probability
            print(f"| {tok:5d} | {tokenizer.decode(tok):15s} | {score.numpy():6.2f} | {np.exp(score.numpy()):.2%}")

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)[len(sequence)+1:]

def generate_test_answer_candidate(question_path, answer_path):
    questions = __load_file(question_path)
    answers = []
    for q in questions:
        answer = generate_text(sequence=q)
        answers.append(answer)
    with open(answer_path, "w") as f:
        f.write("\n".join(answers))

def evaluation_performance():
    reference_responses = __load_file(os.getenv("reference_answers_path"))
    candidate_responses = __load_file(os.getenv("candidate_answers_path"))

    # Tokenization (assuming responses are already tokenized)
    reference_tokenized = [response.split() for response in reference_responses]
    candidate_tokenized = [response.split() for response in candidate_responses]

    # Calculate BLEU score
    bleu_score = corpus_bleu(reference_tokenized, candidate_tokenized, smoothing_function=SmoothingFunction().method1)

    print("BLEU Score:", bleu_score)

    # Calculate ROUGE score
    rouge = evaluate.load('rouge')
    result = rouge.compute(predictions=candidate_responses, references=reference_responses)

    print("Rouge\n    score1: {}, score2: {}\n    rougeL: {}, rougeLsum: {}".format(result["rouge1"], result["rouge2"], result["rougeL"], result["rougeLsum"]))