# Fine-Tuned-GPT-2-with-articles-ground-truth
This repository aims to capture the ground truth of a set of articles related to a specific topic by involving the fine-tuning of GPT models to capture the hidden knowledge graph within the group of articles.

### Introduction

The Fine-Tuned GPT-2 with Articles Ground Truth repository aims to provide a framework for capturing accurate and reliable information from a collection of articles that pertain to a particular subject. By utilizing the GPT-2 language model, this repository enables responses that are relevant and coherent based on the provided input.

PS: If you're wondering how to download articles related to a specific topic, you're welcome to visit this [repository](https://github.com/zz9tf/article_scraper).

### Project process

#### 1. Data preprocessing

In this step, I have used tools like PyPDF2 to clean the extracted text, removing noise such as headers, footers, and any irrelevant content. The following is a part of the data after cleaning: 

![image](https://github.com/zz9tf/Fine-Tuned-GPT-2-with-articles-ground-truth/assets/77183284/227fd32d-64cd-4304-8b68-889092aea97b)

Then, I tokenize the text into smaller chunks to ensure coherence and relevance. I apply techniques such as sentence splitting and paragraph segmentation to organize the data effectively and remove any remaining noise or formatting artifacts.

#### 2. Training and evaluate GPT model

Initialize the GPT-2 model with pre-trained weights. Define the fine-tuning objective, specifying the task (e.g., text generation) and the evaluation metrics. Split the dataset into training, validation, and test sets to monitor the model's performance. Fine-tune the GPT model using techniques like transfer learning, adjusting hyperparameters such as learning rate and batch size. Then, evaluate the fine-tuned model based on human evaluation, with specific questions such as "What is a monosaccharide?"

**Generation samples:**

##### (1) on a small subset of the data

![image](../../Demo/GPT2/QA%20example.png)

The image shown above is generated based on `do_sample=True`, which means the generated text is based on randomly selected possible words. Therefore, the result has significant hallucinations, or may even be unrelated to our question.

```
(.conda) (base) root@autodl-container-7b0e4cb38b-09225263:~/Fine-Tuned-GPT-2-with-articles-ground-truth/code# python main.py evaluate

Evaluating...
[Q] What is the Monosaccharide? 
<start> A: Monosaccharide is a chemical compound that has been shown to be effective in the treatment of various diseases. 
It has been shown to be effective in the treatment of various diseases. 
It has been shown to be effective in the treatment of various diseases. 
It has been shown to be effective in the treatment of various diseases. 
It has been shown to be effective in the treatment of various diseases. 
It has been shown to be effective in the treatment of various diseases. It has been<end>
```
The result shown above is generated based on `do_sample=False`, which means the generated text is based on highest posibility. Therefore, the result is more related to the question we provided. However, since gpt2 is a small model, it lacks the ability to infer long texts, causing it to forget its previous inferences or fail to produce longer and effective text.

##### (2) on bigger datasets
  
<!--![image](https://github.com/zz9tf/Fine-Tuned-GPT-2-with-articles-ground-truth/assets/77183284/d9a6bbe2-8a96-4dfd-a6bf-9bf6e922e5df)-->

```
(.conda) (base) root@autodl-container-7b0e4cb38b-09225263:~/Fine-Tuned-GPT-2-with-articles-ground-truth/code# python main.py evaluate

Evaluating...
[Q] What is the Monosaccharide?
<start>A. Glucose
B. Fructose
C. Sucrose
D. Maltose
CHECK GOOGLE PLA Y BOOKS FOR ANSWERS KEYSPRACTICEBOOK»NOTFORSALE2.2 Structure and Function of the Cell 254
8.Which of the following is not a constituent of
the cell membrane?
A. Carbohydrates
B. Lipids
C. Proteins
D. NucleicAcids
9<end>
```

This result is generated by a large dataset. The reason it considered my question as the start of a multiple-choice question and generated the rest of the multiple-choice question is that one document is a collection of exercises for such problems. Comparing with the previous result, the GPT-2 model is more closely related to the data I fed into it. GPT-2 learned better from the feed-in data. However, it also shows a lack of understanding of human questions. It's possible that it doesn't have samples for this task. As mentioned above, its size is not good enough for long sentence tasks. Inspired by the training process of the ChatGPT model, it also needs reinforcement learning to improve its performance to be closer to human daily communication methods.

### Challenges faced currently:

- Low quality input data
- Size is small for desire tasks
- Lack of communication methods that humans are accustomed to

#### Next Steps
- Improve input data quality using other Language Model (LM) techniques
- Enhance input data quality by incorporating additional high-quality corpora
- Increase the volume of data and scale the model according to scaling laws.
- Apply reinforcement learning technique to fine-tune the model

### Reference

This project highly refers this [project](https://python.plainenglish.io/i-fine-tuned-gpt-2-on-100k-scientific-papers-heres-the-result-903f0784fe65?gi=07b7320c472b), feel free to visit it for more details.

### Licence
This project is licensed under the [MIT License](/LICENSE)