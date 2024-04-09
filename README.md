# Fine-Tuned-GPT-2-with-articles-ground-truth
This repository targets at capture the ground truth of a set of articles related to some specific topic.

### Introduction

The Fine-Tuned GPT-2 with Articles Ground Truth repository aims to provide a framework for capturing accurate and reliable information form a collection of articles that pertain to a particular subject. By utilizing the GPT-2 language model, this respository enables to response relevant and coherent context based on the provided input.

PS: If you are wondering how to download articles related to one specific topic, you are weclome to visit this [rep](https://github.com/zz9tf/article_scraper).

### Project process

1. Data preprocessing

In this step, I have used Tools like PyPDF2. Clean the extracted text to remove noise, such as headers, footers, and any irrelevant content. 

![image](https://github.com/zz9tf/Fine-Tuned-GPT-2-with-articles-ground-truth/assets/77183284/227fd32d-64cd-4304-8b68-889092aea97b)

The following is part of data after clean: Then, I tokenize the text into smaller chunks, ensuring coherence and relevance. Apply techniques such as sentence splitting and paragraph segmentation to organize the data effectively. Remove any remaining noise or formatting artifacts.

2. Training and evaluate GPT model

Initialize the GPT-2 model with pre-trained weights. Define the fine-tuning objective, specifying the task (e.g., text generation) and the evaluation metrics. Split the dataset into training, validation, and test sets to monitor the model's performance. Fine-tune the GPT model using techniques like transfer learning, adjusting hyperparameters such as learning rate and batch size. Then, evaluate the fine-tuned model based on human eyes with some specific qeustions like what is "Monosaccharide"

Test samples:

![image](https://github.com/zz9tf/Fine-Tuned-GPT-2-with-articles-ground-truth/blob/main/images/QA%20example.png)
![image](https://github.com/zz9tf/Fine-Tuned-GPT-2-with-articles-ground-truth/assets/77183284/d9a6bbe2-8a96-4dfd-a6bf-9bf6e922e5df)


## Challenging face currently:

1. High quality input data

2. Insufficience evaluation about the performance of the model.

### Coming soon
1. Improving the number of the data and the size of the model according to scaling law.

2. Evaluate Model performance based on appropriate metrics.

### Contributing
Contributions to this repository are welcome! If you would like to contribute, please follow the guidelines outlined in the CONTRIBUTING.md file.

### Reference

This project highly refers this [project](https://python.plainenglish.io/i-fine-tuned-gpt-2-on-100k-scientific-papers-heres-the-result-903f0784fe65?gi=07b7320c472b), feel free to visit it for more details.

### Licence
This project is licensed under the [MIT License](/LICENSE)
