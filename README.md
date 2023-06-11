# Text Classification Project

This repository contains the code and resources for a text classification project aimed at categorizing headlines from the UK. The goal of the project is to classify headlines into various categories based on their content.

## Dataset
The dataset used for this project consists of 1.6 million headlines collected from various sources in the UK. The dataset is available on Kaggle and can be accessed [here](https://www.kaggle.com/datasets/therohk/ireland-historical-news?select=ireland-news-headlines.csv). It includes 103 unique labels, representing the different categories to which the headlines belong.

## Models
The project explores the performance of five different models for text classification:

1. Naive Bayes: This model serves as a baseline and provides a simple yet efficient approach for classification tasks.
2. Sentence Transformer: The Sentence Transformer model employs semantic embeddings to capture the meaning of sentences and achieve powerful text classification results.
3. CLIP Zero-Shot: CLIP is a robust model trained on text-image pairs. In this project, we utilize the zero-shot capabilities of CLIP to classify headlines based on similarities with label descriptions.
4. DistilBERT: DistilBERT is a distilled version of BERT that offers comparable performance while being faster and smaller. It is trained on the same corpus as BERT and performs well on various text classification tasks.
5. GPT2: GPT2 is an advanced text generation model that exhibits strong performance in natural language processing tasks, including text classification.

## Usage
To use this code, please follow the instructions below:

1. Clone the repository to your local machine.
2. Install the required dependencies listed in the `requirements.txt` file.
3. Download the dataset from the provided Kaggle link and place it in the appropriate directory.
4. Preprocess the dataset to prepare it for training. Ensure that the dataset is in the appropriate format and has the necessary labels.
5. Choose the desired model for training by uncommenting the corresponding code section in the main script.
6. Run the training script and monitor the progress and performance of the selected model.
7. Evaluate the trained model using the provided evaluation metrics and techniques.
8. Experiment with different hyperparameters, such as learning rate and batch size, to further improve the model's performance.

Please refer to the individual model files and the project documentation for more detailed instructions and information about each model's implementation.

## Acknowledgments
We would like to thank the contributors and developers of the open-source libraries and models used in this project. Their efforts and contributions have greatly facilitated the development and evaluation of the text classification models.