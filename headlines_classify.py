import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
import logging
logging.basicConfig( level=logging.INFO)
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Models to try
# sentence_transformers

# GPT
# GPT4: anon8231489123/gpt4-x-alpaca-13b-native-4bit-128g
# GPT3: TurkuNLP/gpt3-finnish-small

# BERT
# CNN
# CLIP?



# Load the preprocessed data
data = pd.read_csv('/data/talya/nlp_project/data/clean_df_for_modeling.csv')
#data = data[2000:4000]
# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
#data['label_numerical'] = labelencoder.fit_transform(data['label'])
data[['label_1', 'label_2', 'label_3', 'label_4']] = data['label'].str.split('.', expand=True)
#data['label_numerical'] = labelencoder.fit_transform(data['label_1'])

encoded_labels = labelencoder.fit_transform(data['label_1'])

train_texts, test_texts, train_labels, test_labels = train_test_split(data['headline_text'], encoded_labels, test_size=0.2, random_state=42)
def bert_classify(train_texts, test_texts, train_labels, test_labels):
    logging.info("Loading BERT tokenizer...")

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    #model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

    logging.info("Tokenizing data...")
    # Tokenize the input texts
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

    # Create PyTorch DataLoader for training and testing sets
    logging.info("Creating PyTorch DataLoader...")
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                                torch.tensor(train_encodings['attention_mask']),
                                                torch.tensor(train_labels))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']),
                                                torch.tensor(test_encodings['attention_mask']),
                                                torch.tensor(test_labels))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load the pre-trained BERT model for sequence classification
    logging.info("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Set the optimizer and learning rate
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training loop
    logging.info("Training BERT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    for epoch in range(3):  # adjust the number of epochs as needed
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

    # Evaluation loop
    logging.info("Evaluating BERT model...")
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            test_loss += loss.item()
            _, predicted_labels = torch.max(logits, dim=1)
            correct += (predicted_labels == labels).sum().item()

    accuracy = correct / len(test_dataset)
    report = classification_report(test_labels, predicted_labels=predicted_labels,zero_division=0)

    logging.info('Accuracy:', accuracy)
    logging.info('Classification report:', report)


def gpt_classsify(train_texts, test_texts, train_labels, test_labels):
    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Tokenize the input texts
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

    # Create PyTorch DataLoader for training and testing sets
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                                torch.tensor(train_encodings['attention_mask']),
                                                torch.tensor(train_labels))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']),
                                                torch.tensor(test_encodings['attention_mask']),
                                                torch.tensor(test_labels))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load the pre-trained GPT-2 model for sequence classification
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)

    # Set the optimizer and learning rate
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    for epoch in range(3):  # adjust the number of epochs as needed
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

    # Evaluation loop
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            test_loss += loss.item()
            _, predicted_labels = torch.max(logits, dim=1)
            correct += (predicted_labels == labels).sum().item()

    accuracy = correct / len(test_dataset)
    print('Accuracy:', accuracy)

        

if __name__ == '__main__':
    bert_classify(train_texts, test_texts, train_labels, test_labels)