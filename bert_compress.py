import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer)

# from sentence_transformers import SentenceTransformer

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
fmt = (
    "%(asctime)s - %(process)s-%(thread)-6.6s - %(filename)-15.15s %(funcName)-15.15s "
    "%(lineno)-4.4s - %(levelname)-6.6s - %(message)s"
)
logging.basicConfig(level=logging.INFO, format=fmt)

# model = SentenceTransformer('all-MiniLM-L6-v2')
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

### Create an output directory
output_dir = "./model1_outputs"
if not os.path.exists(output_dir):  ### If the file directory doesn't already exists,
    os.makedirs(output_dir)  ### Make it please


# Models to try
# sentence_transformers

# GPT
# GPT4: anon8231489123/gpt4-x-alpaca-13b-native-4bit-128g
# GPT3: TurkuNLP/gpt3-finnish-small

# BERT
# CNN
# CLIP?


class BertClassifier:
    def __init__(
        self,
        path="/data/talya/nlp_project/data/clean_df_for_modeling.csv",
        model_name="distilbert-base-uncased",
        label="label",
        text="headline_text",
        frac=1,
        quantized_model=True,
        pruning=False,
        distilled_model=False,
    ):

        self.label = label
        # Load the preprocessed data
        self.data = pd.read_csv(path)
        logging.info(f"Original data shape: {self.data.shape}")
        self.data = self.data.sample(frac=frac)
        logging.info(f"Data shape: {self.data.shape}")

        # creating instance of labelencoder
        self.data[["label_1", "label_2", "label_3", "label_4"]] = self.data[
            "label"
        ].str.split(".", expand=True)
        self.data["label_2_levels"] = self.data["label_1"] + "." + self.data["label_2"]
        self.num_labels = self.data[self.label].nunique() + 1
        print("#### num_labels: ", self.num_labels)

        self.model_checkpoint = model_name
        self.labelencoder = LabelEncoder()

        self.encoded_labels = self.labelencoder.fit_transform(self.data[self.label])
        # print(self.data[self.label].value_counts(normalize=True))

        (
            self.train_texts,
            self.test_texts,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(
            self.data[text],
            self.encoded_labels,
            test_size=0.2,
            random_state=42,
        )
        print("train_labels: ", self.train_labels.shape)
        print("test_labels: ", self.test_labels.shape)

        unique_classes = np.unique(self.train_labels)

        class_weights = class_weight.compute_class_weight(
            "balanced", classes=unique_classes, y=self.train_labels
        )
        class_weights_dict = dict(zip(unique_classes, class_weights))
        full_class_weights = [
            class_weights_dict.get(i, 1.0) for i in range(self.num_labels)
        ]
        self.class_weights = torch.FloatTensor(full_class_weights)

        # Load the DistilBERT tokenizer
        logging.info("Loading BERT tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_checkpoint, use_fast=True
        )

        # Load the pre-trained BERT model for sequence classification
        logging.info("Loading BERT model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint, num_labels=self.num_labels
        )
        self.model.to("cpu")
        if quantized_model:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
        if pruning:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(
                        module, name="weight", amount=0.3
                    )  # Apply 30% sparsity pruning

        # Set the optimizer and learning rate
        self.optimizer = AdamW(
            self.model.parameters(), lr=1e-5, no_deprecation_warning=True
        )

    def data_processing(self):
        logging.info("Tokenizing data...")
        # Tokenize the input texts
        train_encodings = self.tokenizer(
            list(self.train_texts), truncation=True, padding=True
        )
        test_encodings = self.tokenizer(
            list(self.test_texts), truncation=True, padding=True
        )

        # Create PyTorch DataLoader for training and testing sets
        logging.info("Creating PyTorch DataLoader...")
        self.train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_encodings["input_ids"]),
            torch.tensor(train_encodings["attention_mask"]),
            torch.tensor(self.train_labels),
        )
        self.test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(test_encodings["input_ids"]),
            torch.tensor(test_encodings["attention_mask"]),
            torch.tensor(self.test_labels),
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=32, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=32, shuffle=False
        )

    def train_bert(
        self, quantized_model=False, pruned_model=False, distilled_model=False
    ):
        # Training loop
        logging.info("Training BERT model...")
        self.model
        self.model.train()

        for epoch in range(3):  # adjust the number of epochs as needed
            logging.info(f" Epoch {epoch + 1} of 3")
            for batch in self.train_loader:
                input_ids, attention_mask, labels = batch

                input_ids = input_ids
                attention_mask = attention_mask
                labels = labels.type(torch.LongTensor)
                labels = labels
                self.optimizer.zero_grad()

                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = F.cross_entropy(
                    outputs.logits, labels, weight=self.class_weights
                )

                # loss = outputs.loss
                logits = outputs.logits
                # if not quantized_model:
                #     loss.backward()
                #     self.optimizer.step()

        # Evaluation loop
        logging.info("Evaluating BERT model...")
        self.model.eval()
        test_loss = 0
        correct = 0
        all_labels = []
        all_predicted_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids
                attention_mask = attention_mask
                labels = labels

                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                # loss = outputs.loss
                loss = F.cross_entropy(
                    outputs.logits, labels, weight=self.class_weights
                )
                logits = outputs.logits

                test_loss += loss.item()
                _, predicted_labels = torch.max(logits, dim=1)
                correct += (predicted_labels == labels).sum().item()
                all_labels.extend(labels.tolist())
                all_predicted_labels.extend(predicted_labels.tolist())

        accuracy = correct / len(self.test_dataset)
        logging.info(f"Accuracy: {accuracy}")

        # Convert tensors to lists
        # all_labels = all_labels.tolist()
        # all_predicted_labels = all_predicted_labels.tolist()

        # Ensure one-dimensional arrays or lists
        all_labels = np.array(all_labels).flatten()
        all_predicted_labels = np.array(all_predicted_labels).flatten()

        report = classification_report(all_labels, all_predicted_labels)
        logging.info(f"Classification report: {report}")

        logging.info("Saving to file...")
        final_df = pd.DataFrame(
            {"true_labels": all_labels, "pred_labels": all_predicted_labels}
        )
        final_df.to_csv(f"/data/talya/nlp_project/prediction_bert_{self.label}.csv")
        report = classification_report(
            all_labels,
            all_predicted_labels,
            target_names=self.data[self.label].unique(),
        )
        logging.info(f"Classification report: {report}")

    def bert_classify_headlines(self):
        self.data_processing()
        self.train_bert()


if __name__ == "__main__":
    bert = BertClassifier(
        label="label_1",
        frac=0.5,
        quantized_model=True,
        pruning=False,
        distilled_model=False,
    )
    bert.bert_classify_headlines()
