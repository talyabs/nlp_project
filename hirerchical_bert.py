import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn import pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from torch.quantization import quantize_dynamic
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fmt = (
    "%(asctime)s - %(process)s-%(thread)-6.6s - %(filename)-15.15s %(funcName)-15.15s "
    "%(lineno)-4.4s - %(levelname)-6.6s - %(message)s"
)
logging.basicConfig(level=logging.INFO, format=fmt)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

### Create an output directory
output_dir = "./model1_outputs"
if not os.path.exists(output_dir):  ### If the file directory doesn't already exists,
    os.makedirs(output_dir)  ### Make it please


class BertClassifier:
    def __init__(
        self,
        model_name,
        path="/data/talya/nlp_project/data/clean_df_for_modeling.csv",
        text="headline_text",
        target_label_level=1,
        frac=1,
        num_epochs=3,
        batch_size=32,
        quantized_model=False,
        pruning=False,
        distilled_model=False,
    ):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size  # TODO: try 64
        self.target_label_level = target_label_level

        self.data = pd.read_csv(path)
        logging.info(f"Original data shape: {self.data.shape}")
        self.data = self.data.sample(frac=frac)
        logging.info(f"Data shape: {self.data.shape}")

        # creating instance of labelencoder
        self.data[["label_1", "label_2", "label_3", "label_4"]] = self.data[
            "label"
        ].str.split(".", expand=True)
        self.num_labels = self.data[f"label_{self.target_label_level}"].nunique()
        print("#### num_labels: ", self.num_labels)

        self.labelencoder = LabelEncoder()
        self.encoded_labels = self.labelencoder.fit_transform(
            self.data[f"label_{self.target_label_level}"]
        )
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

        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.train_labels),
            y=self.train_labels,
        )
        self.class_weights = torch.FloatTensor(class_weights).to(device)

        # Load the pre-trained BERT model for sequence classification
        if self.model_name == "distilbert":
            self.model_checkpoint = "distilbert-base-uncased"
            logging.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_checkpoint, use_fast=True
            )
            logging.info("Loading BERT model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_checkpoint, num_labels=self.num_labels
            )
            self.optimizer = AdamW(
                self.model.parameters(), lr=1e-5, no_deprecation_warning=True
            )

        if not self.model_checkpoint:
            logging.error("No model checkpoint specified")

        if quantized_model:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )

        # Set the optimizer and learning rate

    def data_processing(self):
        logging.info("Tokenizing data...")
        # Tokenize the input texts
        if self.model_name == "distilbert":
            train_encodings = self.tokenizer(
                list(self.train_texts), truncation=True, padding=True
            )
            test_encodings = self.tokenizer(
                list(self.test_texts), truncation=True, padding=True
            )
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

        # Create PyTorch DataLoader for training and testing sets

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def train_bert(
        self, quantized_model=False, pruned_model=False, distilled_model=False
    ):
        # Training loop
        logging.info("Training BERT model...")
        self.model.to(device)
        self.model.train()
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.num_epochs,
        )

        for epoch in range(self.num_epochs):  # adjust the number of epochs as needed
            logging.info(f" Epoch {epoch + 1} of {self.num_epochs}")
            for batch in self.train_loader:
                input_ids, attention_mask, labels = batch

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                self.optimizer.zero_grad()

                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = F.cross_entropy(
                    outputs.logits, labels, weight=self.class_weights
                )
                logits = outputs.logits

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

    def predict_category(self, models, label_levels, text):
        current_text = text
        category_parts = []

        for model, level_labels in zip(models, label_levels):
            nlp = pipeline(
                "text-classification",
                model=model.model,
                tokenizer=model.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
            prediction = nlp(current_text)[0]
            predicted_label = level_labels[prediction["label"].replace("LABEL_", "")]

            category_parts.append(predicted_label)
            current_text = f"{current_text} {predicted_label}"  # Update the text with the predicted label.

        return ".".join(category_parts)

    def evaluate_bert(self):
        logging.info("Evaluating BERT model...")
        self.model.eval()

        all_labels = []
        all_predicted_labels = []

        with torch.no_grad():
            for text, true_label in zip(self.test_texts, self.test_labels):
                # Tokenize the input text
                encoding = self.tokenizer(
                    text, truncation=True, padding=True, return_tensors="pt"
                ).to(device)

                # Get the model logits
                logits = (
                    self.model(
                        encoding["input_ids"], attention_mask=encoding["attention_mask"]
                    )
                    .logits.detach()
                    .cpu()
                    .numpy()
                )

                # Get the predicted label index
                predicted_label_idx = np.argmax(logits, axis=-1)

                all_labels.append(true_label)
                all_predicted_labels.append(predicted_label_idx[0])

        accuracy = sum(
            [
                1 if true == pred else 0
                for true, pred in zip(all_labels, all_predicted_labels)
            ]
        ) / len(all_labels)
        logging.info(f"Accuracy: {accuracy}")

        # Create a DataFrame to store the true labels and predicted labels
        final_df = pd.DataFrame(
            {"true_labels": all_labels, "pred_labels": all_predicted_labels}
        )
        final_df.to_csv(f"./prediction_bert_{self.target_label_level}.csv")

        # Print the classification report
        report = classification_report(all_labels, all_predicted_labels)
        logging.info(f"Classification report: {report}")

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_model(output_dir):
        model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        return model, tokenizer

    def bert_classify_headlines(self):
        self.data_processing()
        self.train_bert()
        self.evaluate_bert()


def hirerchical_model(frac=1):
    logging.info("Starting hirerchical model...")
    logging.info(" ### Level 1 Model Training")
    bert_level1 = BertClassifier(target_label_level=1, frac=frac, model_name=model)
    bert_level1.bert_classify_headlines()
    bert_level1.save_model("output_dir_level1")

    logging.info(" ### Level 2 Model Training")
    bert_level2 = BertClassifier(target_label_level=2, frac=frac, model_name=model)
    bert_level2.bert_classify_headlines()
    bert_level2.save_model("output_dir_level2")

    logging.info(" ### Level 3 Model Training")
    bert_level3 = BertClassifier(target_label_level=3, frac=frac, model_name=model)
    bert_level3.bert_classify_headlines()
    bert_level3.save_model("output_dir_level3")

    logging.info(" ### Level 3 Model Training")
    bert_level4 = BertClassifier(target_label_level=4, frac=frac, model_name=model)
    bert_level4.bert_classify_headlines()
    bert_level4.save_model("output_dir_level4")


if __name__ == "__main__":

    model = "distilbert"

    hirerchical_model(frac=1)
