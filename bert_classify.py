import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from torch.nn.utils import prune
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fmt = (
    "%(asctime)s - %(process)s-%(thread)-6.6s - %(filename)-15.15s %(funcName)-15.15s "
    "%(lineno)-4.4s - %(levelname)-6.6s - %(message)s"
)
logging.basicConfig(level=logging.INFO, format=fmt)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class BertClassifier:
    def __init__(
        self,
        model_name,
        path="/data/talya/nlp_project/data/clean_df_for_modeling.csv",
        label="label",
        text="headline_text",
        frac=1,
        num_epochs=4,
        batch_size=32,
        validation=False,
        quantized_model=False,
        pruning=False,
        distilled_model=False,
        label_threhold=None,
    ):
        self.label = label
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.validation = validation
        self.pruning = pruning
        self.distilled_model = distilled_model

        self.data = pd.read_csv(path)
        logging.info(f"Original data shape: {self.data.shape}")
        self.data = self.data.sample(frac=frac)
        logging.info(f"Data shape: {self.data.shape}")
        original_shape = self.data.shape

        # creating instance of labelencoder
        self.data[["label_1", "label_2", "label_3", "label_4"]] = self.data[
            "label"
        ].str.split(".", expand=True)
        self.data = self.data.drop_duplicates(subset=[text, label])
        logging.info(f"Data shape after duplication removal : {self.data.shape}")
        logging.info(f"Removed {original_shape[0] - self.data.shape[0]} rows")
        logging.info(
            f"Ratio of removed rows: {(original_shape[0] - self.data.shape[0])/original_shape[0]}"
        )

        self.data["label_2_levels"] = self.data["label_1"] + "." + self.data["label_2"]
        self.num_labels = self.data[self.label].nunique()
        print("#### num_labels: ", self.num_labels)

        # Label threshold to labels with under 1%, to be changed to "other"
        if label_threhold:
            label_share = self.data[label].value_counts(normalize=True)
            print("#### label_share: ", label_share)
            self.labels_to_change = label_share[label_share < label_threhold].index
            self.data[label] = self.data[label].apply(self.change_label)
            self.num_labels = self.data[self.label].nunique()
            print("#### after change num_labels: ", self.num_labels)

        self.labelencoder = LabelEncoder()
        self.encoded_labels = self.labelencoder.fit_transform(self.data[self.label])

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
        if self.validation:
            (
                self.train_texts,
                self.val_texts,
                self.train_labels,
                self.val_labels,
            ) = train_test_split(
                self.train_texts,
                self.train_labels,
                test_size=0.5,
                random_state=42,
            )
            logging.info(f"val_labels: {self.val_labels.shape}")
        logging.info(f"train_labels: {self.train_labels.shape}")
        logging.info(f"test_labels: {self.test_labels.shape}")

        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.train_labels),
            y=self.train_labels,
        )
        self.class_weights = torch.FloatTensor(class_weights).to(device)

        # Load the pre-trained BERT model for sequence classification
        if self.model_name == "distilbert":
            self.model_checkpoint = "distilbert-base-uncased"
            # Load the DistilBERT tokenizer
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

        elif self.model_name == "sentence-transformer":
            self.model_checkpoint = "all-MiniLM-L12-v2"
            logging.info("Loading Sentence Transformer model...")
            self.sentence_transformer = SentenceTransformer(self.model_checkpoint)
            self.optimizer = optim.Adam(self.sentence_transformer.parameters(), lr=1e-5)

        if not self.model_checkpoint:
            logging.error("No model checkpoint specified")

    def change_label(self, label):
        if label in self.labels_to_change:
            return "other"
        else:
            return label

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
        elif self.model_name == "sentence-transformer":
            train_embeddings = self.sentence_transformer.encode(list(self.train_texts))
            test_embeddings = self.sentence_transformer.encode(list(self.test_texts))
            self.train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(train_embeddings, dtype=torch.float32),
                torch.tensor(self.train_labels),
            )
            self.test_dataset = torch.utils.data.TensorDataset(
                torch.tensor(test_embeddings, dtype=torch.float32),
                torch.tensor(self.test_labels),
            )
            self.model = nn.Linear(train_embeddings.shape[1], self.num_labels)

        # Create PyTorch DataLoader for training and testing sets
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def train_sentence_tranformer(self):
        logging.info("Training BERT model...")
        self.model.to(device)
        self.model.train()
        for epoch in range(self.num_epochs):  # adjust the number of epochs as needed
            logging.info(f" Epoch {epoch + 1} of {self.num_epochs + 1}")
            for batch in self.train_loader:
                embeddings, labels = batch

                embeddings = embeddings.to(device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                self.optimizer.zero_grad()

                logits = self.model(embeddings)
                loss = F.cross_entropy(logits, labels, weight=self.class_weights)

                loss.backward()
                self.optimizer.step()

        logging.info("Evaluating BERT model...")
        self.model.eval()
        test_loss = 0
        correct = 0
        all_labels = []
        all_predicted_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                embeddings, labels = batch
                embeddings = embeddings.to(device)
                labels = labels.to(device)

                logits = self.model(embeddings)
                loss = F.cross_entropy(logits, labels, weight=self.class_weights)

                test_loss += loss.item()
                _, predicted_labels = torch.max(logits, dim=1)
                correct += (predicted_labels == labels).sum().item()
                all_labels.extend(labels.tolist())
                all_predicted_labels.extend(predicted_labels.tolist())

        accuracy = correct / len(self.test_dataset)
        logging.info(f"Accuracy: {accuracy}")
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

    def train_bert(
        self,
    ):
        # Training loop
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.info("Training BERT model...")
        self.model.to(device)
        self.model.train()
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.num_epochs,
        )

        train_loss_lst = []
        val_loss_lst = []
        train_acc_lst = []
        val_acc_lst = []
        for epoch in range(self.num_epochs):
            total_loss = 0
            total_correct = 0
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
                total_loss += loss.item()
                _, predicted_labels = torch.max(logits, dim=1)
                total_correct += (predicted_labels == labels).sum().item()

            train_loss = total_loss / len(self.train_loader)
            train_accuracy = total_correct / len(self.train_dataset)
            train_loss_lst.append(train_loss)
            train_acc_lst.append(train_accuracy)
            logging.info(
                f"Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}"
            )
        if self.validation:
            self.model.eval()
            val_loss = 0
            val_correct = 0
            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    outputs = self.model(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                    loss = F.cross_entropy(
                        outputs.logits, labels, weight=self.class_weights
                    )
                    val_loss += loss.item()
                    _, predicted_labels = torch.max(outputs.logits, dim=1)
                    val_correct += (predicted_labels == labels).sum().item()
            val_accuracy = val_correct / len(self.val_dataset)
            val_loss_lst.append(val_loss)
            val_acc_lst.append(val_accuracy)
            logging.info(
                f"Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}"
            )
            logging.info(
                f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}"
            )

        # Evaluation loop
        if self.pruning:
            classifier = self.model.classifier
            prune.l1_unstructured(classifier, name="weight", amount=0.2)
            prune.remove(classifier, "weight")

            # One more traning loop after pruning
            total_loss = 0
            total_correct = 0
            logging.info(f" -- Last training after Pruning")
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
                total_loss += loss.item()
                _, predicted_labels = torch.max(logits, dim=1)
                total_correct += (predicted_labels == labels).sum().item()

            train_loss = total_loss / len(self.train_loader)
            train_accuracy = total_correct / len(self.train_dataset)
            train_loss_lst.append(train_loss)
            train_acc_lst.append(train_accuracy)
            logging.info(
                f"Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}"
            )

        logging.info("Evaluating BERT model...")
        self.model.eval()
        test_loss = 0
        correct = 0
        all_labels = []
        all_predicted_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
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

        # Ensure one-dimensional arrays or lists
        all_labels = np.array(all_labels).flatten()
        all_predicted_labels = np.array(all_predicted_labels).flatten()

        logging.info("Saving to file...")
        final_df = pd.DataFrame(
            {"true_labels": all_labels, "pred_labels": all_predicted_labels}
        )
        final_df.to_csv(
            f"/data/talya/nlp_project/prediction_bert_{self.label}_{timestamp}.csv"
        )

        model_file_path = f"/data/talya/nlp_project/bert_model_{timestamp}.pt"
        if self.pruning:
            model_file_path = (
                f"/data/talya/nlp_project/bert_model_{timestamp}_pruning.pt"
            )

        # Save the model to the file
        torch.save(self.model.state_dict(), model_file_path)

        report = classification_report(
            all_labels,
            all_predicted_labels,
            target_names=self.data[self.label].unique(),
        )
        logging.info(f"Classification report: {report}")
        classification_report_file = (
            f"/data/talya/nlp_project/classification_report_{timestamp}.txt"
        )
        with open(classification_report_file, "w") as file:
            file.write(report)

    def bert_classify_headlines(self):
        self.data_processing()
        self.train_bert()

    def sentence_transformer_classify_headlines(self):
        self.data_processing()
        self.train_sentence_tranformer()


if __name__ == "__main__":
    # model = "sentence-transformer"
    model = "distilbert"

    bert = BertClassifier(
        label="label_1",
        frac=1,
        num_epochs=4,
        model_name=model,
        quantized_model=False,
        pruning=True,
        distilled_model=False,
        label_threhold=None,
    )
    if model == "distilbert":
        bert.bert_classify_headlines()
    if model == "sentence-transformer":
        bert.sentence_transformer_classify_headlines()
