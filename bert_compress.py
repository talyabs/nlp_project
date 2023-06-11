import logging
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from transformers import (
    AdamW,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fmt = (
    "%(asctime)s - %(process)s-%(thread)-6.6s - %(filename)-15.15s %(funcName)-15.15s "
    "%(lineno)-4.4s - %(levelname)-6.6s - %(message)s"
)
logging.basicConfig(level=logging.INFO, format=fmt)

# model = SentenceTransformer('all-MiniLM-L6-v2')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

### Create an output directory
output_dir = "./model1_outputs"
if not os.path.exists(output_dir):  ### If the file directory doesn't already exists,
    os.makedirs(output_dir)  ### Make it please


class BertClassifier:
    def __init__(
        self,
        path="/data/talya/nlp_project/data/clean_df_for_modeling.csv",
        model_name="distilbert-base-uncased",
        label="label",
        text="headline_text",
        frac=1,
        epochs=3,
        quantized_model=True,
        pruning=False,
        distilled_model=False,
    ):

        self.label = label
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the preprocessed data
        self.data = pd.read_csv(path)
        logging.info(f"Original data shape: {self.data.shape}")
        self.data = self.data.sample(frac=frac)
        self.distilled_model = distilled_model
        self.quantized_model = quantized_model
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
        if self.quantized_model:
            self.model.to("cpu")
            device = torch.device("cpu")

        if self.distilled_model:
            self.student_model_tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/paraphrase-TinyBERT-L6-v2"
            )
            # self.student_model_checkpoint = AutoModel.from_pretrained(
            #     "sentence-transformers/paraphrase-TinyBERT-L6-v2"
            # )
            self.student_model_checkpoint = (
                "sentence-transformers/paraphrase-TinyBERT-L6-v2"
            )

            self.student_model = AutoModelForSequenceClassification.from_pretrained(
                self.student_model_checkpoint, num_labels=self.num_labels
            )
            self.student_model.to(self.device)
            self.student_optimizer = AdamW(self.student_model.parameters(), lr=1e-5)
            self.model.to(self.device)

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

    def data_processing_distilled_model(self):
        logging.info("Tokenizing data...")
        # Tokenize the input texts
        train_encodings = self.student_model_tokenizer(
            list(self.train_texts), truncation=True, padding=True
        )
        test_encodings = self.student_model_tokenizer(
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

    def train_distilled_model(self):
        # Disable gradient updates for the teacher model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        # Consider using a scheduler for learning rate
        scheduler = torch.optim.lr_scheduler.StepLR(
            self.student_optimizer, step_size=1, gamma=0.95
        )

        # Training loop for distillation
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in self.train_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(self.device)

                # Forward pass for the teacher model
                with torch.no_grad():
                    teacher_outputs = self.model(
                        input_ids, attention_mask=attention_mask
                    )
                teacher_logits = teacher_outputs.logits

                # Forward pass for the student model
                student_outputs = self.student_model(
                    input_ids, attention_mask=attention_mask
                )
                student_logits = student_outputs.logits

                # Compute distillation loss (for example, MSE between teacher and student logits)
                loss = F.mse_loss(student_logits, teacher_logits)

                self.student_optimizer.zero_grad()
                loss.backward()
                self.student_optimizer.step()

                total_loss += loss.item()

            # Adjust learning rate according to scheduler
            scheduler.step()

            print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.train_loader)}")

            # Evaluation loop for distillation
            self.student_model.eval()
            total_eval_loss = 0
            all_true_labels = []  # Collect all true labels
            all_pred_labels = []
            for batch in self.test_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(self.device)

                # Forward pass for the teacher model
                with torch.no_grad():
                    teacher_outputs = self.model(
                        input_ids, attention_mask=attention_mask
                    )
                teacher_logits = teacher_outputs.logits

                # Forward pass for the student model
                with torch.no_grad():
                    student_outputs = self.student_model(
                        input_ids, attention_mask=attention_mask
                    )
                student_logits = student_outputs.logits

                # Compute loss for evaluation
                loss = F.mse_loss(student_logits, teacher_logits)

                total_eval_loss += loss.item()
                pred_labels = torch.argmax(student_logits, dim=1)
                all_true_labels.extend(labels.tolist())
                all_pred_labels.extend(pred_labels.tolist())

            print(f"Evaluation Loss: {total_eval_loss/len(self.test_loader)}")

        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        print(f"Accuracy: {accuracy}")
        report = classification_report(all_true_labels, all_pred_labels)
        print(f"Report: \n{report}")
        report = classification_report(
            all_true_labels,
            all_pred_labels,
            target_names=self.data[self.label].unique(),
        )
        logging.info(f"Classification report: {report}")

        self.student_model.train()

        torch.save(
            self.model.state_dict(), "/data/talya/nlp_project/distiled_model.pth"
        )
        model_path = "/data/talya/nlp_project"
        self.student_model.save_pretrained(model_path)

    def train_bert(self):
        # Training loop
        logging.info("Training BERT model...")
        self.model
        self.model.train()

        for epoch in range(self.epochs):  # adjust the number of epochs as needed
            logging.info(f" Epoch {epoch + 1} of {self.epochs}")
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
                loss.backward()
                self.optimizer.step()
        if self.quantized_model:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
            torch.save(
                self.model.state_dict(), "/data/talya/nlp_project/quantized_model.pth"
            )

        # Evaluation loop
        logging.info("Evaluating BERT model...")
        self.model.eval()
        test_loss = 0
        correct = 0
        num_batches = 0

        all_labels = []
        all_predicted_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                num_batches += 1
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
        avg_test_loss = test_loss / num_batches
        accuracy = correct / len(self.test_dataset)

        logging.info(f"Average Test Loss: {avg_test_loss}")
        logging.info(f"Accuracy: {accuracy}")

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
        if self.distilled_model:
            self.data_processing_distilled_model()
            self.train_distilled_model()
        else:
            self.data_processing()
            self.train_bert()


if __name__ == "__main__":
    bert = BertClassifier(
        label="label_1",
        frac=1,
        quantized_model=False,
        pruning=False,
        distilled_model=True,
    )
    bert.bert_classify_headlines()
