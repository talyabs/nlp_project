import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer

fmt = (
    "%(asctime)s - %(process)s-%(thread)-6.6s - %(filename)-15.15s %(funcName)-15.15s "
    "%(lineno)-4.4s - %(levelname)-6.6s - %(message)s"
)
logging.basicConfig(level=logging.INFO, format=fmt)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, labels_mapping):
        self.labels = [labels_mapping[label] for label in df["label_2_levels"]]
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt",
            )
            for text in df["headline_text"]
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Get a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Get a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(
        self, hidden_size: int, num_classes: int, max_seq_len: int, gpt_model_name: str
    ):
        super(SimpleGPT2SequenceClassifier, self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(hidden_size * max_seq_len, num_classes)

    def forward(self, input_id, mask):
        """
        Args:
                input_id: encoded inputs ids of sent.
        """
        gpt_out, _ = self.gpt2model(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size, -1))
        return linear_output


def train(model, train_data, val_data, learning_rate, epochs, labels_mapping):
    train, val = Dataset(train_data, labels_mapping), Dataset(val_data, labels_mapping)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        logging.info(f"Epoch {epoch_num + 1}")
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            model.zero_grad()

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

            print(
                f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}"
            )


def evaluate(model, test_data):

    test = Dataset(test_data, labels_mapping)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    # Tracking variables
    predictions_labels = []
    true_labels = []

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input["attention_mask"].to(device)
            input_id = test_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

            # add original labels
            true_labels += test_label.cpu().numpy().flatten().tolist()
            # get predicitons to list
            predictions_labels += output.argmax(dim=1).cpu().numpy().flatten().tolist()

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")
    return true_labels, predictions_labels


def gpt_classify():
    df = pd.read_csv("/data/talya/nlp_project/data/clean_df_for_modeling.csv")
    logging.info(f"Data shape: {df.shape}")
    df[["label_1", "label_2", "label_3", "label_4"]] = df["label"].str.split(
        ".", expand=True
    )
    df["label_2_levels"] = df["label_1"] + "." + df["label_2"]

    labels_mapping = dict()
    for idx, label in enumerate(df["label_2_levels"].unique()):
        labels_mapping[label] = idx
    df = df[["headline_text", "label_2_levels"]]
    num_classes = len(labels_mapping)

    EPOCHS = 1
    model = SimpleGPT2SequenceClassifier(
        hidden_size=768, num_classes=num_classes, max_seq_len=128, gpt_model_name="gpt2"
    )
    LR = 1e-5
    np.random.seed(112)
    df_train, df_val, df_test = np.split(
        df.sample(frac=1, random_state=35), [int(0.8 * len(df)), int(0.9 * len(df))]
    )

    logging.info(
        f"Train: {len(df_train)}, Validation: {len(df_val)}, Test: {len(df_test)}"
    )
    logging.info("Training the model...")
    train(model, df_train, df_val, LR, EPOCHS, labels_mapping)
    # save trained model
    torch.save(
        model.state_dict(),
        "/data/talya/nlp_project/models/gpt2-text-classifier-model.pt",
    )
    logging.info("Evaluating the model...")
    true_labels, pred_labels = evaluate(model, df_test)
    final_df = pd.DataFrame({"true_labels": true_labels, "pred_labels": pred_labels})
    logging.info("Saving to file...")
    final_df.to_csv("/data/talya/nlp_project/prediction_gpt2_first_level.csv")


if __name__ == "__main__":
    gpt_classify()
