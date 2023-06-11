import logging

import clip
import pandas as pd
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32")
fmt = (
    "%(asctime)s - %(process)s-%(thread)-6.6s - %(filename)-15.15s %(funcName)-15.15s "
    "%(lineno)-4.4s - %(levelname)-6.6s - %(message)s"
)
logging.basicConfig(level=logging.INFO, format=fmt)


# Load CLIP model and preprocessor
def data_prep(path, frac=1, label="label"):
    data = pd.read_csv(path)
    logging.info(f"Original data shape: {data.shape}")
    data = data.sample(frac=frac)
    logging.info(f"Data shape: {data.shape}")
    data[["label_1", "label_2", "label_3", "label_4"]] = data["label"].str.split(
        ".", expand=True
    )
    labels = data[label]
    unique_labels = data[label].unique()
    headlines = data["headline_text"]
    return headlines, labels, unique_labels


def classify_headline(headline, categories, model):
    # logging.info(f"Classifying headline: {headline}")
    tokenized_headline = clip.tokenize(headline).to(device)
    tokenized_categories = clip.tokenize(categories).to(device)

    with torch.no_grad():
        headline_embedding = model.encode_text(tokenized_headline)
        category_embeddings = model.encode_text(tokenized_categories)
        cos_sim = torch.nn.functional.cosine_similarity(
            headline_embedding, category_embeddings, dim=-1
        )

    top_category_idx = cos_sim.argmax().item()
    # logging.info(f"Predicted category: {categories[top_category_idx]}")
    return categories[top_category_idx]


def clip_zerp_shot(headlines, labels, unique_labels):
    logging.info("Starting CLIP zero-shot classification")

    # Classify all headlines
    predicted_labels = []
    for headline in tqdm(headlines):
        predicted_labels.append(classify_headline(headline, unique_labels, model))

    # Print classification report
    report = classification_report(labels, predicted_labels)
    print(report)


if __name__ == "__main__":
    path = "/data/talya/nlp_project/data/clean_df_for_modeling.csv"
    headlines, labels, unique_labels = data_prep(path, frac=0.1, label="label")
    clip_zerp_shot(headlines, labels, unique_labels)
