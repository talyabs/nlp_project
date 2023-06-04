import logging

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

fmt = (
    "%(asctime)s - %(process)s-%(thread)-6.6s - %(filename)-15.15s %(funcName)-15.15s "
    "%(lineno)-4.4s - %(levelname)-6.6s - %(message)s"
)

logging.basicConfig(level=logging.INFO, format=fmt)


class NBClassifier:
    def __init__(
        self,
        path="/data/talya/nlp_project/data/clean_df_for_modeling.csv",
        label="label",
        text="headline_text",
        frac=1,
    ):

        self.label = label
        self.text = text
        self.frac = frac

        self.data = pd.read_csv(path)
        self.data = self.data.sample(frac=frac)
        logging.info(f"Original data shape: {self.data.shape}")
        self.data[["label_1", "label_2", "label_3", "label_4"]] = self.data[
            "label"
        ].str.split(".", expand=True)
        self.data["label_2_levels"] = self.data.apply(self.concatenate_labels, axis=1)
        self.num_labels = self.data[self.label].nunique() + 1
        print("#### num_labels: ", self.num_labels)
        print("### column   ", self.data.columns)
        exit()

    def concatenate_labels(self, row):
        return (
            row["label_1"] + "." + row["label_2"]
            if not pd.isnull(row["label_2"])
            else row["label_1"]
        )

    def train(self):

        # Replace 'headline' and 'category' with the names of the columns in your data
        headlines = self.data[self.text]
        categories = self.data[self.label]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            headlines, categories, test_size=0.2, random_state=42
        )

        # Create a pipeline with CountVectorizer and MultinomialNB
        pipeline = Pipeline(
            [
                ("vectorizer", CountVectorizer(stop_words="english")),
                ("classifier", MultinomialNB()),
            ]
        )
        # X_train = X_train.apply(eval)
        # X_test = X_test.apply(eval)

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # one time with clean_lemmatized_headlines
    # one time with noraml text
    nb = NBClassifier(text="clean_lemmatized_headlines", frac=1, label="label")
    nb.train()
