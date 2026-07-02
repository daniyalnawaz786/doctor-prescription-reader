"""Evaluate the fine-tuned model on the test split (CER / WER / accuracy)."""

import os

import pandas as pd
from tqdm.auto import tqdm

from src import config, metrics
from src.predict import PrescriptionReader


def evaluate(base_path, model_dir=config.MODEL_DIR):
    test_df = pd.read_csv(os.path.join(base_path, "Testing", "testing_labels.csv"))
    image_dir = os.path.join(base_path, "Testing", "testing_words")

    reader = PrescriptionReader(model_dir)

    references, hypotheses = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        img_path = os.path.join(image_dir, row[config.IMAGE_COL])
        references.append(str(row[config.MEDICINE_COL]))
        hypotheses.append(reader.predict(img_path))

    print(f"CER            : {metrics.cer(references, hypotheses):.4f}")
    print(f"WER            : {metrics.wer(references, hypotheses):.4f}")
    print(f"Exact accuracy : {metrics.exact_match_accuracy(references, hypotheses):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", required=True)
    args = parser.parse_args()
    evaluate(args.base_path)
